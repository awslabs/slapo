# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
.. currentmodule:: slapo

Debugging with Print
====================

Although Slapo only traces a sub-module when we have to schedule its computational
graph, it is still annoying to debug the numerical correctness of traced sub-modules.
One important reason is that the traced sub-module becomes a GraphModule, which
computational graph is the traced IR graph in torch.fx. It means the forward function
of the sub-module is only evaluated when generating torch.fx IR graph instead of
the model execution. Therefore, we cannot print the intermediate values of
the sub-module during runtime.

To solve this problem, we provide a custom module ``Print`` in Slapo. This module
is marked as a leaf in our tracer, which means it will be preserved in the traced
graph and can be evaluated in runtime.

In this turorial, we will show how to use ``Print`` to print the intermediate
values of a sub-module.
"""
# %%
# We first import the Slapo package. Make sure you have already installed PyTorch.

import torch
import torch.nn as nn
import slapo

# %%
# We define a MLP module that consists of two linear layers and a GELU activation
# as an example in this tutorial. You can notice that we add a ``Print`` module
# to print the intermediate output of the first Linear layer.


class MLPWithPrint(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.print = slapo.op.Print()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, data):
        out = self.linear1(data)
        out = self.print(out, "linear1 shape\n", out.shape, "\nvalue\n", out)
        out = self.activation(out)
        out = self.linear2(out)
        return out


# You may feel the usage of `self.print` looks weird. This is because `self.print`
# has to return a tensor, and the returned tensor has to be consumed by the
# next operator/module, making it a part of the dataflow graph; otherwise
# `self.print` will be removed by the tracer because it is dead code.
# From a dataflow's point of view, you can treat `out = self.print(out, ...)` as
# a statement of identical assignment (i.e., `out = out`).

# Starting from the second argument are the arguments of normal Python `print`.
# However, you have to make sure the values you printed are evaluated lazily.
# Specifically, in this example, we specify `out` in the 3rd argument instead of
# a part of the string in 2nd argument, so that it will be evaluated in runtime.
# We will show some incorrect usages of `self.print` in the end of this tutorial.

# %%
# Now let's create a schedule and trace the module.

model = MLPWithPrint(4)
sch = slapo.create_schedule(model)
sch.trace()

# And here is the traced torch.fx graph. We can see that `self.print` becomes
# an operator in the graph with the output of linear1 as its arguments.
print(sch.mod.code)

# We then build and execute the model:
model, _ = slapo.build(sch, init_weights=False)
data = torch.randn((2, 2, 4))
model(data)

# The linear1's output is printed!

# %%
# On the other hand, as we have mentioned above, the print won't work properly
# if the values you want to print are evaluated when tracing. Here is an example
# that shows incorrect usages of `self.print`.


class MLPWithWrongPrint(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.print = slapo.op.Print()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, data):
        out = self.linear1(data)
        out = self.print(out, f"print1: {out}")
        out = self.print(out, "print2: %s" % str(out))
        self.print(out, f"print3: {out}")
        out = self.activation(out)
        out = self.linear2(out)
        return out


# %%
# Again we create a schedule and trace the module.

model = MLPWithWrongPrint(4)
sch = slapo.create_schedule(model)
sch.trace()

# And here is the traced torch.fx graph.
print(sch.mod.code)

# We can see that the string to be prined in print1 and print2 are evaluated
# and fixed after tracing. Therefore, the printed values are always like "Proxy(...)"
# even if we execute the model:
model, _ = slapo.build(sch, init_weights=False)
data = torch.randn((2, 2, 4))
model(data)

# Also, print3 disappeared in the graph, because its return value is not consumed
# by the next operator/module and thus is dead code.
