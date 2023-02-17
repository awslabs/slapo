# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
.. currentmodule:: slapo

Optimize MLP Module on Multi-Device
===================================

This guide uses the multi-layer perceptron (MLP) module, one of the 
basin components in Transformer-based models, as an example to show
how we can leverage Slapo to optimize its performance on multiple devices.
We will cover tensor parallelism, synchronization, and operator fusion
in this tutorial.
"""

# %%
# We first import the necessary packages. Make sure you have already installed
# the PyTorch framework.

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import slapo

# %%
# Since we will use multiple GPUs to run the model, we need to initialize the distributed
# backend. We only initialize the CPU backend for illustration purpose. Users can
# initialize the NCCL backend on GPU by passing in ``backend="nccl"``, and change
# the actual number of devices accordingly.

slapo.env.setup(rank=0, world_size=1, backend="gloo")
print(f"rank: {dist.get_rank()}, world_size: {dist.get_world_size()}")

# %%
# Model Definition
# ----------------
#
# We first define a MLP module that consists of two linear layers and a GELU activation,
# which is a basic component in Transformer-based models like GPT. Users can instantiate
# the module as usual.


class MLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, data):
        out = self.linear1(data)
        out = self.activation(out)
        out = self.linear2(out)
        return out


model = MLP(1024)

# %%
# Create Model Schedule
# ---------------------
#
# We then create a default schedule ``sch`` for the model. Users can always check the
# corresponding PyTorch model by calling ``sch.mod``.

sch = slapo.create_schedule(model)
print(sch.mod)

# %%
# Tensor Parallelism
# ------------------
#
# Here comes the most important part of transforming the single-device model to
# a parallelized one. Slapo provides a ``.shard()`` primitive to realize tensor
# parallelism. Users can specify the name of the tensor and the axis to shard the
# tensor along. We follow the convention of `Megatron-LM <https://arxiv.org/abs/1909.08053>`_
# to shard the weight :math:`A` in the first linear layer by column, and the
# weight :math:`B` in the second linear layer by row. Basically, the computation
# becomes as follows:
#
# .. math::
#   f(XA)B = f\left(X\begin{bmatrix}A_1 & A_2\end{bmatrix}\right) \begin{bmatrix}B_1 \\ B_2\end{bmatrix} =f(XA_1)B_1 + f(XA_2)B_2
#
# where :math:`X` is the input tensor. Since PyTorch's ``nn.Linear`` module by default
# transposes the weight matrix, ``axis=0`` means sharding the output dimension.
# As each device only holds a part of the result, we need to synchronize the results
# at the end of both forward and backward pass. We can also use ``.sync()`` to specify the
# synchronization point and strategy. Here we use ``all_reduce`` to synchronize the results
# after the second linear layer during forward pass, and insert another ``all_reduce``
# before the first linear layer during backward pass. Users only need to write the following
# several lines of code to realize complex tensor parallelism but have no need to care about
# the low-level implementation details.

sch["linear1"].shard("weight", axis=0)
sch["linear1"].shard("bias", axis=0)
sch["linear2"].shard("weight", axis=1)
sch["linear2"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")
sch["linear1"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
print(sch.mod)

# %%
# From the above output, we can see that the weight and bias of the linear layers
# are correctly sharded, where the output dimension of the first linear layer
# becomes half of the original one, and each device only holds half of the
# weight.

# %%
# Operator Fusion
# ---------------
#
# Another optimization we can do is to fuse the GELU activation with the first
# linear layer. We can use ``.decompose()`` to decompose the linear layer into
# a matrix multiplication and a bias addition. As shown in the output below,
# the ``nn.Linear`` layer is replaced with the predefined ``LinearWithSeparateBias``
# module.

sch["linear1"].decompose()
print(sch.mod)

# %%
# To enable operator fusion, we need a static dataflow graph. Here, we explicitly
# call ``.trace()`` to trace the module and break the linear layer into two separate
# multiply and add operators. Users can easily determine whether they want their
# dataflow graph to be flattened or not by just passing in a flag.

sch.trace(flatten=True)
print(sch.mod)

# %%
# Later, we define a pattern for matching the bias addition and GELU activation.
# Notice Slapo supports different types of patterns, including subgraphs with multiple
# inputs and fuzzy matching, which provides users enough flexibility to express
# their subgraphs.


def pattern(x, bias):
    x = F.gelu(bias + x)
    return x


subgraph = sch.find(pattern)
print(subgraph)

# %%
# As expected, the subgraph consists of two nodes, one for the bias addition and
# the other for the GELU activation. We can then fuse the subgraph into a single
# node by calling ``.fuse()``. By default, Slapo will use TorchScript (nvFuser)
# as the backend compiler.

sch.fuse(subgraph, compiler="TorchScript", name="BiasGeLU")
print(sch.mod)

# %%
# Build the Optimized Model
# -------------------------
#
# We can see the previous sharding optimization is still preserved, and the fused
# kernel is correctly inserted into the hierarchical module definition and the
# corresponding dataflow graph.
#
# Finally, we can build the optimized model by calling ``.build()``.

opt_model, _ = slapo.build(sch, init_weights=False)
