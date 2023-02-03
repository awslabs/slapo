"""
.. currentmodule:: slapo

Quick Start
===========

This guide walks through the key functionality of Slapo.
We will use the BERT model in `HuggingFace Hub <https://github.com/huggingface/transformers>`_ as an example and leverage Slapo to optimize its performance.
"""
# %%
# Optimize PyTorch model with Slapo
# ---------------------------------
# We first import the Slapo package. Make sure you have already installed the PyTorch package.

import slapo

# %%
# We then load a BERT model implemented in PyTorch from HuggingFace Hub.
# This is the model definition part.

from transformers import BertLMHeadModel, AutoConfig

config = AutoConfig.from_pretrained("bert-large-uncased")
bert = BertLMHeadModel(config)

# %%
# After we have the model defintion, we can create a default schedule `sch`.
# Later on, all the optimizations will be conducted on this schedule,
# and we do not need to directly modify the original model.
# The original module is stored in the :class:`~slapo.Schedule`, and can be accessed by `sch.mod`.

sch = slapo.create_schedule(bert)
print(sch.mod)

# %%
# From the above output, we can see that the original hierachical structure
# is preserved in Slapo schedule. Actually, since we have not added any
# optimizations, the model should be exactly the same as the vanilla PyTorch one.
# Users can leverage this structure to access the inner modules to
# conduct optimizations. For example, we can use the following code to access
# the first attention layer.

subsch = sch["bert.encoder.layer.0.attention"]
print(subsch.mod)

# %%
# The output submodule is just a part of the original one, but this helps users
# to quickly locate the submodule they need.

subsch = sch["bert.encoder.layer.0.intermediate"]
print(subsch.mod)

# %%
# We next optimize the Feed-Forward Network (FFN) part. We try to conduct operator
# fusion for linear bias and GeLU function. As we want to conduct other optimizations
# for linear weight (e.g., sharding), we cannot fuse it with consequential operators.
# Therefore, we need to decompose the bias from the `nn.Linear` module, 

subsch["dense"].decompose()
print(subsch.mod)

# %%
# Since operator fusion requires a static dataflow graph, we call the `.trace()` function
# to obtain the graph. As we want to get the inner operators of the linear layer, we also
# need to specify the `flatten` keyword in order to let the tracer trace into it.

subsch.trace(flatten=True)
print(subsch.mod)

# %%
# After we obtain the dataflow graph, we can define the fusion pattern and leverage
# `.find()` primitive to retrieve the subgraph.
# 
# .. note::
#   :class: margin
# 
#   The returned nodes are reprented in a tuple, where the first element is the
#   path of the node (i.e., its parent module's name), and the second element is
#   the actual `fx.Node`.

import torch.nn.functional as F
def bias_gelu_pattern(x, bias):
    return F.gelu(x + bias)

subgraphs = subsch.find(bias_gelu_pattern)
print(subgraphs)

# %%
# The output is a list of subgraphs that contains a list of nodes satisfying the
# pattern requirement. For this case, there are two operators, named `add` and `gelu`.
# That is what we want for the pattern. We then pass it into TorchScript compiler 
# and fuse the operators.

subsch.fuse(subgraphs, compiler="TorchScript", name="FusedBiasGeLU")
print(subsch.mod)

# %%
# We can see from the above result that the linear bias and GeLU function are indeed
# fused together and form a new module named `FusedBiasGeLU_0`.
