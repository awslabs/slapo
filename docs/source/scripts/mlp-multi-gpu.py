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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import slapo
from slapo.logger import get_logger

logger = get_logger()

dist.init_process_group("nccl")
logger.info(f"rank: {dist.get_rank()}, world_size: {dist.get_world_size()}", ranks=0)

# %%
# The Attention module consists of SelfAttention and Projection modules, where
# SelfAttention takes in the hidden states and passes it through three different
# linear layers to generate the query, key and value tensors. Then, those tensors
# will be performed the following scaled dot-product attention:
#
# .. math::
#    \mathrm{CoreAttention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^{\mathrm{T}}}{\sqrt(d_k)}\right) * V
#
# where :math:`d_k` is the hidden dimension. Finally, the output of the attention
# module will be passed through a linear projection layer, added with the residual
# connection, and conducted a layer norm to generate the final output.
# The following code shows the implementation of the Attention module.


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
sch = slapo.create_schedule(model)
logger.info(sch.mod, ranks=0)

# %%
sch["linear1"].shard("weight", axis=0)
sch["linear1"].shard("bias", axis=0)
sch["linear1"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
sch["linear2"].shard("weight", axis=1)
sch["linear2"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")
logger.info(sch.mod, ranks=0)

# %%
sch["linear1"].decompose()
logger.info(sch.mod, ranks=0)

# %%
sch.trace(flatten=True)
logger.info(sch.mod, ranks=0)

# %%
def pattern(x, bias):
    x = F.gelu(bias + x)
    return x


subgraph = sch.find(pattern)
logger.info(subgraph, ranks=0)

# %%
sch.fuse(subgraph, compiler="TorchScript", name="BiasGeLU")
logger.info(sch.mod, ranks=0)

# %%
opt_model, _ = slapo.build(sch, init_weights=False)
