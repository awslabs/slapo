"""
.. currentmodule:: slapo

Optimize Attention Module on A Single GPU
=========================================

This guide uses the `Attention <https://arxiv.org/abs/1706.03762>`_ module,
the core and most time-consuming module in Transformer-based models, as an
example to show how we can leverage Slapo to optimize its performance on
a single GPU.
"""

# %%
# We first import the necessary packages. Make sure you have already installed
# the PyTorch framework.

import torch
import torch.nn as nn
import torch.nn.functional as F
import slapo

# %%
# The Attention module is consisted of SelfAttention and Projection modules, where
# SelfAttention takes in the hidden states and pass it through three diffferent
# linear layers to generate the query, key and value tensors. Then, those tensors
# will be performed the following scaled dot-product attention:
# 
# $$ \mathrm{CoreAttention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt(d_k)}\right) * V $$
# 
# where $d_k$ is the hidden dimension. Finally, the output of the attention module
# will be passed through a linear projection layer, added with the residual
# connection, and conducted a layer norm to generate the final output.
# The following code shows the implementation of the Attention module.

def scaled_dot_product(q, k, v):
    # (bs, head, seq, hs // head)
    d_k = q.shape[-1]
    attn_score = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(d_k)
    # (bs, head, seq, seq)
    attn_probs = F.softmax(attn_score, dim=-1)
    attn_probs = F.dropout(attn_probs, 0.1)
    # (bs, head, seq, hs // head)
    attn = torch.matmul(attn_probs, v)
    return attn


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(1024, 1024)
        self.k_proj = nn.Linear(1024, 1024)
        self.v_proj = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(0.1)
        self.n_heads = 16

    def permute_for_scores(self, x):
        # x: (batch_size, seq_len, hidden_size)
        new_shape = x.shape[:-1] + (self.n_heads, -1)
        x = x.view(new_shape)
        # output: (bs, head, seq, hs // head)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # hidden_states: (batch_size, seq_len, hidden_size)
        # qkv layers
        q = self.permute_for_scores(self.q_proj(hidden_states))
        k = self.permute_for_scores(self.k_proj(hidden_states))
        v = self.permute_for_scores(self.v_proj(hidden_states))
        # core attention
        output = scaled_dot_product(q, k, v)
        # output: (bs, seq, head, hs // head)
        output.permute(0, 2, 1, 3)
        output.view(output.shape[0], output.shape[1], -1)
        return output


class Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(1024)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states, input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = SelfAttention()
        self.proj = Projection()

    def forward(self, hidden_states):
        self_output = self.self_attn(hidden_states)
        attention_output = self.proj(self_output, hidden_states)
        return attention_output

# %%
# Users can instantiate the model based on the above definition as usual.

model = Attention()

# %%
# Later, we pass the model to Slapo and create a default schedule for it.
# The schedule always includes the original or the transformed module.
# Users can check the module by calling the ``mod`` attribute.

sch = slapo.create_schedule(model)
print(sch.mod)

# %%
# As we can see, Slapo works seamlessly with the PyTorch models and preserves
# the hierachical structure of the original model. As we have not added any
# optimizations, the module is exactly the same as the original one.
# This is also the idea of progressive optimization -- we only apply optimizations
# to a small part of the model at a time and do not affect other parts.
# If no optimizations are applied, then no changes will be made to the model, which
# is different from the traditional static graph optimization employed by deep
# learning compilers.
# 
# In the following, we will show how to gradually apply optimizations to the model.
# Since the three linear layers in the SelfAttention module are independent, we
# can merge them into a single linear layer to reduce the number of GEMM
# operations, and thus reduce the GPU kernel launch overheads.
# 
# The first thing to do is to find those three linear layers and the consequential
# operations in the model. Slapo provides an easy-to-use API to help users
# define the pattern and find the corresponding module or subgraph in the model.
# We can define a subgraph pattern function as shown below. The ``call_module``
# function will try to match a call node that satisfies the user-defined
# constraint in the dataflow graph. The first argument specifies the name of the
# module, where regular expression is supported, so the it can support fuzzy
# matching in this case. The latter arguments are the arguments of the call node.
# Using this function, we can use just one line of code to match the three linear
# layers. Also, we need to incorporate the ``view`` and ``permute`` operations,
# which should also be fused together instead of doing three times separately.

from slapo.pattern import call_module

def pattern(x):
    x = call_module(r"[qkv]_proj", x)
    new_shape = x.shape[:-1] + (16, -1)
    x = x.view(new_shape)
    return x.permute(0, 2, 1, 3)

# %%
# After defining the pattern, we can use the ``.find()`` primitive to find the
# corresponding subgraph in the model. It will return a list of subgraphs that
# match the pattern. In our case, there will be three subgraphs, one for each
# linear layer and the ``view`` and ``permute`` operations.

subgraphs = sch.find(pattern)
print(subgraphs)

# %%
# We can later define a fused QKV layer to replace the three linear layers.
