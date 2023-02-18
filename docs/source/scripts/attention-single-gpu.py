# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
.. currentmodule:: slapo

Optimize Attention Module on A Single Device
============================================

This guide uses the `Attention <https://arxiv.org/abs/1706.03762>`_ module,
the core and most time-consuming module in Transformer-based models, as an
example to show how we can leverage Slapo to optimize its performance on
a single device. We will cover module tracing, pattern matching, operator
fusion, and partial module replacement in this tutorial.
"""

# %%
# We first import the necessary packages. Make sure you have already installed
# the PyTorch framework.

import torch
import torch.nn as nn
import torch.nn.functional as F
import slapo

# %%
# Model Definition
# ----------------
#
# The Attention module consists of SelfAttention and Projection modules, where
# SelfAttention takes in the hidden states and passes it through three different
# linear layers to generate the query, key and value tensors. Then, those tensors
# will be performed the following scaled dot-product attention:
#
# .. math::
#    \mathrm{CoreAttention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^{\mathrm{T}}}{\sqrt{d_k}}\right) \cdot V
#
# where :math:`d_k` is the hidden dimension. Finally, the output of the attention
# module will be passed through a linear projection layer, added with the residual
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
    def __init__(self, hidden_size, n_heads):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.n_heads = n_heads

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
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super().__init__()
        self.self_attn = SelfAttention(hidden_size, n_heads)
        self.proj = Projection(hidden_size)

    def forward(self, hidden_states):
        self_output = self.self_attn(hidden_states)
        attention_output = self.proj(self_output, hidden_states)
        return attention_output


# %%
# Users can instantiate the model based on the above definition as usual.

model = Attention(hidden_size=1024, n_heads=16)

# %%
# Create Model Schedule
# ---------------------
#
# Later, we pass the model to Slapo and create a default schedule for it.
# The schedule always includes the original or the transformed module.
# Users can check the module by calling the ``mod`` attribute.

sch = slapo.create_schedule(model)
print(sch.mod)

# %%
# As we can see, Slapo works seamlessly with the PyTorch models and preserves
# the hierarchical structure of the original model. As we have not added any
# optimizations, the module is exactly the same as the original one.
# We can easily obtain the submodules by passing the module name to the schedule,
# which will return a new schedule for the submodule.

attn_sch = sch["self_attn"]
print(attn_sch.mod)

# %%
# This is also the idea of progressive optimization -- we only apply optimizations
# to a small part of the model at a time and do not affect other parts.
# If no optimizations are applied, then no changes will be made to the model, which
# is different from the traditional static graph optimization employed by deep
# learning compilers.
#
# In the following, we will show how to gradually apply optimizations to the model.

# %%
# Optimize SelfAttention Module
# -----------------------------
#
# Replace QKV Linear Layers
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Since the three linear layers in the SelfAttention module are independent, we
# can merge them into a single linear layer to reduce the number of GEMM
# operations, and thus reduce the kernel launch overheads.
#
# The first thing to do is to find those three linear layers and the consequential
# operations in the model. Slapo provides an easy-to-use API to help users
# define the pattern and find the corresponding module or subgraph in the model.
# We can define a subgraph pattern function as shown below. The ``call_module``
# function will try to match a call node that satisfies the user-defined
# constraint in the dataflow graph. The first argument specifies the name of the
# module, where regular expression is supported, so it can support fuzzy
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
# corresponding subgraph in the model.

qkv_subgraphs = attn_sch.find(pattern)

# %%
# The primitive basically does two things. First, it will `implicitly` trace the submodule
# into a static subgraph. Currently, we use `torch.fx <https://pytorch.org/docs/stable/fx.html>`_
# as the IR, so the traced module will become a ``torch.fx.GraphModule``, and we can also
# see the forward function of it.

print(attn_sch.mod)

# %%
# Second, the ``.find()`` primitive will return a list of subgraphs that
# match the pattern. In our case, there will be three subgraphs, one for each
# linear layer and the consequential ``view`` and ``permute`` operations.

print(qkv_subgraphs)

# %%
# Then, we define a fused QKV module as follows and instantiate it.


class FusedQKV(nn.Module):
    def __init__(self, hidden_size, n_heads) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.fused_linear = nn.Linear(hidden_size, hidden_size * 3)

    def permute_for_scores(self, x):
        new_shape = x.shape[:-1] + (self.n_heads, -1)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        qkv = self.fused_linear(hidden_states)
        reshaped_qkv = self.permute_for_scores(qkv)
        q, k, v = torch.split(reshaped_qkv, 1, dim=-1)
        q = torch.squeeze(q, -1).contiguous()
        k = torch.squeeze(k, -1).contiguous()
        v = torch.squeeze(v, -1).contiguous()
        return [q, k, v]


fused_qkv = FusedQKV(hidden_size=1024, n_heads=16)

# %%
# We can replace the subgraphs with the fused QKV module by calling the
# ``.replace()`` primitive. The first argument is the new module,
# and the second argument is the subgraph to be replaced.
# After replacing the subgraph, we can check the model again to see the
# changes.

attn_sch.replace(fused_qkv, qkv_subgraphs)
print(attn_sch.mod)

# %%
# From the above output, we can see there is a new module called ``FusedQKV_0``
# with :math:`3\times` ``out_features`` compared to the original linear layer.
# The corresponding forward function is also changed to leverage the fused
# module.

# %%
# Replace Scaled Dot-Product Attention
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Next, we still use the ``.find()`` primitive to find the core attention function
# and replace it with a more efficient implementation. Different from the QKV example
# that requires us to explicitly write the fuzzy pattern, we can directly write
# a function with the identical computation subgraph as the pattern. Since the
# ``scaled_dot_product`` function has been defined previously, we can reuse it
# and pass it into ``.find()``.

core_attn_subgraph = attn_sch.find(scaled_dot_product)
print(core_attn_subgraph)

# %%
# We can use the ``FlashAttentionOp`` provided by Slapo that makes use 
# of `flash attention <https://arxiv.org/abs/2205.14135>`_ kernels from
# `xFormers <https://github.com/facebookresearch/xformers>`_ and
# `flash-attention <https://github.com/HazyResearch/flash-attention>`_ libraries
# to replace the core attention. We directly import and replace the subgraph
# with ``FlashAttentionOp``.
# Notice, since the ``scaled_dot_product`` function we defined above only accepts
# the ``query``, ``key``, and ``value`` tensors, while ``FlashAttentionOp`` requires
# five arguments, so we need to explicitly pass ``None`` to the ``attention_mask``
# argument, and set the dropout probability ``p`` to 0.1 by setting the ``concrete_args``.
# 
# .. note::
#   :class: margin
#
#   We use ``native_xformers`` in this tutorial to demonstrate the functionality.
#   In reality, users can choose ``cutlass``, ``triton``, or ``cuda`` kernels to achieve
#   better performance, while the latter two only support NVIDIA V100 GPU.
#   Please refer to `slapo.op.attention.FlashAttentionOp` for more details.

from slapo.op.attention import FlashAttentionOp

flash_attn = FlashAttentionOp(attn_op_name="native_xformers", apply_causal_mask=False)
attn_sch.replace(
    flash_attn, core_attn_subgraph, concrete_args={"attention_mask": None, "p": 0.1}
)
print(attn_sch.mod)

# %%
# Again, the ``FlashAttentionOp`` is attached to the GraphModule, and the forward
# function becomes much simpler to call those two submodules.

# %%
# Optimize the Projection Module
# ------------------------------
#
# We then optimize the ``Projection`` module. A common practice is to fuse the
# dropout and the layer norm layer with those element-wise addition operations.
# We first create a subschedule for the ``Projection`` module.

proj_sch = sch["proj"]
print(proj_sch.mod)

# %%
# As we want to fuse the linear bias with the consequential layers, we need to
# decompose the linear layer into two separate matrix multiplication and bias add operations.
# In Slapo, this is easy to achieve by simply calling ``.decompose()`` on the linear module.
#
# .. note::
#   :class: margin
#
#   The default `nn.Linear <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_ module
#   in PyTorch will directly pass both weight and bias to the backend
#   `F.linear <https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html>`_ function,
#   and dispatch it to the corresponding C/CUDA library, so there is no way to fuse the bias if we
#   do not take it apart. Another reason for decomposing is that we can still optimize the
#   ``weight`` parameter later (e.g., sharding) even though the ``bias`` may be fused.

proj_sch["dense"].decompose()
print(proj_sch.mod)

# %%
# We can see the ``Linear`` module changed into ``LinearWithSeparateBias``, and other submodules
# remain the same. Next, we need to `explicitly` call the ``.trace()`` primitive to trace the module
# into a static subgraph. It gives us more control over the traced module. For example, we can
# pass in the ``flatten`` flag to let the tracer gets into each submodule so that the bias add
# can be depicted as a node in the subgraph.

proj_sch.trace(flatten=True)
print(proj_sch.mod)

# %%
# We can again define the fusion pattern as follows. Here the pattern includes three input arguments,
# Slapo can still handle it correctly and grab all the required nodes in the subgraph.


def ln_pattern(x, bias, residual):
    return F.layer_norm(F.dropout(x + bias) + residual, 1024)


ln_subgraph = proj_sch.find(ln_pattern)
print(ln_subgraph)

# %%
# For this case of vertical fusion, Slapo provides a ``.fuse()`` primitive to easily fuse the subgraph.
# Users can specify the backend fusion compiler and the name of the fused module. By default, Slapo
# will use TorchScript with nvFuser to fuse the subgraph.

proj_sch.fuse(ln_subgraph, compiler="TorchScript", name="FusedLayerNorm")
print(proj_sch.mod)

# %%
# As shown in the above output, the ``FusedLayerNorm`` module is attached to the GraphModule, and
# only ``torch._C._nn.Linear`` and ``FusedLayerNorm`` are called in the forward function.

# %%
# Build the Optimized Model
# -------------------------
#
# Finally, we finish all the optimizations for Attention module on a single device.
# We can pass the schedule into ``sch.build`` to build the optimized model for execution.
# It returns the optimized model and a default optimizer. We can print out the top-level module
# to see the changes. The optimizations are clearly reflected in the new module, and we still
# keep the module hierarchy, which greatly enhances the readability and debuggability of the code.

opt_model, _ = slapo.build(sch, init_weights=False)
print(opt_model)
