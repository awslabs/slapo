# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention module using high efficient CUDA kernels.

The flash-attention kernel is tested with:
https://github.com/jfc4050/flash-attention/commit/3676bd2

The xFormers kernel is tested with:
https://github.com/facebookresearch/xformers/commit/48a77cc

If you encounter an error when using above kernels, please check if the
commit hash is the same as the one we tested with.
"""
# pylint: disable=too-many-arguments, too-many-instance-attributes
from __future__ import annotations

import math
from functools import partial
from typing import Optional

import torch
from torch import nn

from ..logger import get_logger
from ..utils.common import importlib_or_none

logger = get_logger()

ATTN_GLOBAL_MSGS = set()


def warning_once(msg):
    """Log the warning message only once."""
    if msg not in ATTN_GLOBAL_MSGS:
        logger.warning(msg)
        ATTN_GLOBAL_MSGS.add(msg)


def flash_attn_ref(
    q,
    k,
    v,
    bias=None,
    causal=False,
    dropout_p=0.0,
    softmax_scale=None,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_mask=None,
    upcast=True,
    reorder_ops=False,
):
    """The functional equivalent of FlashAttentionTriton for correctness checking.
    Source: https://github.com/jfc4050/flash-attention/commit/f52868287ca9bd3ac1598dad6ce818358c1beafc

    Parameters
    ----------
    q : torch.Tensor
        Shape: (batch_size, seqlen_q, nheads, head_dim)
    k : torch.Tensor
        Shape: (batch_size, seqlen_k, nheads, head_dim)
    v : torch.Tensor
        Shape: (batch_size, seqlen_k, nheads, head_dim)
    bias : Optional[torch.Tensor]
        Shape: (batch_size, nheads, seqlen_q, seqlen_k)
    causal : bool
        Whether to apply lower triangular causal mask.
    dropout_p: float
        The dropout probability.
    softmax_scale : Optional[float]
        The softmax scale. If None, use 1 / sqrt(d).
    query_padding_mask : Optional[torch.Tensor]
        Shape: (batch_size, seqlen_q)
    key_padding_mask : Optional[torch.Tensor]
        (batch_size, seqlen_k)
    dropout_mask: Optional[torch.Tensor]
        The dropout mask. Shape: (batch_size, nheads, seqlen_q, seqlen_k)
    upcast : bool
        Whether to cast all inputs to fp32, do all computation in fp32, then cast
        output back to fp16/bf16.
    reorder_ops : bool
        whether to change the order of operations (scaling k instead of scaling k, etc.)
        without changing the math. This is to estimate the numerical error from
        operation reordering.

    Returns
    -------
    torch.Tensor
        Shape: (batch_size, seqlen_q, nheads, head_dim)
    """
    # pylint: disable=invalid-unary-operand-type
    assert softmax_scale is None, "softmax_scale is not supported"
    einops = importlib_or_none("einops")
    assert einops is not None, "einops is not installed"
    rearrange = einops.rearrange

    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if bias is not None:
        scores = (scores + bias).to(dtype=scores.dtype)
    if key_padding_mask is not None:
        scores.masked_fill_(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf")
        )
    if causal:
        causal_mask = torch.triu(
            torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device), 1
        )
        scores.masked_fill_(causal_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
        attention = attention.masked_fill(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0
        )
    return output.to(dtype=dtype_og)


def xformers_ref(q, k, v, attn_bias, p=0.0, scale=None):
    """The native PyTorch implementation of attention with the same signature as the
    attention implemented in xformers. This is used mainly to check the correctness
    of the xformers implementation.

    Parameters
    ----------
    q : torch.Tensor
        Shape: (batch_size, seqlen_q, nheads, head_dim)
    k : torch.Tensor
        Shape: (batch_size, seqlen_k, nheads, head_dim)
    v : torch.Tensor
        Shape: (batch_size, seqlen_k, nheads, head_dim)
    attn_bias : Optional[torch.Tensor]
        Shape: (batch_size, nheads, seqlen_q, seqlen_k)
    p : float
        The dropout probability.
    scale : Optional[float]
        The softmax scale. If None, use 1 / sqrt(d).

    Returns
    -------
    torch.Tensor
        Shape: (batch_size, seqlen_q, nheads, head_dim)
    """
    xformers_ops = importlib_or_none("xformers.ops")
    assert xformers_ops is not None, "xformers is not installed"
    assert q.ndim == 4

    def attention_bmk(q, k, v, attn_bias=None, p=0.0, scale=None):
        assert q.ndim == 3
        q = q.float()
        k = k.float()
        v = v.float()

        scale = scale if scale is not None else (1 / q.shape[-1] ** 0.5)
        q = q * scale

        attn = q @ k.transpose(-2, -1)
        if attn_bias is not None:
            if attn_bias.ndim == 4:
                assert q.shape[0] == attn_bias.shape[0] * attn_bias.shape[1]
                attn_bias = attn_bias.reshape([-1, *attn_bias.shape[2:]])
            attn = attn + attn_bias.float()
        attn = attn.softmax(-1).to(q.dtype)
        if p > 0:
            attn = torch.nn.functional.dropout(attn, p=p)
        return attn @ v

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape(
            [t.shape[0] * t.shape[2], t.shape[1], t.shape[3]]
        )

    if isinstance(attn_bias, xformers_ops.AttentionBias):
        attn_bias = attn_bias.materialize(
            (q.shape[0], q.shape[2], q.shape[1], k.shape[1]),
            device=q.device,
            dtype=q.dtype,
        ).reshape([q.shape[0] * q.shape[2], q.shape[1], k.shape[1]])
    out = attention_bmk(T(q), T(k), T(v), attn_bias, p, scale=scale)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))


def validate_sm_version(name, min_sm, max_sm=None):
    """Validate the sm version.

    Parameters
    ----------
    name : str
        The name of the kernel.
    min_sm : tuple[int, int]
        The minimum sm version.
    max_sm : Optional[tuple[int, int]]
        The maximum sm version. If None, the maximum sm version is not checked.
    """
    allow_range = f"sm_{min_sm[0]}{min_sm[1]}"
    allow_range += f"-sm_{max_sm[0]}{max_sm[1]}" if max_sm is not None else "+"

    cuda_sm = torch.cuda.get_device_capability("cuda")
    if cuda_sm < min_sm or (max_sm is not None and cuda_sm > max_sm):
        raise RuntimeError(
            f"{name} is only supported on GPUs with {allow_range} "
            f"but got sm_{cuda_sm[0]}{cuda_sm[1]}"
        )


def get_xfoemers_attn_op_by_name(attn_name):
    """Get the xformers attention operator by name."""
    xformers_ops = importlib_or_none("xformers.ops")
    if xformers_ops is None:
        raise RuntimeError("xformers is not installed")

    ops = [
        (xformers_ops.fmha.cutlass.FwOp, xformers_ops.fmha.cutlass.BwOp),
        (xformers_ops.fmha.flash.FwOp, xformers_ops.fmha.flash.BwOp),
        (xformers_ops.fmha.triton.FwOp, xformers_ops.fmha.triton.BwOp),
        (xformers_ops.fmha.small_k.FwOp, xformers_ops.fmha.small_k.BwOp),
    ]
    target_op = None
    if attn_name is not None and attn_name != "auto":
        for op in ops:
            if f"{attn_name}F" == op[0].NAME:
                target_op = op
                break
        else:
            raise ValueError(f"Unknown attention op name: {attn_name}")
    return partial(xformers_ops.memory_efficient_attention, op=target_op)


class FlashAttentionOp(nn.Module):
    """A wrapper module that processes HF attention mask to flash attention mask.

    Parameters
    ----------
    attn_op_name : str
        The name of the attention operator. Can be "native_xformers",
        "native_flash_attn", "triton", "cuda", "cutlass", or "auto". "triton"
        and "cuda" uses the kernel from flash-attention; while
        "cutlass" and "auto" use the kernel from xFormers.
    apply_causal_mask : bool
        Whether to apply causal mask.
    scale : Optional[float]
        The softmax scale. If None, use 1 / sqrt(d).
    """

    def __init__(self, attn_op_name, apply_causal_mask, scale=None):
        super().__init__()
        self.attn_op_name = attn_op_name
        self.apply_causal_mask = apply_causal_mask
        self.scale = scale
        self.pkg = None

        if attn_op_name == "native_xformers":
            self.pkg = "xformers"
            self.attn_fn = partial(xformers_ref, scale=scale)
        elif attn_op_name == "native_flash_attn":
            self.pkg = "flash_attn"
            self.attn_fn = partial(
                flash_attn_ref,
                query_padding_mask=None,
                key_padding_mask=None,
                dropout_mask=None,
                upcast=True,
                reorder_ops=False,
            )
        elif attn_op_name == "triton":
            self.pkg = "flash_attn"
            validate_sm_version("flash_attn_triton", (8, 0))
            flash_attn_triton = importlib_or_none("flash_attn.flash_attn_triton")
            if flash_attn_triton is None:
                raise RuntimeError("flash_attn is not installed")
            self.attn_fn = flash_attn_triton.flash_attn_func
        elif attn_op_name == "cuda":
            self.pkg = "flash_attn"
            validate_sm_version("flash_attn_unpadded_func", (8, 0))
            flash_attn_interface = importlib_or_none("flash_attn.flash_attn_interface")
            if flash_attn_interface is None:
                raise RuntimeError("flash_attn is not installed")
            self.attn_fn = flash_attn_interface.flash_attn_unpadded_func
        else:
            self.pkg = "xformers"
            # When op=None, the xformers attention op will be automatically selected.
            self.attn_fn = partial(
                get_xfoemers_attn_op_by_name(attn_op_name), scale=scale
            )

        # Different kernels have different requirements on the bias layout.
        self.bias_layout = "b11k" if self.pkg == "flash_attn" else "bhqk"

    def forward(self, query_layer, key_layer, value_layer, attention_mask, p):
        if self.pkg == "xformers":
            if self.apply_causal_mask:
                xformers_ops = importlib_or_none("xformers.ops")
                attn_bias = xformers_ops.fmha.attn_bias.LowerTriangularMask()
                if attention_mask is not None:
                    attn_bias = attn_bias.add_bias(attention_mask)
            else:
                attn_bias = attention_mask

            ret = self.attn_fn(query_layer, key_layer, value_layer, attn_bias, p=p)
        else:
            assert self.pkg == "flash_attn"
            if self.attn_op_name != "native_flash_attn" and attention_mask is not None:
                warning_once(
                    "WARNING: bias gradient is not supported yet. "
                    "The given mask will be ignored"
                )
                attn_bias = None
            else:
                attn_bias = attention_mask

            if self.attn_op_name == "triton":
                ret = self.attn_fn(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_bias,  # bias
                    self.apply_causal_mask,  # causal
                    p,  # dropout_p
                    self.scale,  # softmax_scale
                )
            else:
                assert self.attn_op_name == "cuda"
                # CUDA kernel in flash-attention requires qkv to be in
                # [B x S, H, D] layout.
                batch_size, seq_len, num_heads, head_size = query_layer.shape
                query_layer, key_layer, value_layer = [
                    x.reshape(batch_size * seq_len, num_heads, head_size)
                    for x in (query_layer, key_layer, value_layer)
                ]
                cu_seqlens = torch.arange(
                    0,
                    (batch_size + 1) * seq_len,
                    step=seq_len,
                    dtype=torch.int32,
                    device=query_layer.device,
                )
                ret = self.attn_fn(
                    query_layer,
                    key_layer,
                    value_layer,
                    cu_seqlens,
                    cu_seqlens,
                    seq_len,
                    seq_len,
                    p,
                    causal=self.apply_causal_mask,
                    softmax_scale=self.scale,
                )
                ret = ret.reshape(batch_size, seq_len, num_heads, head_size)
        ret = ret.to(query_layer.dtype)
        return ret


class FlashAttention(nn.Module):
    """A HuggingFace self attention module with flash attention kernels.
    Note that this module has limited supports to specialized processing,
    documetned as follows:

    - Only support absolute positional embeddings.
    - Do not support cross attention.
    - Do not support head mask, encoder_attention_mask, and output attention.

    We organize the Attention module as follows:

    - Attention
        - SelfAttention
            - Q, K, V
            - CoreAttention
        - Projection
            - OutDense
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        attn_op_name="auto",
        bias=True,
        output_proj=True,
        fused_qkv=False,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple "
                f"of the number of attention heads ({num_attention_heads})"
            )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.fused_qkv = fused_qkv
        if fused_qkv:
            self.qkv = nn.Linear(hidden_size, 3 * self.all_head_size, bias=bias)
        else:
            self.query = nn.Linear(hidden_size, self.all_head_size, bias=bias)
            self.key = nn.Linear(hidden_size, self.all_head_size, bias=bias)
            self.value = nn.Linear(hidden_size, self.all_head_size, bias=bias)

        self.output_proj = output_proj
        self.attn_pdrop = attn_pdrop

        if self.output_proj:
            self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
            self.resid_dropout = nn.Dropout(resid_pdrop)

        self.attn_op_name = attn_op_name
        self.attn_op = FlashAttentionOp(attn_op_name, self.output_proj)
        self.bias_layout = self.attn_op.bias_layout

    @staticmethod
    def layout_attention_mask(mask, num_attention_heads):
        # (B, 1, 1, S) -> (B, H, S, S)
        # Note that we use expand instead of repeat to avoid actual memory copy.
        mask = mask.expand(-1, num_attention_heads, mask.shape[-1], -1)
        return mask.contiguous()

    def reshape_for_scores(self, x: torch.Tensor):
        """Copy from transpose_for_scores but without the transpose"""
        new_x_shape = x.size()[:-1] + (
            -1,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x

    def forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[tuple[torch.Tensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.Tensor]:
        if encoder_hidden_states is not None or encoder_attention_mask is not None:
            raise NotImplementedError(
                "FlashAttention does not support cross attention yet."
            )
        if output_attentions:
            raise NotImplementedError(
                "FlashAttention does not support output attention yet."
            )
        if head_mask is not None:
            raise NotImplementedError("FlashAttention does not support head mask yet.")

        if self.fused_qkv:
            # (B, S, 3 * T * head_size) -> (B, S, T, 3 * head_size)
            # - split -> (B, S, T, head_size)
            # where T is #heads and we use -1 to cover the sharding case.
            layers = self.qkv(hidden_states)
            new_shape = layers.size()[:-1] + (-1, 3 * self.attention_head_size)
            layers = layers.view(new_shape)
            query_layer, key_layer, value_layer = layers.split(
                self.attention_head_size, dim=-1
            )
            query_layer = torch.squeeze(query_layer, -1)
            key_layer = torch.squeeze(key_layer, -1)
            value_layer = torch.squeeze(value_layer, -1)
        else:
            query_layer = self.query(hidden_states)
            key_layer = self.key(hidden_states)
            value_layer = self.value(hidden_states)
            query_layer = self.reshape_for_scores(query_layer)
            key_layer = self.reshape_for_scores(key_layer)
            value_layer = self.reshape_for_scores(value_layer)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key, key_layer), dim=-2)
            value_layer = torch.cat((past_value, value_layer), dim=-2)

        if attention_mask is not None and self.bias_layout == "bhqk":
            # Required bias layout: [batch_size, #heads, seq_length, seq_length].
            # The input shape is [batch_size, 1, 1, seq_length].
            # In other words, we need to broadcast other dimensions manually.
            attention_mask = self.layout_attention_mask(
                attention_mask, self.num_attention_heads
            )

        context_layer = self.attn_op(
            query_layer.contiguous(),
            key_layer.contiguous(),
            value_layer.contiguous(),
            attention_mask,
            p=self.attn_pdrop,
        )
        context_layer = context_layer.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(new_context_layer_shape)

        if self.output_proj:
            context_layer = self.out_proj(context_layer)
            context_layer = self.resid_dropout(context_layer)

        if use_cache:
            outputs = (context_layer, (key_layer, value_layer))
        else:
            outputs = (context_layer, None)
        return outputs
