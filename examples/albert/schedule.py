# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import torch
import torch.distributed as dist
import torch.nn as nn

from slapo import init_empty_weights
from typing import Optional
from slapo.pattern import call_module
from slapo.op import FlashAttention


def trace_attention(
    sch, config, attn_path="encoder.albert_layer_groups.N.albert_layers.N.attention"
):
    cnt = 0
    for idx in range(config.num_hidden_layers):
        sub_sch = sch[attn_path.replace("N", str(idx))]
        input_names = ["hidden_states", "attention_mask"]
        sig = inspect.signature(sub_sch.mod.forward)
        concrete_args = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.name not in input_names
        }
        if sub_sch.trace(tracer="pytorch", concrete_args=concrete_args):
            cnt += 1
    return cnt


def fix_attention_mask_shape(sch):
    # Attention mask may needed to be expanded from (B, 1, 1, S)
    # to (B, H, S, S), where H is sharded.
    ops = sch.find_node(
        lambda node: node.op == "call_method" and node.target == "expand"
    )

    def new_expand(tensor, *args):
        # (B, 1, 1, S) -> (B, H, S, S)
        assert len(args) == 4
        out = tensor.expand(args[0], args[1] // sch.world_size, *args[2:])
        return out.contiguous()

    for op in ops:
        sch.replace(new_expand, op[1])


def replace_and_shard_attention(
    sch,
    config,
    attn_path="encoder.albert_layer_groups.N.albert_layers.N",
    delay_init=True,
    disable_flash_attn=False,
):
    attn_op_name = "native_xformers" if disable_flash_attn else "triton"
    init_config = dict(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        output_proj=True,
        attn_pdrop=config.attention_probs_dropout_prob,
        resid_pdrop=config.hidden_dropout_prob,
        attn_op_name=attn_op_name,
        fused_qkv=True,
        bias=True,
        world_size=sch.world_size,
    )

    class AlbertXFAttention(nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            try:
                self.self_attn = FlashAttention(**kwargs)
            except Exception as err:
                if kwargs["attn_op_name"] == "native_xformers":
                    raise RuntimeError(
                        f"Failed to create native attention: {err}"
                    ) from None

                # Failed to use the triton kernel. This may due to unsupported
                # GPU (< sm_75) or flash-attention is not installed. Fallback
                # to xFormers' cutlass.
                kwargs["attn_op_name"] = "cutlass"
                self.self_attn = FlashAttention(**kwargs)

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None,
            head_mask: Optional[torch.FloatTensor] = None,
            output_attentions: bool = False,
        ):
            outputs = self.self_attn(hidden_states, attention_mask)
            context_layer = outputs[0]
            return (context_layer, None)

    cnt = 0
    for idx in range(1):  # use layer group
        prefix = attn_path.replace("N", str(idx))
        with init_empty_weights(enable=delay_init):
            attn = AlbertXFAttention(config, **init_config)
        sch[f"{prefix}.attention"].replace(attn)

        if sch.world_size > 1:
            sch[f"{prefix}.attention.self_attn"].trace(
                tracer="pytorch",
                leaf_modules=["FlashAttentionOp"],
                concrete_args={
                    "layer_past": None,
                    "head_mask": None,
                    "encoder_hidden_states": None,
                    "encoder_attention_mask": None,
                    "use_cache": False,
                    "output_attentions": False,
                },
            )
            sub_sch = sch[f"{prefix}.attention"]
            sub_sch["self_attn.qkv"].shard("weight", axis=0)
            sub_sch["self_attn.qkv"].shard("bias", axis=0)
            sub_sch["self_attn.qkv"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
            fix_attention_mask_shape(sub_sch["self_attn"])
            sub_sch["self_attn.out_proj"].shard("weight", axis=1)
            sub_sch["self_attn.out_proj"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")
        cnt += 1

    return cnt


def shard_word_embedding(sch, vocab_size, word_embed_name="embeddings.word_embeddings"):
    if sch.world_size == 1:
        return

    # Embedding
    sch[word_embed_name].shard("weight", axis=0)
    # Build the mask
    vocab_start_index = sch.rank * vocab_size // sch.world_size
    vocab_end_index = (sch.rank + 1) * vocab_size // sch.world_size

    def fwd_pre_hook(_module, _input):
        # Mask the input
        input_mask = (_input[0] < vocab_start_index) | (_input[0] >= vocab_end_index)
        masked_input = _input[0].clone() - vocab_start_index
        masked_input[input_mask] = 0
        return masked_input

    sch[word_embed_name].sync(mode="fwd_pre", sync_op_or_fn=fwd_pre_hook)

    def fwd_post_hook(_module, _input, output):
        # Mask the output embedding
        input_mask = (_input[0] < vocab_start_index) | (_input[0] >= vocab_end_index)
        output[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=sch.group)
        return output

    sch[word_embed_name].sync(mode="fwd_post", sync_op_or_fn=fwd_post_hook)


def fuse_bias_gelu(sch, config, path="encoder.albert_layer_groups.N.albert_layers.N"):
    def bias_gelu_pattern(x, bias):
        x = bias + x
        x = call_module("activation", x)
        return x

    for idx in range(1):
        subsch = sch[path.replace("N", str(idx))]
        subsch["ffn"].decompose()
        subsch.trace(
            flatten=True,
            leaf_modules=[
                "AlbertAttention",
                "AlbertXFAttention",
                "NewGELUActivation",
                "GELUActivation",
            ],
        )

        subgraphs = subsch.find(bias_gelu_pattern)
        assert len(subgraphs) == 1
        assert len(subgraphs[0]) == 2
        subsch.fuse(subgraphs, compiler="TorchScript", name="FusedBiasGeLU")


def shard_mlp(
    sch,
    config,
    path="encoder.albert_layer_groups.N.albert_layers.N",
    fc_names=["ffn", "ffn_output"],
):
    if sch.world_size == 1:
        return

    for idx in range(1):  # use layer group
        prefix = path.replace("N", str(idx))
        sch[f"{prefix}.{fc_names[0]}"].shard("weight", axis=0)
        sch[f"{prefix}.{fc_names[0]}"].shard("bias", axis=0)
        sch[f"{prefix}.{fc_names[0]}"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
        sch[f"{prefix}.{fc_names[1]}"].shard("weight", axis=1)
        sch[f"{prefix}.{fc_names[1]}"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")


def checkpoint(
    sch, config, path="encoder.albert_layer_groups.N.albert_layers.N", ckpt_ratio=1.0
):
    if ckpt_ratio == 0.0:
        return

    n_ckpt = int(config.num_hidden_layers * ckpt_ratio)
    # TODO: Only checkpoint part of the layers
    for idx in range(1):  # use layer group
        sch[path.replace("N", str(idx))].checkpoint()
    return n_ckpt


def broadcast_input(sch):
    def broadcast_input(inputs):
        for inp in inputs:
            dist.broadcast(inp, src=0, group=sch.group)
        return inputs

    sch.sync(mode="fwd_pre", sync_op_or_fn=broadcast_input)
