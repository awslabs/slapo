# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import torch
import torch.nn as nn
from torch.distributed import distributed_c10d as dist

import slapo
from slapo import init_empty_weights
from slapo.pattern import call_module
from slapo.op.linear import FusedQKV
from slapo import init_empty_weights, get_cuda_rng_tracker


def trace_attention(sch, config, attn_path="h.N.attn.attention"):
    cnt = 0
    for idx in range(config.num_layers):
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


def replace_and_shard_attention(
    sch,
    config,
    attn_path="h.N.attn.attention",
    delay_init=True,
    disable_flash_attn=False,
    sequence_parallel=False,
):
    from epoi.inject.policy.gpt import InjectHFGPTAttentionPolicy
    from epoi.ops.xformers_attn import GenericSelfAttention, MemoryEfficientAttentionOp

    try:
        # Backward compatibility
        from epoi.ops.flash_attention import FlashSelfAttention, FlashAttentionTritonOp
    except ImportError:
        FlashSelfAttention = None
        FlashAttentionTritonOp = None

    cuda_sm = torch.cuda.get_device_capability("cuda")
    if not disable_flash_attn and FlashSelfAttention is not None and cuda_sm == (8, 0):
        SelfAttentionModule = FlashSelfAttention
        AttentionOp = FlashAttentionTritonOp
        attn_op_name = "triton"
    else:
        SelfAttentionModule = GenericSelfAttention
        AttentionOp = MemoryEfficientAttentionOp
        attn_op_name = "native" if disable_flash_attn else "cutlass"

    class SelfAttention(nn.Module):
        """A wrapper to align the original GPTNeoAttention forward signature."""

        def __init__(self, **kwargs):
            super().__init__()
            self.module = SelfAttentionModule(**kwargs)

        def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
        ):
            outputs = self.module(hidden_states, attention_mask, layer_past, use_cache)
            # FIXME: The original output is (hidden_states, None) where the None
            # is present_key_value and only used by in inference.
            return outputs[:1]

    class MemoryEfficientAttentionWithRNGOp(AttentionOp):
        def forward(self, query_layer, key_layer, value_layer, attention_mask, p):
            with get_cuda_rng_tracker().fork():
                return super().forward(
                    query_layer, key_layer, value_layer, attention_mask, p
                )

    num_layers, num_heads, hidden_size = (
        config.num_layers,
        config.num_heads,
        config.hidden_size,
    )

    cnt = 0
    for idx in range(num_layers):
        sub_sch = sch[attn_path.replace("N", str(idx))]
        init_config = InjectHFGPTAttentionPolicy.gen_init_config_from_object(
            sub_sch.mod, attn_op_name=attn_op_name
        )
        with init_empty_weights(enable=delay_init):
            new_mod = SelfAttention(**init_config)
        sub_sch.replace(new_mod)
        sub_sch.trace(
            tracer="pytorch",
            leaf_modules=[AttentionOp.__name__],
            concrete_args={
                "layer_past": None,
                "use_cache": False,
            },
        )

        def pattern(x: torch.Tensor) -> torch.Tensor:
            x = call_module("query|key|value", x)
            new_x_shape = x.size()[:-1] + (num_heads, hidden_size)
            x = x.view(new_x_shape)
            return x

        subgraphs = sub_sch["module"].find(pattern)
        assert len(subgraphs) == 3
        with init_empty_weights(enable=delay_init):
            new_fused_qkv = FusedQKV(hidden_size, num_heads, sch.world_size)
        sub_sch["module"].replace(new_fused_qkv, subgraphs)
        if sch.world_size > 1:
            sub_sch["module.FusedQKV_0.fused_linear"].shard("weight", axis=0)
            sub_sch["module.FusedQKV_0.fused_linear"].shard("bias", axis=0)
            sub_sch["module.out_proj"].shard("weight", axis=1)

            # Attention mask is broadcasted from (B, 1, 1, S) to (B, H, S, S),
            # where H is sharded.
            ops = sub_sch["module"].find_node(
                lambda node: node.op == "call_method" and node.target == "repeat"
            )
            assert len(ops) == 1

            def new_repeat(tensor, *args):
                # (B, 1, 1, S) -> (B, H, S, S)
                assert len(args) == 4
                out = tensor.repeat(args[0], args[1] // sch.world_size, *args[2:])
                return out.contiguous()

            sub_sch["module"].replace(new_repeat, ops[0][1])

            if sequence_parallel:
                sub_sch["module.FusedQKV_0.fused_linear"].sync(
                    mode="fwd_pre", sync_op_or_fn="all_gather", axis=1
                )

                sub_sch["module.out_proj"].sync(
                    mode="fwd_post", sync_op_or_fn="reduce_scatter", axis=1
                )
            else:
                # Shard qkv and output projection.
                sub_sch["module.FusedQKV_0.fused_linear"].sync(
                    mode="bwd_post", sync_op_or_fn="all_reduce"
                )
                sub_sch["module.out_proj"].sync(
                    mode="fwd_post", sync_op_or_fn="all_reduce"
                )

                # In this case, the attention dropout in between has to
                # use different random seeds.
                new_op = MemoryEfficientAttentionWithRNGOp(
                    sub_sch["module"]["attn_op"].mod.attn_op_name,
                    sub_sch["module"]["attn_op"].mod.apply_causal_mask,
                )
                sub_sch["module"]["attn_op"].replace(new_op)

        cnt += 1

    return cnt


def shard_word_embedding(
    sch,
    head_sch,
    vocab_size,
    word_embed_name="wte",
    pos_embed_name="wpe",
    final_ln_name="ln_f",
    sequence_parallel=False,
):
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
        output = slapo.sharding.reduce_forward_output(output, sch.group)
        return output

    sch[word_embed_name].sync(mode="fwd_post", sync_op_or_fn=fwd_post_hook)

    if sequence_parallel:
        sch[word_embed_name].sync(mode="fwd_post", sync_op_or_fn="scatter", axis=1)
        sch[pos_embed_name].sync(mode="fwd_post", sync_op_or_fn="scatter", axis=1)
        sch[final_ln_name].sync(
            mode="fwd_post",
            sync_op_or_fn="all_gather",
            axis=1,
            tensor_parallel_output_grad=False,
        )

    # Shard output embedding.
    if head_sch is not None:
        head_sch.shard("weight", axis=0)
        head_sch.sync(mode="bwd_post", sync_op_or_fn="all_reduce")


def replace_and_shard_mlp(
    sch,
    config,
    path="h.N.mlp",
    fc_names=["c_fc", "c_proj"],
    delay_init=True,
    sequence_parallel=False,
):
    from epoi.inject.policy.gpt import InjectHFGPTMLPPolicy

    for idx in range(config.num_layers):
        prefix = path.replace("N", str(idx))
        if config.activation_function in ["gelu", "gelu_new"]:
            sub_sch = sch[prefix]
            with init_empty_weights(enable=delay_init):
                new_mod = InjectHFGPTMLPPolicy.init_from_object(sub_sch.mod)
            sub_sch.replace(new_mod)
            sub_sch.trace(leaf_modules=["FusedBiasGELU", "FusedBiasNewGELU"])

            if sch.world_size > 1:
                sub_sch["fc_in"].shard("weight", axis=0)
                sub_sch["act"].shard("bias", axis=0)
                sub_sch["fc_out"].shard("weight", axis=1)

                if sequence_parallel:
                    sub_sch["fc_in"].sync(
                        mode="fwd_pre", sync_op_or_fn="all_gather", axis=1
                    )
                    sub_sch["fc_out"].sync(
                        mode="fwd_post", sync_op_or_fn="reduce_scatter", axis=1
                    )
                else:
                    sub_sch["fc_in"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
                    sub_sch["fc_out"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")

        elif sch.world_size > 1:
            sch[f"{prefix}.{fc_names[0]}"].shard("weight", axis=0)
            sch[f"{prefix}.{fc_names[0]}"].shard("bias", axis=0)
            sch[f"{prefix}.{fc_names[1]}"].shard("weight", axis=1)

            if sequence_parallel:
                sch[f"{prefix}.{fc_names[0]}"].sync(
                    mode="fwd_pre", sync_op_or_fn="all_gather", axis=1
                )
                sch[f"{prefix}.{fc_names[1]}"].sync(
                    mode="fwd_post", sync_op_or_fn="reduce_scatter", axis=1
                )
            else:
                sch[f"{prefix}.{fc_names[0]}"].sync(
                    mode="bwd_post", sync_op_or_fn="all_reduce"
                )
                sch[f"{prefix}.{fc_names[1]}"].sync(
                    mode="fwd_post", sync_op_or_fn="all_reduce"
                )


def checkpoint(sch, config, path="h.N", ckpt_ratio=1.0):
    if ckpt_ratio == 0.0:
        return

    def order_args_fn(*args, **kwargs):
        assert len(args) == 1
        attention_mask = kwargs.get("attention_mask", None)
        head_mask = kwargs.get("head_mask", None)
        output_attentions = kwargs.get("output_attentions", False)
        # Forward: (
        #   hidden_states,
        #   layer_past,
        #   attention_mask,
        #   head_mask,
        #   use_cache,
        #   output_attentions
        # )
        return (args[0], None, attention_mask, head_mask, False, output_attentions)

    n_ckpt = int(config.num_hidden_layers * ckpt_ratio)
    for idx in range(n_ckpt):
        sch[path.replace("N", str(idx))].checkpoint(order_args_fn=order_args_fn)
    return n_ckpt


def broadcast_input(sch):
    group_src_rank = dist.get_global_rank(sch.group, 0)

    def _broadcast_input(module, inputs):
        for inp in inputs:
            if inp is not None:
                dist.broadcast(inp, src=group_src_rank, group=sch.group)
        return inputs

    sch.sync(mode="fwd_pre", sync_op_or_fn=_broadcast_input)
