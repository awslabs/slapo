# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace OPT with model schedule."""
# pylint: disable=logging-fstring-interpolation, unused-argument

import inspect

import torch
from torch import nn
import torch.distributed as dist

from ..schedule import create_schedule
from ..initialization import init_empty_weights
from ..op import FlashAttention, FusedMLP
from ..pattern import call_module
from ..logger import get_logger

logger = get_logger("OPT")


def schedule_model(
    model,
    config,
    prefix="",
    attn_op_name="cuda",
    fp16=True,
    ckpt_ratio=0.0,
    group=None,
    bcast_input=False,
    pipeline_cuts=None,
    delay_init=True,
):

    logger.info("Scheduling OPT", ranks=0)

    if fp16:
        logger.info("Change model dtype to fp16", ranks=0)
        model.half()

    sch = create_schedule(model, group=group)

    # Replace self attention with flash attention, and shard QKV/output
    # if MP group > 1.
    if attn_op_name == "native_xformers":
        logger.info("Disabled Flash Attention", ranks=0)
    cnt, applied_attn_op_name = replace_and_shard_attention(
        sch[prefix],
        config,
        delay_init=delay_init,
        attn_op_name=attn_op_name,
    )
    logger.info(
        f"Replace {cnt} attention layers with {applied_attn_op_name} op", ranks=0
    )

    # Shard other parameters if MP group > 1.
    if sch.world_size > 1:
        replace_and_shard_mlp(sch[prefix], config, delay_init=delay_init)
        head_sch = sch["lm_head"] if "lm_head" in sch else None
        shard_word_embedding(sch[prefix], head_sch, config.vocab_size)

        # Broadcast input to all devices within the MP group.
        # This is not required when running on Megatron.
        if bcast_input:
            broadcast_input(sch)

    # Insert activation checkpoints.
    if ckpt_ratio > 0.0:
        n_ckpt = checkpoint(sch[prefix], config, ckpt_ratio=ckpt_ratio)
        logger.info(f"Checkpointing {n_ckpt} layers", ranks=0)

    # Cut pipeline stages.
    if pipeline_cuts:
        input_names = ["input_ids", "attention_mask", "position_ids"]
        sig = inspect.signature(model.forward)
        concrete_args = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.name not in input_names
        }
        sch.trace_for_pipeline(
            f"{prefix}", tracer="huggingface", concrete_args=concrete_args
        )
        _prefix = f"{prefix}." if prefix else ""
        for cut in pipeline_cuts:
            sch[f"{_prefix}h.{cut}"].cut_pipeline_stage()

    return sch


def trace_attention(sch, config, attn_path="h.N.attn.attention"):
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
        sch.replace(new_expand, op)


def replace_and_shard_attention(
    sch,
    config,
    attn_path="decoder.layers.N.self_attn",
    delay_init=True,
    attn_op_name="cuda",
):
    init_config = dict(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        output_proj=True,
        attn_pdrop=config.attention_dropout,
        resid_pdrop=config.dropout,
        attn_op_name=attn_op_name,
        fused_qkv=True,
        bias=True,
    )

    class Attention(nn.Module):
        """A wrapper to align the original OPT forward signature."""

        def __init__(self, **kwargs):
            super().__init__()
            try:
                self.module = FlashAttention(**kwargs)
            except Exception as err:
                if kwargs["attn_op_name"] == "native_xformers":
                    raise RuntimeError(
                        f"Failed to create native attention: {err}"
                    ) from None

                # Failed to use the triton kernel. This may due to unsupported
                # GPU (< sm_75) or flash-attention is not installed. Fallback
                # to xFormers' cutlass.
                kwargs["attn_op_name"] = "cutlass"
                self.module = FlashAttention(**kwargs)

        def forward(
            self,
            hidden_states,
            past_key_value=None,
            attention_mask=None,
            layer_head_mask=None,
            output_attentions=False,
            use_cache=False,
        ):
            outputs = self.module(hidden_states, attention_mask, past_key_value)
            # FIXME: The original output is (hidden_states, None) where the None
            # is present_key_value and only used by in inference.
            # OPT output is (hidden_states, self_attn_weights, present_key_value)
            return outputs[0], None, None

    cnt = 0
    attn_op = []
    for idx in range(config.num_hidden_layers):
        sub_sch = sch[attn_path.replace("N", str(idx))]
        with init_empty_weights(enable=delay_init):
            new_mod = Attention(**init_config)
            attn_op.append(new_mod.module.attn_op_name)
        sub_sch.replace(new_mod)

        if sch.world_size > 1:
            sub_sch.trace(
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
            sub_sch["module.qkv"].shard("weight", axis=0)
            sub_sch["module.qkv"].shard("bias", axis=0)
            fix_attention_mask_shape(sub_sch["module"])
            sub_sch["module.qkv"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
            sub_sch["module.out_proj"].shard("weight", axis=1)
            sub_sch["module.out_proj"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")
        cnt += 1

    # Check if all attention ops are the same.
    attn_op = list(set(attn_op))
    if len(attn_op) > 1:
        raise RuntimeError(
            f"The attention op is not consistent across layers, including {attn_op}"
        )

    return cnt, attn_op[0]


def remove_cast(sch, config, attn_path="h.N.attn.attention"):
    """[Untested] Remove .to(torch.float32) in GPT-Neo attention to align
    HF and Megatron GPT-2 behavior.
    """
    cnt = 0
    for idx in range(config.num_hidden_layers):
        sub_sch = sch[attn_path.replace("N", str(idx))]
        ops = sub_sch.find_node(
            lambda node: node.op == "call_method"
            and node.target == "to"
            and len(node.args) == 2
            and node.args[1] == torch.float32
        )

        for op in ops:
            sub_sch.replace(lambda x, *args: x, op)
            cnt += 1
    return cnt


def shard_word_embedding(
    sch, head_sch, vocab_size, word_embed_name="decoder.embed_tokens"
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
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=sch.group)
        return output

    sch[word_embed_name].sync(mode="fwd_post", sync_op_or_fn=fwd_post_hook)

    # Shard output embedding.
    if head_sch is not None:
        head_sch.shard("weight", axis=0)


# pylint: disable=dangerous-default-value
def replace_and_shard_mlp(
    sch,
    config,
    path="decoder.layers.N",
    fc_names=["fc1", "fc2"],
    delay_init=True,
    disable_fuse_bias_gelu=True,
):
    for idx in range(config.num_hidden_layers):
        prefix = path.replace("N", str(idx))
        replaced_new_mlp = False
        if config.activation_function in {"gelu", "gelu_new"}:
            if disable_fuse_bias_gelu:
                sub_sch = sch[prefix]
                inter_size, hidden_size = sub_sch.mod.c_fc.weight.shape
                with init_empty_weights(enable=delay_init):
                    new_mod = FusedMLP(
                        hidden_size,
                        inter_size,
                        config.activation_function,
                        config.resid_dropout,
                    )
                sub_sch.replace(new_mod)
                sub_sch.trace(leaf_modules=["FusedBiasGELU", "FusedBiasNewGELU"])
            else:

                def bias_gelu_pattern(x, bias):
                    x = x + bias
                    x = call_module("activation_fn", x)
                    return x

                subsch = sch[prefix]
                subsch["fc1"].decompose()
                subsch.trace(flatten=True)

                subgraphs = subsch.find(bias_gelu_pattern)
                assert len(subgraphs) == 1
                assert len(subgraphs[0]) == 2
                subsch.fuse(subgraphs, compiler="TorchScript", name="FusedBiasGeLU")
        if sch.world_size > 1:
            if replaced_new_mlp:
                sub_sch["fc_in"].shard("weight", axis=0)
                sub_sch["act"].shard("bias", axis=0)
                sub_sch["fc_in"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
                sub_sch["fc_out"].shard("weight", axis=1)
                sub_sch["fc_out"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")
            else:
                sch[f"{prefix}.{fc_names[0]}"].shard("weight", axis=0)
                sch[f"{prefix}.{fc_names[0]}"].shard("bias", axis=0)
                sch[f"{prefix}.{fc_names[0]}"].sync(
                    mode="bwd_post", sync_op_or_fn="all_reduce"
                )
                sch[f"{prefix}.{fc_names[1]}"].shard("weight", axis=1)
                sch[f"{prefix}.{fc_names[1]}"].sync(
                    mode="fwd_post", sync_op_or_fn="all_reduce"
                )


def checkpoint(sch, config, path="decoder.layers.N", ckpt_ratio=1.0):
    if ckpt_ratio == 0.0:
        return 0

    n_ckpt = int(config.num_hidden_layers * ckpt_ratio)
    for idx in range(n_ckpt):
        sch[path.replace("N", str(idx))].checkpoint()
    return n_ckpt


def broadcast_input(sch):
    def broadcast(inputs):
        for inp in inputs:
            dist.broadcast(inp, src=0, group=sch.group)
        return inputs

    sch.sync(mode="fwd_pre", sync_op_or_fn=broadcast)