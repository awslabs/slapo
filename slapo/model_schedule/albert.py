# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace Albert with model schedule."""
# pylint: disable=unused-argument

import inspect
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from ..schedule import create_schedule
from ..initialization import init_empty_weights
from ..pattern import call_module
from ..op import FlashAttention
from ..logger import get_logger
from .registry import register_schedule


@register_schedule()
def _apply_schedule(
    model,
    **sch_config,
):
    model_config = sch_config.get("model_config", None)
    if model_config is None:
        raise ValueError(
            "Model config is not specified in sch_config. Please provide `model_config` in the kwarg."
        )
    try:
        model_name = model_config._name_or_path
    except Exception:
        model_name = model_config.get("_name_or_path", None)
    logger = get_logger(f"{model_name}")

    # Change data type.
    fp16 = sch_config.get("fp16", False)
    bf16 = sch_config.get("bf16", False)
    if fp16 and bf16:
        raise ValueError("Cannot use both fp16 and bf16")
    if fp16:
        logger.info("Change model dtype to fp16", ranks=0)
        model.half()
    elif bf16:
        logger.info("Change model dtype to bf16", ranks=0)
        model.bfloat16()
    else:
        logger.info("Use fp32 as default model dtype", ranks=0)

    group = sch_config.get("group", None)
    sch = create_schedule(model, group=group)
    logger.info(
        "Scheduling %s with TP=%d, config: %s",
        model_name,
        sch.world_size,
        sch_config,
        ranks=0,
    )

    # Tensor parallelism.
    logger.info("Shard model parameters", ranks=0)
    prefix = sch_config.get("prefix", "")
    delay_init = sch_config.get("delay_init", True)
    # Replace self attention with flash attention, and shard QKV/output
    # if MP group > 1.
    attn_op_name = sch_config.get("attn_op_name", "cuda")
    if attn_op_name == "native_xformers":
        logger.info("Disabled Flash Attention", ranks=0)
    cnt, applied_attn_op_name = replace_and_shard_attention(
        sch[prefix],
        model_config,
        delay_init=delay_init,
        attn_op_name=attn_op_name,
    )
    logger.info(
        "Replace %d attention layers with %s op", cnt, applied_attn_op_name, ranks=0
    )

    # Operator fusion
    if not sch_config.get("disable_fuse_bias_gelu", True):
        fuse_bias_gelu(sch[prefix], model_config)
        logger.info("Fused Bias+GeLU", ranks=0)

    # Shard other parameters if MP group > 1.
    if sch.world_size > 1:
        shard_mlp(sch[prefix], model_config)
        shard_word_embedding(sch[prefix], model_config.vocab_size)

    if sch.world_size > 1 and sch_config.get("bcast_input", False):
        # Broadcast input to all devices within the MP group.
        # This is not required when running on Megatron.
        logger.info("Broadcast input to all devices", ranks=0)
        broadcast_input(sch)

    # Insert activation checkpoints.
    ckpt_ratio = sch_config.get("ckpt_ratio", 0.0)
    if ckpt_ratio > 0.0:
        prefix = sch_config.get("prefix", "")
        checkpoint_method = sch_config.get("checkpoint_method", "uniform")
        logger.info("Checkpoint ratio: %.2f", ckpt_ratio, ranks=0)
        n_ckpt = checkpoint(
            sch[prefix],
            model_config,
            ckpt_ratio=ckpt_ratio,
            checkpoint_method=checkpoint_method,
        )
        logger.info("Checkpointed %d layers", n_ckpt, ranks=0)

    # Pipeline parallelism.
    if sch_config.get("pipeline_cuts", None):
        logger.info("Generate pipeline schedule", ranks=0)
        generate_pipeline_schedule(sch, sch_config)

    return sch


def generate_pipeline_schedule(sch, sch_config):
    pipeline_cuts = sch_config.get("pipeline_cuts", None)
    prefix = sch_config.get("prefix", "")
    # Cut pipeline stages.
    if pipeline_cuts:
        input_names = [
            "input_ids",
            "attention_mask",
            "token_type_ids",
        ]
        sig = inspect.signature(sch.mod.forward)
        concrete_args = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.name not in input_names
        }
        _prefix = f"{prefix}." if prefix else ""
        sch.trace_until(
            f"{_prefix}encoder", tracer="huggingface", concrete_args=concrete_args
        )
        for cut in pipeline_cuts:
            sch[f"{_prefix}encoder.layer.{cut}"].cut_pipeline_stage()

    return sch


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
    model_config,
    attn_path="encoder.albert_layer_groups.N.albert_layers.N",
    delay_init=True,
    attn_op_name="cuda",
):
    init_config = dict(
        hidden_size=model_config.hidden_size,
        num_attention_heads=model_config.num_attention_heads,
        output_proj=True,
        attn_pdrop=model_config.attention_probs_dropout_prob,
        resid_pdrop=model_config.hidden_dropout_prob,
        attn_op_name=attn_op_name,
        fused_qkv=True,
        bias=True,
    )

    class AlbertXFAttention(nn.Module):
        def __init__(self, **kwargs):
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
    attn_op = []
    for idx in range(1):  # use layer group
        prefix = attn_path.replace("N", str(idx))
        with init_empty_weights(enable=delay_init):
            attn = AlbertXFAttention(**init_config)
            attn_op.append(attn.self_attn.attn_op_name)
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
            sub_sch["self_attn.out_proj"].sync(
                mode="fwd_post", sync_op_or_fn="all_reduce"
            )
        cnt += 1

    # Check if all attention ops are the same.
    attn_op = list(set(attn_op))
    if len(attn_op) > 1:
        raise RuntimeError(
            f"The attention op is not consistent across layers, including {attn_op}"
        )

    return cnt, attn_op[0]


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


def fuse_bias_gelu(
    sch, model_config, path="encoder.albert_layer_groups.N.albert_layers.N"
):
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


# pylint: disable=dangerous-default-value
def shard_mlp(
    sch,
    model_config,
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
    sch,
    model_config,
    path="encoder.albert_layer_groups.N.albert_layers.N",
    ckpt_ratio=1.0,
    checkpoint_method="uniform",
):
    if checkpoint_method != "uniform":
        raise NotImplementedError(
            f"Checkpoint method {checkpoint_method} is not supported yet."
        )
    if ckpt_ratio == 0.0:
        return 0

    n_ckpt = int(model_config.num_hidden_layers * ckpt_ratio)
    # TODO: Only checkpoint part of the layers
    for idx in range(1):  # use layer group
        sch[path.replace("N", str(idx))].checkpoint()
    return n_ckpt


def broadcast_input(sch):
    def broadcast(inputs):
        for inp in inputs:
            dist.broadcast(inp, src=0, group=sch.group)
        return inputs

    sch.sync(mode="fwd_pre", sync_op_or_fn=broadcast)
