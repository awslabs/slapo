# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace T5 with model schedule."""
# pylint: disable=unused-argument, import-error

import inspect

import torch
from torch import nn
import torch.distributed as dist

from ..schedule import create_schedule
from ..initialization import init_empty_weights
from ..pattern import call_module
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
    prefix = sch_config.get("prefix", "")
    if sch.world_size > 1:
        logger.info("Shard model parameters", ranks=0)
        replace_and_shard_attention(sch, model_config, sch_config)
        shard_mlp(sch[prefix], model_config, "encoder.block.N.layer.1.DenseReluDense")
        shard_mlp(sch[prefix], model_config, "decoder.block.N.layer.2.DenseReluDense")
        shard_word_embedding(sch[prefix], model_config.vocab_size)

    if sch.world_size > 1 and sch_config.get("bcast_input", False):
        # Broadcast input to all devices within the MP group.
        # This is not required when running on Megatron.
        logger.info("Broadcast input to all devices", ranks=0)
        broadcast_input(sch)

    # Insert activation checkpoints.
    ckpt_ratio = sch_config.get("ckpt_ratio", 0.0)
    if ckpt_ratio > 0.0:
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


def replace_and_shard_attention(sch, model_config, sch_config):
    model_name = model_config._name_or_path
    logger = get_logger(model_name)
    prefix = sch_config.get("prefix", "")
    delay_init = sch_config.get("delay_init", True)
    # Replace self attention with flash attention, and shard QKV/output
    # if MP group > 1.
    attn_op_name = sch_config.get("attn_op_name", "cuda")
    disable_flash_attn = attn_op_name == "native_xformers"
    cnt, fix_shape_cnt = _replace_and_shard_attention(
        sch[prefix],
        model_config,
        "encoder.block.N.layer.0.SelfAttention",
        delay_init=delay_init,
        disable_flash_attn=disable_flash_attn,
    )
    logger.info(
        "Replace %d encoder self attention patterns with %d shape fixing",
        cnt,
        fix_shape_cnt,
        ranks=0,
    )
    cnt, fix_shape_cnt = _replace_and_shard_attention(
        sch[prefix],
        model_config,
        "decoder.block.N.layer.0.SelfAttention",
        delay_init=delay_init,
        disable_flash_attn=disable_flash_attn,
    )
    logger.info(
        "Replace %d decoder self attention patterns with %d shape fixing",
        cnt,
        fix_shape_cnt,
        ranks=0,
    )
    cnt, fix_shape_cnt = _replace_and_shard_attention(
        sch[prefix],
        model_config,
        "decoder.block.N.layer.1.EncDecAttention",
        cross_attn=True,
        delay_init=delay_init,
        disable_flash_attn=disable_flash_attn,
    )
    logger.info(
        "Replace %d decoder cross attention patterns with %d shape fixing",
        cnt,
        fix_shape_cnt,
        ranks=0,
    )


def checkpoint(
    sch,
    model_config,
    path="",
    ckpt_ratio=1.0,
    checkpoint_method="uniform",
):
    if checkpoint_method != "uniform":
        raise NotImplementedError(
            f"Checkpoint method {checkpoint_method} is not supported yet."
        )
    n_ckpt = 0
    if ckpt_ratio > 0.0:
        n_ckpt += _checkpoint(
            sch, model_config, "encoder.block.N", ckpt_ratio=ckpt_ratio
        )
        n_ckpt += _checkpoint(
            sch, model_config, "decoder.block.N", ckpt_ratio=ckpt_ratio
        )
    return n_ckpt


def generate_pipeline_schedule(sch, sch_config):
    pipeline_cuts = sch_config.get("pipeline_cuts", None)
    prefix = sch_config.get("prefix", "")
    # Cut pipeline stages. For example, [[11], [11]] means to cut
    # encoder.block.11, decoder.block.11. And we always cut between encoder/decoder,
    # so there will be 4 stages in total.
    if pipeline_cuts:
        assert len(pipeline_cuts) == 2
        input_names = [
            "decoder_input_ids",
            "input_ids",
            "decoder_attention_mask",
            "attention_mask",
        ]
        sig = inspect.signature(sch.mod.forward)
        concrete_args = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.name not in input_names
        }
        _prefix = f"{prefix}." if prefix else ""
        sch.trace_until(
            [f"{_prefix}encoder", f"{_prefix}decoder"],
            tracer="huggingface",
            concrete_args=concrete_args,
        )
        for cut in pipeline_cuts[0]:
            sch[f"{_prefix}encoder.block.{cut}"].cut_pipeline_stage()
        sch[f"{_prefix}encoder"].cut_pipeline_stage()
        for cut in pipeline_cuts[1]:
            sch[f"{_prefix}decoder.block.{cut}"].cut_pipeline_stage()

    return sch


def fix_position_bias_shape(sch, delay_init=True):
    # Target EPOI T5 attention module.
    cnt = 0

    # Case 1: If position_bias is not given and the layer
    # does not have relative position bias, it generates zeros
    # with ZeroBiasLike that takes (n_heads).
    if "zero_bias_like" in sch:
        from epoi.ops.xformers_attn import ZeroBiasLike

        old_mod = sch["zero_bias_like"].mod
        with init_empty_weights(enable=delay_init):
            new_mod = ZeroBiasLike(old_mod.n_heads // sch.world_size)
        sch["zero_bias_like"].replace(new_mod)
        cnt += 1

    # Case 2: If position_bias is not given and the layer
    # has relative position bias, it generates bias with RelativeBias
    # that takes (n_buckets, max_dist, n_heads, is_decoder).
    if "relative_attention_bias" in sch:
        from epoi.ops.xformers_attn import RelativeBias

        old_mod = sch["relative_attention_bias"].mod
        new_bias_mod = RelativeBias(
            old_mod.relative_attention_num_buckets,
            old_mod.relative_attention_max_distance,
            old_mod.n_heads // sch.world_size,
            old_mod.is_decoder,
        )
        sch["relative_attention_bias"].replace(new_bias_mod)
        cnt += 1
    return cnt


def _replace_and_shard_attention(
    sch,
    model_config,
    attn_path,
    cross_attn=False,
    delay_init=True,
    disable_flash_attn=False,
):
    from epoi.inject.policy.t5 import InjectHFT5AttentionPolicy
    from epoi.ops.xformers_attn import T5Attention

    num_layers, num_heads, hidden_size, d_kv = (
        model_config.num_hidden_layers,
        model_config.num_attention_heads,
        model_config.hidden_size,
        model_config.d_kv,
    )

    cnt = 0
    fix_shape_cnt = 0
    for idx in range(num_layers):
        prefix = attn_path.replace("N", str(idx))
        sub_sch = sch[f"{prefix}"]
        init_config = InjectHFT5AttentionPolicy.gen_init_config_from_object(sub_sch.mod)
        if disable_flash_attn:
            init_config["attn_op_name"] = "native"
        with init_empty_weights(enable=delay_init):
            new_mod = T5Attention(**init_config)
        sub_sch.replace(new_mod)
        concrete_args = {
            "layer_head_mask": None,
            "past_key_value": None,
            "layer_past": None,
            "use_cache": False,
            "output_attentions": False,
        }
        if not cross_attn:
            concrete_args["key_value_states"] = None
        if idx == 0:
            # The first layer of encoder and decoder generates position bias
            # from scratch.
            concrete_args["position_bias"] = None
        sub_sch.trace(
            tracer="pytorch",
            leaf_modules=["MemoryEfficientAttentionOp", "RelativeBias", "ZeroBiasLike"],
            concrete_args=concrete_args,
        )

        if cross_attn:
            # Cross attention can only fuse k, v, because k, v are taking encoder status
            # while q is taking the current hidden status.
            class FusedKV(nn.Module):
                def __init__(self, num_heads, d_model, d_kv) -> None:
                    super().__init__()
                    self.hidden_size = d_model
                    self.num_heads = num_heads
                    self.key_value_proj_dim = d_kv
                    self.inner_dim = num_heads * self.key_value_proj_dim
                    self.fused_linear = nn.Linear(
                        self.hidden_size, self.inner_dim * 2, bias=False
                    )

                def reshape_for_scores(self, x):
                    new_x_shape = x.size()[:-1] + (
                        self.num_heads // sch.world_size,
                        self.key_value_proj_dim,
                        2,
                    )
                    x = x.view(new_x_shape)
                    return x.contiguous()

                def forward(self, hidden_states):
                    kv = self.fused_linear(hidden_states)
                    reshaped_qkv = self.reshape_for_scores(kv)
                    k, v = torch.split(reshaped_qkv, 1, dim=-1)
                    k = torch.squeeze(k, -1).contiguous()
                    v = torch.squeeze(v, -1).contiguous()
                    return [k, v]

            class ShardableQ(nn.Module):
                def __init__(self, num_heads, d_model, d_kv) -> None:
                    super().__init__()
                    self.hidden_size = d_model
                    self.num_heads = num_heads
                    self.key_value_proj_dim = d_kv
                    self.inner_dim = num_heads * self.key_value_proj_dim
                    self.query = nn.Linear(hidden_size, self.inner_dim, bias=False)

                def reshape_for_scores(self, x):
                    new_x_shape = x.size()[:-1] + (
                        self.num_heads // sch.world_size,
                        self.key_value_proj_dim,
                    )
                    x = x.view(new_x_shape)
                    return x.contiguous()

                def forward(self, hidden_states):
                    states = self.query(hidden_states)
                    states = self.reshape_for_scores(states)
                    return states

            def pattern_kv(x: torch.Tensor) -> torch.Tensor:
                x = call_module("key|value", x)
                new_x_shape = x.size()[:-1] + (num_heads, d_kv)
                x = x.view(new_x_shape)
                return x

            subgraphs = sub_sch.find(pattern_kv)
            assert len(subgraphs) == 2
            with init_empty_weights(enable=delay_init):
                new_fused_kv = FusedKV(num_heads, hidden_size, d_kv)
            sub_sch.replace(new_fused_kv, subgraphs)

            def pattern_q(x: torch.Tensor) -> torch.Tensor:
                x = call_module("query", x)
                new_x_shape = x.size()[:-1] + (num_heads, d_kv)
                x = x.view(new_x_shape)
                return x

            subgraphs = sub_sch.find(pattern_q)
            assert len(subgraphs) == 1
            with init_empty_weights(enable=delay_init):
                new_q = ShardableQ(num_heads, hidden_size, d_kv)
            sub_sch.replace(new_q, subgraphs)
            if sch.world_size > 1:
                sub_sch["FusedKV_0.fused_linear"].shard("weight", axis=0)
                sub_sch["FusedKV_0.fused_linear"].sync(
                    mode="bwd_post", sync_op_or_fn="all_reduce"
                )

                # q is not fused so we shard it along.
                sub_sch["ShardableQ_0.query"].shard("weight", axis=0)
                sub_sch["ShardableQ_0.query"].sync(
                    mode="bwd_post", sync_op_or_fn="all_reduce"
                )
        else:
            # Self attention can fuse q, k, v.

            class FusedQKV(nn.Module):
                def __init__(self, num_heads, d_model, d_kv) -> None:
                    super().__init__()
                    self.hidden_size = d_model
                    self.num_heads = num_heads
                    self.key_value_proj_dim = d_kv
                    self.inner_dim = num_heads * self.key_value_proj_dim
                    self.fused_linear = nn.Linear(
                        self.hidden_size, self.inner_dim * 3, bias=False
                    )

                def reshape_for_scores(self, x):
                    new_x_shape = x.size()[:-1] + (
                        self.num_heads // sch.world_size,
                        self.key_value_proj_dim,
                        3,
                    )
                    x = x.view(new_x_shape)
                    return x.contiguous()

                def forward(self, hidden_states):
                    qkv = self.fused_linear(hidden_states)
                    reshaped_qkv = self.reshape_for_scores(qkv)
                    q, k, v = torch.split(reshaped_qkv, 1, dim=-1)
                    q = torch.squeeze(q, -1).contiguous()
                    k = torch.squeeze(k, -1).contiguous()
                    v = torch.squeeze(v, -1).contiguous()
                    return [q, k, v]

            def pattern(x: torch.Tensor) -> torch.Tensor:
                x = call_module("query|key|value", x)
                new_x_shape = x.size()[:-1] + (num_heads, d_kv)
                x = x.view(new_x_shape)
                return x

            subgraphs = sub_sch.find(pattern)
            assert len(subgraphs) == 3
            new_fused_qkv = FusedQKV(num_heads, hidden_size, d_kv)
            sub_sch.replace(new_fused_qkv, subgraphs)
            if sch.world_size > 1:
                sub_sch["FusedQKV_0.fused_linear"].shard("weight", axis=0)
                sub_sch["FusedQKV_0.fused_linear"].sync(
                    mode="bwd_post", sync_op_or_fn="all_reduce"
                )

        if sch.world_size > 1:
            fix_shape_cnt += fix_position_bias_shape(sub_sch)
            sch[f"{prefix}.out"].shard("weight", axis=1)
            sch[f"{prefix}.out"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")
        cnt += 1

    return cnt, fix_shape_cnt


def shard_word_embedding(sch, vocab_size, word_embed_name="shared"):
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


# pylint: disable=dangerous-default-value
def shard_mlp(sch, model_config, path, fc_names=["wi", "wo"]):
    if sch.world_size == 1:
        return
    assert not model_config.is_gated_act, "Gated activation is not supported yet."

    for idx in range(model_config.num_hidden_layers):
        prefix = path.replace("N", str(idx))
        sch[f"{prefix}.{fc_names[0]}"].shard("weight", axis=0)
        sch[f"{prefix}.{fc_names[0]}"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
        sch[f"{prefix}.{fc_names[1]}"].shard("weight", axis=1)
        sch[f"{prefix}.{fc_names[1]}"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")


def _checkpoint(sch, model_config, path, ckpt_ratio=1.0):
    if ckpt_ratio == 0.0:
        return 0

    n_ckpt = int(model_config.num_hidden_layers * ckpt_ratio)
    for idx in range(n_ckpt):
        sch[path.replace("N", str(idx))].checkpoint()
    return n_ckpt


def broadcast_input(sch):
    def broadcast(inputs):
        for inp in inputs:
            dist.broadcast(inp, src=0, group=sch.group)
        return inputs

    sch.sync(mode="fwd_pre", sync_op_or_fn=broadcast)
