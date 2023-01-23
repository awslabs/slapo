# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import torch
import torch.nn as nn
import torch.distributed as dist
from slapo import init_empty_weights


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
        new_mod = RelativeBias(
            old_mod.relative_attention_num_buckets,
            old_mod.relative_attention_max_distance,
            old_mod.n_heads // sch.world_size,
            old_mod.is_decoder,
        )
        sch["relative_attention_bias"].replace(new_mod)
        cnt += 1
    return cnt


def replace_and_shard_attention(
    sch, config, attn_path, cross_attn=False, delay_init=True
):
    from epoi.inject.policy.t5 import InjectHFT5AttentionPolicy
    from epoi.ops.xformers_attn import T5Attention

    num_layers, num_heads, hidden_size, d_kv = (
        config.num_hidden_layers,
        config.num_attention_heads,
        config.hidden_size,
        config.d_kv,
    )

    cnt = 0
    fix_shape_cnt = 0
    for idx in range(num_layers):
        prefix = attn_path.replace("N", str(idx))
        sub_sch = sch[f"{prefix}"]
        init_config = InjectHFT5AttentionPolicy.gen_init_config_from_object(sub_sch.mod)
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
                    return [states]

            def pattern(x: torch.Tensor) -> torch.Tensor:
                new_x_shape = x.size()[:-1] + (num_heads, d_kv)
                x = x.view(new_x_shape)
                return x

            subgraphs = sub_sch.find("key|value", pattern)
            assert len(subgraphs) != 0
            with init_empty_weights(enable=delay_init):
                new_fused_kv = FusedKV(num_heads, hidden_size, d_kv)
            sub_sch.replace(new_fused_kv, subgraphs)

            subgraphs = sub_sch.find("query", pattern)
            assert len(subgraphs) != 0
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
                new_x_shape = x.size()[:-1] + (num_heads, d_kv)
                x = x.view(new_x_shape)
                return x

            subgraphs = sub_sch.find("query|key|value", pattern)
            assert len(subgraphs) != 0
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


def shard_mlp(sch, config, path, fc_names=["wi", "wo"]):
    if sch.world_size == 1:
        return
    assert not config.is_gated_act, "Gated activation is not supported yet."

    for idx in range(config.num_hidden_layers):
        prefix = path.replace("N", str(idx))
        sch[f"{prefix}.{fc_names[0]}"].shard("weight", axis=0)
        sch[f"{prefix}.{fc_names[0]}"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
        sch[f"{prefix}.{fc_names[1]}"].shard("weight", axis=1)
        sch[f"{prefix}.{fc_names[1]}"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")


def checkpoint(sch, config, path, ckpt_ratio=1.0):
    if ckpt_ratio == 0.0:
        return

    n_ckpt = int(config.num_hidden_layers * ckpt_ratio)
    for idx in range(n_ckpt):
        sch[path.replace("N", str(idx))].checkpoint()
    return n_ckpt


def broadcast_input(sch):
    def broadcast_input(inputs):
        for inp in inputs:
            dist.broadcast(inp, src=0, group=sch.group)
        return inputs

    sch.sync(mode="fwd_pre", sync_op_or_fn=broadcast_input)
