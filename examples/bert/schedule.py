# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import torch
import torch.distributed as dist
import torch.nn as nn

from slapo import init_empty_weights


def trace_attention(sch, config, attn_path="encoder.layer.N.attention"):
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
    # EPOI attention module uses repeat to process attention mask to
    # align xformer attention mask shape:
    # (B, 1, 1, S) -repeat->  (B, H, S, S) -reshape-> (B x H, S, S),
    # so we need to replace "repeat" wit the sharded H.
    ops = sch.find_node(
        lambda node: node.op == "call_method"
        and node.target == "repeat"
        and len(node.args) == 5  # args[0] is self
        and node.args[1] == 1
        and node.args[-1] == 1
    )

    def new_repeat(tensor, *old_args):
        assert len(old_args) == 4
        new_args = (old_args[0],) + (old_args[1] // sch.world_size,) + old_args[2:]
        return tensor.repeat(*new_args)

    for op in ops:
        sch.replace(new_repeat, op[1])


def replace_and_shard_attention(
    sch,
    config,
    attn_path="encoder.layer.N.attention",
    delay_init=True,
    disable_flash_attn=False,
):
    from epoi.inject.policy.bert import InjectHFBertSelfAttentionPolicy
    from epoi.ops.xformers_attn import GenericSelfAttention

    class SelfAttention(nn.Module):
        """A wrapper to align the original BertSelfAttention forward signature."""

        def __init__(self, **kwargs):
            super().__init__()
            self.module = GenericSelfAttention(**kwargs)

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
        ):
            outputs = self.module(hidden_states, attention_mask, past_key_value, False)
            # FIXME: The original output is (hidden_states, None) where the None
            # is present_key_value and only used by decoder (e.g., GPT).
            return outputs[:1]

    num_layers, num_heads, hidden_size = (
        config.num_hidden_layers,
        config.num_attention_heads,
        config.hidden_size,
    )

    cnt = 0
    for idx in range(num_layers):
        prefix = attn_path.replace("N", str(idx))
        sub_sch = sch[f"{prefix}.self"]
        init_config = InjectHFBertSelfAttentionPolicy.gen_init_config_from_object(
            sub_sch.mod
        )
        if disable_flash_attn:
            init_config["attn_op_name"] = "native"
        with init_empty_weights(enable=delay_init):
            new_mod = SelfAttention(**init_config)
        sub_sch.replace(new_mod)
        sub_sch.trace(
            tracer="pytorch",
            leaf_modules=["MemoryEfficientAttentionOp"],
            concrete_args={
                "past_key_value": None,
                "layer_past": None,
                "use_cache": False,
            },
        )

        class FusedQKV(nn.Module):
            def __init__(self, hidden_size, num_heads) -> None:
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_size = hidden_size // num_heads
                self.fused_linear = nn.Linear(
                    hidden_size, self.num_heads * self.head_size * 3
                )

            def reshape_for_scores(self, x):
                new_x_shape = x.size()[:-1] + (
                    self.num_heads // sch.world_size,
                    self.head_size,
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
            new_x_shape = x.size()[:-1] + (num_heads, hidden_size)
            x = x.view(new_x_shape)
            return x

        subgraphs = sub_sch["module"].find("query|key|value", pattern)
        assert len(subgraphs) != 0
        with init_empty_weights(enable=delay_init):
            new_fused_qkv = FusedQKV(hidden_size, num_heads)
        sub_sch["module"].replace(new_fused_qkv, subgraphs)
        if sch.world_size > 1:
            sub_sch["module.FusedQKV_0.fused_linear"].shard("weight", axis=0)
            sub_sch["module.FusedQKV_0.fused_linear"].shard("bias", axis=0)
            sub_sch["module.FusedQKV_0.fused_linear"].sync(
                mode="bwd_post", sync_op_or_fn="all_reduce"
            )
            fix_attention_mask_shape(sub_sch["module"])
            sch[f"{prefix}.output.dense"].shard("weight", axis=1)
            sch[f"{prefix}.output.dense"].sync(
                mode="fwd_post", sync_op_or_fn="all_reduce"
            )
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


def shard_mlp(sch, config, path="encoder.layer.N", fc_names=["intermediate", "output"]):
    if sch.world_size == 1:
        return

    for idx in range(config.num_hidden_layers):
        prefix = path.replace("N", str(idx))
        sch[f"{prefix}.{fc_names[0]}.dense"].shard("weight", axis=0)
        sch[f"{prefix}.{fc_names[0]}.dense"].shard("bias", axis=0)
        sch[f"{prefix}.{fc_names[0]}.dense"].sync(
            mode="bwd_post", sync_op_or_fn="all_reduce"
        )
        sch[f"{prefix}.{fc_names[1]}.dense"].shard("weight", axis=1)
        sch[f"{prefix}.{fc_names[1]}.dense"].sync(
            mode="fwd_post", sync_op_or_fn="all_reduce"
        )


def checkpoint(sch, config, path="encoder.layer.N", ckpt_ratio=1.0):
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
