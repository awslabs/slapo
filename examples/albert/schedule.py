# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import torch
import torch.distributed as dist
import torch.nn as nn

from slapo import init_empty_weights
from typing import Optional


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
    # EPOI attention module uses repeat to process attention mask to
    # align xformer attention mask shape:
    # (B, 1, 1, S) -repeat->  (B, H, S, S) -reshape-> (B x H, S, S),
    # so we need to replace "repeat" wit the sharded H.
    ops = sch.find(
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
    attn_path="encoder.albert_layer_groups.N.albert_layers.N",
    delay_init=True,
):
    from epoi.inject.policy.bert import InjectHFBertSelfAttentionPolicy
    from epoi.ops.xformers_attn import GenericSelfAttention
    from transformers.activations import ACT2FN
    from transformers.pytorch_utils import apply_chunking_to_forward

    # TODO: Use subgraph matching to obtain FFN module
    class FFN(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.chunk_size_feed_forward = config.chunk_size_feed_forward
            self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
            self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
            self.activation = ACT2FN[config.hidden_act]
            self.dropout = nn.Dropout(float(config.hidden_dropout_prob))
            self.seq_len_dim = 1
            self.full_layer_layer_norm = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )

        def forward(self, context_layer: torch.Tensor):
            ffn_output = apply_chunking_to_forward(
                self.ff_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                context_layer,
            )
            hidden_states = self.full_layer_layer_norm(ffn_output + context_layer)

            return (hidden_states,)

        def ff_chunk(self, attention_output: torch.Tensor) -> torch.Tensor:
            ffn_output = self.ffn(attention_output)
            ffn_output = self.activation(ffn_output)
            ffn_output = self.ffn_output(ffn_output)
            return ffn_output

    class AlbertXFAttention(nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            self.self_attn = GenericSelfAttention(**kwargs)
            self.output_dropout = nn.Dropout(float(config.hidden_dropout_prob))
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None,
            head_mask: Optional[torch.FloatTensor] = None,
            output_attentions: bool = False,
        ):
            outputs = self.self_attn(hidden_states, attention_mask)
            context_layer = outputs[0]

            projected_context_layer = self.dense(context_layer)
            projected_context_layer_dropout = self.output_dropout(
                projected_context_layer
            )
            layernormed_context_layer = self.LayerNorm(
                hidden_states + projected_context_layer_dropout
            )
            return (layernormed_context_layer, None)

    class AlbertLayer(nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            self.config = config
            self.attention = AlbertXFAttention(config, **kwargs)
            if dist.get_world_size() == 1:
                # FIXME: Avoid hardcoding
                device = "cuda"
                output = torch.ones((8, 512, 1024), dtype=torch.float16, device=device)
                self.ffn = FFN(config).half().cuda()
                self.ffn = torch.jit.trace(self.ffn, [output])
            else:
                self.ffn = FFN(config)

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
        ):
            attention_output = self.attention(
                hidden_states, attention_mask, head_mask, output_attentions
            )
            res = self.ffn(attention_output[0])
            return res

    cnt = 0
    for idx in range(1):  # use layer group
        prefix = attn_path.replace("N", str(idx))
        sub_sch = sch[f"{prefix}.attention"]
        init_config = InjectHFBertSelfAttentionPolicy.gen_init_config_from_object(
            sub_sch.mod
        )
        sch[f"{prefix}"].replace(AlbertLayer(config, **init_config))

        num_layers, num_heads, hidden_size = (
            config.num_hidden_layers,
            config.num_attention_heads,
            config.hidden_size,
        )

        sch[f"{prefix}.attention.self_attn"].trace(
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

        sub_sch = sch[f"{prefix}.attention"]
        subgraphs = sub_sch["self_attn"].find("query|key|value", pattern)
        assert len(subgraphs) != 0
        with init_empty_weights(enable=delay_init):
            new_fused_qkv = FusedQKV(hidden_size, num_heads)
        sub_sch["self_attn"].replace(new_fused_qkv, subgraphs)
        if sch.world_size > 1:
            sub_sch["self_attn.FusedQKV_0.fused_linear"].shard("weight", axis=0)
            sub_sch["self_attn.FusedQKV_0.fused_linear"].shard("bias", axis=0)
            sub_sch["self_attn.FusedQKV_0.fused_linear"].sync(mode="backward")
            fix_attention_mask_shape(sub_sch["self_attn"])
            sub_sch["dense"].shard("weight", axis=1)
            sub_sch["dense"].sync(mode="forward")
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

    def fw_pre_hook(_input):
        # Mask the input
        input_mask = (_input[0] < vocab_start_index) | (_input[0] >= vocab_end_index)
        masked_input = _input[0].clone() - vocab_start_index
        masked_input[input_mask] = 0
        return masked_input

    sch[word_embed_name].hook("fw_pre", fw_pre_hook)

    def fw_post_hook(_input, output):
        # Mask the output embedding
        input_mask = (_input[0] < vocab_start_index) | (_input[0] >= vocab_end_index)
        output[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=sch.group)
        return output

    sch[word_embed_name].hook("fw_post", fw_post_hook)


def shard_mlp(
    sch,
    config,
    path="encoder.albert_layer_groups.N.albert_layers.N",
    fc_names=["ffn.ffn", "ffn.ffn_output"],
):
    if sch.world_size == 1:
        return

    for idx in range(1):  # use layer group
        prefix = path.replace("N", str(idx))
        sch[f"{prefix}.{fc_names[0]}"].shard("weight", axis=0)
        sch[f"{prefix}.{fc_names[0]}"].shard("bias", axis=0)
        sch[f"{prefix}.{fc_names[0]}"].sync(mode="backward")
        sch[f"{prefix}.{fc_names[1]}"].shard("weight", axis=1)
        sch[f"{prefix}.{fc_names[1]}"].sync(mode="forward")


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

    sch.hook("fw_pre", broadcast_input)
