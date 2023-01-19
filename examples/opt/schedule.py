# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import torch
import torch.nn as nn
import torch.distributed as dist

from slapo import init_empty_weights


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


def replace_qkv(sch, config, attn_path="h.N.attn.attention"):
    """Untested."""
    num_layers, num_heads, hidden_size = (
        config.num_hidden_layers,
        config.num_heads,
        config.hidden_size,
    )

    class FusedQKV(nn.Module):
        def __init__(self, hidden_size, num_heads) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_size = hidden_size // num_heads
            self.fused_linear = nn.Linear(hidden_size, num_heads * self.head_size * 3)

        def transpose_for_scores(self, x):
            new_x_shape = x.size()[:-1] + (
                self.num_heads // sch.world_size,
                self.head_size,
                3,
            )
            x = x.view(new_x_shape)
            return x.permute(0, 2, 1, 3, 4)

        def forward(self, hidden_states):
            qkv = self.fused_linear(hidden_states)
            transposed_qkv = self.transpose_for_scores(qkv)
            q, k, v = torch.split(transposed_qkv, 1, dim=-1)
            q = torch.squeeze(q, -1)
            k = torch.squeeze(k, -1)
            v = torch.squeeze(v, -1)
            return [q, k, v]

    def pattern(x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (num_heads, hidden_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    cnt = 0
    for idx in range(num_layers):
        sub_sch = sch[attn_path.replace("N", str(idx))]
        subgraphs = sub_sch.find("k_proj|q_proj|v_proj", pattern)
        assert subgraphs, "Cannot find QKV pattern"
        new_mod = FusedQKV(hidden_size, num_heads)
        sub_sch.replace(new_mod, subgraphs)
        cnt += 1
    return cnt


def fix_attention_mask_shape(sch):
    # EPOI attention module uses repeat to process attention mask to
    # align xformer attention mask shape:
    # (B, 1, Tgt_S, Src_S)
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
        new_args = (old_args[0],) + (old_args[1] // sch.world_size, 1, 1)
        return tensor.repeat(*new_args)

    for op in ops:
        sch.replace(new_repeat, op[1])


def replace_and_shard_attention(
    sch, config, attn_path="decoder.layers.N.self_attn", delay_init=True
):
    from epoi.inject.policy.gpt import InjectHFGPTAttentionPolicy
    from epoi.ops.xformers_attn import GenericSelfAttention

    class SelfAttention(nn.Module):
        """A wrapper to align the original GPTNeoAttention forward signature."""

        def __init__(self, **kwargs):
            super().__init__()
            self.module = GenericSelfAttention(**kwargs)

        def forward(
            self,
            hidden_states,
            past_key_value=None,
            attention_mask=None,
            layer_head_mask=None,
            output_attentions=False,
            use_cache=False,
        ):
            outputs = self.module(
                hidden_states, attention_mask, past_key_value, use_cache
            )
            # FIXME: The original output is (hidden_states, None) where the None
            # is present_key_value and only used by in inference.
            # OPT output is (hidden_states, self_attn_weights, present_key_value)
            return outputs[0], None, None

    num_layers, num_heads, hidden_size = (
        config.num_hidden_layers,
        config.num_attention_heads,
        config.hidden_size,
    )

    cnt = 0
    for idx in range(num_layers):
        sub_sch = sch[attn_path.replace("N", str(idx))]
        init_config = InjectHFGPTAttentionPolicy.gen_init_config_from_object(
            sub_sch.mod
        )
        with init_empty_weights(enable=delay_init):
            new_mod = SelfAttention(**init_config)
        sub_sch.replace(new_mod)
        sub_sch.trace(
            tracer="pytorch",
            leaf_modules=["MemoryEfficientAttentionOp"],
            concrete_args={
                "layer_past": None,
                "use_cache": False,
            },
        )
        fix_attention_mask_shape(sub_sch["module"])

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
            sub_sch["module.FusedQKV_0.fused_linear"].sync(mode="backward")
            sub_sch["module.out_proj"].shard("weight", axis=1)
            sub_sch["module.out_proj"].sync(
                mode="forward", sync_op="reduce_scatter", axis=1
            )
            sub_sch["module.resid_dropout"].sync(
                mode="forward", sync_op="all_gather", axis=1
            )
        cnt += 1

    return cnt


def remove_cast(sch, config, attn_path="h.N.attn.attention"):
    """[Untested] Remove .to(torch.float32) in GPT-Neo attention to align
    HF and Megatron GPT-2 behavior.
    """
    cnt = 0
    for idx in range(config.num_hidden_layers):
        sub_sch = sch[attn_path.replace("N", str(idx))]
        ops = sub_sch.find(
            lambda node: node.op == "call_method"
            and node.target == "to"
            and len(node.args) == 2
            and node.args[1] == torch.float32
        )

        for op in ops:
            sub_sch.replace(lambda x, *args: x, op[1])
            cnt += 1
    return cnt


def shard_word_embedding(sch, vocab_size, word_embed_name="decoder.embed_tokens"):
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


def shard_qkv(
    sch,
    config,
    attn_path="h.N.attn.attention",
    qkv_name="FusedQKV_0",
    out_proj_name="out_proj",
):
    """Untested."""
    num_layers = config.num_hidden_layers

    def fix_shape_after_shard(path):
        # Fix shape of view ops after sharding.
        import operator

        sub_sch = sch[path]
        ops = sub_sch.find(
            lambda node: node.target == "view"
            and len(node.args) == 2
            and node.args[0].target == "contiguous"
            and isinstance(node.args[1], torch.fx.Node)
            and node.args[1].target == operator.add
        )

        def new_view(tensor, old_shape):
            new_shape = old_shape[:-1] + (-1,)
            return tensor.view(new_shape)

        for op in ops:
            sub_sch.replace(new_view, op)

    for idx in range(num_layers):
        prefix = attn_path.replace("N", str(idx))
        sch[f"{prefix}.{qkv_name}.fused_linear"].shard("weight", axis=0)
        sch[f"{prefix}.{qkv_name}.fused_linear"].shard("bias", axis=0)
        sch[f"{prefix}.{qkv_name}.fused_linear"].sync(mode="backward")

        sch[f"{prefix}.{out_proj_name}"].shard("weight", axis=1)
        sch[f"{prefix}.{out_proj_name}"].sync(mode="forward")
        fix_shape_after_shard(prefix)


def replace_and_shard_mlp(
    sch, config, path="decoder.layers.N", fc_names=["fc1", "fc2"], delay_init=True
):
    from epoi.inject.policy.gpt import InjectHFGPTMLPPolicy

    for idx in range(config.num_hidden_layers):
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
                sub_sch["fc_in"].sync(mode="backward")
                sub_sch["fc_out"].shard("weight", axis=1)
                sub_sch["fc_out"].sync(mode="forward")
        elif sch.world_size > 1:
            sch[f"{prefix}.{fc_names[0]}"].shard("weight", axis=0)
            sch[f"{prefix}.{fc_names[0]}"].shard("bias", axis=0)
            sch[f"{prefix}.{fc_names[0]}"].sync(mode="backward")
            sch[f"{prefix}.{fc_names[1]}"].shard("weight", axis=1)
            sch[f"{prefix}.{fc_names[1]}"].sync(mode="forward")


def checkpoint(sch, config, path="decoder.layers.N", ckpt_ratio=1.0):
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

    sch.hook("fw_pre", broadcast_input)
