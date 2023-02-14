# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace GPT-Neo with model schedule."""
# pylint: disable=unused-argument

import inspect

from torch import nn
from torch.distributed import distributed_c10d as dist

from ..schedule import create_schedule
from ..initialization import init_empty_weights
from ..random import get_cuda_rng_tracker
from ..op import FlashAttention, FlashAttentionOp, FusedMLP
from ..sharding import reduce_forward_output
from ..logger import get_logger
from .registry import register_schedule_method


@register_schedule_method()
def apply_schedule(
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
    shard_parameters(sch, model_config, sch_config)

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


def shard_parameters(sch, model_config, sch_config):
    model_name = model_config._name_or_path
    logger = get_logger(model_name)
    prefix = sch_config.get("prefix", "")
    delay_init = sch_config.get("delay_init", True)
    sequence_parallel = sch_config.get("sequence_parallel", False)
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
        sequence_parallel=sequence_parallel,
    )
    logger.info(
        "Replace %d attention layers with %s op", cnt, applied_attn_op_name, ranks=0
    )

    # Shard other parameters if MP group > 1.
    if sch.world_size > 1:
        replace_and_shard_mlp(
            sch[prefix],
            model_config,
            delay_init=delay_init,
            sequence_parallel=sequence_parallel,
        )
        head_sch = sch["lm_head"] if "lm_head" in sch else None
        shard_word_embedding(
            sch[prefix],
            head_sch,
            model_config.vocab_size,
            sequence_parallel=sequence_parallel,
        )


def generate_pipeline_schedule(sch, sch_config):
    pipeline_cuts = sch_config.get("pipeline_cuts", None)
    prefix = sch_config.get("prefix", "")
    # Cut pipeline stages.
    if pipeline_cuts:
        input_names = ["input_ids", "attention_mask", "position_ids"]
        sig = inspect.signature(sch.mod.forward)
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


def trace_attention(sch, model_config, attn_path="h.N.attn.attention"):
    cnt = 0
    for idx in range(model_config.num_layers):
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
    model_config,
    attn_path="h.N.attn.attention",
    delay_init=True,
    attn_op_name="cuda",
    sequence_parallel=False,
):
    init_config = dict(
        hidden_size=model_config.hidden_size,
        num_attention_heads=model_config.num_attention_heads,
        output_proj=True,
        attn_pdrop=model_config.attention_dropout,
        resid_pdrop=model_config.resid_dropout,
        attn_op_name=attn_op_name,
        fused_qkv=True,
        bias=False,  # GPT-Neo does not use bias in attention.
    )

    class Attention(nn.Module):
        """A wrapper to align the original GPTNeoAttention forward signature."""

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
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
        ):
            """Match the original GPTNeoAttention forward signature."""
            outputs = self.module(
                hidden_states,
                attention_mask,
                layer_past,
                head_mask,
                None,
                None,
                use_cache,
                output_attentions,
            )
            # FIXME: The original output is (hidden_states, None) where the None
            # is present_key_value and only used by in inference.
            return outputs[:1]

    class AttentionOpWithRNG(FlashAttentionOp):
        def forward(self, query_layer, key_layer, value_layer, attention_mask, p):
            with get_cuda_rng_tracker().fork():
                return super().forward(
                    query_layer, key_layer, value_layer, attention_mask, p
                )

    cnt = 0
    attn_op = []
    for idx in range(model_config.num_layers):
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
            sub_sch["module.out_proj"].shard("weight", axis=1)
            fix_attention_mask_shape(sub_sch["module"])

            if sequence_parallel:
                sub_sch["module.qkv"].sync(
                    mode="fwd_pre", sync_op_or_fn="all_gather", axis=1
                )

                sub_sch["module.out_proj"].sync(
                    mode="fwd_post", sync_op_or_fn="reduce_scatter", axis=1
                )
            else:
                # Shard qkv and output projection.
                sub_sch["module.qkv"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
                sub_sch["module.out_proj"].sync(
                    mode="fwd_post", sync_op_or_fn="all_reduce"
                )

                # In this case, the attention dropout in between has to
                # use different random seeds.
                new_op = AttentionOpWithRNG(
                    sub_sch["module"]["attn_op"].mod.attn_op_name,
                    sub_sch["module"]["attn_op"].mod.apply_causal_mask,
                    sub_sch["module"]["attn_op"].mod.scale,
                )
                sub_sch["module"]["attn_op"].replace(new_op)

        cnt += 1

    # Check if all attention ops are the same.
    attn_op = list(set(attn_op))
    if len(attn_op) > 1:
        raise RuntimeError(
            f"The attention op is not consistent across layers, including {attn_op}"
        )

    return cnt, attn_op[0]


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
        output = reduce_forward_output(output, sch.group)
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


# pylint: disable=dangerous-default-value
def replace_and_shard_mlp(
    sch,
    model_config,
    path="h.N.mlp",
    fc_names=["c_fc", "c_proj"],
    delay_init=True,
    sequence_parallel=False,
):
    for idx in range(model_config.num_layers):
        prefix = path.replace("N", str(idx))
        if model_config.activation_function in {"gelu", "gelu_new"}:
            sub_sch = sch[prefix]
            inter_size, hidden_size = sub_sch.mod.c_fc.weight.shape
            with init_empty_weights(enable=delay_init):
                new_mod = FusedMLP(
                    hidden_size,
                    inter_size,
                    model_config.activation_function,
                    model_config.resid_dropout,
                )
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


def checkpoint(
    sch, model_config, path="h.N", ckpt_ratio=1.0, checkpoint_method="uniform"
):
    if ckpt_ratio == 0.0:
        return 0

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

    n_ckpt = int(model_config.num_hidden_layers * ckpt_ratio)
    if checkpoint_method == "head":
        for idx in range(n_ckpt):
            sch[path.replace("N", str(idx))].checkpoint(order_args_fn=order_args_fn)
    elif checkpoint_method == "uniform" and ckpt_ratio > 0:
        for idx in range(
            0, model_config.num_hidden_layers, max(1, int(1 / ckpt_ratio))
        ):
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
