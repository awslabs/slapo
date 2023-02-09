# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import inspect

import torch.nn as nn
from torch.distributed import distributed_c10d as dist

import slapo
from slapo import init_empty_weights
from slapo.op import FlashSelfAttention, FlashAttentionOp, FusedMLP
from slapo import init_empty_weights, get_cuda_rng_tracker


def replace_attention(
    sch,
    config,
    attn_path="h.N.attn",
    delay_init=True,
    disable_flash_attn=False,
):
    """Replace the attention module with flash attention.

    Parameters
    ----------
    sch : slapo.Schedule
        The schedule of the entire model.
    config : GPT2Config
        The model configuration.
    attn_path : str
        The path to the attention module.
    delay_init : bool
        Whether to delay the initialization of the new module.
    disable_flash_attn : bool
        Whether to disable the flash attention.

    Returns
    -------
    tuple[int, str]
        The number of attention layers replaced and the name of the attention.
    """
    attn_op_name = "native_xformers" if disable_flash_attn else "triton"
    init_config = dict(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        is_decoder=True,
        attn_pdrop=config.attn_pdrop,
        resid_pdrop=config.resid_pdrop,
        attn_op_name=attn_op_name,
        fused_qkv=True,
    )

    class SelfAttention(nn.Module):
        """A wrapper to align the original GPTNeoAttention forward signature."""

        def __init__(self, **kwargs):
            super().__init__()
            try:
                self.module = FlashSelfAttention(**kwargs)
            except Exception as err:
                if kwargs["attn_op_name"] == "native_xformers":
                    raise RuntimeError(
                        f"Failed to create native attention: {err}"
                    ) from None

                # Failed to use the triton kernel. This may due to unsupported
                # GPU (< sm_75) or flash-attention is not installed. Fallback
                # to xFormers' cutlass.
                kwargs["attn_op_name"] = "cutlass"
                self.module = FlashSelfAttention(**kwargs)

        def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
        ):
            outputs = self.module(
                hidden_states,
                layer_past,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                use_cache,
                output_attentions,
            )
            # FIXME: The original output is (hidden_states, None) where the None
            # is present_key_value and only used by in inference.
            return outputs[:1]

    cnt = 0
    attn_op = []
    for idx in range(config.num_hidden_layers):
        sub_sch = sch[attn_path.replace("N", str(idx))]
        with init_empty_weights(enable=delay_init):
            new_mod = SelfAttention(**init_config)
            attn_op.append(new_mod.module.attn_op_name)
        sub_sch.replace(new_mod)
        cnt += 1

    # Check if all attention ops are the same.
    attn_op = list(set(attn_op))
    if len(attn_op) > 1:
        raise RuntimeError(
            f"The attention op is not consistent across layers, including {attn_op}"
        )

    return cnt, attn_op


def replace_mlp(
    sch,
    config,
    path="h.N.mlp",
    delay_init=True,
):
    """Replace the MLP module with a fused MLP module.

    Parameters
    ----------
    sch : slapo.Schedule
        The schedule of the entire model.
    config : GPT2Config
        The model configuration.
    path : str
        The path to the MLP module.
    delay_init : bool
        Whether to delay the initialization of the new module.

    Returns
    -------
    int
        The number of MLP layers replaced.
    """
    for idx in range(config.num_hidden_layers):
        prefix = path.replace("N", str(idx))
        sub_sch = sch[prefix]
        hidden_size, inter_size = sub_sch.mod.c_fc.weight.shape
        with init_empty_weights(enable=delay_init):
            new_mod = FusedMLP(
                hidden_size,
                inter_size,
                config.activation_function,
                config.resid_pdrop,
            )
        sub_sch.replace(new_mod)
    return config.num_hidden_layers


def gen_embedding_hooks(sch, vocab_size):
    """Generate hooks for input embedding layer to deal with word embedding sharding.

    Parameters
    ----------
    sch : slapo.Schedule
        The schedule of the entire model.
    vocab_size : int
        The total vocabulary size.

    Returns
    -------
    tuple(callable, callable)
        The forward pre-hook and post-hook for the embedding layer.
    """
    vocab_start_index = sch.rank * vocab_size // sch.world_size
    vocab_end_index = (sch.rank + 1) * vocab_size // sch.world_size

    def fwd_pre_hook(_module, _input):
        # Mask the input
        input_mask = (_input[0] < vocab_start_index) | (_input[0] >= vocab_end_index)
        masked_input = _input[0].clone() - vocab_start_index
        masked_input[input_mask] = 0
        return masked_input

    def fwd_post_hook(_module, _input, output):
        # Mask the output embedding
        input_mask = (_input[0] < vocab_start_index) | (_input[0] >= vocab_end_index)
        output[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs
        output = slapo.sharding.reduce_forward_output(output, sch.group)
        return output

    return fwd_pre_hook, fwd_post_hook


def fix_attention_mask_shape(sch):
    """A utility function to fix the attention mask shape.
    The input attention mask shape is (B, 1, 1, S) where S is the sequence length.
    However, xFormers kernels expect (B, H, S, S) where H is the number of heads.
    Since we shard H to be H // world_size, we need to fix this expand op.

    Parameters
    ----------
    sch : slapo.Schedule
        The schedule of the entire model.
    """
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


class AttentionOpWithRNG(FlashAttentionOp):
    def forward(self, query_layer, key_layer, value_layer, attention_mask, p):
        with get_cuda_rng_tracker().fork():
            return super().forward(
                query_layer, key_layer, value_layer, attention_mask, p
            )


def shard(
    sch,
    head_sch,
    config,
    shard_target,
    sequence_parallel=False,
):
    """Shard the model for tensor parallelism. This function assumes
    the attention layers are already replaced with the Slapo ops.

    Parameters
    ----------
    sch : slapo.Schedule
        The schedule of the transformer model.
    head_sch : slapo.Schedule
        The schedule of the lm_head.
    config: GPT2Config
        The configuration of the model.
    shard_target : list[str]
        The sharding target. This function shards the corresponding layers
        specified in this target. It could include "embed", "attention", "mlp".
    sequence_parallel : bool
        Whether to use sequence parallelism. Default False.
    """

    if sch.world_size == 1:
        return

    # Shard input embedding.
    if "embed" in shard_target:
        word_embed_name, pos_embed_name, final_ln_name = "wte", "wpe", "ln_f"

        sch[word_embed_name].shard("weight", axis=0)
        fwd_pre_hook, fwd_post_hook = gen_embedding_hooks(sch, config.vocab_size)
        sch[word_embed_name].sync(mode="fwd_pre", sync_op_or_fn=fwd_pre_hook)
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

    # Shard attention.
    if "attention" in shard_target:
        path = "h.N.attn"

        for idx in range(config.num_hidden_layers):
            sub_sch = sch[path.replace("N", str(idx))]
            sub_sch["module.qkv"].shard("weight", axis=0)
            sub_sch["module.qkv"].shard("bias", axis=0)
            sub_sch["module.out_proj"].shard("weight", axis=1)

            # Fix attention mask shape to consider the sharded size. This requires
            # the attention module to be in static graph, and thus we need to trace it.
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

    # Shard MLP.
    if "mlp" in shard_target:
        path = "h.N.mlp"

        for idx in range(config.num_hidden_layers):
            prefix = path.replace("N", str(idx))
            sub_sch = sch[prefix]

            # The name of first linear, the linear that bias belongs to,
            # and last linear layers.
            # fc_names = ["c_fc", "c_fc", "c_proj"]
            # FIXME: If MLP is not replaced, the weights in GPT2MLP are transposed
            # and we don't handle that right now.
            assert "act" in sub_sch
            # When the MLP is replaced, the bias of the first linear
            # is splitted to the new module of fused activation layer.
            fc_names = ["fc_in", "act", "fc_out"]

            sub_sch[fc_names[0]].shard("weight", axis=0)
            sub_sch[fc_names[1]].shard("bias", axis=0)
            sub_sch[fc_names[2]].shard("weight", axis=1)

            if sequence_parallel:
                sub_sch[fc_names[0]].sync(
                    mode="fwd_pre", sync_op_or_fn="all_gather", axis=1
                )
                sub_sch[fc_names[2]].sync(
                    mode="fwd_post", sync_op_or_fn="reduce_scatter", axis=1
                )
            else:
                sub_sch[fc_names[0]].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
                sub_sch[fc_names[2]].sync(mode="fwd_post", sync_op_or_fn="all_reduce")


def checkpoint(sch, config, path="h.N", ckpt_ratio=1.0, checkpoint_method="uniform"):
    """Add activation checkpointing to the model. The ckpt_ratio specifies
    the ratio of the attention layers to be checkpointed. For example, if
    ckpt_ratio is 0.5, then half of the attention layers will be checkpointed.

    Parameters
    ----------
    sch : slapo.Schedule
        The schedule of the model.
    config : GPT2Config
        The configuration of the model.
    path : str
        The path to the attention layer. Default: "h.N.attn".
    ckpt_ratio : float
        The ratio of the attention layers to be checkpointed. Default: 1.0.
    """
    if ckpt_ratio == 0.0:
        return

    def order_args_fn(*args, **kwargs):
        assert len(args) == 1
        attention_mask = kwargs.get("attention_mask", None)
        head_mask = kwargs.get("head_mask", None)
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
        encoder_attention_mask = kwargs.get("encoder_attention_mask", None)
        output_attentions = kwargs.get("output_attentions", False)
        # Forwards: (
        #   hidden_states,
        #   layer_past,
        #   attention_mask,
        #   head_mask,
        #   encoder_hidden_states,
        #   encoder_attention_mask,
        #   use_cache,
        #   output_attentions
        # )
        return (
            args[0],
            None,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            False,
            output_attentions,
        )

    n_ckpt = int(config.num_hidden_layers * ckpt_ratio)
    if checkpoint_method == "head":
        for idx in range(n_ckpt):
            sch[path.replace("N", str(idx))].checkpoint(order_args_fn=order_args_fn)
    elif checkpoint_method == "uniform" and ckpt_ratio > 0:
        for idx in range(0, config.num_hidden_layers, max(1, int(1 / ckpt_ratio))):
            sch[path.replace("N", str(idx))].checkpoint(order_args_fn=order_args_fn)
    return n_ckpt


def broadcast_input(sch):
    """Add a hook in the beinning of the model to broadcast the input.
    This is used when tensor parallelism is used and the runtime does not
    broadcast the input automatically.

    Parameters
    ----------
    sch : slapo.Schedule
        The schedule of the model.
    """
    group_src_rank = dist.get_global_rank(sch.group, 0)

    def _broadcast_input(module, inputs):
        for inp in inputs:
            if inp is not None:
                dist.broadcast(inp, src=group_src_rank, group=sch.group)
        return inputs

    sch.sync(mode="fwd_pre", sync_op_or_fn=_broadcast_input)


def pipeline(sch, prefix, pipeline_cuts):
    """Trace the top module of the model and cut the pipeline stages.

    Parameters
    ----------
    sch : slapo.Schedule
        The schedule to be cut.
    prefix : str
        The prefix of the top module.
    pipeline_cuts : list[str]
        The list of attention layer index indicating cut points.
    """
    input_names = ["input_ids", "attention_mask", "position_ids"]
    sig = inspect.signature(sch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    sch.trace_for_pipeline(
        f"{prefix}", tracer="huggingface", concrete_args=concrete_args
    )
    _prefix = f"{prefix}." if prefix else ""
    for cut in pipeline_cuts:
        sch[f"{_prefix}h.{cut}"].cut_pipeline_stage()
