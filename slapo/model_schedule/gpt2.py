# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace GPT-2 with model schedule."""
# pylint: disable=unused-argument
import inspect
import json

from torch import nn
from torch.distributed import distributed_c10d as dist

from ..schedule import create_schedule
from ..op import FlashAttention, FusedMLP, LinearWithSyncFunc
from ..initialization import init_empty_weights
from ..sharding import reduce_forward_output
from ..logger import get_logger
from .registry import register_schedule


@register_schedule()
def _apply_schedule(
    model,
    **sch_config,
):
    # Extract configs.
    model_config = sch_config.get("model_config", None)
    fp16 = sch_config.get("fp16", False)
    bf16 = sch_config.get("bf16", False)
    group = sch_config.get("group", None)
    prefix = sch_config.get("prefix", "")
    bcast_input = sch_config.get("bcast_input", False)
    ckpt_ratio = sch_config.get("ckpt_ratio", 0.0)
    checkpoint_method = sch_config.get("checkpoint_method", "uniform")
    pipeline_cuts = sch_config.get("pipeline_cuts", None)
    sequence_parallel = sch_config.get("sequence_parallel", False)

    # Validate config.
    if model_config is None:
        raise ValueError(
            "Model config is not specified in sch_config. Please provide "
            "`model_config` in the kwarg."
        )
    if fp16 and bf16:
        raise ValueError("Cannot use both fp16 and bf16")

    # Get model name and setup logger.
    try:
        model_name = model_config._name_or_path
    except Exception:
        model_name = model_config.get("_name_or_path", None)
    logger = get_logger(f"{model_name}")

    # Change data type.
    if fp16:
        logger.info("Change model dtype to fp16", ranks=0)
        model.half()
    elif bf16:
        logger.info("Change model dtype to bf16", ranks=0)
        model.bfloat16()
    else:
        logger.info("Use fp32 as default model dtype", ranks=0)

    sch = create_schedule(model, group=group)
    logger.info(
        "Scheduling %s with TP=%d, config: %s",
        model_name,
        sch.world_size,
        get_sch_config_str(sch_config),
        ranks=0,
    )

    # Replace modules.
    logger.info("Replace Attention and MLP modules", ranks=0)
    for name, cnt in replace_modules(sch[prefix], model_config, sch_config).items():
        logger.info("Replace %d %s", cnt, name, ranks=0)

    # Tensor parallelism.
    if sch.world_size > 1:
        logger.info("Shard model parameters", ranks=0)
        head_sch = sch["lm_head"] if "lm_head" in sch else None
        shard_target = ["embed", "attention", "mlp"]
        log_list = shard_parameters(
            sch[prefix],
            head_sch,
            model_config,
            shard_target,
            sequence_parallel=sequence_parallel,
        )
        for msg in log_list:
            logger.info(msg, ranks=0)

        if bcast_input:
            # Broadcast input to all devices within the MP group.
            # This is not required when running on Megatron.
            logger.info("Broadcast input to all devices", ranks=0)
            broadcast_input(sch)

        if sequence_parallel:
            annotate_layernorm_and_bias(sch)

    # Insert activation checkpoints.
    if ckpt_ratio > 0.0:
        logger.info("Checkpoint ratio: %.2f", ckpt_ratio, ranks=0)
        n_ckpt = checkpoint(
            sch[prefix],
            model_config,
            ckpt_ratio=ckpt_ratio,
            checkpoint_method=checkpoint_method,
        )
        logger.info("Checkpointed %d layers", n_ckpt, ranks=0)

    # Pipeline parallelism.
    if pipeline_cuts:
        logger.info("Generate pipeline schedule", ranks=0)
        generate_pipeline_schedule(sch, sch_config)

    return sch


def get_sch_config_str(sch_config):
    """A utility to generate a pretty string of the schedule config.

    Parameters
    ----------
    sch_config : dict
        The model schedule configuration.

    Returns
    -------
    str
        A pretty string of the schedule config.
    """

    def json_default(obj):
        if hasattr(obj, "to_dict"):
            # The type of model config is PretrainedConfig,
            # so we first convert it to dict to let json.dumps
            # interpret its formats correctly.
            return obj.to_dict()
        return repr(obj)

    # model_config_dict = sch_config["model_config"].to_dict()
    return json.dumps(sch_config, indent=2, sort_keys=True, default=json_default)


def replace_modules(sch, model_config, sch_config):
    """Replace modules in the model with efficient implementations.

    Parameters
    ----------
    sch : slapo.Schedule
        The schedule of the entire model.
    model_config : GPT2Config
        The model configuration.
    sch_config : dict
        The model schedule configuration.

    Returns
    -------
    log_dict : dict
        A dict of log messages in the format of {replaced module info: count}.
    """
    delay_init = sch_config.get("delay_init", True)
    log_dict = {}

    # Replace self attention with flash attention.
    attn_op_name = sch_config.get("attn_op_name", "cuda")
    cnt, applied_attn_op_name = replace_attention(
        sch,
        model_config,
        delay_init=delay_init,
        attn_op_name=attn_op_name,
    )
    log_dict[f"attention layers with {applied_attn_op_name} op"] = cnt

    # Replace MLP with fused kernels.
    cnt = replace_mlp(sch, model_config, delay_init=delay_init)
    log_dict["MLP layers"] = cnt
    return log_dict


def replace_attention(
    sch,
    model_config,
    attn_path="h.N.attn",
    delay_init=True,
    attn_op_name="cuda",
):
    """Replace the attention module with flash attention.

    Parameters
    ----------
    sch : slapo.Schedule
        The schedule of the entire model.
    model_config : GPT2Config
        The model configuration.
    attn_path : str
        The path to the attention module.
    delay_init : bool
        Whether to delay the initialization of the new module.
    attn_op_name : str
        The name of attention op to use. Can be "native_xformers", "cutlass",
        "cuda", or "triton". Except for "native_xformers", all ops implement
        Flash Attention. If specified "cuda" or "triton" but they are not
        available (e.g., missing required library or unsupported GPU arch),
        it will fallback to "cutlass".

    Returns
    -------
    tuple[int, str]
        The number of attention layers replaced and the name of the attention.
    """
    init_config = dict(
        hidden_size=model_config.hidden_size,
        num_attention_heads=model_config.num_attention_heads,
        output_proj=True,
        attn_pdrop=model_config.attn_pdrop,
        resid_pdrop=model_config.resid_pdrop,
        attn_op_name=attn_op_name,
        fused_qkv=True,
    )

    class Attention(nn.Module):
        """A wrapper to align the original GPTAttention forward signature."""

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
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
        ):
            outputs = self.module(
                hidden_states,
                attention_mask,
                layer_past,
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
    for idx in range(model_config.num_hidden_layers):
        sub_sch = sch[attn_path.replace("N", str(idx))]
        with init_empty_weights(enable=delay_init):
            new_mod = Attention(**init_config)
            attn_op.append(new_mod.module.attn_op_name)
        sub_sch.replace(new_mod)
        cnt += 1

    # Check if all attention ops are the same.
    attn_op = list(set(attn_op))
    if len(attn_op) > 1:
        raise RuntimeError(
            f"The attention op is not consistent across layers, including {attn_op}"
        )

    return cnt, attn_op[0]


def replace_mlp(
    sch,
    model_config,
    path="h.N.mlp",
    delay_init=True,
):
    """Replace the MLP module with a fused MLP module.

    Parameters
    ----------
    sch : slapo.Schedule
        The schedule of the entire model.
    model_config : GPT2Config
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
    for idx in range(model_config.num_hidden_layers):
        prefix = path.replace("N", str(idx))
        sub_sch = sch[prefix]
        hidden_size, inter_size = sub_sch.mod.c_fc.weight.shape
        with init_empty_weights(enable=delay_init):
            new_mod = FusedMLP(
                hidden_size,
                inter_size,
                model_config.activation_function,
                model_config.resid_pdrop,
            )
        sub_sch.replace(new_mod)
    return model_config.num_hidden_layers


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
        # Mask the output embedding. Note that the input is already masked.
        output[_input[0] == 0, :] = 0.0
        # Reduce across all the model parallel GPUs
        output = reduce_forward_output(output, sch.group)
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
        sch.replace(new_expand, op)


def shard_parameters(
    sch,
    head_sch,
    model_config,
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
    model_config: GPT2Config
        The configuration of the model.
    shard_target : list[str]
        The sharding target. This function shards the corresponding layers
        specified in this target. It could include "embed", "attention", "mlp".
    sequence_parallel : bool
        Whether to use sequence parallelism. Default False.
    """
    log_list = []

    if sch.world_size == 1:
        return log_list

    # Shard input embedding.
    if "embed" in shard_target:
        word_embed_name, pos_embed_name, final_ln_name = "wte", "wpe", "ln_f"

        sch[word_embed_name].shard("weight", axis=0)
        fwd_pre_hook, fwd_post_hook = gen_embedding_hooks(sch, model_config.vocab_size)
        sch[word_embed_name].sync(mode="fwd_pre", sync_op_or_fn=fwd_pre_hook)
        sch[word_embed_name].sync(mode="fwd_post", sync_op_or_fn=fwd_post_hook)
        log_list.append("Shard input embedding")

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
            log_list.append("Shard output embedding")
        else:
            log_list.append(
                "Skip sharding output embedding because lm_head is not given"
            )
    else:
        log_list.append("Skip sharding embedding layers")

    # Shard attention.
    if "attention" in shard_target:
        path = "h.N.attn"

        for idx in range(model_config.num_hidden_layers):
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

                # In this case, only the attention dropout in between has to
                # use different random seeds.
                sub_sch["module"]["attn_op"].fork_rng()

        log_list.append(f"Shard {model_config.num_hidden_layers} attention layers")
    else:
        log_list.append("Skip sharding attention layers")

    # Shard MLP.
    if "mlp" in shard_target:
        path = "h.N.mlp"

        for idx in range(model_config.num_hidden_layers):
            prefix = path.replace("N", str(idx))
            sub_sch = sch[prefix]

            # FIXME: If MLP is not replaced, the weights in GPT2MLP are transposed
            # and the fc names are different. We don't handle that right now.
            fc_names = ["fc_in", "fc_out"]

            sub_sch[fc_names[0]].shard("weight", axis=0)
            sub_sch[fc_names[0]].shard("bias", axis=0)
            sub_sch[fc_names[1]].shard("weight", axis=1)

            if sequence_parallel:
                sub_sch[fc_names[0]].sync(
                    mode="fwd_pre", sync_op_or_fn="all_gather", axis=1
                )
                sub_sch[fc_names[1]].sync(
                    mode="fwd_post", sync_op_or_fn="reduce_scatter", axis=1
                )
            else:
                sub_sch[fc_names[0]].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
                sub_sch[fc_names[1]].sync(mode="fwd_post", sync_op_or_fn="all_reduce")

        log_list.append(f"Shard {model_config.num_hidden_layers} MLP layers")
    else:
        log_list.append("Skip sharding MLP layers")

    return log_list


def annotate_layernorm_and_bias(sch):
    """Annotate parameters that require additional allreduce on tensor parallel group
    when sequence parallelism is turned on. This is specific for DeepSpeed pipeline
    runtime.

    Parameters
    ----------
    sch : slapo.Schedule
        The schedule of the model.
    """
    for sub_sch in sch.child.values():
        if isinstance(sub_sch.mod, nn.LayerNorm):
            for name, _ in sub_sch.mod.named_parameters(recurse=False):
                sub_sch.annotate(name, "replicated_param", True)
        if issubclass(sub_sch.mod.__class__, LinearWithSyncFunc):
            sub_sch.annotate("bias", "replicated_param", True)
        annotate_layernorm_and_bias(sub_sch)


def checkpoint(
    sch, model_config, path="h.N", ckpt_ratio=1.0, checkpoint_method="uniform"
):
    """Add activation checkpointing to the model. The ckpt_ratio specifies
    the ratio of the attention layers to be checkpointed. For example, if
    ckpt_ratio is 0.5, then half of the attention layers will be checkpointed.

    Parameters
    ----------
    sch : slapo.Schedule
        The schedule of the model.
    model_config : GPT2Config
        The configuration of the model.
    path : str
        The path to the attention layer. Default: "h.N.attn".
    ckpt_ratio : float
        The ratio of the attention layers to be checkpointed. Default: 1.0.
    checkpoint_method : str
        The checkpointing method. Default: "uniform".
    """
    if ckpt_ratio == 0.0:
        return 0

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
    """Add a hook in the beginning of the model to broadcast the input.
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


def generate_pipeline_schedule(sch, sch_config):
    """Trace the top module of the model and cut the pipeline stages.

    Parameters
    ----------
    sch : slapo.Schedule
        The schedule to be cut.
    sch_config : dict
        The configuration of the schedule. Should contain the following keys:
        prefix : str
            The prefix of the top module.
        pipeline_cuts : list[str]
            The list of attention layer index indicating cut points.
    """
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
        sch.trace_until(f"{prefix}", tracer="huggingface", concrete_args=concrete_args)
        _prefix = f"{prefix}." if prefix else ""
        for cut in pipeline_cuts:
            sch[f"{_prefix}h.{cut}"].cut_pipeline_stage()
