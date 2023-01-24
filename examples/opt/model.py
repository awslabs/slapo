# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace OPT with model schedule."""
import inspect

import slapo
from slapo.logger import get_logger
from schedule import (
    broadcast_input,
    checkpoint,
    remove_cast,
    replace_and_shard_mlp,
    replace_and_shard_attention,
    replace_qkv,
    shard_qkv,
    shard_word_embedding,
    trace_attention,
)

logger = get_logger("OPT")


def schedule_model(
    model,
    config,
    prefix="",
    disable_flash_attn=False,
    fp16=True,
    ckpt_ratio=0.0,
    group=None,
    bcast_input=False,
    pipeline_cuts=None,
    delay_init=True,
):

    logger.info(f"Scheduling OPT", ranks=0)

    if fp16:
        logger.info("Change model dtype to fp16", ranks=0)
        model.half()

    sch = slapo.create_schedule(model, group=group)

    # Replace self attention with flash attention, and shard QKV/output
    # if MP group > 1.
    attn_path, out_proj_name = "decoder.layers.N.self_attn", "out_proj"
    if not disable_flash_attn:
        cnt = replace_and_shard_attention(sch[prefix], config, delay_init=delay_init)
        logger.info(f"Replace {cnt} attention patterns", ranks=0)
    else:
        # FIXME: This path is not working because our tracer cannot trace
        # OPT without tracing the whole model.
        raise NotImplementedError("OPT without fusion")
        cnt = trace_attention(sch, config, attn_path)
        logger.info(f"Traced {cnt} attention layesr", ranks=0)
        assert cnt > 0
        cnt = remove_cast(sch, config, attn_path)
        logger.info(f"Remove {cnt} .to(torch.float32) ops", ranks=0)
        cnt = replace_qkv(sch, config, attn_path)
        logger.info(f"Replace {cnt} QKV patterns", ranks=0)
        if sch.world_size > 1:
            shard_qkv(sch, config, attn_path, out_proj_name=out_proj_name)

    # Shard other parameters if MP group > 1.
    if sch.world_size > 1:
        replace_and_shard_mlp(sch[prefix], config, delay_init=delay_init)
        tie_sch = sch["lm_head"] if "lm_head" in sch else None
        shard_word_embedding(sch[prefix], tie_sch, config.vocab_size)

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
