# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace OPT with model schedule."""
import inspect

import slapo
from slapo.logger import get_logger
from schedule import (
    broadcast_input,
    checkpoint,
    replace_and_shard_mlp,
    replace_and_shard_attention,
    shard_word_embedding,
)

logger = get_logger("OPT")


def schedule_model(
    model,
    config,
    prefix="",
    attn_op_name="cuda",
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
    if attn_op_name == "native_xformers":
        logger.info("Disabled Flash Attention", ranks=0)
    cnt, applied_attn_op_name = replace_and_shard_attention(
        sch[prefix],
        config,
        delay_init=delay_init,
        attn_op_name=attn_op_name,
    )
    logger.info(
        f"Replace {cnt} attention layers with {applied_attn_op_name} op", ranks=0
    )

    # Shard other parameters if MP group > 1.
    if sch.world_size > 1:
        replace_and_shard_mlp(sch[prefix], config, delay_init=delay_init)
        head_sch = sch["lm_head"] if "lm_head" in sch else None
        shard_word_embedding(sch[prefix], head_sch, config.vocab_size)

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
