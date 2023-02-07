# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Albert with model schedule."""
import inspect

import slapo
from slapo.logger import get_logger
from schedule import (
    broadcast_input,
    checkpoint,
    replace_and_shard_attention,
    fuse_bias_gelu,
    shard_mlp,
    shard_word_embedding,
)

logger = get_logger("Albert")


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
    disable_fuse_bias_gelu=False,
    delay_init=True,
):
    logger.info("Scheduling Albert", ranks=0)

    if fp16:
        logger.info("Change model dtype to fp16", ranks=0)
        model.half()

    sch = slapo.create_schedule(model, group=group)

    # Replace self attention with flash attention, and shard QKV/output
    # if MP group > 1.
    if disable_flash_attn:
        logger.info("Disabled Flash Attention", rank=0)
    cnt = replace_and_shard_attention(
        sch[prefix],
        config,
        delay_init=delay_init,
        disable_flash_attn=disable_flash_attn,
    )
    logger.info(f"Replace {cnt} attention patterns", ranks=0)

    # Operator fusion
    if disable_fuse_bias_gelu:
        fuse_bias_gelu(sch[prefix], config)
        logger.info(f"Fused Bias+GeLU", ranks=0)

    # Shard other parameters if MP group > 1.
    if sch.world_size > 1:
        shard_mlp(sch[prefix], config)
        shard_word_embedding(sch[prefix], config.vocab_size)

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
        input_names = [
            "input_ids",
            "attention_mask",
            "token_type_ids",
        ]
        sig = inspect.signature(model.forward)
        concrete_args = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.name not in input_names
        }
        _prefix = f"{prefix}." if prefix else ""
        sch.trace_for_pipeline(
            f"{_prefix}encoder", tracer="huggingface", concrete_args=concrete_args
        )
        for cut in pipeline_cuts:
            sch[f"{_prefix}encoder.layer.{cut}"].cut_pipeline_stage()

    return sch
