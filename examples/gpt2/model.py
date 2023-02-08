# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace GPT-2 with model schedule."""

import slapo
from slapo.logger import get_logger
from schedule import (
    broadcast_input,
    checkpoint,
    replace_mlp,
    replace_attention,
    shard,
    pipeline,
)

logger = get_logger("GPT")


def schedule_model(
    model,
    config,
    prefix="",
    disable_flash_attn=False,
    fp16=True,
    bf16=False,
    ckpt_ratio=0.0,
    group=None,
    bcast_input=False,
    pipeline_cuts=None,
    delay_init=True,
    sequence_parallel=False,
):
    if fp16:
        logger.info("Change model dtype to fp16", ranks=0)
        model.half()
    elif bf16:
        logger.info("Change model dtype to bf16", ranks=0)
        model.bfloat16()

    sch = slapo.create_schedule(model, group=group)
    logger.info(f"Scheduling GPT with TP={sch.world_size}", ranks=0)

    # Replace self attention with flash attention.
    if disable_flash_attn:
        logger.info("Disabled Flash Attention", ranks=0)
    cnt, attn_op_name = replace_attention(
        sch[prefix],
        config,
        delay_init=delay_init,
        disable_flash_attn=disable_flash_attn,
    )
    logger.info(f"Replace {cnt} attention layers with {attn_op_name} op", ranks=0)

    # Replace MLP with fused kernels.
    cnt = replace_mlp(sch[prefix], config, delay_init=delay_init)
    logger.info(f"Replaced {cnt} MLP layers", ranks=0)

    # Shard parameters if MP group > 1.
    if sch.world_size > 1:
        head_sch = sch["lm_head"] if "lm_head" in sch else None
        shard_target = ["embed", "attention", "mlp"]
        shard(
            sch[prefix],
            head_sch,
            config,
            shard_target,
            sequence_parallel=sequence_parallel,
        )

        # Broadcast input to all devices within the MP group.
        # This is not required if it will be done by the training framework.
        if bcast_input:
            broadcast_input(sch)

    # Insert activation checkpoints.
    if ckpt_ratio > 0.0:
        n_ckpt = checkpoint(sch[prefix], config, ckpt_ratio=ckpt_ratio)
        logger.info(f"Checkpointing {n_ckpt} layers", ranks=0)

    # Cut pipeline stages.
    if pipeline_cuts:
        pipeline(sch, prefix, pipeline_cuts)

    return sch
