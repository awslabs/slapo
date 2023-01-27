# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace GPT-Neo with model schedule."""
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

logger = get_logger("GPT")


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
    assert "GPT2" not in config.architectures[0], "GPT-2 schedule is not working"

    if fp16:
        logger.info("Change model dtype to fp16", ranks=0)
        model.half()

    sch = slapo.create_schedule(model, group=group)
    logger.info(f"Scheduling GPT with TP={sch.world_size}", ranks=0)

    # Replace self attention with flash attention, and shard QKV/output
    # if MP group > 1.
    attn_path, out_proj_name = "h.N.attn.attention", "out_proj"
    if disable_flash_attn:
        logger.info("Disabled Flash Attention", rank=0)
    cnt = replace_and_shard_attention(
        sch[prefix],
        config,
        delay_init=delay_init,
        disable_flash_attn=disable_flash_attn,
    )
    logger.info(f"Replace {cnt} attention patterns", ranks=0)

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
