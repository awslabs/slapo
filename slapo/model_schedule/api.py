# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .registry import get_schedule_method
from ..schedule import create_schedule
from ..logger import get_logger


def apply_schedule(
    model,
    model_config,
    **sch_config,
):
    model_name = model_config._name_or_path
    short_name = model_name.split("/")[-1].split("-")[0]
    short_name = "gpt_neo" if short_name == "gpt" else short_name
    logger = get_logger(f"{model_name}")
    logger.info(f"Scheduling {model_name}", ranks=0)

    # Change data type.
    fp16 = sch_config.get("fp16", False)
    bf16 = sch_config.get("bf16", False)
    if fp16 and bf16:
        raise ValueError("Cannot use both fp16 and bf16")
    elif fp16:
        logger.info("Change model dtype to fp16", ranks=0)
        model.half()
    elif bf16:
        logger.info("Change model dtype to bf16", ranks=0)
        model.bfloat16()
    else:
        logger.info("Use fp32 as default model dtype", ranks=0)

    group = sch_config.get("group", None)
    sch = create_schedule(model, group=group)
    logger.info(f"Scheduling {model_name} with TP={sch.world_size}", ranks=0)

    # Tensor parallelism.
    shard_params_fn = get_schedule_method(short_name, "shard_parameters")
    if shard_params_fn:
        logger.info("Shard model parameters", ranks=0)
        shard_params_fn(sch, model_config, sch_config)

    if sch.world_size > 1 and sch_config.get("bcast_input", False):
        # Broadcast input to all devices within the MP group.
        # This is not required when running on Megatron.
        broadcast_input_fn = get_schedule_method(short_name, "broadcast_input")
        if broadcast_input_fn:
            logger.info("Broadcast input to all devices", ranks=0)
            broadcast_input_fn(sch)

    # Insert activation checkpoints.
    ckpt_ratio = sch_config.get("ckpt_ratio", 0.0)
    if ckpt_ratio > 0.0:
        prefix = sch_config.get("prefix", "")
        checkpoint_method = sch_config.get("checkpoint_method", "uniform")
        checkpoint_fn = get_schedule_method(short_name, "checkpoint")
        if checkpoint_fn:
            logger.info(f"Checkpoint ratio: {ckpt_ratio}", ranks=0)
            n_ckpt = checkpoint_fn(
                sch[prefix],
                model_config,
                ckpt_ratio=ckpt_ratio,
                checkpoint_method=checkpoint_method,
            )
            logger.info(f"Checkpointed {n_ckpt} layers", ranks=0)

    # Pipeline parallelism.
    if sch_config.get("pipeline_cuts", None):
        pipeline_fn = get_schedule_method(short_name, "generate_pipeline_schedule")
        if pipeline_fn:
            logger.info("Generate pipeline schedule", ranks=0)
            pipeline_fn(sch, model_config, sch_config)

    return sch
