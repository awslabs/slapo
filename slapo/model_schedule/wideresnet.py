# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""TorchVision WideResNet with model schedule."""

import inspect

import torch
from torch import nn
import torch.distributed as dist

from ..schedule import create_schedule
from .registry import register_schedule
from ..logger import get_logger


@register_schedule()
def _apply_schedule(
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

    # Operator fusion
    prefix = sch_config.get("prefix", "")
    if sch_config.get("fuse_conv", False):
        fuse_conv_bn(sch[prefix], model_config)

    # Tensor parallelism.
    if sch.world_size > 1:
        logger.info("Shard model parameters", ranks=0)
        shard_layers(sch[prefix], model_config)

    if sch.world_size > 1 and sch_config.get("bcast_input", False):
        # Broadcast input to all devices within the MP group.
        # This is not required when running on Megatron.
        logger.info("Broadcast input to all devices", ranks=0)
        broadcast_input(sch)

    # Insert activation checkpoints.
    ckpt_ratio = sch_config.get("ckpt_ratio", 0.0)
    if ckpt_ratio > 0.0:
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


def generate_pipeline_schedule(sch, sch_config):
    pipeline_cuts = sch_config.get("pipeline_cuts", None)
    prefix = sch_config.get("prefix", "")
    # Cut pipeline stages.
    if pipeline_cuts:
        assert len(pipeline_cuts) == 4
        input_names = ["x"]
        sig = inspect.signature(sch.mod.forward)
        concrete_args = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.name not in input_names
        }
        _prefix = f"{prefix}." if prefix else ""

        leaves = [f"{_prefix}layer{idx + 1}" for idx in range(4) if pipeline_cuts[idx]]
        sch.trace_until(
            leaves,
            tracer="pytorch",
            concrete_args=concrete_args,
        )
        for idx, cuts in enumerate(pipeline_cuts):
            for cut in cuts:
                sch[f"{_prefix}layer{idx + 1}"][str(cut)].cut_pipeline_stage()

    return sch


def fuse_conv_bn(sch, model_config):
    from torchvision.models.resnet import Bottleneck

    inplanes = 64
    expansion = 4
    all_planes = [64, 128, 256, 512]
    all_strides = [1, 2, 2, 2]
    in_sizes = [112, 56, 28, 14]
    base_width, num_layers = model_config["block_size"]
    for i in range(4):
        planes = all_planes[i]
        layers = num_layers[i]
        stride = all_strides[i]
        in_size = in_sizes[i]
        if stride != 1 or inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * expansion),
            )
        data = torch.rand((8, inplanes, in_size, in_size), dtype=torch.float).cuda()
        new_block = torch.jit.trace(
            Bottleneck(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                base_width=base_width,
            ).cuda(),
            data,
        )
        sch[f"layer{i+1}"]["0"].replace(new_block)
        inplanes = planes * expansion
        data = torch.rand(
            (8, inplanes, in_size // 2, in_size // 2), dtype=torch.float
        ).cuda()
        for j in range(1, layers):
            new_block = torch.jit.trace(
                Bottleneck(
                    inplanes=inplanes,
                    planes=planes,
                    base_width=base_width,
                ).cuda(),
                data,
            )
            sch[f"layer{i+1}"][f"{j}"].replace(new_block)
    print(sch.mod)


def shard_layers(sch, model_config):
    if sch.world_size == 1:
        return

    _, n_layers = model_config["block_size"]
    for idx, n_layer in enumerate(n_layers):
        for lidx in range(n_layer):
            # ResNet implements layers using nn.Sequential instead of
            # ModuleList, so we cannot use sch[f"{layer_path}.{lidx}"]
            # to access the layer. This is a constraint of the current
            # hierarchical schedule.
            sub_sch = sch[f"layer{idx + 1}"][str(lidx)]
            # Forward: partitioned output.
            # Backward: allreduce.
            sub_sch["conv1"].shard("weight", axis=0)
            sub_sch["conv1"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")

            # We choose not allgather, so we need to shard bn as well.
            sub_sch["bn1"].shard("weight", axis=0)
            sub_sch["bn1"].shard("bias", axis=0)
            sub_sch["bn1"].shard("running_mean", axis=0)
            sub_sch["bn1"].shard("running_var", axis=0)

            # Forward: partial output (need allreduce)
            # Backward: do nothing.
            sub_sch["conv2"].shard("weight", axis=1)
            sub_sch["conv2"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")

            # Forward: partitioned output (followed by allgather).
            # Backward: allreduce.
            sub_sch["conv3"].shard("weight", axis=0)
            sub_sch["conv3"].sync(
                mode="fwd_post",
                sync_op_or_fn="all_gather",
                axis=1,
                tensor_parallel_output_grad=False,
            )
            sub_sch["conv3"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")


def checkpoint(
    sch,
    model_config,
    ckpt_ratio=1.0,
    checkpoint_method="uniform",
):
    if checkpoint_method != "uniform":
        raise NotImplementedError(
            f"Checkpoint method {checkpoint_method} is not supported yet."
        )
    if ckpt_ratio == 0.0:
        return 0

    _, n_layers = model_config["block_size"]

    total_ckpt = 0
    for idx, n_layer in enumerate(n_layers):
        n_ckpt = int(n_layer * ckpt_ratio)
        total_ckpt += n_ckpt

        for lidx in range(n_ckpt):
            # ResNet implements layers using nn.Sequential instead of
            # ModuleList, so we cannot use sch[f"{layer_path}.{lidx}"]
            # to access the layer. This is a constraint of the current
            # hierarchical schedule.
            sch[f"layer{idx + 1}"][str(lidx)].checkpoint()

    return total_ckpt


def broadcast_input(sch):
    def broadcast(inputs):
        for inp in inputs:
            dist.broadcast(inp, src=0, group=sch.group)
        return inputs

    sch.sync(mode="fwd_pre", sync_op_or_fn=broadcast)
