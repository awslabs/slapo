# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional
import inspect

import torch
from torch.nn import Module
from torchvision.models.resnet import Bottleneck, ResNet
from transformers.utils import ModelOutput

import slapo
from slapo.logger import get_logger
from schedule import fuse_conv_bn, checkpoint, broadcast_input, shard_layers

logger = get_logger("WideResNet")


@dataclass
class ResNetOutput(ModelOutput):
    output: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None


class ResNetWithLoss(Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x, labels: Optional[torch.Tensor] = None):
        output = self.model(x)
        loss = None
        if labels is not None:
            loss = self.loss_fn(output, labels.squeeze())
        return ResNetOutput(output=output, loss=loss)


def schedule_model(
    model,
    config,
    prefix="",
    fp16=False,
    ckpt_ratio=0.0,
    group=None,
    bcast_input=False,
    pipeline_cuts=None,
):
    if fp16:
        logger.info("Change model dtype to fp16", ranks=0)
        model.half()

    sch = slapo.create_schedule(model, group=group)
    logger.info(f"Scheduling Wide-ResNet with TP={sch.world_size}", ranks=0)

    if sch.world_size > 1:
        # Shard layers.
        shard_layers(sch[prefix], config)

        # Broadcast input to all devices within the MP group.
        # This is not required when running on Megatron.
        if bcast_input:
            broadcast_input(sch)
    else:
        fuse_conv_bn(sch[prefix], config)

    # Insert activation checkpoints.
    if ckpt_ratio > 0.0:
        n_ckpt = checkpoint(sch[prefix], config, ckpt_ratio=ckpt_ratio)
        logger.info(f"Checkpointing {n_ckpt} layers", ranks=0)

    # Cut pipeline stages.
    if pipeline_cuts:
        assert len(pipeline_cuts) == 4
        input_names = ["x"]
        sig = inspect.signature(model.forward)
        concrete_args = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.name not in input_names
        }
        _prefix = f"{prefix}." if prefix else ""

        leaves = [f"{_prefix}layer{idx + 1}" for idx in range(4) if pipeline_cuts[idx]]
        sch.trace_for_pipeline(
            leaves,
            tracer="pytorch",
            concrete_args=concrete_args,
        )
        for idx, cuts in enumerate(pipeline_cuts):
            for cut in cuts:
                sch[f"{_prefix}layer{idx + 1}"][str(cut)].cut_pipeline_stage()

    return sch


def get_model(width_per_group, layers):
    model = ResNet(Bottleneck, layers, width_per_group=width_per_group)
    return ResNetWithLoss(model, torch.nn.CrossEntropyLoss())


def get_model_config(name) -> Module:
    """wideresnet-size."""
    if not name.startswith("wideresnet"):
        raise ValueError(
            f"Invalid model name: {name}. Expected to start with wideresnet"
        )
    _, size = name.split("-")

    if size == "250M":
        return (128, [6, 8, 46, 6])
    if size == "1.3B":
        return (320, [6, 8, 46, 6])
    if size == "2.4B":
        return (448, [6, 8, 46, 6])
    if size == "3B":
        return (512, [6, 8, 46, 6])
    if size == "4.6B":
        return (640, [6, 8, 46, 6])
    raise ValueError(f"Unsupported model size: {size}")
