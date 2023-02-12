# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""TorchVision WideResNet with model schedule."""

import inspect

import torch
from torch import nn
import torch.distributed as dist

from .registry import register_schedule_method

MODEL_SHORT_NAME = "wideresnet"


@register_schedule_method(MODEL_SHORT_NAME)
def shard_parameters(sch, model_config, sch_config):
    prefix = sch_config.get("prefix", "")

    # Operator fusion
    if sch_config.get("fuse_conv", False):
        fuse_conv_bn(sch[prefix], model_config)

    if sch.world_size > 1:
        # Shard layers.
        shard_layers(sch[prefix], model_config)


@register_schedule_method(MODEL_SHORT_NAME)
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
        sch.trace_for_pipeline(
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


@register_schedule_method(MODEL_SHORT_NAME)
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


@register_schedule_method(MODEL_SHORT_NAME)
def broadcast_input(sch):
    def broadcast(inputs):
        for inp in inputs:
            dist.broadcast(inp, src=0, group=sch.group)
        return inputs

    sch.sync(mode="fwd_pre", sync_op_or_fn=broadcast)
