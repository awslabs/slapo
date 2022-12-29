# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch.distributed as dist


def shard_layers(sch, config):
    if sch.world_size == 1:
        return

    _, n_layers = config
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
            sub_sch["conv1"].sync(mode="backward")

            # We choose not allgather, so we need to shard bn as well.
            sub_sch["bn1"].shard("weight", axis=0)
            sub_sch["bn1"].shard("bias", axis=0)
            sub_sch["bn1"].shard("running_mean", axis=0)
            sub_sch["bn1"].shard("running_var", axis=0)

            # Forward: partial output (need allreduce)
            # Backward: do nothing.
            sub_sch["conv2"].shard("weight", axis=1)
            sub_sch["conv2"].sync(mode="forward") # forward allreduce only

            # Forward: partitioned output (followed by allgather).
            # Backward: allreduce.
            sub_sch["conv3"].shard("weight", axis=0)
            sub_sch["conv3"].sync(mode="both")


def checkpoint(sch, config, ckpt_ratio=1.0):
    if ckpt_ratio == 0.0:
        return

    _, n_layers = config

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
    def broadcast_input(inputs):
        for inp in inputs:
            dist.broadcast(inp, src=0, group=sch.group)
        return inputs

    sch.hook("fw_pre", broadcast_input)
