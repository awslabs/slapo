# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test DeepSpeed Pipeline."""
import os
import pytest

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import deepspeed

import slapo
from slapo.framework_dialect.deepspeed.pipeline import (
    get_ds_config,
    create_dist_group_for_pipeline,
)


class LinearReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        return x


class Model(nn.Module):
    def __init__(self, num_layers=12, has_relu=False):
        super().__init__()
        if not has_relu:
            self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(num_layers)])
        else:
            self.layers = nn.ModuleList([LinearReLU() for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_pipeline_2stages_pp_dp():
    deepspeed.init_distributed(dist_backend="nccl")
    if dist.get_world_size() != 4:
        pytest.skip("This test requires 4 GPUs.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    with slapo.init_empty_weights():
        model = Model(12)
    # If total number of devices is 4, then num_dp = 2
    num_pp = 2
    num_mp = 1
    num_dp = dist.get_world_size() // (num_pp * num_mp)
    topology, group = create_dist_group_for_pipeline(num_pp=num_pp, num_mp=num_mp)
    sch = slapo.create_schedule(model, group=group)
    sch.trace_until("")

    bs = 8
    ds_config_dict = get_ds_config(
        batch_size=bs,
        micro_batch_size_per_gpu=bs // num_dp,
        fp16=False,
    )
    inp = torch.randn(bs, 10, device=dist.get_rank())
    label = torch.randint(0, 10, (bs,), dtype=torch.long, device=dist.get_rank())
    with slapo.Verify(
        sch,
        example_inputs=[inp],
        example_outputs=label,
        loss_fn=F.cross_entropy,
        topology=topology,
        config=ds_config_dict,
    ):
        sch["layers.5"].cut_pipeline_stage()


def test_pipeline_2stages_pp_tp():
    deepspeed.init_distributed(dist_backend="nccl")
    if dist.get_world_size() != 4:
        pytest.skip("This test requires 4 GPUs.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    with slapo.init_empty_weights():
        model = Model(2, has_relu=True)
    num_pp = 2
    num_mp = 2
    num_dp = dist.get_world_size() // (num_pp * num_mp)
    topology, group = create_dist_group_for_pipeline(num_pp=num_pp, num_mp=num_mp)
    sch = slapo.create_schedule(model, group=group)
    sch.trace_until("")

    bs = 8
    ds_config_dict = get_ds_config(
        batch_size=bs,
        micro_batch_size_per_gpu=bs // num_dp,
        fp16=False,
    )
    inp = torch.randn(bs, 10, device=dist.get_rank())
    label = torch.randint(0, 10, (bs,), dtype=torch.long, device=dist.get_rank())
    with slapo.Verify(
        sch,
        example_inputs=[inp],
        example_outputs=label,
        loss_fn=F.cross_entropy,
        topology=topology,
        config=ds_config_dict,
    ):
        # TODO: Fix layers.0/1 indexing bug
        sch["layers.0.linear"].shard("weight", axis=0)
        sch["layers.0.linear"].shard("bias", axis=0)
        sch["layers.1.linear"].shard("weight", axis=1)
        sch["layers.1.linear"].sync("fwd_post", sync_op_or_fn="all_reduce")
        sch["layers.0.linear"].sync("bwd_post", sync_op_or_fn="all_reduce")
        sch["layers.0"].cut_pipeline_stage()


def test_pipeline_4stages_pp():
    deepspeed.init_distributed(dist_backend="nccl")
    if dist.get_world_size() != 4:
        pytest.skip("This test requires 4 GPUs.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    with slapo.init_empty_weights():
        model = Model(12)
    topology, group = create_dist_group_for_pipeline(num_pp=4, num_mp=1)
    sch = slapo.create_schedule(model, group=group)
    sch.trace_until("")

    bs = 8
    ds_config_dict = get_ds_config(
        batch_size=bs,
        micro_batch_size_per_gpu=bs,
        fp16=False,
    )
    inp = torch.randn(bs, 10, device=dist.get_rank())
    label = torch.randint(0, 10, (bs,), dtype=torch.long, device=dist.get_rank())
    with slapo.Verify(
        sch,
        example_inputs=[inp],
        example_outputs=label,
        loss_fn=F.cross_entropy,
        topology=topology,
        config=ds_config_dict,
    ):
        sch["layers.2"].cut_pipeline_stage()
        sch["layers.5"].cut_pipeline_stage()
        sch["layers.8"].cut_pipeline_stage()


def test_pipeline_2stages_pp_tp_dp():
    deepspeed.init_distributed(dist_backend="nccl")
    if dist.get_world_size() != 8:
        pytest.skip("This test requires 8 GPUs.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    with slapo.init_empty_weights():
        model = Model(2, has_relu=True)
    num_pp = 2
    num_mp = 2
    num_dp = dist.get_world_size() // (num_pp * num_mp)
    print("num_dp:", num_dp, "num_pp:", num_pp, "num_mp:", num_mp)
    topology, group = create_dist_group_for_pipeline(num_pp=num_pp, num_mp=num_mp)
    sch = slapo.create_schedule(model, group=group)
    sch.trace_until("")

    bs = 8
    ds_config_dict = get_ds_config(
        batch_size=bs,
        micro_batch_size_per_gpu=bs // num_dp,
        fp16=False,
    )
    inp = torch.randn(bs, 10, device=dist.get_rank())
    label = torch.randint(0, 10, (bs,), dtype=torch.long, device=dist.get_rank())
    with slapo.Verify(
        sch,
        example_inputs=[inp],
        example_outputs=label,
        loss_fn=F.cross_entropy,
        topology=topology,
        config=ds_config_dict,
    ):
        # TODO: Fix layers.0/1 indexing bug
        sch["layers.0.linear"].shard("weight", axis=0)
        sch["layers.0.linear"].shard("bias", axis=0)
        sch["layers.1.linear"].shard("weight", axis=1)
        sch["layers.1.linear"].sync("fwd_post", sync_op_or_fn="all_reduce")
        sch["layers.0.linear"].sync("bwd_post", sync_op_or_fn="all_reduce")
        sch["layers.0"].cut_pipeline_stage()


if __name__ == "__main__":
    test_pipeline_2stages_pp_dp()
    test_pipeline_2stages_pp_tp()
    test_pipeline_4stages_pp()
    test_pipeline_2stages_pp_tp_dp()
