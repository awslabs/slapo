# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test verification."""
# pylint: disable=unused-argument

import os
import copy
import pytest

import torch
from torch import nn
import torch.nn.functional as F

import slapo
from slapo.pattern import call_module


def test_vertical_fusion():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, 3)

        def forward(self, x):
            x = self.conv(x)
            x = F.relu(x)
            x = x + 1
            return x

    mod = Model().cuda()
    sch = slapo.create_schedule(mod)

    def pattern(x: torch.Tensor):
        x = call_module("conv", x)
        x = F.relu(x)
        x = x + 1
        return x

    subgraph = sch.find(pattern)
    assert len(subgraph[0]) == 3
    inp = torch.randn((1, 3, 32, 32), requires_grad=True).cuda()
    with slapo.verify(sch, inp):
        sch.fuse(subgraph, compiler="TorchScript", name="FusedReLU")


def test_bias_gelu():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1024, 1024)
            self.gelu = nn.GELU()

        def forward(self, x):
            x = self.linear(x)
            x = self.gelu(x)
            return x

    mod = Model().cuda()
    sch = slapo.create_schedule(mod)

    sch["linear"].decompose()
    sch.trace(flatten=True)

    def pattern(x, bias):
        x = F.gelu(bias + x)
        return x

    subgraph = sch.find(pattern)
    assert len(subgraph[0]) == 2
    inp = torch.randn((1, 16, 1024, 1024), requires_grad=True)
    with slapo.verify(sch, inp):
        sch.fuse(subgraph, compiler="TorchScript", name="BiasGeLU")
    assert isinstance(sch["BiasGeLU_0"].mod, torch.jit.ScriptModule)


# def test_linear():
#     import torch.distributed as dist
#     dist.init_process_group(backend="nccl")
def test_linear(init_dist):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(20, 30)
            self.linear2 = torch.nn.Linear(30, 40)

        def forward(self, data):
            out = self.linear1(data)
            out = self.linear2(out)
            return out

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    model = Model()

    sch = slapo.create_schedule(copy.deepcopy(model))
    # FIXME: Use random input
    with slapo.verify(sch, [torch.ones(10, 20)], device=f"cuda:{local_rank}"):
        sch["linear1"].shard("weight", axis=0)
        sch["linear1"].shard("bias", axis=0)
        sch["linear1"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
        sch["linear2"].shard("weight", axis=1)
        sch["linear2"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")

    # sch = slapo.create_schedule(copy.deepcopy(model))
    # with pytest.raises(Exception):
    #     with slapo.verify(sch, [torch.ones(10, 20)], device=f"cuda:{local_rank}"):
    #         sch["linear1"].shard("weight", axis=0)
    #         sch["linear1"].shard("bias", axis=0)
    #         sch["linear1"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
    #         sch["linear2"].shard("weight", axis=1)


def test_meta(init_dist):
# def test_meta_distributed():
#     import torch.distributed as dist
#     dist.init_process_group(backend="nccl")
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(20, 30)
            self.linear2 = torch.nn.Linear(30, 40)

        def forward(self, data):
            out = self.linear1(data)
            out = self.linear2(out)
            return out

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    with slapo.init_empty_weights():
        model = Model()

    sch = slapo.create_schedule(copy.deepcopy(model))
    with slapo.verify(sch, [torch.ones((10, 20))], device=f"cuda:{local_rank}"):
        sch["linear1"].shard("weight", axis=0)
        sch["linear1"].shard("bias", axis=0)
        sch["linear1"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
        sch["linear2"].shard("weight", axis=1)
        sch["linear2"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")


if __name__ == "__main__":
    pytest.main([__file__])
