# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test sharding primitive. Note that this test has to be invoked by torchrun. For example:
torchrun --nproc_per_node 2 -m pytest test_shard.py
"""
import os
import copy
import pytest

import torch
import torch.distributed as dist
import slapo


def init_dist():
    torch.manual_seed(9999)
    try:
        dist.init_process_group(backend="nccl")
    except Exception:
        pytest.skip(f"Skip {__file__} because torch.distributed is not initialized")


def verify_weights(module: torch.nn.Module):
    for param in module.parameters():
        torch.testing.assert_close(
            param.data, torch.zeros_like(param).cuda(param.device)
        )


def init_module(module: torch.nn.Module):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.zero_()
        if module.bias is not None:
            module.bias.data.zero_()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(20, 20)
        self.linear2 = torch.nn.Linear(20, 40, bias=False)

    def forward(self, data):
        out = self.linear1(data)
        out = self.linear2(out)
        return out


def test_singlegpu_consolidation():
    device = 0
    torch.cuda.set_device(device)
    with slapo.init_empty_weights(enable=True):
        model = Model()

    sch = slapo.create_schedule(copy.deepcopy(model))
    sch_model, _ = slapo.build(sch, init_weights=init_module)

    sch_model.cuda(device)
    verify_weights(sch_model)


def test_multigpu_consolidation():
    init_dist()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    with slapo.init_empty_weights(enable=True):
        model = Model()

    sch = slapo.create_schedule(copy.deepcopy(model))
    sch["linear1"].shard("weight", axis=0)
    sch["linear1"].shard("bias", axis=0)
    sch["linear1"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
    sch["linear2"].shard("weight", axis=1)
    sch["linear2"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")
    sch_model, _ = slapo.build(sch, init_weights=init_module)

    sch_model.cuda(local_rank)
    verify_weights(sch_model)


if __name__ == "__main__":
    pytest.main([__file__])
