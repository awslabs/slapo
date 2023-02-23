# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modification: Megatron-LM.
# See https://github.com/NVIDIA/Megatron-LM/blob/main/tests/tensor_parallel/test_random.py
"""
Test fork_rng primitive. Note that this test has to be invoked by torchrun.
See ci/task_unit_tests.sh for an example.
"""
# pylint: disable=unused-argument

import os

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

import slapo
from slapo import set_random_seed


def test_dropout(init_dist):
    def verify(model, data, rank, local_rank, world_size, all_close):
        out = model(data)

        outs = [
            torch.zeros(data.shape, dtype=data.dtype).cuda(local_rank)
            for _ in range(world_size)
        ]
        dist.all_gather(outs, out.contiguous())
        if rank == 0:
            if all_close:
                for out in outs[1:]:
                    torch.testing.assert_close(out, outs[0])
            else:
                for out in outs[1:]:
                    with pytest.raises(AssertionError):
                        torch.testing.assert_close(out, outs[0])

    class Model(nn.Module):
        def forward(self, x):
            return F.dropout(x, p=0.5)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    set_random_seed(123, None, None, local_rank)

    model = Model()

    data = torch.randn((10, 20), requires_grad=True).cuda(local_rank)
    dist.broadcast(data, src=0)

    # Without fork_rng, the dropout mask is the same across all ranks.
    sch = slapo.create_schedule(model)
    sch_model, _ = slapo.build(sch)
    sch_model.cuda(local_rank)
    verify(sch_model, data, rank, local_rank, world_size, all_close=True)

    # With fork_rng, the dropout mask is different across all ranks.
    sch = slapo.create_schedule(model)
    sch.fork_rng()
    sch_model, _ = slapo.build(sch)
    sch_model.cuda(local_rank)
    verify(sch_model, data, rank, local_rank, world_size, all_close=False)


if __name__ == "__main__":
    pytest.main([__file__])
