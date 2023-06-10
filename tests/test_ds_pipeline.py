# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test DeepSpeed Pipeline."""
import pytest

import os
import torch
from torch import nn
import deepspeed

import slapo
from slapo.framework_dialect.deepspeed.pipeline import (
    get_ds_config,
    create_dist_group_for_pipeline,
)
from slapo.random import set_random_seed
import torch.distributed as dist
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(12)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_pipeline():
    deepspeed.init_distributed(dist_backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    with slapo.init_empty_weights():
        model = Model()
    topology, group = create_dist_group_for_pipeline(num_pp=2, num_mp=1)
    sch = slapo.create_schedule(model, group=group)
    sch.trace_until("")

    bs = 8
    ds_config_dict = get_ds_config(
        batch_size=bs,
        micro_batch_size_per_gpu=bs,
        fp16=False,
    )
    set_random_seed(2013)
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


if __name__ == "__main__":
    # pytest.main([__file__])
    test_pipeline()
