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
    orig_sch = slapo.create_schedule(model)
    sch.trace_until("")
    sch["layers.5"].cut_pipeline_stage()
    bs = 8
    ds_config_dict = get_ds_config(
        batch_size=bs,
        micro_batch_size_per_gpu=bs,
        fp16=False,
    )
    original_model, _ = slapo.build(orig_sch)
    original_model.to(dist.get_rank())
    set_random_seed(2023)
    inp = torch.randn(bs, 10, device=dist.get_rank())
    label = torch.randint(0, 10, (bs,), dtype=torch.long, device=dist.get_rank())
    if dist.get_world_size() > 1:
        dist.broadcast(inp, src=0)
        dist.broadcast(label, src=0)
    original_output = original_model(inp)
    original_output = F.cross_entropy(original_output, label)
    print("original output: ", original_output)
    original_state_dict = original_model.state_dict()

    def init_weights(mod, path):
        for name, _ in mod.named_parameters(recurse=False):
            old_name = ".".join(path.split(".")[1:]).replace("_", ".") + "." + name
            setattr(
                mod,
                name,
                nn.Parameter(
                    original_state_dict[old_name].detach().to(dist.get_rank())
                ),
            )

    model, _ = slapo.build(
        sch,
        init_weights=init_weights,
        topology=topology,
        target="deepspeed",
        config=ds_config_dict,
        loss_fn=nn.CrossEntropyLoss(),
    )
    model.to(dist.get_rank())
    train_iter = iter([(inp, label) for _ in range(100)])
    output = model.train_batch(data_iter=train_iter)
    print("new ouput", output)
    if dist.get_rank() == 1:
        assert torch.allclose(original_output, output)


if __name__ == "__main__":
    # pytest.main([__file__])
    test_pipeline()
