# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import torch
import torch.nn as nn
import slapo
from torch.testing._internal.distributed._shard.sharded_tensor._test_ops_common import (
    clone_module_parameter,
)
from slapo.utils.report import report_memory

import deepspeed
from deepspeed.utils import RepeatingLoader


class MLP(nn.Module):
    def __init__(self, dim: int = 1024):
        super().__init__()
        intermediate_dim = dim * 2
        self.dense_1 = nn.Linear(dim, intermediate_dim)
        self.activation = nn.ReLU()
        self.dense_2 = nn.Linear(intermediate_dim, dim, bias=None)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        return x


class Top(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()

    def forward(self, x):
        return self.mlp(x).mean()


def _weight_override(module_dst, module_src):
    module_dst.mlp.dense_1.weight = clone_module_parameter(
        module_src.mlp.dense_1, "weight"
    )
    module_dst.mlp.dense_1.bias = clone_module_parameter(module_src.mlp.dense_1, "bias")
    module_dst.mlp.dense_2.weight = clone_module_parameter(
        module_src.mlp.dense_2, "weight"
    )
    # module_dst.mlp.dense_2.bias = clone_module_parameter(module_src.mlp.dense_2, "bias")


def train(args):
    rank = args.local_rank
    print(f"Running basic MLP example on rank {rank}.")

    print("Use deepspeed to initialize")
    deepspeed.init_distributed(dist_backend="nccl")

    # === Model execution schedule ===
    model = Top()
    # local_model = Top()
    # _weight_override(local_model, model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002)

    # Create a default schedule
    sch = slapo.create_schedule(model, optimizer, args.world_size, rank)

    # Partition parameters
    # Cannot be used with pipeline parallelism

    sch["mlp.activation"].cut_pipeline_stage()
    ds_config_dict = {
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {"type": "AdamW", "params": {"lr": 0.0001}},
    }
    ds_model, optimizer = slapo.build(
        sch, target="deepspeed", config=ds_config_dict, loss_fn=None
    )
    print(sch.gm)
    opt_inp = torch.rand((2048, 1024), requires_grad=True).cuda(rank)
    label = torch.zeros((1,)).cuda(rank)
    loader = RepeatingLoader([((opt_inp,), label)])
    data_iter = iter(loader)

    loss = ds_model.train_batch(data_iter=data_iter)
    print("Pass")


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    # This is passed in via cmd
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=n_gpus)
    parser.add_argument("--iter_nums", type=int, default=5)
    parser.add_argument(
        "--checkpoint", action="store_true", help="Enable gradient checkpointing"
    )
    args = parser.parse_args()
    train(args)
