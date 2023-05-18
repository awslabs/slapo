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
    with slapo.Verify(sch, inp):
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
    with slapo.Verify(sch, inp):
        sch.fuse(subgraph, compiler="TorchScript", name="BiasGeLU")
    assert isinstance(sch["BiasGeLU_0"].mod, torch.jit.ScriptModule)


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
    with slapo.Verify(sch, [torch.rand(10, 20)], device=f"cuda:{local_rank}"):
        sch["linear1"].shard("weight", axis=0)
        sch["linear1"].shard("bias", axis=0)
        sch["linear1"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
        sch["linear2"].shard("weight", axis=1)
        sch["linear2"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")

    sch = slapo.create_schedule(copy.deepcopy(model))
    with pytest.raises(Exception):
        with slapo.Verify(sch, [torch.rand(10, 20)], device=f"cuda:{local_rank}"):
            sch["linear1"].shard("weight", axis=0)
            sch["linear1"].shard("bias", axis=0)
            sch["linear1"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
            sch["linear2"].shard("weight", axis=1)


def test_meta_distributed(init_dist):
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
    with slapo.Verify(sch, [torch.rand((10, 20))], device=f"cuda:{local_rank}"):
        sch["linear1"].shard("weight", axis=0)
        sch["linear1"].shard("bias", axis=0)
        sch["linear1"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
        sch["linear2"].shard("weight", axis=1)
        sch["linear2"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")


def test_bert(init_dist):
    from transformers import BertLMHeadModel, AutoConfig
    import inspect

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    config = AutoConfig.from_pretrained("bert-large-uncased")
    with slapo.init_empty_weights():
        model = BertLMHeadModel(config)

    sch = slapo.create_schedule(model)

    def fix_attention_mask_shape(sch):
        input_names = ["hidden_states"]
        sig = inspect.signature(sch.mod.forward)
        concrete_args = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.name not in input_names
        }
        sch.trace(
            recursive=False, flatten=True, tracer="pytorch", concrete_args=concrete_args
        )
        ops = sch.find_node(
            lambda node: node.op == "call_method" and node.target == "view"
        )
        assert len(ops) == 4  # q,k,v,context_layer

        def new_view(tensor, args):
            if len(args) == 4:  # q,k,v
                new_shape = (args[0], args[1], args[2] // sch.world_size, -1)
            else:  # context_layer
                new_shape = (args[0], args[1], args[2] // sch.world_size)
            out = tensor.view(new_shape)
            return out

        for op in ops:
            sch.replace(new_view, op)

    bs = 2
    seq = 512
    input_ids = torch.ones(bs, seq, dtype=torch.long, device=sch.rank)
    with slapo.Verify(sch, [input_ids], eval_mode=True):
        for i in range(config.num_hidden_layers):
            # shard attention
            subsch = sch[f"bert.encoder.layer.{i}.attention.self"]
            subsch["query"].shard("weight", axis=0)
            subsch["query"].shard("bias", axis=0)
            subsch["key"].shard("weight", axis=0)
            subsch["key"].shard("bias", axis=0)
            subsch["value"].shard("weight", axis=0)
            subsch["value"].shard("bias", axis=0)
            fix_attention_mask_shape(subsch)
            subsch = sch[f"bert.encoder.layer.{i}.attention.output"]
            subsch["dense"].shard("weight", axis=1)
            subsch["dense"].sync("fwd_post", sync_op_or_fn="all_reduce")
            # shard MLP
            subsch = sch[f"bert.encoder.layer.{i}"]
            with slapo.Verify(
                subsch, [torch.randn(bs, seq, config.hidden_size)], enable=False
            ):
                subsch["intermediate.dense"].shard("weight", axis=0)
                subsch["intermediate.dense"].shard("bias", axis=0)
                subsch["output.dense"].shard("weight", axis=1)
                subsch["output.dense"].sync("fwd_post", sync_op_or_fn="all_reduce")


if __name__ == "__main__":
    pytest.main([__file__])
