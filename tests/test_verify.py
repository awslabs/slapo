# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test verification."""

import pytest

import slapo
from slapo.pattern import call_module

import torch
from torch import nn
import torch.nn.functional as F


def test_verify_replace():
    class SubMod(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1024, 1024)
            self.activation = nn.ReLU()

        def forward(self, x):
            x = self.linear(x)
            x = self.activation(x)
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1024, 1024)
            self.act1 = nn.ReLU()
            self.fc2 = nn.Linear(1024, 1024)
            self.act2 = nn.ReLU()
            self.submod = SubMod()

        def forward(self, x):
            x = self.fc1(x)
            x = self.act1(x)
            x = self.fc2(x)
            x = self.act2(x)
            x = self.submod(x)
            return x

    model = Model()
    sch = slapo.create_schedule(model)

    def pattern(x):
        x = call_module("fc1", x)
        x = F.relu(x)
        return x

    subgraph = sch.find(pattern)
    print("Subgraph:", subgraph)
    example_inputs = [torch.randn(1, 1024)]
    with slapo.verify(example_inputs=example_inputs):
        new_mod = SubMod()
        sch.replace(new_mod, subgraph)


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
    with slapo.verify(inp):
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
    with slapo.verify(inp):
        sch.fuse(subgraph, compiler="TorchScript", name="BiasGeLU")
    assert isinstance(sch["BiasGeLU_0"].mod, torch.jit.ScriptModule)


if __name__ == "__main__":
    # test_verify_replace()
    test_vertical_fusion()
    test_bias_gelu()
    # pytest.main([__file__])
