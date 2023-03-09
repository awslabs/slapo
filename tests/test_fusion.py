# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test operator fusion."""

# pylint: disable=comparison-with-callable
import copy
import operator
import pytest

import torch
from torch import nn
import torch.nn.functional as F

import slapo
from slapo.pattern import call_module

from .utils import reset_random_seeds


def test_decompose():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 20)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.linear(x)
            x = self.relu(x)
            return x

    mod = Model().cuda()
    sch = slapo.create_schedule(copy.deepcopy(mod))

    sch["linear"].decompose()
    sch.trace(flatten=True)

    sch_model, _ = slapo.build(sch, init_weights=False)
    print(sch_model)

    inp = torch.randn((32, 10), requires_grad=True).cuda()
    out = sch_model(inp)
    out_ref = mod(inp)
    torch.testing.assert_close(out, out_ref)


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
    sch = slapo.create_schedule(copy.deepcopy(mod))

    def pattern(x: torch.Tensor):
        x = call_module("conv", x)
        x = F.relu(x)
        x = x + 1
        return x

    subgraph = sch.find(pattern)
    assert len(subgraph[0]) == 3
    sch.fuse(subgraph, compiler="TorchScript", name="FusedReLU")

    sch_model, _ = slapo.build(sch, init_weights=False)
    print(sch_model)

    inp = torch.randn((1, 3, 32, 32), requires_grad=True).cuda()
    out = sch_model(inp)
    out_ref = mod(inp)
    torch.testing.assert_close(out, out_ref)


def test_fallback_fusion():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, 3)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = x + 1
            return x

    mod = Model().cuda()
    sch = slapo.create_schedule(copy.deepcopy(mod))

    def pattern(x: torch.Tensor):
        x = call_module("conv", x)
        x = F.relu(x)
        return x

    subgraph = sch.find(pattern)
    assert len(subgraph[0]) == 2
    sch.fuse(subgraph, compiler=None, name="FusedConvReLU")
    assert sch.mod.FusedConvReLU_0.__class__.__name__ == "FusedConvReLU"
    assert isinstance(getattr(sch.mod.FusedConvReLU_0, "0"), nn.Conv2d)
    assert isinstance(getattr(sch.mod.FusedConvReLU_0, "1"), nn.ReLU)

    sch_model, _ = slapo.build(sch, init_weights=False)
    print(sch_model)

    inp = torch.randn((1, 3, 32, 32), requires_grad=True).cuda()
    out = sch_model(inp)
    out_ref = mod(inp)
    torch.testing.assert_close(out, out_ref)


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
    sch = slapo.create_schedule(copy.deepcopy(mod))

    sch["linear"].decompose()
    sch.trace(flatten=True)
    print(sch.mod.graph)

    def pattern(x, bias):
        x = F.gelu(bias + x)
        return x

    subgraph = sch.find(pattern)
    assert len(subgraph[0]) == 2
    assert subgraph[0][0][1].target == operator.add
    assert subgraph[0][1][1].target == "gelu"
    sch.fuse(subgraph, compiler="TorchScript", name="BiasGeLU")
    assert isinstance(sch["BiasGeLU_0"].mod, torch.jit.ScriptModule)

    sch_model, _ = slapo.build(sch, init_weights=False)
    print(sch_model)

    inp = torch.randn((1, 16, 1024, 1024), requires_grad=True).cuda()
    out = sch_model(inp)
    out_ref = mod(inp)
    torch.testing.assert_close(out, out_ref)


def test_bias_layernorm():
    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1024, 1024)
            self.dropout = nn.Dropout(0.2)
            self.ln = nn.LayerNorm(1024)

        def forward(self, x, residual):
            x = self.linear(x)
            x = self.dropout(x)
            x = self.ln(x + residual)
            return x

    mod = Projection().cuda()
    sch = slapo.create_schedule(copy.deepcopy(mod))
    torch.testing.assert_allclose(sch.mod.linear.bias, mod.linear.bias)

    sch["linear"].decompose()
    sch.trace(flatten=True)
    print(sch.mod)

    def pattern(x, bias, residual):
        return F.layer_norm(F.dropout(x + bias) + residual, 1024)

    subgraph = sch.find(pattern)
    assert len(subgraph) == 1
    assert len(subgraph[0]) == 4
    assert subgraph[0][0][1].target == operator.add
    assert subgraph[0][1][1].target == "dropout"
    assert subgraph[0][2][1].target == operator.add
    assert subgraph[0][3][1].target == "ln"

    sch.fuse(subgraph, compiler="TorchScript", name="FusedLN")
    assert isinstance(sch["FusedLN_0"].mod, torch.jit.ScriptModule)

    sch_model, _ = slapo.build(sch, init_weights=False)
    torch.testing.assert_allclose(sch_model.linear.weight, mod.linear.weight)
    print(sch_model)

    inp = torch.randn((1, 16, 1024, 1024), requires_grad=True).cuda()
    resid = torch.randn((1, 16, 1024, 1024), requires_grad=True).cuda()
    reset_random_seeds()
    out = sch_model(inp, resid)

    reset_random_seeds()
    out_ref = mod(inp, resid)
    torch.testing.assert_close(out, out_ref)


if __name__ == "__main__":
    pytest.main([__file__])
