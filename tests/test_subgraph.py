# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test operator fusion."""

# pylint: disable=comparison-with-callable
import operator
import pytest
import torch
from torch import nn
import torch.nn.functional as F

import slapo
from slapo.pattern import Pattern, ModulePattern, call_module


def test_exact_match():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, 3)
            self.bn = nn.BatchNorm2d(32)

        def forward(self, x):
            x = F.relu(self.conv(x)) + x
            return x

    sch = slapo.create_schedule(Model())

    def pattern(x: torch.Tensor):
        # ReLU + residual add
        return F.relu(x) + x

    subgraph = sch.find(pattern)[0]
    assert len(subgraph) == 2
    # pylint: disable=comparison-with-callable
    assert subgraph[0][1].target == F.relu
    assert subgraph[1][1].target == operator.add


def test_functional_module_match():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, 3)
            self.bn = nn.BatchNorm2d(32)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.conv(x)) + x
            return x

    sch = slapo.create_schedule(Model())

    def test_F():
        def pattern(x: torch.Tensor):
            # ReLU + residual add
            return F.relu(x) + x

        subgraph = sch.find(pattern)[0]
        assert len(subgraph) == 2
        assert subgraph[0][1].op == "call_module" and subgraph[0][1].target == "relu"
        assert subgraph[1][1].target == operator.add

    def test_Pat():
        def pattern(x: torch.Tensor):
            # ReLU + residual add
            return call_module("relu", x) + x

        subgraph = sch.find(pattern)[0]
        assert len(subgraph) == 2
        assert subgraph[0][1].op == "call_module" and subgraph[0][1].target == "relu"
        assert subgraph[1][1].target == operator.add

    test_F()
    test_Pat()


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


def test_relu_bn():
    sch = slapo.create_schedule(LeNet5(10))

    class ReLUBNPattern(Pattern):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(16)
            self.relu = nn.ReLU()

        # pylint: disable=arguments-differ
        def forward(self, x: torch.Tensor):
            return self.relu(self.bn(x))

    subgraph = sch["layer2"].find(ReLUBNPattern())[0]
    assert len(subgraph) == 2
    assert isinstance(sch["layer2"].get_module(subgraph[0][1].target), nn.BatchNorm2d)
    assert isinstance(sch["layer2"].get_module(subgraph[1][1].target), nn.ReLU)


def test_relu_bn_functional():
    sch = slapo.create_schedule(LeNet5(10))

    class ReLUBNPattern2(Pattern):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(16)
            self.relu = nn.ReLU()

        # pylint: disable=arguments-differ
        def forward(self, x: torch.Tensor):
            return F.relu(self.bn(x))

    subgraph = sch["layer1"].find(ReLUBNPattern2())[0]
    assert len(subgraph) == 2
    assert isinstance(sch["layer1"].get_module(subgraph[0][1].target), nn.BatchNorm2d)
    assert isinstance(sch["layer1"].get_module(subgraph[1][1].target), nn.ReLU)


def test_linear_relu():
    sch = slapo.create_schedule(LeNet5(10))

    class LinearReLUPattern(Pattern):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 20)
            self.relu = nn.ReLU()

        # pylint: disable=arguments-differ
        def forward(self, x: torch.Tensor):
            x = self.linear(x)
            x = self.relu(x)
            return x

    subgraph = sch.find(LinearReLUPattern())
    assert len(subgraph) == 2
    for i in range(2):
        assert isinstance(sch.get_module(subgraph[i][0][1].target), nn.Linear)
        assert isinstance(sch.get_module(subgraph[i][1][1].target), nn.ReLU)


def test_two_paths():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 10)
            self.fc2 = nn.Linear(10, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x1 = self.fc1(x)
            x2 = self.fc2(x)
            x = self.relu(x1 + x2)
            return x

    sch = slapo.create_schedule(Model())

    def pattern(x1, x2):
        x = F.relu(x1 + x2)
        return x

    subgraph = sch.find(pattern)[0]
    assert len(subgraph) == 2
    assert subgraph[0][1].target == operator.add
    assert subgraph[1][1].target == "relu"


def test_tree_pattern():
    class Model(nn.Module):
        def forward(self, x, y, z):
            a = x + y
            b = y - z
            c = a * b
            return c

    sch = slapo.create_schedule(Model())

    def pattern(x, y, z):
        return (x + y) * (y - z)

    subgraph = sch.find(pattern)[0]
    assert len(subgraph) == 3
    assert subgraph[0][1].name == "add"
    assert subgraph[1][1].name == "sub"
    assert subgraph[2][1].name == "mul"


def test_horizontal_pattern():
    class CoreAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32)
            self.k_proj = nn.Linear(32, 32)
            self.v_proj = nn.Linear(32, 32)
            self.dropout = nn.Dropout(0.2)

        def permute(self, x):
            return x.permute(0, 2, 1, 3)

        def forward(self, x):
            q = self.permute(self.q_proj(x))
            k = self.permute(self.k_proj(x))
            v = self.permute(self.v_proj(x))
            return self.dropout(F.softmax(q * k)) * v

    attn = CoreAttention()
    sch = slapo.create_schedule(attn)

    def pattern(x):
        x = call_module(r"[qkv]_proj", x)
        return x.permute(0, 2, 1, 3)

    subgraph = sch.find(pattern)
    assert len(subgraph) == 3
    assert len(subgraph[-1]) == 2
    assert subgraph[0][0][1].target == "q_proj"
    assert subgraph[1][0][1].target == "k_proj"
    assert subgraph[2][0][1].target == "v_proj"
    assert subgraph[0][1][1].target == "permute"


def test_pattern_call_module_class():
    sch = slapo.create_schedule(LeNet5(10))

    class LinearReLUPattern(slapo.Pattern):
        def __init__(self):
            super().__init__()
            self.fc = ModulePattern(r"fc?")
            self.relu = nn.ReLU()

        # pylint: disable=arguments-differ
        def forward(self, out: torch.Tensor):
            out = self.fc(out)
            out = self.relu(out)
            return out

    subgraph = sch.find(LinearReLUPattern())
    assert len(subgraph) == 2
    for i in range(2):
        assert isinstance(sch.get_module(subgraph[i][0][1].target), nn.Linear)
        assert isinstance(sch.get_module(subgraph[i][1][1].target), nn.ReLU)

    def pattern(x: torch.Tensor):
        x = call_module(r"fc?", x)
        x = F.relu(x)
        return x

    subgraph = sch.find(pattern)
    assert len(subgraph) == 2
    for i in range(2):
        assert isinstance(sch.get_module(subgraph[i][0][1].target), nn.Linear)
        assert isinstance(sch.get_module(subgraph[i][1][1].target), nn.ReLU)


if __name__ == "__main__":
    pytest.main([__file__])
