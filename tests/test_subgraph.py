# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test operator fusion."""

import operator
import pytest
import torch
from torch import nn
import torch.nn.functional as F

import slapo


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

    subgraph = sch.find("conv", pattern)[0]
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

    def pattern(x: torch.Tensor):
        # ReLU + residual add
        return F.relu(x) + x

    subgraph = sch.find("conv", pattern)[0]
    assert len(subgraph) == 2
    assert subgraph[0][1].op == "call_module" and subgraph[0][1].target == "relu"
    # pylint: disable=comparison-with-callable
    assert subgraph[1][1].target == operator.add


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

    class ReLUBNPattern(slapo.Pattern):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(16)
            self.relu = nn.ReLU()

        # pylint: disable=arguments-differ
        def forward(self, x: torch.Tensor):
            return self.relu(self.bn(x))

    subgraph = sch["layer2"].find("0", ReLUBNPattern())[0]
    assert len(subgraph) == 2
    assert isinstance(sch["layer2"].get_module(subgraph[0][1].target), nn.BatchNorm2d)
    assert isinstance(sch["layer2"].get_module(subgraph[1][1].target), nn.ReLU)


def test_relu_bn_functional():
    sch = slapo.create_schedule(LeNet5(10))

    class ReLUBNPattern2(slapo.Pattern):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(16)
            self.relu = nn.ReLU()

        # pylint: disable=arguments-differ
        def forward(self, x: torch.Tensor):
            return F.relu(self.bn(x))

    subgraph = sch["layer1"].find("0", ReLUBNPattern2())[0]
    assert len(subgraph) == 2
    assert isinstance(sch["layer1"].get_module(subgraph[0][1].target), nn.BatchNorm2d)
    assert isinstance(sch["layer1"].get_module(subgraph[1][1].target), nn.ReLU)


# def test_linear_relu():
#     sch = slapo.create_schedule(LeNet5(10))

#     class LinearReLUPattern(slapo.Pattern):
#         def __init__(self):
#             super().__init__()
#             self.relu = nn.ReLU()

#         # pylint: disable=arguments-differ
#         def forward(self, out: torch.Tensor):
#             out = self.relu(out)
#             return out

#     subgraph = sch.find(r"fc.?", LinearReLUPattern())
#     assert len(subgraph) == 2
#     for i in range(2):
#         assert isinstance(sch.get_module(subgraph[i][0][1].target), nn.Linear)
#         assert isinstance(sch.get_module(subgraph[i][1][1].target), nn.ReLU)

#     def pattern(x: torch.Tensor):
#         x = F.relu(x)
#         return x

#     subgraph = sch.find(r"fc.?", pattern)
#     assert len(subgraph) == 2
#     for i in range(2):
#         assert isinstance(sch.get_module(subgraph[i][0][1].target), nn.Linear)
#         assert isinstance(sch.get_module(subgraph[i][1][1].target), nn.ReLU)


if __name__ == "__main__":
    pytest.main([__file__])
