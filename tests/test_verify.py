# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test verification."""

import pytest

import slapo
import torch
from torch import nn


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

    example_inputs = [torch.randn(1, 1024)]
    with slapo.verify(example_inputs=example_inputs):
        sch["fc1"].replace(nn.Linear(1024, 1024))


if __name__ == "__main__":
    test_verify_replace()
    # pytest.main([__file__])
