# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test utilities."""

import pytest
from torch import nn

import slapo


def test_submodule():
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

    mod = Model()
    sch = slapo.create_schedule(mod)
    subschs = dict(sch.named_schedules())
    for name in (
        "fc1",
        "act1",
        "fc2",
        "act2",
        "submod",
        "submod.linear",
        "submod.activation",
    ):
        assert name in subschs


if __name__ == "__main__":
    pytest.main([__file__])
