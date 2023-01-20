# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test pipeline partition related logic."""
import pytest

from torch import nn
import slapo


def test_analyze_tie_weights():
    class Stage0(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(10, 10)
            self.linear = nn.Linear(10, 10, bias=False)

        def forward(self, x):
            return self.linear(self.wte(x))

    class StageN(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10, bias=False)

        def forward(self, x):
            return self.linear(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.stage0 = Stage0()
            self.stage1 = StageN()
            self.stage2 = StageN()

        def forward(self, x):
            return self.stage2(self.stage1(self.stage0(x)))

    with slapo.init_empty_weights():
        model = Model()
        # Tie weights
        model.stage1.linear.weight = model.stage0.wte.weight
        model.stage2.linear.weight = model.stage0.wte.weight

    tie_weights = slapo.pipeline.analyze_tie_weights(model)
    assert len(tie_weights) == 1
    assert len(tie_weights[0]) == 3
    assert ("stage0.wte.weight", 0) in tie_weights[0]
    assert ("stage1.linear.weight", 1) in tie_weights[0]
    assert ("stage2.linear.weight", 2) in tie_weights[0]


if __name__ == "__main__":
    pytest.main([__file__])
