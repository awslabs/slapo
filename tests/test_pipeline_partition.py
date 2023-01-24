# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test pipeline partition related logic."""
import pytest
from mock import MagicMock

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

    # Analyze tie weights for a normal model.
    tie_weights = slapo.pipeline.analyze_tie_weights(model, False)

    assert len(tie_weights) == 1
    val = list(tie_weights.values())[0]
    assert len(val) == 3
    assert ("stage0.wte.weight", 0) in val
    assert ("stage1.linear.weight", 0) in val
    assert ("stage2.linear.weight", 0) in val

    # Analyze tie weights for a pipeline model. In this case,
    # the forward in top module only runs each pipeline stage sequentially.
    tie_weights = slapo.pipeline.analyze_tie_weights(model, True)

    assert len(tie_weights) == 1
    val = list(tie_weights.values())[0]
    assert len(val) == 3
    assert ("wte.weight", 0) in val
    assert ("linear.weight", 1) in val
    assert ("linear.weight", 2) in val


def test_analyze_tie_ranks():
    # Mock deepspeed.runtime.pipe.topology.PipeModelDataParallelTopology
    # This mocked topology assumes pp=2, tp=1, dp=1.
    topology = MagicMock()
    topology.filter_match = lambda pipe: [pipe]

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

        def forward(self, x):
            return self.stage1(self.stage0(x))

    with slapo.init_empty_weights():
        model = Model()
        # Tie weights
        model.stage1.linear.weight = model.stage0.wte.weight

    tie_weights = list(slapo.pipeline.analyze_tie_weights(model, True).values())
    tie_ranks = slapo.model_dialect.deepspeed.pipeline.analyze_tie_ranks(
        tie_weights, topology
    )

    assert len(tie_ranks) == 1
    assert len(tie_ranks[0]) == 1
    assert len(tie_ranks[0][0]) == 2
    assert tie_ranks[0][0] == [0, 1]


if __name__ == "__main__":
    pytest.main([__file__])
