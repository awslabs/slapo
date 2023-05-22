# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test different resharding schemes on MLP. 
Verified by different combinations of resharding schemes. 
"""

import torch
from torch import nn
from torch import fx
import torch.nn.functional as F

import slapo
from slapo.logger import get_logger

logger = get_logger(__name__)

# Config for verification
p = 8
bs = 8
seq_len = 1024
hidden_size = 1024


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


def test_mlp():
    with slapo.init_empty_weights():
        mlp = MLP(hidden_size)

    sch = slapo.create_schedule(mlp)
    sch.trace()
    assert isinstance(sch.mod, fx.GraphModule)

    from slapo.sharding import Solver

    sol = Solver(sch.mod, p=p)
    results, max_cost = sol.solve([torch.randn(bs, seq_len, hidden_size)])
    # fc1: SRxRR->SR
    # fc2: SRxRR->SR->RR
    assert results["fc1_0"] == 2
    assert results["fc1_1"] == 0
    assert results["fc2_0"] == 2
    assert results["fc2_1"] == 0
    assert max_cost == seq_len * hidden_size / p


if __name__ == "__main__":
    test_mlp()
