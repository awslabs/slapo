# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument
"""
Test different resharding schemes on MLP. 
Verified by different combinations of resharding schemes. 
"""

import os
import copy
import argparse

import torch
from torch import nn
from torch import fx
import torch.nn.functional as F

import slapo
from slapo.logger import get_logger

logger = get_logger(__name__)

# Config for verification
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


with slapo.init_empty_weights():
    mlp = MLP(hidden_size)

sch = slapo.create_schedule(mlp)
sch.trace()
assert isinstance(sch.mod, fx.GraphModule)

from slapo.sharding import Solver

sol = Solver(sch.mod, p=8)
sol.solve([torch.randn(bs, seq_len, hidden_size)])
