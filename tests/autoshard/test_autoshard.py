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
from slapo.sharding import Solver

logger = get_logger(__name__)

# Config for verification
p = 8
bs = 8
seq_len = 512
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

    sol = Solver(sch.mod, p=p)
    results, max_cost = sol.solve([torch.randn(bs, seq_len, hidden_size)])
    # fc1: SRxRR->SR
    # fc2: SRxRR->SR->RR
    assert results["fc1_0"] == 2
    assert results["fc1_1"] == 0
    assert results["fc2_0"] == 2
    assert results["fc2_1"] == 0
    assert max_cost == (bs * seq_len * hidden_size / p + 1)


def test_attn():
    from transformers import BertLMHeadModel, AutoConfig
    import inspect

    config = AutoConfig.from_pretrained("bert-large-uncased")
    with slapo.init_empty_weights():
        model = BertLMHeadModel(config)
    logger.info(config, ranks=0)

    sch = slapo.create_schedule(model)
    input_names = ["hidden_states"]
    i = 0
    subsch = sch[f"bert.encoder.layer.{i}"]
    sig = inspect.signature(subsch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    subsch.trace(
        recursive=False,
        flatten=True,
        tracer="pytorch",
        concrete_args=concrete_args,
    )
    logger.info(subsch.mod.graph, ranks=0)

    sol = Solver(subsch.mod, p=p)
    _, max_cost = sol.solve([torch.randn(bs, seq_len, hidden_size)])
    assert max_cost == 3 * (bs * seq_len * hidden_size / p) + 4


if __name__ == "__main__":
    test_mlp()
    test_attn()
