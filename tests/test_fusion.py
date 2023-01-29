# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test operator fusion."""

import pytest
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import slapo


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
        x = F.relu(x)
        x = x + 1
        return x

    subgraph = sch.find("conv", pattern, include_start_op=False)
    sch.fuse(subgraph, compiler="TorchScript", name="FusedReLU")

    sch_model, _ = slapo.build(sch, init_weights=False)
    print(sch_model)

    inp = torch.randn((1, 3, 32, 32), requires_grad=True).cuda()
    out = sch_model(inp)
    out_ref = mod(inp)
    torch.testing.assert_close(out, out_ref)


if __name__ == "__main__":
    pytest.main([__file__])
