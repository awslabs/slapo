# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test annotate primitive. Note that this test has to be invoked by torchrun.
See ci/task_unit_tests.sh for an example.
"""
# pylint: disable=unused-argument

import os

import pytest
import torch

import slapo


def test_shard_annotate(init_dist):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(32, 32, bias=False)
            self.layer2 = torch.nn.Linear(32, 32, bias=False)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            return x

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    model = Model()

    with slapo.init_empty_weights():
        sch = slapo.create_schedule(model)

    sch["layer1"].shard("weight", axis=0)
    sch["layer1"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
    sch["layer2"].shard("weight", axis=1)
    sch["layer2"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")
    sch_model, _ = slapo.build(sch)

    assert hasattr(sch_model.layer1.weight, "orig_shape")
    assert hasattr(sch_model.layer2.weight, "orig_shape")


def test_arbitrary_annotate():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(32, 32, bias=False)
            self.layer2 = torch.nn.Linear(32, 32, bias=False)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            return x

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    model = Model()

    with slapo.init_empty_weights():
        sch = slapo.create_schedule(model)

    sch["layer1"].annotate("weight", "foo", "bar")
    assert "foo" in sch.metadata.param_tags

    sch_model, _ = slapo.build(sch)
    assert sch_model.layer1.weight.foo == "bar"
    assert not hasattr(sch_model.layer2.weight, "foo")


if __name__ == "__main__":
    pytest.main([__file__])
