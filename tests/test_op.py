# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test custom ops. Note that this test has to be invoked by torchrun since
most custom ops are for tensor parallelism.
"""
import os
import pytest

import torch
from torch import nn
from torch import distributed as dist

from slapo import op
from slapo.random import set_random_seed


def test_dropout(init_dist):
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    data = torch.rand(10, 10).cuda(local_rank)
    dist.broadcast(data, src=0)

    # The custom dropout should throw error if set_random_seed is not called.
    with pytest.raises(Exception):
        op.DropoutWithTensorParallel(p=0.5)(data)

    set_random_seed(123, tp_rank=local_rank)

    # Assuming all devices are in the same TP group, the native dropout
    # should produce the same output on all devices.
    out = nn.Dropout(p=0.5)(data)
    out_reduced = out.clone()
    dist.all_reduce(out_reduced)
    torch.testing.assert_close(
        out * world_size,
        out_reduced,
        msg=lambda msg: f"output mismatch\n{msg}",
    )

    # The custom dropout should produce different outputs on different devices
    # even they are in the same TP group.
    out = op.DropoutWithTensorParallel(p=0.5)(data)
    out_reduced = out.clone()
    dist.all_reduce(out_reduced)
    with pytest.raises(Exception):
        torch.testing.assert_close(out * world_size, out_reduced)


if __name__ == "__main__":
    pytest.main([__file__])
