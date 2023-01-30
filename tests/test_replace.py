# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test replace primitives."""
import pytest

from torch import nn

import slapo
from slapo.utils.common import get_hooks


def test_transfer_hook():
    """Test whether the hooks are transferred to the new replaced module."""
    # pylint: disable=unused-argument

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            return self.linear(x)

    model = Model()

    def fwd_pre_hook(mod, inp):
        return inp

    def fwd_post_hook(mod, inp, out):
        return out

    def bwd_post_hook(mod, grad_inp, grad_out):
        return grad_inp

    sch = slapo.create_schedule(model)

    # Directly register hooks instead of using .sync, because
    # we do not want to test .sync in this test.
    sch.mod.register_forward_pre_hook(fwd_pre_hook)
    sch.mod.register_forward_hook(fwd_post_hook)
    sch.mod.register_backward_hook(bwd_post_hook)
    all_hooks = get_hooks(sch.mod)
    assert len(all_hooks["fwd_pre"]) == 1
    assert len(all_hooks["fwd_post"]) == 1
    assert len(all_hooks["bwd_post"]) == 1

    sch.trace()

    all_hooks = get_hooks(sch.mod)
    assert len(all_hooks["fwd_pre"]) == 1
    assert len(all_hooks["fwd_post"]) == 1
    assert len(all_hooks["bwd_post"]) == 1


if __name__ == "__main__":
    pytest.main([__file__])
