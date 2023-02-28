# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modification: Megatron-LM.
# See https://github.com/NVIDIA/Megatron-LM/blob/main/tests/tensor_parallel/test_random.py
"""
Test random state managements. Note that this test has to be invoked by torchrun.
See ci/task_unit_tests.sh for an example.
"""

import os

import pytest
import torch

from slapo.random import (
    _CUDA_RNG_STATE_TRACKER,
    CudaRNGStatesTracker,
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)


def test_cuda_rng_states_tracker():
    rng_tracker = CudaRNGStatesTracker()
    rng_tracker.set_states({"state1": 1234})
    assert rng_tracker.get_states()["state1"] == 1234

    rng_tracker.reset()
    assert not rng_tracker.get_states()

    seed = 1111
    rng_tracker.add("state2", seed)
    with pytest.raises(Exception):
        assert rng_tracker.add("state3", seed)

    with pytest.raises(Exception):
        assert rng_tracker.add("state2", 111)

    assert rng_tracker.get_states()["state2"] is not None
    with pytest.raises(Exception):
        assert ()

    rng_tracker.fork("state2")
    torch.cuda.manual_seed(seed)
    rng_state = torch.cuda.get_rng_state()
    assert torch.equal(rng_tracker.get_states()["state2"], rng_state)


def test_model_parallel_seed():
    assert torch.cuda.initial_seed() != 123

    local_rank = int(os.environ["LOCAL_RANK"])
    tp_seed = model_parallel_cuda_manual_seed(
        123, tp_rank=local_rank, always_enable_tp_seed=False
    )
    assert _CUDA_RNG_STATE_TRACKER.get_states()["model-parallel-rng"] is not None

    # Outside the context, the seed should be the same.
    assert torch.cuda.initial_seed() == 123

    # Inside the context, the seed should be different.
    with get_cuda_rng_tracker().fork():
        assert torch.cuda.initial_seed() == tp_seed


if __name__ == "__main__":
    pytest.main([__file__])
