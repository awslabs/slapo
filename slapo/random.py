# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modification: Megatron-LM.
# See https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/random.py
"""Random seed and states management."""

import contextlib
import random

import numpy as np
import torch
from torch.cuda import _lazy_call

# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = "model-parallel-rng"


def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.

    Paramters
    ---------
    new_state : torch.ByteTensor
        The desired state.

    device : int
        The GPU device to set the state for. If -1, the current device is used.
    """
    if device == -1:
        device = torch.device("cuda")
    elif isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)

    def cb():
        idx = device.index
        if idx is None:
            idx = torch.cuda.current_device()
        default_generator = torch.cuda.default_generators[idx]
        default_generator.set_state(new_state)

    _lazy_call(cb)


class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.
    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self):
        # Map from a string name to the cuda rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """Set to the initial state (no tracker)."""
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception(f"seed {seed} already exists")
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception(f"cuda rng state {name} already exists")
        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)
        self.states_[name] = torch.cuda.get_rng_state()
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise RuntimeError(
                f"cuda rng state {name} is not added. "
                "Did you call 'set_random_seed'?"
            )
        # Store current rng state.
        orig_cuda_rng_state = torch.cuda.get_rng_state()
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Update the current rng state for later use.
            self.states_[name] = torch.cuda.get_rng_state()
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()


def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    return _CUDA_RNG_STATE_TRACKER


def model_parallel_cuda_manual_seed(seed, tp_rank, always_enable_tp_seed):
    """Initialize model parallel cuda seed.
    This function should be called after the model parallel is
    initialized. Also, no torch.cuda.manual_seed should be called
    after this function. Basically, this is replacement for that
    function.
    Two sets of RNG states are tracked:

    - default state: This is for data parallelism and is the same among a
        set of model parallel GPUs but different across different model parallel groups.
        This is used for example for dropout in the non-tensor-model-parallel regions.
    - tensor-model-parallel state: This state is different among a set of model
        parallel GPUs, but the same across data parallel groups. This is used for
        example for dropout in model parallel regions.

    Parameters
    ----------
    seed : int
        Random seed.
    tp_rank : int
        Tensor model parallel rank.
    always_enable_tp_seed : bool
        Always enable tensor model parallel seed. This is used when sequence
        parallelism is enabled and all dropouts should use different seeds
        even they are in the same TP group. Default is False, meaning that
        tensor model parallel seed is only enabled with get_cuda_rng_tracker().fork().

    Returns
    -------
    int
        Tensor model parallel seed of this rank.
    """
    # 2718 is just for fun and any POSITIVE value will work.
    tensor_model_parallel_seed = seed + 2718 + tp_rank

    _CUDA_RNG_STATE_TRACKER.reset()
    # Set the default state.
    if always_enable_tp_seed:
        torch.cuda.manual_seed(tensor_model_parallel_seed)
    else:
        torch.cuda.manual_seed(seed)
    # and model parallel state.
    _CUDA_RNG_STATE_TRACKER.add(
        _MODEL_PARALLEL_RNG_TRACKER_NAME, tensor_model_parallel_seed
    )
    return tensor_model_parallel_seed


def is_random_seed_set():
    """Check if random seed is set."""
    return bool(_CUDA_RNG_STATE_TRACKER.get_states())


def set_random_seed(
    seed=2013, dp_rank=None, pp_rank=None, tp_rank=None, always_enable_tp_seed=False
):
    """Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed. Default is 2013.
    dp_rank : Optional[int]
        Data parallel rank. Default is None means no data parallelism.
    pp_rank : Optional[int]
        Pipeline parallel rank. Default is None means no pipeline parallelism.
    tp_rank : Optional[int]
        Tensor model parallel rank. Default is None means no tensor parallelism.
    always_enable_tp_seed : bool
        Always enable tensor model parallel seed. This is used when sequence
        parallelism is enabled and all dropouts should use different seeds
        even they are in the same TP group. Default is False, meaning that
        tensor model parallel seed is only enabled with get_cuda_rng_tracker().fork().

    Returns
    -------
    int
        Random seed of this rank.
    """
    # Ensure each pipeline stage uses different seed.
    if pp_rank is not None:
        seed += 100 * pp_rank

    # Ensure each data parallel group uses different seed.
    if dp_rank is not None:
        seed += 10 * dp_rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # In above cases, devices in the same TP group should have the same seed.
    # However, we may need them to have different seeds for some cases, so
    # here we maintain different seeds for each device in TP group separately.
    if torch.cuda.device_count() > 0 and tp_rank is not None:
        model_parallel_cuda_manual_seed(seed, tp_rank, always_enable_tp_seed)

    return seed
