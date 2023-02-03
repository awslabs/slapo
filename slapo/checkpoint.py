# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modification: Megatron-LM.
# See https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/random.py
"""Model checkpoints and activation checkpointing with the consideration
of 3D parallelism and random states.
"""
import torch
from torch.utils.checkpoint import detach_variable
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .random import get_cuda_rng_tracker, is_random_seed_set, _set_cuda_rng_state


class CheckpointFunctionWithRNGTracker(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
    two main changes:
        1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
        2) the states in the model parallel tracker are also properly
            tracked/set/reset.
    """

    # pylint: disable=abstract-method, arguments-differ

    @staticmethod
    def forward(ctx, run_function, *args):
        ctx.run_function = run_function

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        with torch.no_grad():
            outputs = run_function(*args)

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for idx, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(idx)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        # We detach the tensor inputs to make sure we hold a reference to
        # the tensor data. This is needed because when pipeline is enabled,
        # the tensor data may be released by the pipeline engine as it does
        # not know that the tensor is used in the backward pass.
        ctx.save_for_backward(*detach_variable(tuple(tensor_inputs)))

        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )
        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        # Fill in inputs with appropriate saved tensors.
        for idx, tidx in enumerate(tensor_indices):
            inputs[tidx] = tensors[idx]

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.cuda.get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # Compute the forward pass.
        detached_inputs = detach_variable(tuple(inputs))
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for idx, output in enumerate(outputs):
            if torch.is_tensor(output) and output.requires_grad:
                outputs_with_grad.append(output)
                args_with_grad.append(args[idx])
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in detached_inputs
        )
        return (None,) + grads


def checkpoint(function, *args, use_reentrant=True, **kwargs):
    """Checkpoint a model or part of the model. See PyTorch checkpoint
    for details about behaviors and arguments. The only difference is
    when the random seed is set by Slapo, the checkpoint function will
    also track the random states and restore them properly.

    TODO: The implementation in Megatron-LM has a mode to distribute
    the saved activations across model parallel groups to further reduce
    the memory footprint. This is not implemented here yet.
    """
    if not is_random_seed_set():
        return torch_checkpoint(function, *args, use_reentrant=use_reentrant, **kwargs)
    return CheckpointFunctionWithRNGTracker.apply(function, *args)
