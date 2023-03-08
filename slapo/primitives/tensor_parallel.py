# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor parallel primitives."""
# pylint: disable=arguments-differ

import functools
from collections import OrderedDict

from ..random import get_cuda_rng_tracker
from ..sharding import apply_shard_method, apply_sync_method, new_or_get_tied_param
from .base import Primitive, register_primitive


@register_primitive()
class ShardPrimitive(Primitive):
    """Shard a parameter along the given axis by the world size.
    The shape of the parameter axis being sharded must be divisible by the world size.

    Parameters
    ----------
    tensor_name: str
        The name of the parameter to shard.
    axis: int
        The axis to shard on.
    """

    @staticmethod
    def name():
        return "shard"

    @staticmethod
    def apply(sch, tensor_name: str, axis: int):
        def _shard(name, tensor):
            assert axis < len(tensor.shape)
            # TODO: Support arbitrary size sharding
            if tensor.shape[axis] % sch.world_size != 0:
                raise RuntimeError(
                    f"Parameter/Buffer {name} in {sch.path} cannot be sharded "
                    f"along axis {axis} with size {tensor.shape[axis]} "
                    f"by {sch.world_size}"
                )
            sharded_size = tensor.shape[axis] // sch.world_size
            return (
                tensor.detach().split(sharded_size, dim=axis)[sch.rank].contiguous(),
                sharded_size,
            )

        try:
            param = sch.mod.get_parameter(tensor_name)
            new_tensor, sharded_size = _shard(tensor_name, param)
            new_param = new_or_get_tied_param(sch, param, new_tensor)
            sch.mod.register_parameter(tensor_name, new_param)

            # Save the original size of the parameter for consolidation.
            sch.annotate(tensor_name, "orig_shape", param.shape)
        except AttributeError:
            buffer = sch.mod.get_buffer(tensor_name)
            new_buffer, sharded_size = _shard(tensor_name, buffer)
            sch.mod.register_buffer(tensor_name, new_buffer)

        # Add metadata for sync and check. FIXME: A validation mechanism to check this.
        # 1. Whether the param is already sharded in different axis.
        # 2. Whether the output syncing method is conflict.
        try:
            sch.metadata.primitives["shard"][tensor_name] = axis
        except KeyError:
            raise RuntimeError(
                f"Parameter/Buffer {tensor_name} in {sch.path} is already "
                f"sharded along axis {sch.metadata.primitives['shard'][tensor_name]}"
            ) from None

        apply_shard_method("postproc", sch, tensor_name, sharded_size, axis)
        apply_shard_method(
            "infer_and_set_output_type", sch, tensor_name, sharded_size, axis
        )

    @staticmethod
    def init_metadata():
        return OrderedDict()


@register_primitive()
class SyncPrimitive(Primitive):
    """Synchronize the tensor across multiple devices.
    Since the underlying implementation is registering a PyTorch hook
    to the target module, the mode could be "fwd_pre", "fwd_post", "bwd_post".
    The following are some example use cases:

    Case 1: (replica x, shard_out w) -> partition output -> allgather
            -> full output -> (replica x, shard_out w).
        In this case, since forward uses all-gather to get a full output,
        backward must have a split to match the shape, and
        allreduce is also required for x.grad, so we use:
        ```python
        sch["out_prj"].shard("weight", axis=0)
        sch["out_prj"].sync(mode="fwd_post", sync_op_or_fn="all_gather", axis=1)
        sch["out_prj"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
        ```

    Case 2: (replica x, shard_out w) -> partition output -> (shard x, shard_in w).
        In this case, backward still needs allreduce, so we use:
        ```python
        sch["out_prj"].shard("weight", axis=0)
        sch["out_prj"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
        ```

    Case 3: (shard x, shard_in w) -> partial sum -> allreduce
            -> (replica x, shard_out w).
        In this case, backward does not need allreduce, so mode should be 'forward'.
        ```python
        sch["out_prj"].shard("weight", axis=1)
        sch["out_prj"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")
        ```

    Case 4: (shard x, shard_in w) -> partial sum -> reduce-scatter
            -> ... -> allgather -> full output.
        This case breaks the allreduce in case 3 to reduce-scatter and allgather,
        which is called "sequence parallelism". In this case, we also need
        to specify the allgather point in kwargs, so we use:
        ```python
        sch["out_prj"].shard("weight", axis=1)
        sch["out_prj"].sync(mode="fwd_post", sync_op_or_fn="reduce_scatter", axis=1)
        sch["dropout"].sync(mode="fwd_post", sync_op_or_fn="all_gather", axis=1)
        ```

    Case 5: Custom sync function.
        We may need additional logic when syncing the output. In this case,
        we could use a custom sync function. Here is an example of sharding
        a word embedding:
        ```python
        sch["wte"].shard("weight", axis=0)

        def fwd_pre_hook(_module, _input):
            ...
        def fwd_post_hook(_module, _input, output):
            ...
        sch["wte"].sync(mode="fw_pre", sync_op_or_fn=fwd_pre_hook)
        sch["wte"].sync(mode="fw_post", sync_op_or_fn=fwd_post_hook)
        ```

    Parameters
    ----------
    mode: str
        Where to sync the output. Could be "fwd_pre", "fwd_post", or "bwd_post".
    sync_op_or_fn: Union[str, Callable]
        The sync_op_or_fn (e.g., all_gather, all_reduce, reduce_scatter) or
        hook function.
    kwargs: Dict[str, Any]
        Additional arguments. For example, if sync_op_or_fn is specified,
        axis is required for reduce_scatter and all_gather. Note that the axis
        is the axis of the output tensor, not the input or weight tensor.
    """

    @staticmethod
    def name():
        return "sync"

    @staticmethod
    def apply(sch, mode, sync_op_or_fn, **kwargs):
        apply_sync_method(sch, mode, sync_op_or_fn, **kwargs)


@register_primitive()
class ForkRNGPrimitive(Primitive):
    """Fork a random number generator (RNG) for the given module.
    This is required when the module is within a parallel region, meaning that
    its input is partitioned across multiple device in a tensor parallel group.
    In this case, the RNG needs to be forked to ensure that the random numbers
    generated by the module are different across different devices; otherwise,
    the module will generate the same random numbers on all devices, which results
    in a repeated pattern across the final aggregated output and may hurt the
    training convergence.
    """

    @staticmethod
    def name():
        return "fork_rng"

    @staticmethod
    def apply(sch):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with get_cuda_rng_tracker().fork():
                    return func(*args, **kwargs)

            return wrapper

        sch.mod.forward = decorator(sch.mod.forward)
        sch.mod.traceable = False
