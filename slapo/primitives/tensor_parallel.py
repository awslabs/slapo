# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor parallel primitives."""
# pylint: disable=arguments-differ

from collections import OrderedDict
from functools import partial

from torch import nn

from ..sharding import (
    all_gather_forward_output,
    postproc_sharding,
    reduce_backward_grad,
    reduce_forward_output,
    reduce_scatter_forward_output,
    scatter_forward_output,
)
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
            if param in sch.metadata.tie_weights:
                if id(sch.metadata.tie_weights[param]) != id(param):
                    # This parameter is tied to another parameter, and the other
                    # parameter is already sharded. In this case we directly
                    # register the sharded parameter to the module to keep them tied.
                    if new_tensor.shape != sch.metadata.tie_weights[param].shape:
                        raise RuntimeError(
                            f"Parameter {tensor_name} in {sch.path} is tied, "
                            "but they have different sharded shapes: "
                            f"{new_tensor.shape} vs "
                            f"{sch.metadata.tie_weights[param].shape}"
                        )
                    new_param = sch.metadata.tie_weights[param]
                else:
                    # The first parameter in this tie group is sharded.
                    new_param = nn.Parameter(new_tensor)
                    sch.metadata.tie_weights[param] = new_param
            else:
                new_param = nn.Parameter(new_tensor)
            # Tag param with model parallel attribute, used for grad clipping
            new_param.tensor_model_parallel = True
            # Save the original size of the parameter for consolidation.
            new_param.orig_shape = param.shape
            sch.mod.register_parameter(tensor_name, new_param)
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

        def set_output_type(output_type, gather_axis=None):
            try:
                sch.metadata.primitives["shard"]["output_type"] = output_type
            except KeyError:
                raise RuntimeError(
                    f"Output type of {sch.path} is already "
                    f"{sch.metadata.primitives['shard']['output_type']}, but "
                    f"{output_type} is requested"
                ) from None

            if gather_axis is not None:
                try:
                    sch.metadata.primitives["shard"]["gather_axis"] = gather_axis
                except KeyError:
                    raise RuntimeError(
                        f"Output of {sch.path} has to be gathered along axis "
                        f"{sch.metadata.primitives['shard']['gather_axis']}, but "
                        f"{gather_axis} is requested"
                    ) from None

        out_type, out_part_axis = postproc_sharding(
            sch.mod, tensor_name, sharded_size, axis
        )
        if out_type is not None:
            set_output_type(out_type, gather_axis=out_part_axis)

    @staticmethod
    def init_metadata():
        return OrderedDict()


def _validate_sync_op(sch, mode, sync_op_or_fn, axis=None):
    """A helper function to validate the user given sync_op_or_fn."""
    if mode == "fwd_post" and sync_op_or_fn == "scatter":
        if "output_type" in sch.metadata.primitives["shard"]:
            raise ValueError(
                "Output of {sch.path} cannot be scatter along axis {axis}, "
                "if its parameter is sharded"
            )

    if "output_type" not in sch.metadata.primitives["shard"]:
        return
    output_type = sch.metadata.primitives["shard"]["output_type"]

    if mode == "fwd_post" and sync_op_or_fn == "all_gather":
        if output_type == "partition":
            gather_axis = sch.metadata.primitives["shard"]["gather_axis"]
            if gather_axis != axis:
                raise ValueError(
                    f"Output of {sch.path} has to be gathered along axis "
                    f"{gather_axis}, but {axis} is requested"
                )
        else:
            raise ValueError("Cannot all-gather a full output")
    elif mode == "fwd_post" and sync_op_or_fn == "reduce_scatter":
        if output_type == "partition":
            raise ValueError("Cannot reduce-scatter a partition output")
    elif mode == "fwd_pre" and sync_op_or_fn == "all_gather":
        if output_type == "partial":
            raise ValueError(
                "Cannot all-gather a partition input since the operator "
                "with parameter sharded in the input dimension expects "
                "partitioned input"
            )
    elif sync_op_or_fn == "all_reduce":
        if mode == "fwd_post" and output_type == "partition":
            raise ValueError("Cannot all-reduce a partition output")


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
        # Generate the hook if sync_op_or_fn is a string.
        if isinstance(sync_op_or_fn, str):
            if mode == "fwd_post":
                sync_fn = None
                axis = kwargs.get("axis", 0)
                if sync_op_or_fn == "all_gather":
                    tensor_parallel_output_grad = kwargs.get(
                        "tensor_parallel_output_grad", True
                    )
                    _validate_sync_op(sch, mode, sync_op_or_fn, axis)
                    sync_fn = partial(
                        all_gather_forward_output,
                        dim=axis,
                        group=sch.group,
                        tensor_parallel_output_grad=tensor_parallel_output_grad,
                    )
                elif sync_op_or_fn == "reduce_scatter":
                    _validate_sync_op(sch, mode, sync_op_or_fn)
                    sync_fn = partial(
                        reduce_scatter_forward_output, dim=axis, group=sch.group
                    )
                elif sync_op_or_fn == "scatter":
                    _validate_sync_op(sch, mode, sync_op_or_fn)
                    sync_fn = partial(scatter_forward_output, dim=axis, group=sch.group)
                elif sync_op_or_fn == "all_reduce":
                    _validate_sync_op(sch, mode, sync_op_or_fn)
                    sync_fn = partial(reduce_forward_output, group=sch.group)
                else:
                    raise ValueError(
                        f"Invalid sync_op_or_fn {sync_op_or_fn} for mode {mode} "
                        "in {sch.path}."
                    )

                def hook_fn(_module, _input, output):
                    output = sync_fn(output)
                    return output

            elif mode == "fwd_pre":
                sync_fn = None
                axis = kwargs.get("axis", 0)
                if sync_op_or_fn == "all_gather":
                    tensor_parallel_output_grad = kwargs.get(
                        "tensor_parallel_output_grad", True
                    )
                    _validate_sync_op(sch, mode, sync_op_or_fn, axis)
                    sync_fn = partial(
                        all_gather_forward_output,
                        dim=axis,
                        group=sch.group,
                        tensor_parallel_output_grad=tensor_parallel_output_grad,
                    )
                else:
                    raise ValueError(
                        f"Invalid sync_op_or_fn {sync_op_or_fn} for mode {mode} "
                        "in {sch.path}."
                    )

                def hook_fn(_module, _input):
                    _input = sync_fn(_input[0])
                    return _input

            elif mode == "bwd_post":
                # We register this hook to forward pre hook, and
                # use an autograd function to do the sync in backward.
                # This is to avoid using backward hook which semantic is not clear.
                if sync_op_or_fn == "all_reduce":
                    _validate_sync_op(sch, mode, sync_op_or_fn)
                    sync_fn = partial(reduce_backward_grad, group=sch.group)
                    mode = "fwd_pre"
                else:
                    raise ValueError(
                        f"Invalid sync_op_or_fn {sync_op_or_fn} for mode {mode} "
                        "in {sch.path}."
                    )

                def hook_fn(_module, _input):
                    _input = sync_fn(_input[0])
                    return _input

            else:
                raise ValueError(
                    f"Unsupported combination of mode {mode} and "
                    f"sync_op_or_fn {sync_op_or_fn}. Please specify "
                    "sync_op_or_fn as a hook function."
                )
        else:
            hook_fn = sync_op_or_fn

        if mode == "fwd_pre":
            sch.mod.register_forward_pre_hook(hook_fn)
        elif mode == "fwd_post":
            sch.mod.register_forward_hook(hook_fn)
        elif mode == "bwd_post":
            sch.mod.register_full_backward_hook(hook_fn)
        else:
            raise ValueError(f"Unsupported mode {mode}.")
