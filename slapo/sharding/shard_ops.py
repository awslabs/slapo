# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sharding methods for specific modules."""
# pylint: disable=unused-argument
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

try:
    from transformers.modeling_utils import Conv1D
except ImportError:
    Conv1D = None

from .sync_ops import (
    all_gather_forward_output,
    reduce_backward_grad,
    reduce_forward_output,
    reduce_scatter_forward_output,
    scatter_forward_output,
)
from ..initialization import init_empty_weights

SHARD_METHODS = {}


def apply_shard_method(method_type, sch, param_name, sharded_size, axis):
    """Apply sharding method to the module. If the module does not have
    registered sharding methods, the default sharding method is applied.

    Parameters
    ----------
    method_type: str
        The type of the sharding method. It can be "preproc", "postproc", or
        "infer_output_type".
    sch: Schedule
        The schedule of the module to be sharded.
    param_name: str
        The name of the parameter to shard.
    sharded_size: int
        The size of the sharded dimension.
    axis: int
        The axis to shard on.
    """
    module_cls = sch.mod.__class__
    method = SHARD_METHODS[module_cls] if module_cls in SHARD_METHODS else ShardMethod
    return getattr(method, method_type)(sch, param_name, sharded_size, axis)


def apply_sync_method(sch, mode, sync_op_or_fn, **kwargs):
    """Apply syncing method to the module. If the module does not have
    registered syncing methods, the default syncing method is applied.

    Parameters
    ----------
    sch: Schedule
        The schedule of the module to be synced.
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
    module_cls = sch.mod.__class__
    method = SHARD_METHODS[module_cls] if module_cls in SHARD_METHODS else ShardMethod
    return getattr(method, "sync")(sch, mode, sync_op_or_fn, **kwargs)


def register_shard_method(module_cls):
    """Register sharding methods of a module."""

    def decorator(shard_method):
        # Do nothing if the module class is None. This happens when
        # the module to be registered is not imported.
        if module_cls is None:
            return shard_method

        if not issubclass(shard_method, ShardMethod):
            raise ValueError(f"Invalid sharding method {shard_method} for {module_cls}")
        if module_cls in SHARD_METHODS:
            raise ValueError(f"The sharding methods of {module_cls} already registered")
        SHARD_METHODS[module_cls] = shard_method
        return shard_method

    return decorator


def _validate_sync(sch, mode, sync_op_or_fn, axis=None):
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


def _gen_sync_func_from_str(sch, mode, sync_op_or_fn, **kwargs):
    sync_fn = None
    if mode == "fwd_post":
        axis = kwargs.get("axis", 0)
        if sync_op_or_fn == "all_gather":
            tensor_parallel_output_grad = kwargs.get(
                "tensor_parallel_output_grad", True
            )
            _validate_sync(sch, mode, sync_op_or_fn, axis)
            sync_fn = partial(
                all_gather_forward_output,
                dim=axis,
                group=sch.group,
                tensor_parallel_output_grad=tensor_parallel_output_grad,
            )
        elif sync_op_or_fn == "reduce_scatter":
            _validate_sync(sch, mode, sync_op_or_fn)
            sync_fn = partial(reduce_scatter_forward_output, dim=axis, group=sch.group)
        elif sync_op_or_fn == "scatter":
            _validate_sync(sch, mode, sync_op_or_fn)
            sync_fn = partial(scatter_forward_output, dim=axis, group=sch.group)
        elif sync_op_or_fn == "all_reduce":
            _validate_sync(sch, mode, sync_op_or_fn)
            sync_fn = partial(reduce_forward_output, group=sch.group)
        else:
            raise ValueError(
                f"Invalid sync_op_or_fn {sync_op_or_fn} for mode {mode} "
                "in {sch.path}."
            )
    elif mode == "fwd_pre":
        axis = kwargs.get("axis", 0)
        if sync_op_or_fn == "all_gather":
            tensor_parallel_output_grad = kwargs.get(
                "tensor_parallel_output_grad", True
            )
            _validate_sync(sch, mode, sync_op_or_fn, axis)
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
    elif mode == "bwd_post":
        # We register this hook to forward pre hook, and
        # use an autograd function to do the sync in backward.
        # This is to avoid using backward hook which semantic is not clear.
        if sync_op_or_fn == "all_reduce":
            _validate_sync(sch, mode, sync_op_or_fn)
            sync_fn = partial(reduce_backward_grad, group=sch.group)
            mode = "fwd_pre"
        else:
            raise ValueError(
                f"Invalid sync_op_or_fn {sync_op_or_fn} for mode {mode} "
                "in {sch.path}."
            )
    else:
        raise ValueError(
            f"Unsupported combination of mode {mode} and "
            f"sync_op_or_fn {sync_op_or_fn}. Please specify "
            "sync_op_or_fn as a hook function."
        )
    return mode, sync_fn


def new_or_get_tied_param(sch, old_param, new_tensor):
    """TBA"""
    if old_param in sch.metadata.tie_weights:
        if id(sch.metadata.tie_weights[old_param]) != id(old_param):
            # This parameter is tied to another parameter, and the other
            # parameter is already sharded. In this case we directly
            # register the sharded parameter to the module to keep them tied.
            if new_tensor.shape != sch.metadata.tie_weights[old_param].shape:
                raise RuntimeError(
                    f"Parameter in {sch.path} is tied, "
                    "but they have different sharded shapes: "
                    f"{new_tensor.shape} vs "
                    f"{sch.metadata.tie_weights[old_param].shape}"
                )
            new_param = sch.metadata.tie_weights[old_param]
        else:
            # The first parameter in this tie group is sharded.
            new_param = nn.Parameter(new_tensor)
            sch.metadata.tie_weights[old_param] = new_param
    else:
        new_param = nn.Parameter(new_tensor)
    return new_param


class ShardMethod:
    @staticmethod
    def preproc(sch, param_name, sharded_size, axis):
        pass

    @staticmethod
    def postproc(sch, param_name, sharded_size, axis):
        pass

    @staticmethod
    def infer_output_type(sch, param_name, sharded_size, axis):
        return None, None

    @staticmethod
    def infer_and_set_output_type(sch, param_name, sharded_size, axis):
        """Infer the output type of the module after sharding, and set it
        to the schedule metadata in order to validate the sync operations.

        Parameters
        ----------
        sch: Schedule
            The schedule of the module to be sharded.
        param_name: str
            The name of the parameter to shard.
        sharded_size: int
            The size of the sharded dimension.
        axis: int
            The axis to shard on.
        """
        output_type, gather_axis = apply_shard_method(
            "infer_output_type", sch, param_name, sharded_size, axis
        )

        if output_type is not None:
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

    @staticmethod
    def sync(sch, mode, sync_op_or_fn, **kwargs):
        # Generate the hook if sync_op_or_fn is a string.
        if isinstance(sync_op_or_fn, str):
            mode, sync_fn = _gen_sync_func_from_str(sch, mode, sync_op_or_fn, **kwargs)
            if mode == "fwd_post":

                def hook_fn(_module, _input, output):
                    output = sync_fn(output)
                    return output

            elif mode == "fwd_pre":

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


@register_shard_method(nn.Linear)
class ShardLinear(ShardMethod):
    """Sharding methods for linear layer.
    It adjusts the input or output feature size to reflect the shard size,
    and returns the output type (partial or partition) after sharding.
    When sharding along the input feature dimension, we override the default sync
    function to insert the the sync op before the bias addition.
    """

    class LinearWithSyncFunc(nn.Linear):
        def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            sync_fn=None,
        ):
            super().__init__(in_features, out_features, bias, device, dtype)
            self.sync_fn = sync_fn

        def forward(self, x):
            x = F.linear(x, self.weight, None)
            if self.sync_fn is not None:
                x = self.sync_fn(x)
            if self.bias is not None:
                x = x + self.bias
            return x

    @staticmethod
    def postproc(sch, param_name, sharded_size, axis):
        if axis == 0:
            sch.mod.out_features = sharded_size
        # axis == 1
        sch.mod.in_features = sharded_size

    @staticmethod
    def infer_output_type(sch, param_name, sharded_size, axis):
        if axis == 0:
            # Note that the axis is the axis of the output
            return ("partition", 1)
        # axis == 1
        return ("partial", None)

    @staticmethod
    def sync(sch, mode, sync_op_or_fn, **kwargs):
        # In the following two cases, we simply fallback to the default syncing method:
        # 1. If the output type is not specified, meaning that this is "fwd_pre"
        #    syncing. In this case, we don't need special handling for the linear.
        # 2. If the output is partitioned, we don't need to insert the sync op
        #    before the bias addition.
        if (
            "output_type" not in sch.metadata.primitives["shard"]
            or sch.metadata.primitives["shard"]["output_type"] == "partition"
        ):
            ShardMethod.sync(sch, mode, sync_op_or_fn, **kwargs)
            return

        if not isinstance(sync_op_or_fn, str):
            raise ValueError(
                "Only support string sync_op_or_fn for linear layer with input "
                f"feature dimension sharded, but got {sync_op_or_fn}"
            )

        mode, sync_fn = _gen_sync_func_from_str(sch, mode, sync_op_or_fn, **kwargs)
        if mode != "fwd_post":
            raise ValueError(
                "Only support mode fwd_post when syncing a linear with input feature "
                f"sharded, but got {mode}"
            )

        # If the output is partial, we need to insert the sync op
        # before the bias addition.
        with init_empty_weights(enable=(sch.mod.weight.device == torch.device("meta"))):
            new_mod = ShardLinear.LinearWithSyncFunc(
                sch.mod.in_features,
                sch.mod.out_features,
                sch.mod.bias is not None,
                sch.mod.weight.device,
                sch.mod.weight.dtype,
                sync_fn,
            )
        new_mod.register_parameter("weight", sch.mod.weight)
        new_mod.register_parameter("bias", sch.mod.bias)
        sch.replace(new_mod)

        # Deal with tied weights.
        # new_param = new_or_get_tied_param(
        #     sch, sch.mod.get_parameter("weight"), new_mod.weight
        # )
        # sch.mod.register_parameter("weight", new_param)


@register_shard_method(nn.Conv2d)
class ShardConv2d(ShardMethod):
    """Sharding methods for conv2d layer.
    It adjusts the input or output channel number to reflect the shard size,
    and returns the output type (partial or partition) after sharding.
    """

    @staticmethod
    def postproc(sch, param_name, sharded_size, axis):
        axes = [1, 0] if sch.mod.transposed else [0, 1]
        if axis == axes[0]:
            sch.mod.out_channels = sharded_size
        if axis == axes[1]:
            sch.mod.in_channels = sharded_size

    @staticmethod
    def infer_output_type(sch, param_name, sharded_size, axis):
        axes = [1, 0] if sch.mod.transposed else [0, 1]
        if axis == axes[0]:
            return ("partition", 1)
        if axis == axes[1]:
            return ("partial", None)

        raise NotImplementedError


@register_shard_method(Conv1D)
class ShardConv1D(ShardMethod):
    """Sharding methods for Conv1D layer. Note that
    Conv1D has a transposed weight (input features, output features) compared to Linear.
    It adjusts the input or output feature size to reflect the shard size,
    and returns the output type (partial or partition) after sharding.
    """

    @staticmethod
    def postproc(sch, param_name, sharded_size, axis):
        if axis == 1 or param_name == "bias":
            sch.mod.nf = sharded_size

    @staticmethod
    def infer_output_type(sch, param_name, sharded_size, axis):
        if axis == 1 or param_name == "bias":
            # Note that the axis is the axis of the output.
            return ("partition", 1)
        # axis == 0
        return ("partial", None)


@register_shard_method(nn.BatchNorm2d)
class ShardBatchNorm2d(ShardMethod):
    """Sharding methods for BatchNorm2d layer.
    It adjusts the feature number to reflect the shard size,
    and returns the output type (partition along axis=1).
    """

    @staticmethod
    def postproc(sch, param_name, sharded_size, axis):
        if axis != 0:
            raise ValueError("BatchNorm2d only supports sharding on axis 0")
        sch.mod.num_features = sharded_size

    @staticmethod
    def infer_output_type(sch, param_name, sharded_size, axis):
        if axis != 0:
            raise ValueError("BatchNorm2d only supports sharding on axis 0")
        return ("partition", 1)
