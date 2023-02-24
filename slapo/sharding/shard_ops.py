# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sharding methods for specific modules."""
# pylint: disable=unused-argument
import torch
from torch import nn

try:
    from transformers.modeling_utils import Conv1D
except ImportError:
    Conv1D = None

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
    def validate_sync(sch, mode, sync_op_or_fn, axis=None):
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


@register_shard_method(nn.Linear)
class ShardLinear(ShardMethod):
    """Sharding methods for linear layer.
    It adjusts the input or output feature size to reflect the shard size,
    and returns the output type (partial or partition) after sharding.
    When sharding along the input feature dimension, it replces the linear layer
    with a custom version that separates the bias, so that allreduce can be applied
    to the partial output before adding the bias.
    """

    # class LinearForShardingInFeature(nn.Module):
    #     """Wrap `nn.Linear` to separate bias to support sharding
    #     weights along the input feature dimension. In this case, we have to
    #     allreduce the partial outputs before adding the bias to maintain
    #     the numerical correctness.

    #     Arguments are the same as the inputs of `nn.Linear`
    #     """

    #     def __init__(self, in_features, out_features, bias=True):
    #         super().__init__()
    #         self.linear_without_bias = nn.Linear(in_features, out_features, bias=False)
    #         if bias:
    #             self.bias = nn.Parameter(torch.Tensor(out_features))
    #         else:
    #             self.register_parameter("bias", None)

    #     def forward(self, x):
    #         x = self.linear_without_bias(x)
    #         if self.bias is not None:
    #             x = x + self.bias
    #         return x

    @staticmethod
    def preproc(sch, param_name, sharded_size, axis):
        pass

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


@register_shard_method(nn.Conv2d)
class ShardConv2d(ShardMethod):
    """Sharding methods for conv2d layer.
    It adjusts the input or output channel number to reflect the shard size,
    and returns the output type (partial or partition) after sharding.
    """

    @staticmethod
    def preproc(sch, param_name, sharded_size, axis):
        pass

    @staticmethod
    def postproc(sch, param_name, sharded_size, axis):
        axes = [1, 0] if sch.mod.transposed else [0, 1]
        if axis == axes[0]:
            sch.mod.out_channels = sharded_size
        if axis == axes[1]:
            sch.mod.in_channels = sharded_size

        raise NotImplementedError

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
    def preproc(sch, param_name, sharded_size, axis):
        pass

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
    def preproc(sch, param_name, sharded_size, axis):
        pass

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
