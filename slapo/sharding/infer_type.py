# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output type inference function registration for shardable ops."""
# pylint: disable=unused-argument
from torch import nn

try:
    from transformers.modeling_utils import Conv1D
except ImportError:
    Conv1D = None

OUTPUT_INFER_FN = {}


def register_output_infer_fn(module_cls):
    """Register an output inference function for an op."""

    def decorator(rule_fn):
        # Do nothing if the module class is None. This happens when
        # the module to be registered is not imported.
        if module_cls is None:
            return rule_fn

        if module_cls in OUTPUT_INFER_FN:
            raise ValueError(f"{module_cls} already registered for output inference")
        OUTPUT_INFER_FN[module_cls] = rule_fn
        return rule_fn

    return decorator


def get_all_rules():
    """Get all registered ops with output inference functions."""
    return OUTPUT_INFER_FN


def get_output_type_after_sharding(module, param_name, sharded_size, axis):
    """Get the output type (partition or partial) after sharding the module
    along the given axis.

    Parameters
    ----------
    module: torch.nn.Module
        The module to shard.
    param_name: str
        The name of the parameter to shard.
    sharded_size: int
        The size of the sharded dimension.
    axis: int
        The axis to shard on.

    Returns
    -------
    tuple
        The output type and the axis (if the output is partitioned).
        The output type will be None if the output inference function is not
        registered.
    """
    module_cls = module.__class__
    if module_cls in OUTPUT_INFER_FN:
        return OUTPUT_INFER_FN[module_cls](module, param_name, sharded_size, axis)
    return None, None


@register_output_infer_fn(nn.Linear)
def _linear(module, param_name, sharded_size, axis):
    """Output inference for linear layer.
    It adjusts the input or output feature size to reflect the shard size,
    and returns the output type (partial or partition) after sharding.

    Parameters
    ----------
    module: torch.nn.Linear
        The linear layer to shard.
    param_name: str
        The name of the parameter to shard.
    sharded_size: int
        The size of the sharded dimension.
    axis: int
        The axis to shard on.

    Returns
    -------
    tuple
        The output type and the axis (if the output is partitioned).
    """
    if axis == 0:
        module.out_features = sharded_size
        # Note that the axis is the axis of the output
        return ("partition", 1)
    # axis == 1
    module.in_features = sharded_size
    return ("partial", None)


@register_output_infer_fn(nn.Conv2d)
def _conv2d(module, param_name, sharded_size, axis):
    """Output inference for conv2d layer.
    It adjusts the input or output channel number to reflect the shard size,
    and returns the output type (partial or partition) after sharding.

    Parameters
    ----------
    module: torch.nn.Conv2d
        The layer to shard.
    param_name: str
        The name of the parameter to shard.
    sharded_size: int
        The size of the sharded dimension.
    axis: int
        The axis to shard on.

    Returns
    -------
    tuple
        The output type and the axis (if the output is partitioned).
    """
    axes = [1, 0] if module.transposed else [0, 1]
    if axis == axes[0]:
        module.out_channels = sharded_size
        return ("partition", 1)
    if axis == axes[1]:
        module.in_channels = sharded_size
        return ("partial", None)

    raise NotImplementedError


@register_output_infer_fn(nn.BatchNorm2d)
def _batchnorm2d(module, param_name, sharded_size, axis):
    """Output inference for BatchNorm2d layer.
    It adjusts the feature number to reflect the shard size,
    and returns the output type (partition along axis=1).

    Parameters
    ----------
    module: torch.nn.BatchNorm2d
        The layer to shard.
    param_name: str
        The name of the parameter to shard.
    sharded_size: int
        The size of the sharded dimension.
    axis: int
        The axis to shard on.

    Returns
    -------
    tuple
        The output type and the axis (if the output is partitioned).
    """
    if axis != 0:
        raise ValueError("BatchNorm2d only supports sharding on axis 0")
    module.num_features = sharded_size
    return ("partition", 1)


@register_output_infer_fn(Conv1D)
def _conv1d(module, param_name, sharded_size, axis):
    """Output inference for Conv1D layer. Note that Conv1D has a transposed
    weight (input features, output features) compared to Linear.
    It adjusts the input or output feature size to reflect the shard size,
    and returns the output type (partial or partition) after sharding.

    Parameters
    ----------
    module: torch.nn.Linear
        The linear layer to shard.
    param_name: str
        The name of the parameter to shard.
    sharded_size: int
        The size of the sharded dimension.
    axis: int
        The axis to shard on.

    Returns
    -------
    tuple
        The output type and the axis (if the output is partitioned).
    """
    if axis == 1 or param_name == "bias":
        module.nf = sharded_size
        # Note that the axis is the axis of the output.
        return ("partition", 1)
    # axis == 0
    return ("partial", None)
