# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from enum import Enum
import torch
from torch import fx
import torch.nn as nn

from ..registry import register_model_dialect
from ...logger import get_logger, DEBUG, INFO

# Change INFO to DEBUG for more verbose logging.
logger = get_logger("DS-Pipeline", INFO)


class WrappedTypeCode(Enum):
    """Type code for wrapped tensor for inter-GPU communication."""

    TORCH_TENSOR = 0
    SCALAR = 1
    TRUE_NONE = 2
    LIST = 3
    TUPLE = 4


def get_simple_nested_list_str(data):
    """A helper function that prints a nested structure without printing
    tensor values.
    """
    if data is None:
        ret = "None"
    elif isinstance(data, torch.Tensor):
        if data.shape == torch.Size([1]):
            ret = f"scalar({data.item()})"
        else:
            ret = f"tensor{tuple(data.shape)}"
    elif isinstance(data, (list, tuple)):
        ret = ",".join([f"{get_simple_nested_list_str(elt)}" for elt in data])
        ret = f"[{ret}]"
    elif isinstance(data, (int, float)):
        ret = str(data)
    else:
        raise ValueError(f"Unsupported data type {type(data)}")
    return ret


def wrap_to_torch_tensor(data, device):
    if isinstance(data, torch.Tensor):
        return (data, WrappedTypeCode.TORCH_TENSOR)

    if data is None:
        data = 0
        desired_type = WrappedTypeCode.TRUE_NONE
    elif isinstance(data, list):
        desired_type = WrappedTypeCode.LIST
    elif isinstance(data, tuple):
        desired_type = WrappedTypeCode.TUPLE
    elif isinstance(data, (int, float)):
        desired_type = WrappedTypeCode.SCALAR
    else:
        raise ValueError(f"Unsupported data type {type(data)}")
    wrapped = torch.tensor(data, device=device)
    return (wrapped, desired_type)


def unwrap_torch_tensor(tensor, desired_type):
    desired_type = WrappedTypeCode(desired_type)
    if desired_type == WrappedTypeCode.TRUE_NONE:
        return None
    if desired_type == WrappedTypeCode.TORCH_TENSOR:
        return tensor
    if desired_type in [WrappedTypeCode.LIST, WrappedTypeCode.SCALAR]:
        return tensor.tolist()
    if desired_type == WrappedTypeCode.TUPLE:
        return tuple(tensor.tolist())


def flat_and_name_tensor_list(data, name, suffix):
    """Flat the given list and assign the itemized name to each tensor.
    This is mainly used for liveness.
    """
    name_n_value = [(f"{name}{suffix}", data)]
    # Note that the dict keys are droppped.
    values = data.values() if isinstance(data, dict) else data

    if isinstance(values, (list, tuple)) and any(
        [isinstance(t, torch.Tensor) for t in values]
    ):
        for idx, tensor in enumerate(values):
            name_n_value.extend(
                flat_and_name_tensor_list(tensor, name, f"{suffix}.{idx}")
            )
    return name_n_value


def encode_metadata(metadata):
    """Encode metadata to a string.
    The metadata format is:
        [<path>,<type>), (<path>,<type>), ...]
    After encoding: "<path>,<type>|<path>,<type>|..."
    """
    return "|".join([f"{p},{t}" for p, t in metadata])


def decode_metadata(metadata_str):
    """Decode metadata from a string."""
    return [m.split(",") for m in metadata_str.split("|")]


def flatten(outputs, device, path="", metadata=None, ret=None):
    """Flatten nested structure of outputs and make sure every
    output is a torch.Tensor. We maintain a metadata to restore
    the original structure. The metadata format is:
        [<path>,<type>), (<path>,<type>), ...]
    - <path> is the path to the tensor in the nested structure. For example,
     the paths of t1 and t2 in [[t1, t2]] are "0.0" and "0.1". The path of
     t1 and t2 in [[t1], [t2]] is "0.0" and "1.0".
    - <type> is WrappedTypeCode of the tensor.
    Note that len(metadata) == len(ret)
    """
    metadata = metadata if metadata else []
    ret = ret if ret else []

    for idx, output in enumerate(outputs):
        this_path = f"{path}.{idx}" if path else str(idx)
        if isinstance(output, (tuple, list)):
            metadata, ret = flatten(output, device, this_path, metadata, ret)
        elif isinstance(output, dict):
            metadata, ret = flatten(output.values(), device, this_path, metadata, ret)
        else:
            # The output here could be a torch tensor or scalar.
            # The latter will be wrapped to a torch.Tensor,
            # so that it can be passed to the next stage via nccl.
            output, desired_type = wrap_to_torch_tensor(output, device)
            ret.append(output)
            metadata.append((this_path, desired_type.value))
    return metadata, ret


def unflatten(args, metadata):
    """The reverse function of 'flatten'."""
    unordered_args = []
    assert len(args) == len(metadata)

    # Restore nested structure from metadata.
    for arg, (path, desired_type) in zip(args, metadata):
        prev_ptr = None
        curr_ptr = unordered_args
        idx = None
        for token in path.split("."):
            idx = int(token)
            while len(curr_ptr) < idx + 1:
                curr_ptr.append([])
            prev_ptr = curr_ptr
            curr_ptr = curr_ptr[idx]
        arg = unwrap_torch_tensor(arg, int(desired_type))
        prev_ptr[idx] = arg

    # Make everything tuple. Since we usually cut pipeline based on layer,
    # so the tensors to be packed are usually the layer outputs, which are
    # in tuple type. HF models also have some implementations based on tuple
    # output, such as "output[:1] + (None,) + output[1:]".
    def tupleize(data):
        if isinstance(data, (list, tuple)):
            return tuple(tupleize(t) for t in data)
        return data

    return tupleize(unordered_args)


@register_model_dialect("deepspeed", "pipeline_stage")
class DeepSpeedPipeStageWrapper(nn.Module):
    def __init__(
        self,
        mod: fx.GraphModule,
        stage_id: int,
        name: str,
        total_stages: int,
        liveness: dict,
        stage_id_2_arg_names: dict,
    ):
        super().__init__()
        self.mod = mod
        self.stage_id = stage_id
        self.name = name
        self.total_stages = total_stages
        self.last = self.stage_id == self.total_stages - 1

        self.liveness = liveness
        self.stage_id_2_arg_names = stage_id_2_arg_names

    def forward(self, *args, **kwargs):
        # TODO: use kwargs to do mapping
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]

        # Determine the device of this stage.
        device = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                device = arg.device
                break
        else:
            raise ValueError("Cannot retrieve device from pipeline stage inputs")

        # Unpack inputs
        unordered_args = []
        if self.stage_id == 0:
            # The first stage takes the original inputs.
            for arg in args:
                assert not isinstance(arg, dict)
                if isinstance(arg, tuple):
                    unordered_args.extend([item for item in arg])
                else:
                    unordered_args.append(arg)
        else:
            # The other stages take the flattened inputs packed by the previous one.
            args, metadata_str = args[:-1], args[-1]
            metadata_str = metadata_str.cpu().numpy().tobytes().decode()
            metadata = decode_metadata(metadata_str)
            unordered_args = unflatten(args, metadata)

        # The unordered arguments should match the liveness list, and we assign
        # the name of each input argument accordingly.
        liveness = self.liveness[self.stage_id]
        if logger.isEnabledFor(DEBUG):
            logger.debug(
                f"[{self.name}] Args ({len(unordered_args)}): "
                f"{get_simple_nested_list_str(unordered_args)}, "
                f"liveness ({len(liveness)}): {liveness}",
            )
        assert len(unordered_args) == len(
            liveness
        ), f"{len(unordered_args)} vs. {len(liveness)}"
        name_2_live_tensor = dict(zip(liveness, unordered_args))
        if logger.isEnabledFor(DEBUG):
            logger.debug(
                f"[{self.name}] Live tensors before forward: "
                f"{list(name_2_live_tensor.keys())}",
            )

        # Make forward argiments from live tensors to align the submodule arguments.
        ordered_args = []
        for arg_name in self.stage_id_2_arg_names[self.stage_id]:
            assert (
                arg_name in liveness
            ), f"[{self.name}] Arg {arg_name} not found in liveness list: {liveness}"
            idx = liveness.index(arg_name)
            ordered_args.append(unordered_args[idx])

        for value in kwargs.values():
            ordered_args += [value]

        if logger.isEnabledFor(DEBUG):
            logger.debug(
                f"[{self.name}] Ordered forward args: "
                f"{get_simple_nested_list_str(ordered_args)}",
            )

        # Forward pass.
        fwd_outputs = self.mod(*ordered_args)

        # Direct return if this is the last stage.
        if self.last:
            return fwd_outputs

        # Add output tensors to live tensor set.
        name_2_live_tensor.update(flat_and_name_tensor_list(fwd_outputs, self.name, ""))
        if logger.isEnabledFor(DEBUG):
            logger.debug(
                f"[{self.name}] Live tensors after forward: "
                f"{list(name_2_live_tensor.keys())}",
            )

        # Organize output based on the liveness of the next stage.
        outputs = []
        for tensor_name in self.liveness[self.stage_id + 1]:
            assert tensor_name in name_2_live_tensor
            outputs.append(name_2_live_tensor[tensor_name])
        if logger.isEnabledFor(DEBUG):
            logger.debug(
                f"[{self.name}] Output: {get_simple_nested_list_str(outputs)}",
            )

        # Flat and pack outputs for the next stage.
        metadata, ret = flatten(outputs, device)
        if logger.isEnabledFor(DEBUG):
            logger.debug(f"[{self.name}] Flatten: {len(ret)}; metadata: {metadata}")
        ret.append(
            torch.ByteTensor(
                torch.ByteStorage.from_buffer(bytes(encode_metadata(metadata), "utf8"))
            ).to(device)
        )
        ret = tuple(ret)
        return ret

@register_model_dialect("deepspeed", "pipeline_engine")
def deepspeed_pipe_engine(
    stage_modules,
    topology,
    param_dtype,
    **kwargs,
):
    from deepspeed import pipe

    model = pipe.PipelineModule(
        stage_modules,
        topology=topology,
        partition_method="uniform",
        loss_fn=kwargs.get("loss_fn", None),
        param_dtype=param_dtype,
    )
    # TODO: tie weights
    # tie_weight_groups=kwargs.get("tie_weight_groups", None)
    # model.register_tie_weights()
    return model
