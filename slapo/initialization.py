# Copyright 2022 The HuggingFace Team. All rights reserved.
# https://github.com/huggingface/accelerate/blob/main/src/accelerate/big_modeling.py
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager, nullcontext

import torch
from torch import nn

from .utils.versions import is_torch_version


@contextmanager
def init_empty_weights(enable: bool = True, include_buffers: bool = False):
    """
    A context manager under which models are initialized with all parameters on the
    meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.

    Parameters
    ----------
    enable: bool
        Whether or not to enable this context manager.

    include_buffers: bool
        Whether or not to also put all buffers on the meta device while initializing.
    """
    if not is_torch_version(">=", "1.9.0"):
        raise NotImplementedError(
            "Initializing empty weights to a meta device requires torch >= 1.9.0"
        )
    if enable:
        with init_on_device(torch.device("meta"), include_buffers=include_buffers) as f:
            yield f
    else:
        with nullcontext() as f:
            yield f


@contextmanager
def init_on_device(device: torch.device, include_buffers: bool = False):
    """
    A context manager under which models are initialized with all parameters
    on the specified device.

    Parameters
    ----------
    device: torch.device
        Device to initialize all parameters on.

    include_buffers: bool
        Whether or not to also put all buffers on the meta device while initializing.
    """
    # pylint: disable=redefined-variable-type
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None and param.device != device:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(
                module._parameters[name].to(device), **kwargs
            )

    def register_empty_buffer(module, name, buffer):
        old_register_buffer(module, name, buffer)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    # Patch tensor creation
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ("empty", "zeros", "ones", "full")
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(
                torch,
                torch_function_name,
                patch_tensor_constructor(getattr(torch, torch_function_name)),
            )
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
        for (
            torch_function_name,
            old_torch_function,
        ) in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)
