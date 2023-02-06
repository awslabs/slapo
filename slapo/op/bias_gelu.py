# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""BiasGeLU module using with fused kernels."""
# pylint: disable=abstract-method
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    from functorch.compile import memory_efficient_fusion
except ImportError:
    memory_efficient_fusion = None


class BiasGeLUFunction(torch.autograd.Function):
    """Bias+GeLU. Copied from Megatron-LM."""

    # pylint: disable=no-self-argument, arguments-differ

    @torch.jit.script
    def bias_gelu(bias, y):
        x = bias + y
        return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

    # gradient of tanh approximation of gelu
    # gradient of actual gelu is:
    # 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
    @torch.jit.script
    def bias_gelu_back(g, bias, y):
        x = bias + y
        tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
        # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
        ff = 0.5 * x * (
            (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
        ) + 0.5 * (1 + tanh_out)
        return ff * g

    @staticmethod
    # bias is an optional argument
    def forward(ctx, inp, bias):
        ctx.save_for_backward(inp, bias)
        return BiasGeLUFunction.bias_gelu(bias, inp)

    @staticmethod
    def backward(ctx, grad_output):
        inp, bias = ctx.saved_tensors
        tmp = BiasGeLUFunction.bias_gelu_back(grad_output, bias, inp)
        return tmp, tmp


class FusedBiasGELU(torch.nn.Module):
    def __init__(self, size, device=None, dtype=None, prev_weight=None, fused=True):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.bias = torch.nn.Parameter(torch.empty(size, **factory_kwargs))
        self.fused = fused
        self.reset_parameters(prev_weight)

    def reset_parameters(self, prev_weight=None):
        p_range = (0, 1)
        if prev_weight is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(prev_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            p_range = (-bound, bound)
        torch.nn.init.uniform_(self.bias, *p_range)

    def forward(self, inp):
        if self.fused:
            return BiasGeLUFunction.apply(inp, self.bias)
        return F.gelu(inp + self.bias, approximate="none")


def new_gelu(inp):
    """New GELU activation function copied from HuggingFace transformers."""
    return (
        0.5
        * inp
        * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (inp + 0.044715 * torch.pow(inp, 3.0))
            )
        )
    )


def bias_new_gelu(inp, bias):
    return new_gelu(inp + bias)


class FusedBiasNewGELU(torch.nn.Module):
    def __init__(
        self, size, device=None, dtype=None, prev_weight=None, fused=True, aot=True
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.bias = torch.nn.Parameter(torch.empty(size, **factory_kwargs))
        self.fused = fused
        self.reset_parameters(prev_weight)
        if self.fused:
            if aot and memory_efficient_fusion is not None:
                self.func = memory_efficient_fusion(bias_new_gelu)
            else:
                self.func = torch.jit.script(bias_new_gelu)
        else:
            self.func = bias_new_gelu

    def reset_parameters(self, prev_weight=None):
        p_range = (0, 1)
        if prev_weight is not None and len(prev_weight.shape) > 1:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(prev_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            p_range = (-bound, bound)
        torch.nn.init.uniform_(self.bias, *p_range)

    def forward(self, inp):
        return self.func(inp, self.bias)
