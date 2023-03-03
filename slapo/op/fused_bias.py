# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fuse bias with the subsequent ops, such as activation function or dropout."""
# pylint: disable=abstract-method
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


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


def bias_new_gelu(inp: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return new_gelu(inp + bias)


def bias_dropout(
    x: torch.Tensor,
    bias: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> torch.Tensor:
    return F.dropout(x + bias, p=p, training=training, inplace=inplace)
