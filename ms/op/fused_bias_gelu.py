# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# https://github.com/NVIDIA/Megatron-LM/blob/master/megatron/model/fused_bias_gelu.py
import math
import torch
from functorch.compile import memory_efficient_fusion


###### BIAS GELU FUSION/ NO AUTOGRAD ################
# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456
# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))


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
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
        1 + tanh_out
    )
    return ff * g


class BiasGeLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_gelu(bias, input)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_gelu_back(grad_output, bias, input)
        return tmp, tmp


def bias_new_gelu(input, bias):
    """Bias + new GELU activation function copied from HuggingFace transformers."""
    input = input + bias
    return (
        0.5
        * input
        * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    )


class FusedBiasAct(torch.nn.Module):
    def __init__(self, size, act, device=None, dtype=None, prev_weight=None, fused=True, aot=True):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.bias = torch.nn.Parameter(torch.empty(size, **factory_kwargs))
        self.fused = fused
        self.reset_parameters(prev_weight)

        if act == "gelu":
            self.func = BiasGeLUFunction.apply
        elif act == "gelu_new":
            if self.fused:
                if aot:
                    self.func = memory_efficient_fusion(bias_new_gelu)
                else:
                    self.func = torch.jit.script(bias_new_gelu)
            else:
                self.func = bias_new_gelu
        else:
            raise ValueError(f"Unsupported activation {act}")

    @staticmethod
    def check_act_type(act):
        return act in ["gelu", "gelu_new"]

    def reset_parameters(self, prev_weight=None):
        range = (0, 1)
        if prev_weight is not None and len(prev_weight.shape) > 1:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(prev_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            range = (-bound, bound)
        torch.nn.init.uniform_(self.bias, *range)

    def forward(self, input):
        return self.func(input, self.bias)
