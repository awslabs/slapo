# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

try:
    from functorch.compile import memory_efficient_fusion
except ImportError:
    memory_efficient_fusion = None

from .fused_bias import bias_dropout, bias_new_gelu, BiasGeLUFunction


class FusedQKV(nn.Module):
    def __init__(self, hidden_size, num_heads, world_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.fused_linear = nn.Linear(hidden_size, self.num_heads * self.head_size * 3)
        self.world_size = world_size

    def reshape_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_heads // self.world_size,
            self.head_size,
            3,
        )
        x = x.view(new_x_shape)
        return x.contiguous()

    def forward(self, hidden_states):
        qkv = self.fused_linear(hidden_states)
        reshaped_qkv = self.reshape_for_scores(qkv)
        q, k, v = torch.split(reshaped_qkv, 1, dim=-1)
        q = torch.squeeze(q, -1).contiguous()
        k = torch.squeeze(k, -1).contiguous()
        v = torch.squeeze(v, -1).contiguous()
        return [q, k, v]


class LinearWithSeparateBias(nn.Linear):
    """Implementation modified from `nn.Linear`
    Arguments are the same as the inputs of `nn.Linear`
    """

    # pylint: disable=arguments-renamed
    def forward(self, x: Tensor) -> Tensor:
        x = F.linear(x, self.weight, None)
        x = x + self.bias
        return x


class LinearWithAct(nn.Linear):
    """Implementation modified from `nn.Linear`"""

    # pylint: disable=arguments-renamed
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_fn="gelu",
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        if not bias:
            raise ValueError(
                "LinearWithAct requires bias. Please set bias=True or "
                "simply use nn.Linear"
            )

        if act_fn == "gelu":
            self.act = BiasGeLUFunction.apply
        elif act_fn == "gelu_new":
            if memory_efficient_fusion is not None:
                self.act = memory_efficient_fusion(bias_new_gelu)
            else:
                self.act = torch.jit.script(bias_new_gelu)
        else:
            raise NotImplementedError(f"Unsupported activation: {act_fn}")

    def forward(self, x: Tensor) -> Tensor:
        x = F.linear(x, self.weight, None)
        return self.act(x, self.bias)


class LinearWithDropout(nn.Linear):
    """Implementation modified from `nn.Linear`"""

    # pylint: disable=arguments-renamed
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        p=0.5,
        inplace=False,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        if not bias:
            raise ValueError(
                "LinearWithDropout requires bias. Please set bias=True or "
                "simply use nn.Linear"
            )
        self.p = p
        self.inplace = inplace

        # Somehow memory_efficient_fusion generates a different dropout mask
        # against the original dropout function even the random seed is the same.
        # if memory_efficient_fusion is not None:
        #     bias_dropout_func = partial(
        #         bias_dropout, p=p, training=self.training, inplace=inplace
        #     )
        #     self.func = memory_efficient_fusion(bias_dropout_func)
        self.dropout = bias_dropout

    def forward(self, x: Tensor) -> Tensor:
        x = F.linear(x, self.weight, None)
        return self.dropout(x, self.bias, self.p, self.training, self.inplace)
