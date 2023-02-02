# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor


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
