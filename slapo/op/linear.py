# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torch.nn.parameter import Parameter


class LinearWithSeparateBias(nn.Linear):
    """Implementation modified from `nn.Linear`
    This class is also inherited from `nn.Linear` to make sure it will be
    treated as the same class when using `isinstance(...)` in `init_weights`
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        group: dist.ProcessGroup,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LinearWithSeparateBias, self).__init__(
            in_features, out_features, bias=False, device=device, dtype=dtype
        )
        self.world_size = world_size
        self.group = group
        self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = F.linear(x, self.weight, None)
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.group)
        x = x + self.bias
        return x
