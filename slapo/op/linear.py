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
        sync_fn,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            in_features, out_features, bias=False, device=device, dtype=dtype
        )
        self.sync_fn = sync_fn
        self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=arguments-renamed
        x = F.linear(x, self.weight, None)
        # pylint: disable=comparison-with-callable
        if self.sync_fn.func == dist.all_reduce:
            self.sync_fn(x)
        else:
            x = self.sync_fn(x)
        x = x + self.bias
        return x
