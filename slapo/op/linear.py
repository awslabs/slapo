# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from torch import nn
import torch.nn.functional as F
from torch import Tensor

class LinearWithSeparateBias(nn.Linear):
    """Implementation modified from `nn.Linear`
    Arguments are the same as the inputs of `nn.Linear`
    """

    # pylint: disable=arguments-renamed
    def forward(self, x: Tensor) -> Tensor:
        x = F.linear(x, self.weight, None)
        x = x + self.bias
        return x
