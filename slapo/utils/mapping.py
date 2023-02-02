# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Dict

from torch import nn
import torch.nn.functional as F

# pylint: disable=consider-using-alias
MAPPING_FROM_FUNCTIONAL_TO_MODULE: Dict[Callable, Callable] = {
    F.embedding: nn.Embedding,
    F.layer_norm: nn.LayerNorm,
    F.dropout: nn.Dropout,
    F.group_norm: nn.GroupNorm,
    F.linear: nn.Linear,
    F.relu: nn.ReLU,
    F.gelu: nn.GELU,
    F.conv1d: nn.Conv1d,
    F.conv2d: nn.Conv2d,
    F.mse_loss: nn.MSELoss,
    F.cross_entropy: nn.CrossEntropyLoss,
}
