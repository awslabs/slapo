# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import Module
from torchvision.models.resnet import Bottleneck, ResNet
from transformers.utils import ModelOutput


@dataclass
class ResNetOutput(ModelOutput):
    output: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None


class ResNetWithLoss(Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x, labels: Optional[torch.Tensor] = None):
        output = self.model(x)
        loss = None
        if labels is not None:
            loss = self.loss_fn(output, labels.squeeze())
        return ResNetOutput(output=output, loss=loss)


def get_model(width_per_group, layers):
    model = ResNet(Bottleneck, layers, width_per_group=width_per_group)
    return ResNetWithLoss(model, torch.nn.CrossEntropyLoss())


def get_model_config(name) -> Module:
    """wideresnet-size."""
    if not name.startswith("wideresnet"):
        raise ValueError(
            f"Invalid model name: {name}. Expected to start with wideresnet"
        )
    _, size = name.split("-")

    if size == "250M":
        block_size = (128, [6, 8, 46, 6])
    elif size == "1.3B":
        block_size = (320, [6, 8, 46, 6])
    elif size == "2.4B":
        block_size = (448, [6, 8, 46, 6])
    elif size == "3B":
        block_size = (512, [6, 8, 46, 6])
    elif size == "4.6B":
        block_size = (640, [6, 8, 46, 6])
    else:
        raise ValueError(f"Unsupported model size: {size}")

    config = {
        "_name_or_path": name,
        "block_size": block_size,
    }
    return config
