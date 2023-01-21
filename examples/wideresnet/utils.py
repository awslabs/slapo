# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities."""
import torch

from deepspeed.utils import RepeatingLoader


def count_parameters(model):
    try:
        return sum(p.ds_numel for p in model.parameters())
    except:
        return sum(p.numel() for p in model.parameters())


def get_data_loader(micro_batch_size, device, dtype=torch.float):
    loader = RepeatingLoader(
        [
            # First batch
            # (inputs, labels)
            (
                (
                    torch.rand(
                        (micro_batch_size, 3, 224, 224),
                        dtype=dtype,
                        device=device,
                    ),
                ),
                torch.randint(0, 1000, (micro_batch_size,), device=device),
            ),
            # Rest of the batches
            # ...
        ]
    )
    return loader
