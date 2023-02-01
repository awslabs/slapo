# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dropout module."""

from torch import nn

from ..random import get_cuda_rng_tracker


class DropoutWithTensorParallel(nn.Dropout):
    """The dropout that supposed to be used in parallel region.
    In parallel region means the original input tensor is partitioned
    due to tensor parallelism or sequence parallelism. In this case,
    we need to make sure the dropout on each device in the same
    tensor parallel group has DIFFERENT random seed; otherwise each
    partitioned tensor will have the same dropout mask, which may hurt
    the convergence.
    """

    def forward(self, input):
        with get_cuda_rng_tracker().fork():
            return super().forward(input)
