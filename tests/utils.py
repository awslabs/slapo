# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Testing utilities."""

import random
import numpy as np

import torch


def reset_random_seeds(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
