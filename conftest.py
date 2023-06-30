# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import random

import numpy as np
import pytest
import torch
from torch import distributed as dist


def pytest_collection_modifyitems(items):
    """Execute the tests in alphabetical order based on their names.
    This is required for distributed unit tests. If different devices
    are running different tests, then the entire tests will stuck.
    """
    items.sort(key=lambda item: item.name)
    new_items = []
    for item in items:
        # Skip DeepSpeed tests
        if item.parent.name not in ["test_ds_pipeline.py"]:
            new_items.append(item)
    items[:] = new_items
    return items


@pytest.fixture(scope="session")
def init_dist(request):
    """Initialize the distributed group once in the entire test session."""
    try:
        dist.init_process_group(backend="nccl")
    except Exception as err:
        print(f"Skip initializing dist group: {str(err)}")

    def destory_dist():
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    request.addfinalizer(destory_dist)


@pytest.fixture(scope="function", autouse=True)
def random_seed():
    """Set random seed to 1) make the tests deterministic, and 2) make every
    device generate the same weights for tensor parallelism tests.

    Note that if you run pytest with "randomly" plugin enabled, this fixture
    will have no effect. You can disable the plugin with
    pytest -p "no:randomly" ...
    """
    random.seed(9999)
    np.random.seed(9999)
    torch.manual_seed(9999)
