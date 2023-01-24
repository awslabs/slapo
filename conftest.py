# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
from torch import distributed as dist


def pytest_collection_modifyitems(items):
    """Execute the tests in alphabetical order based on their names.
    This is required for distributed unit tests. If different devices
    are running different tests, then the entire tests will stuck.
    """
    items.sort(key=lambda item: item.name)


@pytest.fixture(scope="session", autouse=True)
def init_dist(request):
    """Initialize the distributed group once in the entire test session."""
    torch.manual_seed(9999)
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
