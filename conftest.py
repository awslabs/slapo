# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_collection_modifyitems(items):
    """Execute the tests in alphabetical order based on their names.
    This is required for distributed unit tests. If different devices
    are running different tests, then the entire tests will stuck.
    """
    items.sort(key=lambda item: item.name)