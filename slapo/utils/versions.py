# Copyright 2022 The HuggingFace Team. All rights reserved.
# https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/versions.py
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import operator as op
import sys
from typing import Union

from packaging.version import Version, parse

STR_OPERATION_TO_FUNC = {
    ">": op.gt,
    ">=": op.ge,
    "==": op.eq,
    "!=": op.ne,
    "<=": op.le,
    "<": op.lt,
}


if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

torch_version = parse(importlib_metadata.version("torch"))


def compare_versions(
    library_or_version: Union[str, Version], operation: str, requirement_version: str
):
    """
    Compares a library version to some requirement using a given operation.
    Args:
        library_or_version (`str` or `packaging.version.Version`):
            A library name or a version to check.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`.
        requirement_version (`str`):
            The version to compare the library version against
    """
    if operation not in STR_OPERATION_TO_FUNC:
        raise ValueError(
            f"`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}"
        )
    operation = STR_OPERATION_TO_FUNC[operation]
    if isinstance(library_or_version, str):
        library_or_version = parse(importlib_metadata.version(library_or_version))
    return operation(library_or_version, parse(requirement_version))


def is_torch_version(operation: str, version: str):
    """
    Compares the current PyTorch version to a given reference with an operation.
    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A string version of PyTorch
    """
    return compare_versions(torch_version, operation, version)
