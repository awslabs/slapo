# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model schedule."""

import os
from os.path import abspath, dirname, join, isfile
from inspect import getsourcefile
from importlib import import_module

from .api import apply_schedule

# register all model schedules in the current folder
path = dirname(abspath(getsourcefile(lambda: 0)))
files = [
    f
    for f in os.listdir(path)
    if isfile(join(path, f)) and f not in {"__init__.py", "api.py"}
]
for file in files:
    mod = import_module(f".{file.split('.')[0]}", package="slapo.model_schedule")
    # register the schedule method using the decorator
    if hasattr(mod, "_apply_schedule"):
        getattr(mod, "_apply_schedule")
