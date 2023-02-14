# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model schedule."""

from .api import apply_schedule

# register all model schedules in the current folder
import os
from os.path import abspath, dirname, join, isfile
from inspect import getsourcefile
from importlib import import_module

path = dirname(abspath(getsourcefile(lambda: 0)))
files = [f for f in os.listdir(path) if isfile(join(path, f)) and f != "__init__.py"]
print(path, files)
for file in files:
    mod = import_module(
        ".{}".format(file.split(".")[0]), package="slapo.model_schedule"
    )
    # register the schedule method using the decorator
    try:
        getattr(mod, "apply_schedule")
    except Exception:
        pass
