# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model schedule registration."""
import inspect

# Mapping from model name to schedule method.
SCHEDULE_METHODS = {}


def get_schedule(model_name):
    """Get the schedule method."""
    if model_name not in SCHEDULE_METHODS:
        raise ValueError(f"Schedule for {model_name} does not exist.")
    return SCHEDULE_METHODS[model_name]


def register_schedule():
    """Register a schedule method."""

    def decorator(schedule_method):
        model_name = inspect.getfile(schedule_method).split("/")[-1].split(".")[0]
        if model_name not in SCHEDULE_METHODS:
            SCHEDULE_METHODS[model_name] = schedule_method
        else:
            raise ValueError(f"Schedule method for {model_name} already exists.")
        return schedule_method

    return decorator
