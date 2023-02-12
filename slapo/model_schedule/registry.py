# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model schedule registration."""

# Mapping from model name to schedule method.
SCHEDULE_METHODS = {}


def get_schedule_method(model_name, schedule_name):
    """Get the schedule method."""
    if model_name not in SCHEDULE_METHODS:
        return None
    elif schedule_name not in SCHEDULE_METHODS[model_name]:
        return None
    else:
        return SCHEDULE_METHODS[model_name][schedule_name]


def register_schedule_method(model_name):
    """Register a schedule method."""

    def decorator(schedule_method):
        if model_name not in SCHEDULE_METHODS:
            SCHEDULE_METHODS[model_name] = {}
        if schedule_method.__name__ in SCHEDULE_METHODS[model_name]:
            raise ValueError(
                f"Schedule method {schedule_method.__name__} already registered for {model_name}"
            )
        SCHEDULE_METHODS[model_name][schedule_method.__name__] = schedule_method
        return schedule_method

    return decorator
