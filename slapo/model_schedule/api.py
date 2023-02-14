# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .registry import get_schedule


def apply_schedule(
    model,
    schedule_key,
    **sch_config,
):
    """Apply schedule to a model."""
    schedule_method = get_schedule(schedule_key)
    if schedule_method is None:
        raise ValueError(f"Schedule method for {schedule_key} does not exist.")
    return schedule_method(model, **sch_config)
