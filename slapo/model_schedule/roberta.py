# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace RoBERTa with Slapo schedule."""

from .registry import register_schedule, get_schedule


@register_schedule()
def _apply_schedule(
    model,
    **sch_config,
):
    schedule_fn = get_schedule("bert")
    return schedule_fn(model, **sch_config)
