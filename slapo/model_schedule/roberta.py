# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace RoBERTa with Slapo schedule."""

from .registry import register_schedule_method, get_schedule_method


@register_schedule_method()
def apply_schedule(
    model,
    **sch_config,
):
    schedule_fn = get_schedule_method("bert")
    return schedule_fn(model, **sch_config)
