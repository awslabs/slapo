# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from ..registry import register_model_dialect
from ...logger import get_logger, INFO

logger = get_logger("DS-Engine", INFO)


@register_model_dialect("deepspeed", "runtime_engine")
def init_ds_engine(model, **kwargs):
    """Initialize the DeepSpeed engine."""
    import deepspeed

    if "config" not in kwargs:
        raise ValueError("DeepSpeed config not provided.")

    # pylint: disable=unbalanced-tuple-unpacking
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=kwargs["config"],
        model_parameters=[p for p in model.parameters() if p.requires_grad],
    )
    return model, optimizer
