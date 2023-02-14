# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from ..registry import register_framework_dialect
from ...logger import get_logger, INFO

logger = get_logger("DS-Engine", INFO)


@register_framework_dialect("deepspeed", "runtime_engine")
def init_ds_engine(model, **kwargs):
    """Initialize the DeepSpeed engine."""
    import deepspeed
    from deepspeed.pipe import PipelineModule
    from deepspeed.runtime.pipe.topology import (
        PipeModelDataParallelTopology,
        PipelineParallelGrid,
    )

    if "config" not in kwargs:
        raise ValueError("DeepSpeed config not provided.")

    mpu = kwargs.get("topology", None)
    if isinstance(model, PipelineModule):
        # If the model is already a PipelineModule, the device grid (i.e., mesh)
        # is already configured, so we pass mpu=None to deepspeed.initialize to
        # avoid re-configuration.
        mpu = None
    elif mpu is not None and isinstance(mpu, PipeModelDataParallelTopology):
        mpu = PipelineParallelGrid(topology=mpu)

    # pylint: disable=unbalanced-tuple-unpacking
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=kwargs["config"],
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        mpu=mpu,
    )
    return model, optimizer
