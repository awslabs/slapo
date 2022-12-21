# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Framework dialects."""

from .deepspeed.pipeline import DeepSpeedPipeStageWrapper
from .registry import get_all_dialects, get_dialect_cls
