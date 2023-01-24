# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Framework dialects."""

from .megatron.utils import MegatronLogParser
from .deepspeed.utils import DeepSpeedLogParser
from .deepspeed.pipeline import DeepSpeedPipeStageWrapper
from .deepspeed.engine import init_ds_engine
from .registry import get_all_dialects, get_dialect_cls
