# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model schedule."""

from .api import apply_schedule
from .albert import (
    shard_parameters as albert_shard_parameters,
    generate_pipeline_schedule as albert_generate_pipeline_schedule,
    checkpoint as albert_checkpoint,
    broadcast_input as albert_broadcast_input,
)
from .bert import (
    shard_parameters as bert_shard_parameters,
    generate_pipeline_schedule as bert_generate_pipeline_schedule,
    checkpoint as bert_checkpoint,
    broadcast_input as bert_broadcast_input,
)
from .gpt_neo import (
    shard_parameters as gpt_neo_shard_parameters,
    generate_pipeline_schedule as gpt_neo_generate_pipeline_schedule,
    checkpoint as gpt_neo_checkpoint,
    broadcast_input as gpt_neo_broadcast_input,
)
from .gpt2 import (
    shard_parameters as gpt2_shard_parameters,
    generate_pipeline_schedule as gpt2_generate_pipeline_schedule,
    checkpoint as gpt2_checkpoint,
    broadcast_input as gpt2_broadcast_input,
)
from .opt import (
    shard_parameters as opt_shard_parameters,
    generate_pipeline_schedule as opt_generate_pipeline_schedule,
    checkpoint as opt_checkpoint,
    broadcast_input as opt_broadcast_input,
)
from .roberta import (
    shard_parameters as roberta_shard_parameters,
    generate_pipeline_schedule as roberta_generate_pipeline_schedule,
    checkpoint as roberta_checkpoint,
    broadcast_input as roberta_broadcast_input,
)
from .t5 import (
    shard_parameters as t5_shard_parameters,
    generate_pipeline_schedule as t5_generate_pipeline_schedule,
    checkpoint as t5_checkpoint,
    broadcast_input as t5_broadcast_input,
)
# from .wideresnet import (
#     shard_parameters as wideresnet_shard_parameters,
#     generate_pipeline_schedule as wideresnet_generate_pipeline_schedule,
#     checkpoint as wideresnet_checkpoint,
#     broadcast_input as wideresnet_broadcast_input,
# )
