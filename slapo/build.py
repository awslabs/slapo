# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Functions to build the model based on the schedule."""
from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Optional, Union

import torch
from torch import distributed as dist
from torch import nn

from .utils.common import transfor_param_tags
from .framework_dialect import get_dialect_cls
from .logger import get_logger
from .pipeline import (
    analyze_tie_weights,
    build_pipeline_model,
    generate_pipeline_partition,
)
from .schedule import Schedule

logger = get_logger()


def consolidate_model(
    sch: Schedule,
    target: str,
    param_init_fn: Optional[Callable[[nn.Module], None]] = None,
    **kwargs,
):
    """Consolidate the model weights.
    FIXME: When pipeline is enabled, this function only supports DeepSpeed
    runtime because it relies on DeepSpeed topology. We should use dialects
    in this function to make it general applicable.
    """
    topology = kwargs.get("topology", None)
    if dist.is_initialized() and dist.get_world_size() > sch.world_size:
        if topology is None:
            raise ValueError(
                "topology must be given when there are multiple "
                "tensor paralel groups or pipeline parallelism is used"
            )
        if target != "deepspeed":
            raise ValueError(
                "Only deepspeed runtime is supported for now when there are multiple "
                "tensor paralel groups or pipeline parallelism is used"
            )

    cnt_meta, cnt_materialized = 0, 0
    # Since some parameters are attached to non-leaf modules, we need to
    # fix them layer-by-layer. See the following example:
    # https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/bert/modeling_bert.py#L693
    for _, param in sch.mod.named_parameters(recurse=True):
        if param.device == torch.device("meta"):
            cnt_meta += 1
        else:
            cnt_materialized += 1

    stage_groups = None
    # local rank means the rank in a node
    local_rank = torch.cuda.current_device()
    global_rank = None
    global_ranks = [None]
    if cnt_meta != 0 or cnt_materialized != 0:
        if dist.is_initialized():
            # Tackle with pipeline modules.
            # Even the model does not use meta device, we still need to broadcast
            # the weights to ensure consistency
            global_rank = dist.get_rank()
            if topology is not None:
                # 1st DP: devices in the same bracket are in the same TP group
                #         vertical lines separate different PP stages
                # [0, 1] |
                #        | [4, 5]
                # 2nd DP
                # [2, 3] |
                #        | [6, 7]
                # >>> topo = PipeModelDataParallelTopology(2, 2, 2)
                # >>> topo.get_axis_comm_lists("model")
                # [[0, 1], [2, 3], [4, 5], [6, 7]]
                # >>> topo.get_axis_comm_lists("pipe")
                # [[0, 4], [1, 5], [2, 6], [3, 7]]
                # >>> topo.get_axis_comm_lists("data")
                # [[0, 2], [1, 3], [4, 6], [5, 7]]
                # >>> topo.filter_match(pipe=0)
                # [0, 1, 2, 3]
                # create dist group for broadcasting
                num_pp = topology.get_dim("pipe")
                # each group contains the devices on the same stage
                stage_groups = []
                for i in range(num_pp):
                    stage_groups.append(
                        dist.new_group(ranks=topology.filter_match(pipe=i))
                    )
            else:
                stage_groups = [dist.new_group()]

            global_ranks = list(range(dist.get_world_size()))
    else:
        return sch

    def _init_module(sch: Schedule):
        if param_init_fn:
            param_init_fn(sch.mod)
        elif hasattr(sch.mod, "_init_weights"):
            # `_init_weights` is a HF specific API, see
            # https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/bert/modeling_bert.py#L748
            sch.mod._init_weights(sch.mod)
        elif hasattr(sch.mod, "reset_parameters"):
            sch.mod.reset_parameters()
        else:
            raise RuntimeError(
                f"Module {sch.name} should have `reset_parameters` or "
                "`_init_weights` method or param_init_fn={param_init_fn} needs "
                "to be provided in order to support delay initialization"
            )

    def _consolidate_and_broadcast(sch: Schedule):
        if isinstance(sch.mod, torch.jit.ScriptModule):
            # Scripted module requires the parameters to be initialized in advance,
            # so no need to consolidate
            return 0, 0

        if hasattr(sch, "partition_idx") and topology is not None:
            curr_part_idx = sch.partition_idx
            # topology stores the global ranks
            curr_stage_devices = topology.filter_match(pipe=curr_part_idx)
        else:
            curr_part_idx = 0
            curr_stage_devices = global_ranks

        if global_rank not in curr_stage_devices:
            # do nothing if the target module is NOT on this device group
            return 0, 0

        # Register parameters with the original shape (if sharded) for initialization.
        num_params = 0
        new_param_shapes = {}
        for param_name, param in sch.mod.named_parameters(recurse=False):
            num_params += 1
            new_param_shapes[param_name] = param.shape
            orig_shape = (
                param.orig_shape if hasattr(param, "orig_shape") else param.shape
            )
            new_param = nn.Parameter(
                torch.empty(orig_shape, dtype=param.dtype, device=local_rank)
            )
            sch.mod.register_parameter(
                param_name,
                new_param,
            )
            transfor_param_tags(sch, param, new_param)

        # Use original shape to initialize parameters.
        if global_rank == curr_stage_devices[0] and num_params > 0:
            # only the first device in the PP group needs to initialize the weights
            _init_module(sch)

        # Broadcast complete params from rank 0 to make sure all the TP+DP ranks
        # take the same params.
        if dist.is_initialized():
            curr_stage_group = stage_groups[curr_part_idx]
            for _, param in sch.mod.named_parameters(recurse=False):
                dist.broadcast(param, src=curr_stage_devices[0], group=curr_stage_group)

        # Only keep the partition for this device for sharded params.
        tp_rank = sch.rank
        cnt_shard = 0
        for param_name, param in sch.mod.named_parameters(recurse=False):
            is_found = False
            for idx, new_size in enumerate(new_param_shapes[param_name]):
                if new_size != param.shape[idx]:
                    assert not is_found, "Cannot have two sharded dimensions!"
                    sharded_size = new_size
                    axis = idx
                    is_found = True
            if is_found:
                cnt_shard += 1
                sharded_param = param.detach().split(sharded_size, dim=axis)[tp_rank]
                sharded_param = sharded_param.contiguous()
                new_param = nn.Parameter(sharded_param)
                sch.mod.register_parameter(param_name, new_param)
                transfor_param_tags(sch, param, new_param)

        for subsch in sch.child.values():
            ret = _consolidate_and_broadcast(subsch)
            num_params += ret[0]
            cnt_shard += ret[1]

        return num_params, cnt_shard

    if cnt_meta != 0 or cnt_materialized != 0:
        num_params, cnt_shard = _consolidate_and_broadcast(sch)

    logger.info(
        "Finished consolidating %d parameter tensors with %d being sharded",
        num_params,
        cnt_shard,
    )

    gc.collect()
    torch.cuda.empty_cache()

    return sch


def init_target_engine(sch, target, **kwargs):
    """Initialize the runtime engine for a specific target framework."""
    init_engine_fn = get_dialect_cls("runtime_engine", target, allow_none=True)
    return init_engine_fn(
        sch,
        **kwargs,
    )


def build(
    sch: Schedule,
    target=None,
    init_weights: Optional[Union[bool, Callable]] = True,
    **kwargs,
):
    if sch.metadata.primitives["cut_pipeline_stage"]:
        # pipeline stages will be wrapped into PipeStageWrapper
        sch = generate_pipeline_partition(sch)
        # Re-analyzie tie weights before consolidation.
        sch.metadata.tie_weights = analyze_tie_weights(
            sch.mod, is_pipeline_partitioned=True
        )

    # delay initialization
    if init_weights:
        init_weight_fn = init_weights if isinstance(init_weights, Callable) else None
        sch = consolidate_model(sch, target, init_weight_fn, **kwargs)

    if sch.metadata.primitives["cut_pipeline_stage"] and target is not None:
        # Generate pipeline modules for a particular target.
        model = build_pipeline_model(
            sch,
            target,
            **kwargs,
        )
    else:
        model = sch.mod

    return init_target_engine(model, target, **kwargs)
