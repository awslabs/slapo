# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for the examples."""
import torch
import torch.distributed as dist
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology

from slapo.logger import get_logger

logger = get_logger("Utils")

_groups = []


def get_ds_config(
    batch_size,
    micro_batch_size_per_gpu,
    fp16=True,
    zero_stage=0,
    desc="",
    bf16=False,
    sequence_parallel=False,
):
    # https://github.com/microsoft/DeepSpeed/blob/ff42743/tests/unit/model_parallelism/test_configurable_parallel_pp.py#L20
    logger.info(f"fp16={fp16}, bf16={bf16}")
    config_dict = {
        "help": desc,
        "steps_per_print": 10,
        "optimizer": {"type": "AdamW", "params": {"lr": 0.0001}},
        "fp16": {"enabled": fp16, "initial_scale_power": 12},
        "bf16": {"enabled": bf16},
        "gradient_clipping": 1.0,
        "train_batch_size": batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size_per_gpu,
        "pipeline": {
            "sequence_parallel": sequence_parallel,
        },
        "wall_clock_breakdown": False,
    }

    if zero_stage > 0:
        zero_config_dict = {
            "zero_optimization": {
                "stage": zero_stage,
                "overlap_comm": True,
                "reduce_scatter": True,
                "contiguous_gradients": False,
                "prefetch_bucket_size": 5e8,
            },
            "zero_allow_untested_optimizer": True,
        }
        config_dict.update(zero_config_dict)

    return config_dict


def create_dist_group_for_pipeline(num_pp, num_mp):
    world_size = dist.get_world_size()
    num_dp = world_size // (num_pp * num_mp)
    topology = PipeModelDataParallelTopology(
        num_pp=num_pp, num_mp=num_mp, num_dp=num_dp
    )
    model_groups = topology.get_axis_comm_lists("model")

    global_rank = dist.get_rank()
    group = None

    for g in model_groups:
        proc_group = dist.new_group(ranks=g)
        _groups.append(proc_group)
        if global_rank in g:
            group = proc_group

    return topology, group


def generate_pipeline_cuts(num_layers, num_pp, is_encoder_decoder=False):
    """Evenly partition layers for pipelining. If num_layers is not divisible by
    num_pp, the last num_layers % num_pp partitions will have one more layer.
    If is_encoder_decoder is True, the pipeline stages are evenly
    assigned to encoder and decoder.
    """

    def even_partition(num_layers, num_pp):
        remainder = num_layers % num_pp
        size_list = [num_layers // num_pp] * num_pp

        curr = size_list[0] - 1
        ret = [curr]
        for idx, size in enumerate(size_list):
            size = size + 1 if num_pp - idx - 1 <= remainder else size
            curr += size
            ret.append(curr)
        return ret[: num_pp - 1]

    if is_encoder_decoder:
        if num_pp % 2 != 0:
            raise ValueError("num_pp must be even")
        return [even_partition(num_layers, num_pp // 2) for _ in range(2)]
    return even_partition(num_layers, num_pp)


def train_with_torch(
    model,
    dataloader,
    optimizer,
    loss_fn=None,
    preproc=None,
    postproc=None,
    steps=40,
):
    """The training loop for PyTorch runtime. Note that this simple training loop
    assumes no data parallelism and gradient accumulation.
    """

    for step, batch in enumerate(dataloader):
        inputs, labels = preproc(step, batch) if preproc is not None else batch
        if loss_fn is None:
            loss = model(*inputs, labels=labels).loss
        else:
            loss = loss_fn(model(*inputs).logits, labels)
        loss.backward()
        optimizer.step()
        loss = postproc(step, loss) if postproc is not None else loss

        if step % 10 == 0:
            logger.info(f"step {step} loss: {loss.item()}", ranks=0)

        loss = None
        if step >= steps:
            break


def train_with_deepspeed_engine(
    model,
    dataloader,
    loss_fn=None,
    preproc=None,
    postproc=None,
    steps=40,
):
    """The training loop for DeepSpeedEngine (without pipeline)."""

    device = torch.cuda.current_device()
    for micro_batch_step, batch in enumerate(dataloader):
        inputs, labels = (
            preproc(micro_batch_step, batch) if preproc is not None else batch
        )
        inputs = [inp.to(device) if inp is not None else None for inp in inputs]
        labels = labels.to(device)
        if loss_fn is None:
            loss = model(*inputs, labels=labels).loss
        else:
            loss = loss_fn(model(*inputs).logits, labels)
        model.backward(loss)
        model.step()
        loss = postproc(micro_batch_step, loss) if postproc is not None else loss

        if model.global_steps % 10 == 0 and model.is_gradient_accumulation_boundary():
            logger.info(f"step {model.global_steps} loss: {loss.item()}", ranks=0)

        loss = None
        if model.global_steps >= steps:
            break
