# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
from deepspeed.utils import RepeatingLoader
from transformers import GPTNeoForCausalLM, AutoConfig

import slapo
from slapo.op.cross_entropy import ParallelCrossEntropy
from slapo.utils import report_memory

from gpt_model import schedule_gpt

_groups = []

SINGLE_DEVICE_FOR_DEBUG = False

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def even_partition(num_layers, num_pp):
    """Evenly partition layers for pipelining. If num_layers is not divisible by
    num_pp, the last num_layers % num_pp partitions will have one more layer.
    """
    remainder = num_layers % num_pp
    size_list = [num_layers // num_pp] * num_pp

    curr = size_list[0] - 1
    ret = [curr]
    for idx, size in enumerate(size_list):
        size = size + 1 if num_pp - idx - 1 <= remainder else size
        curr += size
        ret.append(curr)
    return ret[: num_pp - 1]


def create_dist_groups(num_pp, num_mp):
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


def train(args):
    print_rank_0("Use deepspeed to initialize")
    num_pp, num_mp = 1, 1
    rank = args.local_rank
    torch.cuda.set_device(rank)

    if not SINGLE_DEVICE_FOR_DEBUG:
        num_pp, num_mp = 4, 2
        deepspeed.init_distributed(dist_backend="nccl")

    # https://huggingface.co/EleutherAI/gpt-neo-2.7B/blob/main/config.json
    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B")
    # FIXME: This model has vocab size 50257 that cannot be sharded by 2,
    # so we pad it to 50258 in this example. In practice, the tokenizer
    # should be used to pad the vocab size to a multiple of 2.
    config.vocab_size = 50258
    config.use_cache = False
    model = GPTNeoForCausalLM(config)

    topology, group = None, None
    if not SINGLE_DEVICE_FOR_DEBUG:
        topology, group = create_dist_groups(num_pp, num_mp)

    # Evenly partition layers for pipelining.
    if not SINGLE_DEVICE_FOR_DEBUG:
        pipeline_cuts = even_partition(config.num_layers, num_pp)
    else:
        pipeline_cuts = even_partition(config.num_layers, 4)
    print_rank_0(f"Pipeline cuts: {pipeline_cuts}")

    sch = schedule_gpt(
        model,
        config,
        prefix="transformer",
        ckpt_ratio=1 if args.checkpoint else 0,
        bcast_input=True,
        group=group,
        pipeline_cuts=pipeline_cuts,
    )
    if SINGLE_DEVICE_FOR_DEBUG:
        slapo.build(sch)
        assert False

    report_memory(rank)
    device = "cuda:{}".format(rank)
    # https://github.com/microsoft/DeepSpeed/blob/ff427438651943ee473ab37547337f5f3d8c2279/tests/unit/model_parallelism/test_configurable_parallel_pp.py#L20
    ds_config_dict = {
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {"type": "AdamW", "params": {"lr": 0.0001}},
        "fp16": {"enabled": True},
    }

    loss_fct = ParallelCrossEntropy(group=group)

    def loss_fn(outputs, labels):
        prediction_scores = outputs
        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        lm_loss = loss_fct(shifted_prediction_scores, labels)
        lm_loss = lm_loss.contiguous().mean()
        return lm_loss

    model, _ = slapo.build(
        sch,
        topology=topology,
        target="deepspeed",
        config=ds_config_dict,
        loss_fn=loss_fn,
    )
    report_memory(rank)

    bs = 8
    seq_length = 512
    input_dict = {
        "input_ids": torch.zeros(
            bs, seq_length, dtype=torch.long, device=device
        ).random_(model.config.vocab_size),
        "attention_mask": torch.ones(
            bs, seq_length, dtype=torch.float16, device=device, requires_grad=True
        ),
        "position_ids": torch.ones(bs, seq_length, dtype=torch.long, device=device),
        "labels": torch.zeros(bs, seq_length, dtype=torch.long, device=device).random_(
            model.config.vocab_size
        ),
    }

    loader = RepeatingLoader(
        [
            # First batch
            # (inputs, labels)
            (
                (
                    input_dict["input_ids"],
                    input_dict["attention_mask"],
                    input_dict["position_ids"],
                ),
                input_dict["labels"],
            ),
            # Rest of the batches
            # ...
        ]
    )
    data_iter = iter(loader)
    model.train_batch(data_iter=data_iter)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    # This is passed in via cmd
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=n_gpus)
    parser.add_argument("--iter_nums", type=int, default=10)
    parser.add_argument(
        "--checkpoint", action="store_true", help="Enable gradient checkpointing"
    )
    args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    train(args)
