# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
from deepspeed.utils import RepeatingLoader
from transformers import BertLMHeadModel, AutoConfig

import slapo
from slapo.op.cross_entropy import ParallelCrossEntropy
from slapo.utils import report_memory

from bert_model import schedule_bert

_groups = []

SINGLE_DEVICE_FOR_DEBUG = False


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
    print("Use deepspeed to initialize")
    num_pp, num_mp = 1, 1
    rank = args.local_rank
    torch.cuda.set_device(rank)

    if not SINGLE_DEVICE_FOR_DEBUG:
        num_pp, num_mp = 4, 2
        deepspeed.init_distributed(dist_backend="nccl")

    # https://huggingface.co/bert-large-uncased/blob/main/config.json
    bert_config = AutoConfig.from_pretrained("bert-large-uncased")
    bert = BertLMHeadModel(bert_config)

    topology, group = None, None
    if not SINGLE_DEVICE_FOR_DEBUG:
        topology, group = create_dist_groups(num_pp, num_mp)

    sch = schedule_bert(
        bert,
        bert_config,
        prefix="bert",
        ckpt_ratio=1 if args.checkpoint else 0,
        bcast_input=True,
        group=group,
        pipeline_cuts=[5, 11, 17],
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
    bert_input_dict = {
        "input_ids": torch.zeros(
            bs, seq_length, dtype=torch.long, device=device
        ).random_(bert.config.vocab_size),
        "attention_mask": torch.ones(
            bs, seq_length, dtype=torch.float16, device=device, requires_grad=True
        ),
        "token_type_ids": torch.ones(bs, seq_length, dtype=torch.long, device=device),
        "labels": torch.zeros(bs, seq_length, dtype=torch.long, device=device).random_(
            bert.config.vocab_size
        ),
    }

    loader = RepeatingLoader(
        [
            # First batch
            # (inputs, labels)
            (
                (
                    bert_input_dict["input_ids"],
                    bert_input_dict["attention_mask"],
                    bert_input_dict["token_type_ids"],
                ),
                bert_input_dict["labels"],
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
