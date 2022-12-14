# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import inspect
import argparse
from transformers import BertLMHeadModel, AutoConfig
import torch
import torch.nn as nn
import torch.distributed as dist

import slapo
from bert_schedule import (
    replace_layernorm,
    replace_xformer_attention,
    replace_qkv,
    shard_params,
    checkpoint,
    broadcast_input
)
from slapo.utils import report_memory
from slapo.op.cross_entropy import ParallelCrossEntropy

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology

_groups = []

def create_dist_groups(num_pp, num_mp):
    world_size = dist.get_world_size()
    num_dp = world_size // (num_pp * num_mp)
    topology = PipeModelDataParallelTopology(num_pp=num_pp, num_mp=num_mp, num_dp=num_dp)
    model_groups = topology.get_axis_comm_lists('model')

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
    num_pp = 4
    num_mp = 2
    torch.cuda.set_device(args.local_rank)
    deepspeed.init_distributed(dist_backend="nccl")

    # https://huggingface.co/bert-large-uncased/blob/main/config.json
    bert_config = AutoConfig.from_pretrained("bert-large-uncased")
    bert = BertLMHeadModel(bert_config)
    bert.half()

    input_names = list(bert.dummy_inputs.keys())
    input_names += ["attention_mask", "token_type_ids"]
    sig = inspect.signature(bert.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }

    rank = args.local_rank
    topology, group = create_dist_groups(num_pp, num_mp)

    sch = slapo.create_schedule(
        bert,
        None,
        group,
        tracer="huggingface",
        concrete_args=concrete_args,
    )

    if num_mp > 1:
        shard_params(sch, bert_config, prefix="bert")
        broadcast_input(sch)

    if args.checkpoint:
        print("Use gradient checkpoint")
        checkpoint(sch, bert_config, prefix="bert")

    print("Use pipeline parallelism")
    sch["bert.encoder.layer.5"].partition()
    sch["bert.encoder.layer.11"].partition()
    sch["bert.encoder.layer.17"].partition()

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
        lm_loss = loss_fct(
            shifted_prediction_scores, labels
        )
        lm_loss = lm_loss.contiguous().mean()
        return lm_loss

    model, optimizer = slapo.build(
        sch, topology=topology, target="deepspeed", config=ds_config_dict, loss_fn=loss_fn
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

    baseline = model.train_batch(data_iter=data_iter)


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
