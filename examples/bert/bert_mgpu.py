# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import inspect
import argparse
from transformers import BertLMHeadModel, AutoConfig
import numpy as np
import torch
import slapo
from bert_schedule import (
    replace_layernorm,
    replace_xformer_attention,
    replace_qkv,
    shard_params,
    shard_loss,
    broadcast_input,
    checkpoint,
)
from slapo.utils import report_memory
from slapo.env import setup


def train(rank, args):
    setup(rank, args.world_size)
    # https://huggingface.co/bert-large-uncased/blob/main/config.json
    bert_config = AutoConfig.from_pretrained("bert-large-uncased")
    bert = BertLMHeadModel(bert_config)
    optimizer = torch.optim.AdamW(bert.parameters(), lr=0.001)
    bert.half()

    input_names = list(bert.dummy_inputs.keys())
    input_names += ["attention_mask", "token_type_ids", "labels"]
    sig = inspect.signature(bert.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }

    sch = slapo.create_schedule(
        bert,
        optimizer,
        tracer="huggingface",
        concrete_args=concrete_args,
    )
    if not args.checkpoint:
        replace_qkv(sch, bert_config)
        if args.world_size > 1:
            shard_params(sch, bert_config, fused_qkv=True, prefix="bert")
            if "labels" in input_names:
                shard_loss(sch, bert_config)

    else:
        print("Use gradient checkpoint")
        checkpoint(sch, bert_config, prefix="bert")

    if args.world_size > 1:
        broadcast_input(sch)

    report_memory(rank)
    device = "cuda:{}".format(rank)
    model, optimizer = slapo.build(sch)
    print(model)
    model.half()
    model.cuda()
    report_memory(rank)

    bs = 8
    seq_length = 512
    bert_input_dict = {
        "input_ids": torch.zeros(
            bs, seq_length, dtype=torch.long, device=device
        ).random_(bert.config.vocab_size),
        "attention_mask": torch.ones(bs, seq_length, dtype=torch.long, device=device),
        "token_type_ids": torch.ones(bs, seq_length, dtype=torch.long, device=device),
        "labels": torch.zeros(bs, seq_length, dtype=torch.long, device=device).random_(
            bert.config.vocab_size
        ),
    }

    fw_time = []
    bw_time = []
    total_time = []
    for i in range(args.iter_nums):
        start_time = time.time()
        output = model(
            bert_input_dict["input_ids"].cuda(rank),
            bert_input_dict["attention_mask"].cuda(rank),
            bert_input_dict["token_type_ids"].cuda(rank),
            bert_input_dict["labels"].cuda(rank),
        )
        mid_time = time.time()
        output["logits"].mean().backward()
        final_time = time.time()
        optimizer.step()
        fw_time.append(mid_time - start_time)
        bw_time.append(final_time - mid_time)
        total_time.append(final_time - start_time)
        print(
            f"Finish step {i}, fw: {fw_time[-1]:.10f}s, bw: {bw_time[-1]:.10f}s, total: {total_time[-1]:.10f}s"
        )
    fw_avg = np.array(fw_time[1:-1]).mean()
    bw_avg = np.array(bw_time[1:-1]).mean()
    total_avg = np.array(total_time[1:-1]).mean()
    print(
        f"Average fw: {fw_avg*1000:.10f}ms, bw: {bw_avg*1000:.10f}ms, total: {total_avg*1000:.10f}ms"
    )


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    # This is passed in via cmd
    parser.add_argument("--world_size", type=int, default=n_gpus)
    parser.add_argument("--iter_nums", type=int, default=10)
    parser.add_argument(
        "--checkpoint", action="store_true", help="Enable gradient checkpointing"
    )
    args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    slapo.execute(train, args)
