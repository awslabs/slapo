# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Bert with model schedule."""
import inspect

import torch.distributed as dist
import slapo
from bert_schedule import (
    broadcast_input,
    checkpoint,
    replace_and_shard_attention,
    shard_mlp,
    shard_word_embedding,
)


def schedule_bert(
    model,
    config,
    prefix="",
    disable_flash_attn=False,
    fp16=True,
    ckpt_ratio=0.0,
    group=None,
    bcast_input=False,
    pipeline_cuts=None,
    delay_init=True,
):
    def print_rank_0(message):
        """If distributed is initialized, print only on rank 0."""
        if dist.is_initialized():
            if dist.get_rank() == 0:
                print(message, flush=True)
        else:
            print(message, flush=True)

    print_rank_0("Scheduling Bert")

    if fp16:
        print_rank_0("Change model dtype to fp16")
        model.half()

    sch = slapo.create_schedule(model, group=group)

    # Replace self attention with flash attention, and shard QKV/output
    # if MP group > 1.
    if not disable_flash_attn:
        cnt = replace_and_shard_attention(sch[prefix], config, delay_init=delay_init)
        print_rank_0(f"Replace {cnt} attention patterns")
    else:
        raise NotImplementedError("Not implemented yet")

    # Shard other parameters if MP group > 1.
    if sch.world_size > 1:
        shard_mlp(sch[prefix], config)
        shard_word_embedding(sch[prefix], config.vocab_size)

        # Broadcast input to all devices within the MP group.
        # This is not required when running on Megatron.
        if bcast_input:
            broadcast_input(sch)

    # Insert activation checkpoints.
    if ckpt_ratio > 0.0:
        n_ckpt = checkpoint(sch[prefix], config, ckpt_ratio=ckpt_ratio)
        print_rank_0(f"Checkpointing {n_ckpt} layers")

    # Cut pipeline stages.
    if pipeline_cuts:
        input_names = [
            "input_ids",
            "attention_mask",
            "token_type_ids",
        ]
        sig = inspect.signature(model.forward)
        concrete_args = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.name not in input_names
        }
        _prefix = f"{prefix}." if prefix else ""
        sch.trace_for_pipeline(
            f"{_prefix}encoder", tracer="huggingface", concrete_args=concrete_args
        )
        for cut in pipeline_cuts:
            sch[f"{_prefix}encoder.layer.{cut}"].cut_pipeline_stage()

    return sch
