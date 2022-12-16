# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace GPT-Neo with model schedule."""
import inspect

import torch.distributed as dist
import slapo
from gpt_schedule import (
    broadcast_input,
    checkpoint,
    remove_cast,
    replace_and_shard_mlp,
    replace_and_shard_attention,
    replace_qkv,
    shard_qkv,
    shard_word_embedding,
    trace_attention,
)


def schedule_gpt(
    model,
    config,
    prefix="",
    disable_flash_attn=False,
    fp16=True,
    ckpt_ratio=0.0,
    group=None,
    bcast_input=False,
    pipeline_cuts=None,
):
    def print_rank_0(message):
        """If distributed is initialized, print only on rank 0."""
        if dist.is_initialized():
            if dist.get_rank() == 0:
                print(message, flush=True)
        else:
            print(message, flush=True)

    print_rank_0(f"Scheduling GPT")
    assert "GPT2" not in config.architectures[0], "GPT-2 schedule is not working"

    if fp16:
        print_rank_0("Change model dtype to fp16")
        model.half()

    sch = slapo.create_schedule(model, group=group)

    # Replace self attention with flash attention, and shard QKV/output
    # if MP group > 1.
    attn_path, out_proj_name = "h.N.attn.attention", "out_proj"
    if not disable_flash_attn:
        cnt = replace_and_shard_attention(sch[prefix], config)
        print_rank_0(f"Replace {cnt} attention patterns")
    else:
        # FIXME: This path is not working because our tracer cannot trace
        # GPTNeoAttention without tracing the whole model.
        cnt = trace_attention(sch, config, attn_path)
        print_rank_0(f"Traced {cnt} attention layesr")
        assert cnt > 0
        cnt = remove_cast(sch, config, attn_path)
        print_rank_0(f"Remove {cnt} .to(torch.float32) ops")
        cnt = replace_qkv(sch, config, attn_path)
        print_rank_0(f"Replace {cnt} QKV patterns")
        if sch.world_size > 1:
            shard_qkv(sch, config, attn_path, out_proj_name=out_proj_name)

    # Shard other parameters if MP group > 1.
    if sch.world_size > 1:
        replace_and_shard_mlp(sch[prefix], config)
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
        input_names = ["input_ids", "attention_mask", "position_ids"]
        sig = inspect.signature(model.forward)
        concrete_args = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.name not in input_names
        }
        sch.trace_for_pipeline(
            f"{prefix}", tracer="huggingface", concrete_args=concrete_args
        )
        _prefix = f"{prefix}." if prefix else ""
        for cut in pipeline_cuts:
            sch[f"{_prefix}h.{cut}"].cut_pipeline_stage()

    return sch
