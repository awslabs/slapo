# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace T5 with model schedule."""
import inspect

import slapo
import torch.distributed as dist
from t5_schedule import (
    replace_and_shard_attention,
    shard_word_embedding,
    shard_mlp,
    checkpoint,
    broadcast_input,
)


def schedule_t5(
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

    print_rank_0(f"Scheduling T5")

    if fp16:
        print("Change model dtype to fp16")
        model.half()

    sch = slapo.create_schedule(model, group=group)

    # Replace self attention with flash attention, and shard QKV/output
    # if MP group > 1.
    if not disable_flash_attn:
        cnt, fix_shape_cnt = replace_and_shard_attention(
            sch[prefix], config, "encoder.block.N.layer.0.SelfAttention"
        )
        print_rank_0(
            f"Replace {cnt} encoder self attention patterns "
            f"with {fix_shape_cnt} shape fixing"
        )
        cnt, fix_shape_cnt = replace_and_shard_attention(
            sch[prefix], config, "decoder.block.N.layer.0.SelfAttention"
        )
        print_rank_0(
            f"Replace {cnt} decoder self attention patterns "
            f"with {fix_shape_cnt} shape fixing"
        )
        cnt, fix_shape_cnt = replace_and_shard_attention(
            sch[prefix],
            config,
            "decoder.block.N.layer.1.EncDecAttention",
            cross_attn=True,
        )
        print_rank_0(
            f"Replace {cnt} decoder cross attention patterns "
            f"with {fix_shape_cnt} shape fixing"
        )
    else:
        raise NotImplementedError("Not implemented yet")

    # Shard other parameters if MP group > 1.
    if sch.world_size > 1:
        shard_mlp(sch[prefix], config, "encoder.block.N.layer.1.DenseReluDense")
        shard_mlp(sch[prefix], config, "decoder.block.N.layer.2.DenseReluDense")
        shard_word_embedding(sch[prefix], config.vocab_size)

        # Broadcast input to all devices within the MP group.
        # This is not required when running on Megatron.
        if bcast_input:
            broadcast_input(sch)

    if ckpt_ratio > 0.0:
        n_ckpt = checkpoint(
            sch[prefix], config, "encoder.block.N", ckpt_ratio=ckpt_ratio / 2
        )
        n_ckpt += checkpoint(
            sch[prefix], config, "decoder.block.N", ckpt_ratio=ckpt_ratio / 2
        )
        print_rank_0(f"Checkpointing {n_ckpt} layers")

    # Cut pipeline stages. Note that this means to cut
    # encoder.block.11, decoder.block.11. And we always cut between encoder/decoder,
    # so there will be 4 stages in total.
    # pipeline_cuts = [[11], [11]]
    if pipeline_cuts:
        assert len(pipeline_cuts) == 2
        input_names = [
            "decoder_input_ids",
            "input_ids",
            "decoder_attention_mask",
            "attention_mask"
        ]
        sig = inspect.signature(model.forward)
        concrete_args = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.name not in input_names
        }
        _prefix = f"{prefix}." if prefix else ""
        sch.trace_for_pipeline(
            [f"{_prefix}encoder", f"{_prefix}decoder"],
            tracer="huggingface",
            concrete_args=concrete_args,
        )
        for cut in pipeline_cuts[0]:
            sch[f"{_prefix}encoder.block.{cut}"].cut_pipeline_stage()
        sch[f"{_prefix}encoder"].cut_pipeline_stage()
        for cut in pipeline_cuts[1]:
            sch[f"{_prefix}decoder.block.{cut}"].cut_pipeline_stage()

    return sch
