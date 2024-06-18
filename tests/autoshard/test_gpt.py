# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import copy
import inspect
import operator
import argparse

import torch
from torch import nn
from torch import fx
import torch.distributed as dist
from transformers import GPTNeoModel, AutoConfig

import slapo
from slapo.logger import get_logger

logger = get_logger(__name__)

# Config for verification
bs = 4
seq_len = 1024


def perf_model(mod, input_tensor):
    """Measure the performance of a mod with certain resharding schemes"""
    # warmup
    mod.eval()
    # mod.to(torch.float16)
    for _ in range(10):
        mod(input_tensor)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    iters = 40
    for _ in range(iters):
        mod(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    if dist.get_rank() == 0:
        print(f"{start_event.elapsed_time(end_event) / iters:.3f} ms")


def trace_and_find_view(sch, config):
    input_names = ["hidden_states"]
    sig = inspect.signature(sch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    sch.trace(
        recursive=False,
        flatten=True,
        tracer="huggingface",
        concrete_args=concrete_args,
        config=config,
    )
    ops = sch.find_node(
        lambda node: node.op == "call_method"
        and node.target == "view"
        and (
            (node.args[0].op == "call_module" and "proj" in node.args[0].target)
            or (
                len(node.args) > 1
                and isinstance(node.args[1], fx.Node)
                and node.args[1].op == "call_function"
                and node.args[1].target == operator.add
            )
        )
    )
    assert len(ops) == 4  # q,k,v,context_layer
    return ops


def fix_attention_mask_shape_megatron(sch, config):
    ops = trace_and_find_view(sch, config)

    def new_view(tensor, args):
        if len(args) == 4:  # q,k,v
            out = tensor.view(args[0], args[1], args[2] // sch.world_size, args[3])
        else:  # context_layer
            out = tensor.view(args[0], args[1], args[2] // sch.world_size)
        return out

    for op in ops:
        sch.replace(new_view, op)


def scheme_megatron(model, input_ids, config):
    sch = slapo.create_schedule(model)

    enable = True if input_ids.shape[0] == 1 else False
    with slapo.Verify(sch, [input_ids], enable=enable):
        for i in range(config.num_hidden_layers):
            # shard attention
            subsch = sch[f"h.{i}.attn.attention"]
            # no bias for GPTNeo
            subsch["q_proj"].shard("weight", axis=0)
            subsch["k_proj"].shard("weight", axis=0)
            subsch["v_proj"].shard("weight", axis=0)
            subsch["out_proj"].shard("weight", axis=1)
            subsch["out_proj"].sync("fwd_post", sync_op_or_fn="all_reduce")
            fix_attention_mask_shape_megatron(subsch, config)
            # shard MLP
            subsch = sch[f"h.{i}.mlp"]
            subsch["c_fc"].shard("weight", axis=0)
            subsch["c_fc"].shard("bias", axis=0)
            subsch["c_proj"].shard("weight", axis=1)
            subsch["c_proj"].sync("fwd_post", sync_op_or_fn="all_reduce")

    return sch


def scheme_sequence_parallel(model, input_ids, config):
    sch = slapo.create_schedule(model)

    from slapo.sharding.reshard_ops import (
        reshard_SR_to_RR,
        reshard_RS_to_RR,
    )

    def new_matmul(lhs, rhs):
        return torch.matmul(lhs, reshard_RS_to_RR(rhs, sch.group))

    def new_matmul_1(lhs, rhs):
        return torch.matmul(lhs, reshard_SR_to_RR(rhs, sch.group))

    class NewMask(nn.Module):
        def forward(self, query, key, bias):
            query_length, key_length = (
                query.size(-2) * sch.world_size,
                key.size(-2) * sch.world_size,
            )
            size_per_chunk = query_length // sch.world_size
            start_idx = key_length - query_length + size_per_chunk * sch.rank
            end_idx = start_idx + size_per_chunk
            causal_mask = bias[:, :, start_idx:end_idx, :key_length]
            return causal_mask

    enable = True if input_ids.shape[0] == 1 else False
    with slapo.Verify(sch, [input_ids], eval_mode=True, enable=enable):
        sch["drop"].sync(mode="fwd_post", sync_op_or_fn="RR->SR")
        for i in range(config.num_hidden_layers):
            subsch = sch[f"h.{i}.attn.attention"]
            trace_and_find_view(subsch, config)
            ops = subsch.find_node(
                lambda node: node.op == "call_function" and node.target == torch.matmul
            )
            assert len(ops) == 2
            subsch.replace(new_matmul, ops[0])
            subsch.replace(new_matmul_1, ops[1])

            # Need to shard the tril matrix (causal mask)
            def pattern(query, key, bias):
                query_length, key_length = query.size(-2), key.size(-2)
                causal_mask = bias[
                    :, :, key_length - query_length : key_length, :key_length
                ]
                return causal_mask

            ops = subsch.find(pattern)
            subsch.replace(NewMask(), target_ops=[ops[-1]])
        sch[f"ln_f"].sync(mode="fwd_post", sync_op_or_fn="SR->RR")

    return sch


def scheme_activation_stationary(model, input_ids, config):
    sch = slapo.create_schedule(model)
    with slapo.Verify(sch, [input_ids]):
        for i in range(config.num_hidden_layers):
            # shard attention
            subsch = sch[f"bert.encoder.layer.{i}.attention.self"]
            subsch["query"].shard("weight", axis=0)
            subsch["query"].shard("bias", axis=0)
            subsch["key"].shard("weight", axis=0)
            subsch["key"].shard("bias", axis=0)
            subsch["value"].shard("weight", axis=0)
            subsch["value"].shard("bias", axis=0)
            fix_attention_mask_shape_megatron(subsch)
            subsch = sch[f"bert.encoder.layer.{i}.attention.output"]
            # shape here: [4096, 256](RS). Need to matmul with [1024, 1024] (without shard)
            subsch["dense"].sync("fwd_pre", sync_op_or_fn="RS->RR")
            subsch["dense"].shard("weight", axis=0)
            subsch["dense"].shard("bias", axis=0)
            subsch["dense"].sync("fwd_post", sync_op_or_fn="RS->RR")
            # shard MLP
            subsch = sch[f"bert.encoder.layer.{i}"]
            subsch["intermediate.dense"].shard("weight", axis=0)
            subsch["intermediate.dense"].shard("bias", axis=0)
            subsch["intermediate.dense"].sync("fwd_post", sync_op_or_fn="RS->RR")
            subsch["output.dense"].shard("weight", axis=0)
            subsch["output.dense"].shard("bias", axis=0)
            subsch["output.dense"].sync("fwd_post", sync_op_or_fn="RS->RR")

    return sch


def scheme_activation_sharding(model, input_ids, config):
    sch = slapo.create_schedule(model)

    from slapo.sharding.reshard_ops import reshard_RR_to_SR

    def reshard_and_add(dropout, hidden_states):
        """Replace the add operator with reshard_and_add"""
        reshard_hidden_states = reshard_RR_to_SR(hidden_states, sch.group)
        return dropout + reshard_hidden_states

    with slapo.Verify(sch, [input_ids]):
        for i in range(config.num_hidden_layers):
            # shard attention
            subsch = sch[f"bert.encoder.layer.{i}.attention.self"]
            subsch["query"].shard("weight", axis=0)
            subsch["query"].shard("bias", axis=0)
            subsch["key"].shard("weight", axis=0)
            subsch["key"].shard("bias", axis=0)
            subsch["value"].shard("weight", axis=0)
            subsch["value"].shard("bias", axis=0)
            fix_attention_mask_shape_megatron(subsch)
            subsch = sch[f"bert.encoder.layer.{i}.attention.output"]

            subsch.trace(recursive=False, flatten=False, tracer="pytorch")
            ops = subsch.find_node(
                lambda node: node.op == "call_function" and node.target == operator.add
            )
            subsch.replace(reshard_and_add, ops[0])

            # shape here: RS
            subsch["dense"].sync(
                "fwd_pre", sync_op_or_fn="RS->SR"
            )  # LayerNorm will crash for SR x RR = SR
            # shard MLP
            subsch = sch[f"bert.encoder.layer.{i}"]
            subsch["output.LayerNorm"].sync("fwd_post", sync_op_or_fn="SR->RR")

    return sch


def test_schemes(init_dist):
    torch.cuda.set_device(dist.get_rank())
    device = torch.cuda.current_device()

    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-1.3B")
    config.use_cache = False
    with slapo.init_empty_weights():
        model = GPTNeoModel(config)

    schs = []
    input_ids = torch.ones(bs, seq_len, dtype=torch.long, device=device)
    # 1. Slapo-Megatron
    # RR x RS = RS, RS x SR = RR
    schs.append(scheme_megatron(copy.deepcopy(model), input_ids, config))
    # 2. Sequence-Parallel
    # RR->RS x RR = RS, RS x RR = RS->RR
    schs.append(scheme_sequence_parallel(copy.deepcopy(model), input_ids, config))
    # 3. Activation-Stationary
    # RR x RS = RS
    # schs.append(scheme_activation_stationary(copy.deepcopy(model), input_ids, config))
    # # 4. Activation Sharding. SR x RR = SR
    # schs.append(scheme_activation_sharding(copy.deepcopy(model), input_ids, config))
    return schs


if __name__ == "__main__":
    # Create parser
    parser = argparse.ArgumentParser(description="Resharding schemes on GPTNeo")
    # Add arguments
    parser.add_argument("--bs", type=int, help="Batch size", default=4)
    parser.add_argument("--seq", type=int, help="Sequence length", default=1024)
    # Parse the arguments
    args = parser.parse_args()

    bs = args.bs
    seq_len = args.seq

    dist.init_process_group("nccl", world_size=int(os.environ["WORLD_SIZE"]))

    logger.info(
        "Number of GPUs: %d, bs=%d, seq_len=%d; Model: GPTNeo",
        dist.get_world_size(),
        bs,
        seq_len,
        ranks=0,
    )

    input_ids = torch.ones(
        bs, seq_len, dtype=torch.long, device=f"cuda:{dist.get_rank()}"
    )
    schs = test_schemes(None)
    for i, sch in enumerate(schs):
        mod, _ = slapo.build(sch, init_weights=sch.mod._init_weights)
        mod.to(f"cuda:{dist.get_rank()}")
        torch.cuda.empty_cache()
        perf_model(mod, input_ids)
        del mod
