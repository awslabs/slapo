# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument
"""
Test different resharding schemes on MLP. 
Verified by different combinations of resharding schemes. 
"""

import os
import copy
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

import slapo
from slapo.logger import get_logger

logger = get_logger(__name__)

# Config for verification
bs = 8
seq_len = 1024
hidden_size = 1024


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


def perf_model(mod, input_tensor, idx):
    """Measure the performance of a mod with certain resharding schemes"""
    # warmup
    for _ in range(5):
        mod(input_tensor)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    times = 100
    for _ in range(times):
        mod(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    logger.info(
        "Scheme %d: %.3f ms",
        idx + 1,
        start_event.elapsed_time(end_event) / times,
        ranks=0,
    )


def test_schemes(init_dist):
    # check how many GPUs are available
    num_gpus = torch.cuda.device_count()
    logger.info("Available GPUs: %d", num_gpus, ranks=0)

    torch.cuda.set_device(dist.get_rank())
    device = f"cuda:{dist.get_rank()}"

    # ========== Verification =========

    input_tensor_verf = torch.randn(bs, seq_len, hidden_size).to(device=device)

    with slapo.init_empty_weights():
        mlp = MLP(hidden_size)

    # 1. Naive. RR * RR -> RR; RR * RR -> RR
    logger.info("===== 1. Naive RR =====", ranks=0)
    sch_1 = slapo.create_schedule(copy.deepcopy(mlp))
    with slapo.Verify(sch_1, [input_tensor_verf]):
        # do nothing
        pass

    # 2. RR * RS -> RS; RS -> SR (reshard); SR * RR -> SR; SR -> RR (reshard)
    logger.info("===== 2. RS -> SR =====", ranks=0)
    sch_2 = slapo.create_schedule(copy.deepcopy(mlp))
    with slapo.Verify(sch_2, [input_tensor_verf]):
        sch_2["fc1"].shard("weight", axis=0)
        sch_2["fc1"].shard("bias", axis=0)
        sch_2["fc1"].sync("fwd_post", sync_op_or_fn="RS->SR")
        sch_2["fc2"].sync("fwd_post", sync_op_or_fn="SR->RR")

    # 3. RR * RS -> RS; RS -> SR (reshard); SR -> RS (reshard); RS * SR -> RR (with all reduce)
    logger.info("===== 3. SR -> RS =====", ranks=0)
    sch_3 = slapo.create_schedule(copy.deepcopy(mlp))
    with slapo.Verify(sch_3, [input_tensor_verf]):
        sch_3["fc1"].shard("weight", axis=0)
        sch_3["fc1"].shard("bias", axis=0)
        sch_3["fc1"].sync("fwd_post", sync_op_or_fn="RS->SR")
        sch_3["fc1"].sync("fwd_post", sync_op_or_fn="SR->RS")
        sch_3["fc2"].shard("weight", axis=1)
        sch_3["fc2"].sync("fwd_post", sync_op_or_fn="all_reduce")

    # 4. Megatron. RR * RS -> RS; RS * SR -> RR (with all reduce)
    logger.info("===== 4. Megatron =====", ranks=0)
    sch_4 = slapo.create_schedule(copy.deepcopy(mlp))
    with slapo.Verify(sch_4, [input_tensor_verf]):
        sch_4["fc1"].shard("weight", axis=0)
        sch_4["fc1"].shard("bias", axis=0)
        sch_4["fc2"].shard("weight", axis=1)
        sch_4["fc2"].sync("fwd_post", sync_op_or_fn="all_reduce")

    # 5. RR * RS -> RS; RS -> RR (reshard); RR * RS -> RS; RS -> RR (reshard)
    logger.info("===== 5. RR * RS -> RS =====", ranks=0)
    sch_5 = slapo.create_schedule(copy.deepcopy(mlp))
    with slapo.Verify(sch_5, [input_tensor_verf]):
        sch_5["fc1"].shard("weight", axis=0)
        sch_5["fc1"].shard("bias", axis=0)
        sch_5["fc1"].sync("fwd_post", sync_op_or_fn="RS->RR")
        sch_5["fc2"].shard("weight", axis=0)
        sch_5["fc2"].shard("bias", axis=0)
        sch_5["fc2"].sync("fwd_post", sync_op_or_fn="RS->RR")

    # 6. RR * RR -> RR; RR -> RS (reshard); RS * SR -> RR (with all reduce)
    logger.info("===== 6. RR -> RS =====", ranks=0)
    sch_6 = slapo.create_schedule(copy.deepcopy(mlp))
    with slapo.Verify(sch_6, [input_tensor_verf]):
        sch_6["fc1"].sync("fwd_post", sync_op_or_fn="RR->RS")
        sch_6["fc2"].shard("weight", axis=1)
        sch_6["fc2"].sync("fwd_post", sync_op_or_fn="all_reduce")

    # 7. RR * RR -> RR; RR -> SR (reshard); SR * RR -> SR; SR -> RR (reshard)
    logger.info("===== 7. RR -> SR =====", ranks=0)
    sch_7 = slapo.create_schedule(copy.deepcopy(mlp))
    with slapo.Verify(sch_7, [input_tensor_verf]):
        sch_7["fc1"].sync("fwd_post", sync_op_or_fn="RR->SR")
        sch_7["fc2"].sync("fwd_post", sync_op_or_fn="SR->RR")

    # 8. RR -> SR (reshard); SR * RR -> SR; SR * RR -> SR; SR -> RR (reshard)
    logger.info("===== 8. RR -> RS =====", ranks=0)
    sch_8 = slapo.create_schedule(copy.deepcopy(mlp))
    with slapo.Verify(sch_8, [input_tensor_verf]):
        sch_8["fc1"].sync("fwd_pre", sync_op_or_fn="RR->SR")
        sch_8["fc2"].sync("fwd_post", sync_op_or_fn="SR->RR")
    return [sch_1, sch_2, sch_3, sch_4, sch_5, sch_6, sch_7, sch_8]


if __name__ == "__main__":
    dist.init_process_group("nccl", world_size=int(os.environ["WORLD_SIZE"]))

    # Create parser
    parser = argparse.ArgumentParser(description="Resharding schemes on MLP")
    # Add arguments
    parser.add_argument("--bs", type=int, help="Batch size", default=1)
    parser.add_argument("--seq", type=int, help="Sequence length", default=4096)
    parser.add_argument("--d", type=int, help="Hidden size", default=14336)
    # Parse the arguments
    args = parser.parse_args()

    bs = args.bs
    seq_len = args.seq
    hidden_size = args.d

    # =============== Profiling ===============

    logger.info(
        "Number of GPUs: %d, bs=%d, seq_len=%d, hidden_size=%d",
        dist.get_world_size(),
        bs,
        seq_len,
        hidden_size,
    )
    inp = torch.randn(bs, seq_len, hidden_size, device=f"cuda:{dist.get_rank()}")
    schs = test_schemes(None)
    for i, sch in enumerate(schs):
        model, _ = slapo.build(sch)
        perf_model(model, inp, i)
        del model
        torch.cuda.empty_cache()
