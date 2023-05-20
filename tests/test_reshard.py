# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument
"""
Test different resharding schemes on MLP. 
Verified by different combinations of resharding schemes. 
"""

import os
import copy

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

import slapo
from slapo.logger import get_logger

logger = get_logger(__name__)

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
    sch = slapo.create_schedule(copy.deepcopy(mlp))
    with slapo.Verify(sch, [input_tensor_verf]):
        # do nothing
        pass
    mod_1, _ = slapo.build(sch)

    # 2. RR * RS -> RS; RS -> SR (reshard); SR * RR -> SR; SR -> RR (reshard)
    logger.info("===== 2. RS -> SR =====", ranks=0)
    sch = slapo.create_schedule(copy.deepcopy(mlp))
    with slapo.Verify(sch, [input_tensor_verf]):
        sch["fc1"].shard("weight", axis=0)
        sch["fc1"].shard("bias", axis=0)
        sch["fc1"].sync("fwd_post", sync_op_or_fn="RS->SR")
        sch["fc2"].sync("fwd_post", sync_op_or_fn="SR->RR")
    mod_2, _ = slapo.build(sch)

    # 3. RR * RS -> RS; RS -> SR (reshard); SR -> RS (reshard); RS * SR -> RR (with all reduce)
    logger.info("===== 3. SR -> RS =====", ranks=0)
    sch = slapo.create_schedule(copy.deepcopy(mlp))
    with slapo.Verify(sch, [input_tensor_verf]):
        sch["fc1"].shard("weight", axis=0)
        sch["fc1"].shard("bias", axis=0)
        sch["fc1"].sync("fwd_post", sync_op_or_fn="RS->SR")
        sch["fc1"].sync("fwd_post", sync_op_or_fn="SR->RS")
        sch["fc2"].shard("weight", axis=1)
        sch["fc2"].sync("fwd_post", sync_op_or_fn="all_reduce")
    mod_3, _ = slapo.build(sch)

    # 4. Megatron. RR * RS -> RS; RS * SR -> RR (with all reduce)
    logger.info("===== 4. Megatron =====", ranks=0)
    sch = slapo.create_schedule(copy.deepcopy(mlp))
    with slapo.Verify(sch, [input_tensor_verf]):
        sch["fc1"].shard("weight", axis=0)
        sch["fc1"].shard("bias", axis=0)
        sch["fc2"].shard("weight", axis=1)
        sch["fc2"].sync("fwd_post", sync_op_or_fn="all_reduce")
    mod_4, _ = slapo.build(sch)

    # 5. RR * RS -> RS; RS -> RR (reshard); RR * RS -> RS; RS -> RR (reshard)
    logger.info("===== 5. RR * RS -> RS =====", ranks=0)
    sch = slapo.create_schedule(copy.deepcopy(mlp))
    with slapo.Verify(sch, [input_tensor_verf]):
        sch["fc1"].shard("weight", axis=0)
        sch["fc1"].shard("bias", axis=0)
        sch["fc1"].sync("fwd_post", sync_op_or_fn="RS->RR")
        sch["fc2"].shard("weight", axis=0)
        sch["fc2"].shard("bias", axis=0)
        sch["fc2"].sync("fwd_post", sync_op_or_fn="RS->RR")
    mod_5, _ = slapo.build(sch)

    # 6. RR * RR -> RR; RR -> RS (reshard); RS * SR -> RR (with all reduce)
    logger.info("===== 6. RR -> RS =====", ranks=0)
    sch = slapo.create_schedule(copy.deepcopy(mlp))
    with slapo.Verify(sch, [input_tensor_verf]):
        sch["fc1"].sync("fwd_post", sync_op_or_fn="RR->RS")
        sch["fc2"].shard("weight", axis=1)
        sch["fc2"].sync("fwd_post", sync_op_or_fn="all_reduce")
    mod_6, _ = slapo.build(sch)

    # 7. RR * RR -> RR; RR -> SR (reshard); SR * RR -> SR; SR -> RR (reshard)
    logger.info("===== 7. RR -> SR =====", ranks=0)
    sch = slapo.create_schedule(copy.deepcopy(mlp))
    with slapo.Verify(sch, [input_tensor_verf]):
        sch["fc1"].sync("fwd_post", sync_op_or_fn="RR->SR")
        sch["fc2"].sync("fwd_post", sync_op_or_fn="SR->RR")
    mod_7, _ = slapo.build(sch)

    # 8. RR -> SR (reshard); SR * RR -> SR; SR * RR -> SR; SR -> RR (reshard)
    logger.info("===== 8. RR -> RS =====", ranks=0)
    sch = slapo.create_schedule(copy.deepcopy(mlp))
    with slapo.Verify(sch, [input_tensor_verf]):
        sch["fc1"].sync("fwd_pre", sync_op_or_fn="RR->SR")
        sch["fc2"].sync("fwd_post", sync_op_or_fn="SR->RR")
    mod_8, _ = slapo.build(sch)
    return [mod_1, mod_2, mod_3, mod_4, mod_5, mod_6, mod_7, mod_8]


if __name__ == "__main__":
    dist.init_process_group("nccl", world_size=int(os.environ["WORLD_SIZE"]))

    # =============== Profiling ===============

    logger.info(
        "Number of GPUs: %d, bs=%d, seq_len=%d, hidden_size=%d",
        dist.get_world_size(),
        bs,
        seq_len,
        hidden_size,
    )
    inp = torch.randn(bs, seq_len, hidden_size, device=f"cuda:{dist.get_rank()}")
    mods = test_schemes(None)
    for i, model in enumerate(mods):
        perf_model(model, inp, i)
        del model
