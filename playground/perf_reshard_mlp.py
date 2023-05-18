from util_reshard_mlp \
    import reshard_RS_to_SR_post, reshard_SR_to_RS_post, \
    reshard_SR_to_RR_post, reshard_RS_to_RR_post, \
    reshard_RR_to_RS_post, reshard_RR_to_RS_pre, \
    reshard_RR_to_SR_post, reshard_RR_to_SR_pre

import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import slapo
import torch.distributed as dist
import logging
import sys
import argparse

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D, 4*D)
        self.fc2 = nn.Linear(4*D, D)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
def perf_model(mod, input_tensor, times):
    """Measure the performance of a mod with certain resharding schemes
    """
    # warmup
    for _ in range(5):
        mod(input_tensor)

    start_event.record()
    for _ in range(times):
        mod(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    if dist.get_rank() == 0:
        print(f"{start_event.elapsed_time(end_event) / times:.3f} ms")


if __name__ == "__main__":

    # Create parser
    parser = argparse.ArgumentParser(description="Resharding schemes on MLP")
    # Add arguments
    parser.add_argument('--times', type=int, required=True, help='Number of times to run the model')
    parser.add_argument('--bs', type=int, required=True, help='Batch size')
    parser.add_argument('--seq', type=int, help='Sequence length', default=1024)
    parser.add_argument('--d', type=int, help='Model size', default=2048)
    parser.add_argument('--p', type=int, help='Number of processes', default=4)
    # Parse the arguments
    args = parser.parse_args()

    # profile time breakdown
    PROFILE = False

    NUM_PROC = args.p
    # Input and Model Size
    SEQ = args.seq
    D = args.d
    # Performance Testing
    TIMES = args.times
    BS = args.bs

    if PROFILE:
        # Setup Logger
        logger = logging.getLogger("profile_reshard")
        logger.setLevel(logging.INFO)
        # Create a file handler
        handler = logging.FileHandler("profile_reshard.log", mode='w')
        handler.setLevel(logging.INFO)
        # Add the handler to the logger
        logger.addHandler(handler)


    dist.init_process_group("nccl", world_size=NUM_PROC)

    # check how many GPUs are available
    if dist.get_rank() == 0:
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")

    torch.cuda.set_device(dist.get_rank())
    device = f"cuda:{dist.get_rank()}"

    # ========== Scheduling =========
    mlp = MLP()

    # 1. Naive. RR * RR -> RR; RR * RR -> RR
    mlp_naive = copy.deepcopy(mlp).to(device=device)
    sch = slapo.create_schedule(mlp_naive)
    mod_1, _ = slapo.build(sch)

    # 2. RR * RS -> RS; RS -> SR (reshard); SR * RR -> SR; SR -> RR (reshard)
    mlp_2 = copy.deepcopy(mlp).to(device=device)
    sch = slapo.create_schedule(mlp_2)
    sch["fc1"].shard("weight", axis=0)
    sch["fc1"].shard("bias", axis=0)
    sch["fc1"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_SR_post)
    sch["fc2"].sync("fwd_post", sync_op_or_fn=reshard_SR_to_RR_post)
    mod_2, _ = slapo.build(sch)

    # 4. Megatron. RR * RS -> RS; RS * SR -> RR (with all reduce)
    mlp_4 = copy.deepcopy(mlp).to(device=device)
    sch = slapo.create_schedule(mlp_4)
    sch["fc1"].shard("weight", axis=0)
    sch["fc1"].shard("bias", axis=0)
    sch["fc2"].shard("weight", axis=1)
    sch["fc2"].sync("fwd_post", sync_op_or_fn="all_reduce")
    mod_4, _ = slapo.build(sch)

    # 8. RR -> SR (reshard); SR * RR -> SR; SR * RR -> SR; SR -> RR (reshard)
    mlp_8 = copy.deepcopy(mlp).to(device=device)
    sch = slapo.create_schedule(mlp_8)
    sch["fc1"].sync("fwd_pre", sync_op_or_fn=reshard_RR_to_SR_pre)
    sch["fc2"].sync("fwd_post", sync_op_or_fn=reshard_SR_to_RR_post)
    mod_8, _ = slapo.build(sch)

    # 5. RR * RS -> RS; RS -> RR (reshard); RR * RS -> RS; RS -> RR (reshard)
    mlp_5 = copy.deepcopy(mlp).to(device=device)
    sch = slapo.create_schedule(mlp_5)
    sch["fc1"].shard("weight", axis=0)
    sch["fc1"].shard("bias", axis=0)
    sch["fc1"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_RR_post)
    sch["fc2"].shard("weight", axis=0)
    sch["fc2"].shard("bias", axis=0)
    sch["fc2"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_RR_post)
    mod_5, _ = slapo.build(sch)


    # =============== Performance ==============
    # Create cuda events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    if dist.get_rank() == 0:
        print("\n===== Setting ======")
        print(f"Number of GPUs: {dist.get_world_size()}; BS: {BS}, SEQ: {SEQ}, D: {D}, TIMES: {TIMES}\n")

    input_tensor = torch.randn(BS, SEQ, D, device=device)

    mods = [mod_1, mod_2, mod_4, mod_8, mod_5]

    for mod in mods:
        perf_model(mod, input_tensor, times=TIMES)
