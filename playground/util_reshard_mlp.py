# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of different resharding schemes on MLP. 
Verified by different combinations of resharding schemes. 
"""

import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import slapo
import torch.distributed as dist
from slapo.utils.report import profile_perf
import logging
import sys
import argparse


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D, 4 * D)
        self.fc2 = nn.Linear(4 * D, D)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


def reshard_RS_to_SR(in_tensor):
    in_shape = in_tensor.shape
    chunk_shape = list(in_shape[:-2]) + [
        in_shape[-2] // dist.get_world_size(),
        in_shape[-1],
    ]

    splitted_tensor = torch.split(
        in_tensor, in_shape[-2] // dist.get_world_size(), dim=-2
    )

    for i in range(dist.get_world_size()):
        send_tensor = splitted_tensor[i].contiguous()

        if dist.get_rank() != i:
            dist.gather(send_tensor, dst=i, async_op=True)  # send
        else:
            gather_list = [
                torch.empty(chunk_shape, dtype=in_tensor.dtype, device=in_tensor.device)
                for _ in range(dist.get_world_size())
            ]
            handle = dist.gather(send_tensor, gather_list, dst=i, async_op=True)  # recv
    handle.wait()

    ret = torch.cat(gather_list, dim=-1)
    return ret


def reshard_RS_to_SR_post(_module, _input, output):
    return reshard_RS_to_SR(output)


def reshard_RS_to_SR_pre(_module, input):
    return reshard_RS_to_SR(input[0])


def reshard_SR_to_RS(in_tensor):
    in_shape = in_tensor.shape  # [8, 256, 4096]
    # chunk shape = [8, 256, 4096 // p]
    chunk_shape = list(in_shape[:-1]) + [
        in_shape[-1] // dist.get_world_size()
    ]  # [8, 256, 2048]

    splitted_tensor = torch.split(
        in_tensor, in_shape[-1] // dist.get_world_size(), dim=-1
    )  # [8, 256, 2048]

    for i in range(dist.get_world_size()):
        send_tensor = splitted_tensor[i].contiguous()

        if dist.get_rank() != i:
            dist.gather(send_tensor, dst=i, async_op=True)  # send
        else:
            gather_list = [
                torch.empty(chunk_shape, dtype=in_tensor.dtype, device=in_tensor.device)
                for _ in range(dist.get_world_size())
            ]
            handle = dist.gather(send_tensor, gather_list, dst=i, async_op=True)
    handle.wait()

    ret = torch.cat(gather_list, dim=-2)  # [8, 512, 2048]
    return ret


def reshard_SR_to_RS_post(_module, _input, output):
    return reshard_SR_to_RS(output)


def reshard_SR_to_RS_pre(_module, input):
    return reshard_SR_to_RS(input[0])


def reshard_SR_to_RR(in_tensor):
    temp = in_tensor.transpose(0, -2)  # [256, 8, 1024]
    temp = temp.contiguous()
    gather_shape = list(temp.shape)  # [256, 8, 1024]

    gather_shape[0] = dist.get_world_size() * gather_shape[0]  # [512, 8, 1024]

    ret = torch.empty(gather_shape, dtype=temp.dtype, device=in_tensor.device)
    dist.all_gather_into_tensor(ret, temp)  # [512, 8, 1024]

    ret = ret.transpose(0, -2).contiguous()  # [8, 512 1024]
    return ret


def reshard_SR_to_RR_post(_module, _input, output):
    return reshard_SR_to_RR(output)


def reshard_SR_to_RR_pre(_module, input):
    return reshard_SR_to_RR(input[0])


def reshard_RS_to_RR(in_tensor):
    temp = in_tensor.transpose(0, -1).contiguous()  # [1024, 512, 8]
    gather_shape = list(temp.shape)

    gather_shape[0] = dist.get_world_size() * gather_shape[0]  # [2048, 512, 8]

    ret = torch.empty(gather_shape, dtype=temp.dtype, device=in_tensor.device)
    dist.all_gather_into_tensor(ret, temp)  # [2048, 512, 8]
    ret = ret.transpose(0, -1).contiguous()  # [8, 512, 2048]
    return ret


def reshard_RS_to_RR_post(_module, _input, output):
    return reshard_RS_to_RR(output)


def reshard_RS_to_RR_pre(_module, input):
    return reshard_RS_to_RR(input[0])


def reshard_RR_to_RS(in_tensor):
    # get the current rank's tensor. Slice across the last dimension
    shard_dim_size = in_tensor.shape[-1] // dist.get_world_size()
    start_idx = (int)(dist.get_rank() * shard_dim_size)
    end_idx = (int)((dist.get_rank() + 1) * shard_dim_size)

    slices = [slice(None)] * len(in_tensor.shape)
    slices[-1] = slice(start_idx, end_idx)

    # Slice the tensor
    ret = in_tensor[slices]
    return ret


def reshard_RR_to_RS_post(_module, _input, output):
    return reshard_RR_to_RS(output)


def reshard_RR_to_RS_pre(_module, input):
    return reshard_RR_to_RS(input[0])


def reshard_RR_to_SR(in_tensor):
    # get the current rank's tensor. Slice across the 2nd last dimension
    shard_dim_size = in_tensor.shape[-2] // dist.get_world_size()
    start_idx = (int)(dist.get_rank() * shard_dim_size)
    end_idx = (int)((dist.get_rank() + 1) * shard_dim_size)

    slices = [slice(None)] * len(in_tensor.shape)
    slices[-2] = slice(start_idx, end_idx)

    # Slice the tensor
    ret = in_tensor[slices]
    return ret


def reshard_RR_to_SR_post(_module, _input, output):
    return reshard_RR_to_SR(output)


def reshard_RR_to_SR_pre(_module, input):
    return reshard_RR_to_SR(input[0])


if __name__ == "__main__":
    # Create parser
    parser = argparse.ArgumentParser(description="Resharding schemes on MLP")
    # Add arguments
    parser.add_argument(
        "--times", type=int, required=True, help="Number of times to run the model"
    )
    parser.add_argument("--bs", type=int, required=True, help="Batch size")
    parser.add_argument("--seq", type=int, help="Sequence length", default=1024)
    parser.add_argument("--d", type=int, help="Model size", default=2048)
    parser.add_argument("--p", type=int, help="Number of processes", default=4)
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
        handler = logging.FileHandler("profile_reshard.log", mode="w")
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

    # ========== Verification =========

    input_tensor_verf = torch.randn(8, SEQ, D).to(device=device)

    mlp = MLP()

    # 1. Naive. RR * RR -> RR; RR * RR -> RR
    if dist.get_rank() == 0:
        print("===== 1. Naive RR =====")
    mlp_naive = copy.deepcopy(mlp).to(device=device)
    sch = slapo.create_schedule(mlp_naive)
    with slapo.Verify(sch, [input_tensor_verf]):
        # do nothing
        pass
    mod_1, _ = slapo.build(sch)

    # input_tensor = torch.randn(BS, SEQ, D).to(device=device)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ]
    # ) as prof:
    #     mod_1(input_tensor)
    # logger.info(prof.key_averages().table(
    #     sort_by="self_cuda_time_total", row_limit=10))
    # logger.info(prof.key_averages().table(
    #     sort_by="self_cpu_time_total", row_limit=10))

    # sys.exit(0)

    # 2. RR * RS -> RS; RS -> SR (reshard); SR * RR -> SR; SR -> RR (reshard)
    if dist.get_rank() == 0:
        print("===== 2. RS -> SR =====")
    mlp_2 = copy.deepcopy(mlp).to(device=device)
    sch = slapo.create_schedule(mlp_2)
    with slapo.Verify(sch, [input_tensor_verf]):
        sch["fc1"].shard("weight", axis=0)
        sch["fc1"].shard("bias", axis=0)
        sch["fc1"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_SR_post)
        sch["fc2"].sync("fwd_post", sync_op_or_fn=reshard_SR_to_RR_post)
    mod_2, _ = slapo.build(sch)

    # 3. RR * RS -> RS; RS -> SR (reshard); SR -> RS (reshard); RS * SR -> RR (with all reduce)
    if dist.get_rank() == 0:
        print("===== 3. SR -> RS =====")
    mlp_3 = copy.deepcopy(mlp).to(device=device)
    sch = slapo.create_schedule(mlp_3)
    with slapo.Verify(sch, [input_tensor_verf]):
        sch["fc1"].shard("weight", axis=0)
        sch["fc1"].shard("bias", axis=0)
        sch["fc1"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_SR_to_RS_post)
        sch["fc2"].shard("weight", axis=1)
        sch["fc2"].sync("fwd_post", sync_op_or_fn="all_reduce")
    mod_3, _ = slapo.build(sch)

    # 4. Megatron. RR * RS -> RS; RS * SR -> RR (with all reduce)
    if dist.get_rank() == 0:
        print("===== 4. Megatron =====")
    mlp_4 = copy.deepcopy(mlp).to(device=device)
    sch = slapo.create_schedule(mlp_4)
    with slapo.Verify(sch, [input_tensor_verf]):
        sch["fc1"].shard("weight", axis=0)
        sch["fc1"].shard("bias", axis=0)
        sch["fc2"].shard("weight", axis=1)
        sch["fc2"].sync("fwd_post", sync_op_or_fn="all_reduce")
    mod_4, _ = slapo.build(sch)

    # 5. RR * RS -> RS; RS -> RR (reshard); RR * RS -> RS; RS -> RR (reshard)
    if dist.get_rank() == 0:
        print("===== 5. RR * RS -> RS =====")
    mlp_5 = copy.deepcopy(mlp).to(device=device)
    sch = slapo.create_schedule(mlp_5)
    with slapo.Verify(sch, [input_tensor_verf]):
        sch["fc1"].shard("weight", axis=0)
        sch["fc1"].shard("bias", axis=0)
        sch["fc1"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_RR_post)
        sch["fc2"].shard("weight", axis=0)
        sch["fc2"].shard("bias", axis=0)
        sch["fc2"].sync("fwd_post", sync_op_or_fn=reshard_RS_to_RR_post)
    mod_5, _ = slapo.build(sch)

    # 6. RR * RR -> RR; RR -> RS (reshard); RS * SR -> RR (with all reduce)
    if dist.get_rank() == 0:
        print("===== 6. RR -> RS =====")
    mlp_6 = copy.deepcopy(mlp).to(device=device)
    sch = slapo.create_schedule(mlp_6)
    with slapo.Verify(sch, [input_tensor_verf]):
        sch["fc1"].sync("fwd_post", sync_op_or_fn=reshard_RR_to_RS_post)
        sch["fc2"].shard("weight", axis=1)
        sch["fc2"].sync("fwd_post", sync_op_or_fn="all_reduce")
    mod_6, _ = slapo.build(sch)

    # 7. RR * RR -> RR; RR -> SR (reshard); SR * RR -> SR; SR -> RR (reshard)
    if dist.get_rank() == 0:
        print("===== 7. RR -> SR =====")
    mlp_7 = copy.deepcopy(mlp).to(device=device)
    sch = slapo.create_schedule(mlp_7)
    with slapo.Verify(sch, [input_tensor_verf]):
        sch["fc1"].sync("fwd_post", sync_op_or_fn=reshard_RR_to_SR_post)
        sch["fc2"].sync("fwd_post", sync_op_or_fn=reshard_SR_to_RR_post)
    mod_7, _ = slapo.build(sch)

    # 8. RR -> SR (reshard); SR * RR -> SR; SR * RR -> SR; SR -> RR (reshard)
    if dist.get_rank() == 0:
        print("===== 8. RR -> RS =====")
    mlp_8 = copy.deepcopy(mlp).to(device=device)
    sch = slapo.create_schedule(mlp_8)
    with slapo.Verify(sch, [input_tensor_verf]):
        sch["fc1"].sync("fwd_pre", sync_op_or_fn=reshard_RR_to_SR_pre)
        sch["fc2"].sync("fwd_post", sync_op_or_fn=reshard_SR_to_RR_post)
    mod_8, _ = slapo.build(sch)

    # =============== Performance ==============
    # Create cuda events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    def perf_model(mod, input_tensor, times=TIMES):
        """Measure the performance of a mod with certain resharding schemes"""
        start_event.record()
        for _ in range(times):
            mod(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        if dist.get_rank() == 0:
            print(f"{start_event.elapsed_time(end_event) / times:.3f} ms")

    if dist.get_rank() == 0:
        print("\n===== Setting ======")
        print(
            f"Number of GPUs: {dist.get_world_size()}; BS: {BS}, SEQ: {SEQ}, D: {D}, TIMES: {TIMES}\n"
        )

    input_tensor = torch.randn(BS, SEQ, D, device=device)

    # mods = [mod_1, mod_2, mod_4, mod_8, mod_5, mod_3, mod_6, mod_7]
    mods = [mod_1, mod_2, mod_4, mod_8, mod_5]

    for mod in mods:
        perf_model(mod, input_tensor)


# # Mod 2: Ours
# start_event.record()
# for _ in range(TIMES):
#     mod_2(torch.randn(BS, SEQ, D).to(device=device))
# end = time.time()
# if dist.get_rank() == 0:
#     print("Mod 2: ", end - start)

# # Mod 4: Slapo-Megatron
# start_event.record()
# for _ in range(TIMES):
#     mod_4(torch.randn(BS, SEQ, D).to(device=device))
# end = time.time()
# if dist.get_rank() == 0:
#     print("Mod 4: ", end - start)
# # [BS, 512, 1024] (RR) * [1024, 2048] (RS) -> [BS, 512, 2048] (RS);
# # [BS, 512, 2048] (RS) * [2048, 1024] (SR) -> [BS, 512, 1024] (RR) (All reduce)

# # Mod 8: Data Parallelism
# start_event.record()
# for _ in range(TIMES):
#     mod_8(torch.randn(BS, SEQ, D).to(device=device))
# end = time.time()
# if dist.get_rank() == 0:
#     print("Mod 8: ", end - start)

# # Mod 5: Weight Parallelism
# start_event.record()
# for _ in range(TIMES):
#     mod_5(torch.randn(BS, SEQ, D).to(device=device))
# end = time.time()
# if dist.get_rank() == 0:
#     print("Mod 5: ", end - start)

# # Mod 3
# start_event.record()
# for _ in range(TIMES):
#     mod_3(torch.randn(BS, SEQ, D).to(device=device))
# end = time.time()
# if dist.get_rank() == 0:
#     print("Mod 3: ", end - start)

# # Mod 6
# start_event.record()
# for _ in range(TIMES):
#     mod_6(torch.randn(BS, SEQ, D).to(device=device))
# end = time.time()
# if dist.get_rank() == 0:
#     print("Mod 6: ", end - start)

# # Mod 7
# start_event.record()
# for _ in range(TIMES):
#     mod_7(torch.randn(BS, SEQ, D).to(device=device))
# end = time.time()
# if dist.get_rank() == 0:
#     print("Mod 7: ", end - start)
