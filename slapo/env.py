# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def execute(fn, args):
    if isinstance(args, int):
        mp.spawn(fn, args=(args,), nprocs=args, join=True)
    else:
        mp.spawn(fn, args=(args,), nprocs=args.world_size, join=True)
        # if args.world_size > 1:
        #     cleanup()
