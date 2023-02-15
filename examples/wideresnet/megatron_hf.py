# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train Wide-ResNet.
1. This model is not from HuggingFace, but we just use a unified
file name for easy benchmarking.
2. This script does not use Megatron for training but only its utilities.
"""
from datetime import datetime
import os
import time

import torch
import torch.distributed as dist

from torchvision.models.resnet import Bottleneck, ResNet

from megatron import get_args
from megatron import get_timers
from megatron import mpu
from megatron import print_rank_0, print_rank_last
from megatron.initialize import initialize_megatron
from megatron.initialize import set_jit_fusion_options
from megatron.utils import report_memory

from model import get_model_config
from model import get_model as get_wideresnet_model, ResNetWithLoss
from utils import count_parameters, get_data_loader
from examples.utils import train_with_torch

_TRAIN_START_TIME = time.time()


def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_rank_0("[" + string + "] datetime: {} ".format(time_str))


def get_model(
    model_name,
    fp16=False,
    ckpt_ratio=0.0,
    impl="slapo",
    delay_init=True,
):
    config = get_model_config(model_name)
    dtype = torch.float if not fp16 else torch.half
    print_rank_0(config)

    if "slapo" in impl:
        import slapo
        from slapo.model_schedule import apply_schedule

        with slapo.init_empty_weights(enable=delay_init):
            model = get_wideresnet_model(*config["block_size"])
        print_rank_0(model)
        sch = apply_schedule(
            model,
            "wideresnet",
            model_config=config,
            prefix="model",
            fp16=fp16,
            ckpt_ratio=ckpt_ratio,
            fuse_conv=(dist.get_world_size() == 1),
        )
        model, _ = slapo.build(sch)

    elif impl == "torchscript":
        if ckpt_ratio > 0:
            raise RuntimeError("TorchScript cannot support ckpt")

        model = ResNet(Bottleneck, config[1], width_per_group=config[0])
        if fp16:
            model.half()
        model.cuda()

        device = "cuda"
        data = torch.rand(
            (8, 3, 224, 224),
            dtype=dtype,
            device=device,
        )
        model = torch.jit.trace(model, [data])
        model = ResNetWithLoss(model, torch.nn.CrossEntropyLoss())

    elif impl == "eager":
        if ckpt_ratio > 0:
            raise RuntimeError("WideResNet does not have builtin ckpt")
        model = get_wideresnet_model(*config)

    else:
        raise RuntimeError(f"Unrecognized impl `{impl}`")

    if fp16:
        model.half()
    model.cuda()
    return model


def main():
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron()
    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0(
        "time to initialize megatron (seconds): {:.3f}".format(
            time.time() - _TRAIN_START_TIME
        )
    )
    print_datetime("after megatron is initialized")

    args = get_args()
    device = f"cuda:{torch.distributed.get_rank()}"
    timers = get_timers()

    model_name = os.environ.get("MODEL_NAME", None)
    if model_name is None:
        raise RuntimeError("'MODEL_NAME' not found in environment")

    ckpt_ratio = 0.0
    if args.recompute_granularity is not None:
        ckpt_ratio = os.environ.get("ckpt_ratio", 1.0)
        if ckpt_ratio == "selective":
            raise NotImplementedError
        ckpt_ratio = 1.0 if ckpt_ratio == "full" else float(ckpt_ratio)

    impl = os.environ.get("IMPL", None)
    if impl is None:
        raise RuntimeError("'IMPL' not found in environment")

    if args.fp16:
        print_rank_0("args.fp16 is ignored")
    model = get_model(model_name, False, ckpt_ratio, impl)
    print(
        f" > number of parameters on (tensor, pipeline) "
        f"model parallel rank ({mpu.get_tensor_model_parallel_rank()}, 0): "
        f"{count_parameters(model)}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loader = get_data_loader(args.micro_batch_size, device)
    train_iters = args.train_iters

    def preproc(_, batch):
        dist.broadcast(batch[0][0], src=0, group=None)
        dist.broadcast(batch[1], src=0, group=None)
        return batch

    def postproc(step, loss):
        elapsed_time = timers("interval-time").elapsed()
        if step > 0 and step % args.log_interval == 0:
            time_per_iter = elapsed_time / args.log_interval
            log_string = f"elapsed time per iteration (ms): {time_per_iter * 1e3:.1f} |"
            log_string += f" global batch size: {args.micro_batch_size:5d} |"
            print_rank_last(log_string)
        if step == 0:
            report_memory(f"(after 0 iterations)")
        return loss

    timers("interval-time").start()
    train_with_torch(
        model,
        loader,
        optimizer,
        preproc=preproc,
        postproc=postproc,
        steps=train_iters,
    )


if __name__ == "__main__":
    main()
