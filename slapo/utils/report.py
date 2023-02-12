# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import gc

import psutil
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function


def report_memory(msg="", report_gc=False):
    print(
        f"{msg} CPU RAM used: {psutil.virtual_memory()[3] / 1024 / 1024 / 1024:.4f} GiB"
    )
    if not dist.is_initialized():
        return 0.0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gpu_mem = torch.cuda.max_memory_allocated()
    print(f"{msg} GPU rank {dist.get_rank()} " f"used: {gpu_mem / 1024 / 1024:.4f} MiB")
    if report_gc:
        gc.collect()
        tc = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (
                    hasattr(obj, "data") and torch.is_tensor(obj.data)
                ):
                    if dist.get_rank() == 0:
                        print("GC Tensor", type(obj), obj.size())
                    tc += obj.numel()
            except Exception:
                pass
    return gpu_mem


def profile_perf(model, inputs, backward=False):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
    ) as prof:
        with record_function("model_inference_fw"):
            output = model(*inputs)
        if backward:
            with record_function("model_inference_bw"):
                output["logits"].mean().backward()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
    print(
        prof.key_averages(group_by_stack_n=5).table(
            sort_by="self_cuda_time_total", row_limit=10
        )
    )


def calc_decoder_tflops(
    sample_sec,
    num_gpus,
    seq_length,
    num_layers,
    hidden_size,
    vocab_size,
    ckpt_all=False,
):
    """Calculate the decoder TFLOPS.

    Parameters
    ----------
    sample_sec : float
        The number of samples per second.
    num_gpus : int
        The number of GPUs.
    seq_length : int
        The sequence length.
    num_layers : int
        The number of layers.
    hidden_size : int
        The hidden size.
    vocab_size : int
        The vocabulary size.
    ckpt_all : bool
        Whether to checkpoint all layers. Default is False.
        If False, the calculation is based on model FLOPS.

    Returns
    -------
    float
        The decoder TFLOPS.
    """
    if ckpt_all:
        const_a, const_b = 96, 16
    else:
        const_a, const_b = 72, 12

    flops = const_a * sample_sec * seq_length * num_layers * hidden_size**2
    flops *= (
        1
        + seq_length / 6 / hidden_size
        + vocab_size / const_b / num_layers / hidden_size
    )
    return flops / 10**12 / num_gpus
