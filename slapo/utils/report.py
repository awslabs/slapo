# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import psutil
import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity


def report_memory(msg="", report_gc=False):
    print(
        "{} CPU RAM used: {:.4f} GiB".format(
            msg, psutil.virtual_memory()[3] / 1024 / 1024 / 1024
        )
    )
    if not dist.is_initialized():
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(
        "{} GPU rank {} used: {:.4f} MiB".format(
            msg, dist.get_rank(), torch.cuda.max_memory_allocated() / 1024 / 1024
        )
    )
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
            except:
                pass


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
