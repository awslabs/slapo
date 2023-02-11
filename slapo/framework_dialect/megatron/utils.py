# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from ..registry import register_framework_dialect


@register_framework_dialect("megatron", "log_parser")
class MegatronLogParser:
    @staticmethod
    def parse_log(log_filename):
        with open(log_filename, "r", encoding="utf8") as f:
            text = f.read()
        # Find the last number after the key, returns 0 if not exists
        def query(key, last_only=True):
            values = re.findall(key + r": +([\d\.]+)", text)
            if not values:
                return None
            if last_only:
                return float(values[-1])
            return [float(v) for v in values]

        def avg_time(times):
            # Every 5 steps, Megatron reports the average iteration time of
            # the past 5 steps, so we remove the first value (of the first 5 steps)
            # as the warmup if applicable.
            if times is None:
                return 0
            if len(times) == 1:
                return times[0]
            return (sum(times[1:]) * 5) / (5 * (len(times) - 1))

        if "CUDA out of memory" in text:
            print("Out of GPU memory, try a smaller batch size")
            return 0, 0, 0, 1

        iter_times = query(r"elapsed time per iteration \(ms\)", last_only=False)
        if not iter_times:
            print(f'Failed. Check "{log_filename}" to find error')
            return 0, 0, 0, 2

        iter_time = avg_time(iter_times)
        forward_compute_time = avg_time(query("forward-compute", last_only=False))
        backward_compute_time = avg_time(query("backward-compute", last_only=False))
        backward_param_all_reduce_time = avg_time(
            query("backward-params-all-reduce", last_only=False)
        )
        optimizer_time = avg_time(query("optimizer", last_only=False))

        param_per_gpu = query(
            r"parameters on \(tensor, pipeline\) model parallel rank \(0, 0\)"
        )
        avg_samples_per_sec = query("global batch size") / iter_time * 1e3
        gpu_mem = query("max allocated") / 1e3
        print(f"per GPU params\t\t: {param_per_gpu / 1e6:.2f}M")
        print(
            f"Breakdown(ms)\t\t: total {iter_time:.2f}, "
            f"forward {forward_compute_time:.2f}, "
            f"backward {backward_compute_time:.2f}, "
            f"backward-params-all-reduce {backward_param_all_reduce_time:.2f}, "
            f"optimizer {optimizer_time:.2f}"
        )
        # (param_per_gpu, samples_per_sec, gpu_mem, error_code)
        return param_per_gpu, avg_samples_per_sec, gpu_mem, 0
