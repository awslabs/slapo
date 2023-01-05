# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from ..registry import register_model_dialect


@register_model_dialect("deepspeed", "log_parser")
class DeepSpeedLogParser:
    @staticmethod
    def parse_log(log_filename):
        with open(log_filename) as f:
            text = f.read()
        # Find the last number after the key, returns 0 if not exists
        def query(key, last_only=True):
            values = re.findall(key + "=+([\d\.]+)", text)
            if not values:
                return None
            if last_only:
                return float(values[-1])
            return [float(v) for v in values]

        if "CUDA out of memory" in text:
            print("Out of GPU memory, try a smaller batch size")
            return 0, 0, 0, 1

        samples_per_sec = query("SamplesPerSec", last_only=False)
        if not samples_per_sec:
            print(f'Failed. Check "{log_filename}" to find error')
            return 0, 0, 0, 2

        # 1. Every 10 steps, DeepSpeed reports the average samples/sec from beginning.
        # 2. We remove the first value (of the first 10 steps) as the warmup.
        n_records = len(samples_per_sec)
        avg_samples_per_sec = (
            samples_per_sec[-1] * 10 * n_records - samples_per_sec[0] * 10
        ) / (10 * (n_records - 1))

        gpu_mem = query("MaxMemAllocated")

        # (param_per_gpu, samples_per_sec, gpu_mem, error_code)
        return 0, avg_samples_per_sec, gpu_mem, 0
