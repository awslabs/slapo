# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tuning configuration for WideResnet. Example usage (assuming you are under 'benchmark'):
python3 -m slapo.tune --config ../examples/wideresnet/tune_cfg.py \
    --db wideresnet-gpu8.json --error-stop symbol \
    bench_single_node.py slapo-megatron --model wideresnet-250M --gpus 8 \
        --seq-len 1 --batch-size batch_size --gradient-checkpoint ckpt_ratio
"""
import re


def update_space(args, space):
    # Fix GPU number
    n_gpu = int(args["gpus"])
    if "slapo" in args or "slapomegatron" in args or "slapodeepspeed" in args:
        if n_gpu > 1:
            batch_size = space.create_symbol(
                "batch_size", [16 * n_gpu]
            )
        else:
            batch_size = space.create_symbol("batch_size", [52, 56, 60, 64])

        ckpt_ratio_cand = []
        ckpt_ratio_cand += [0.5, 0.34, 0.25, 0]

        space.create_symbol("ckpt_ratio", ckpt_ratio_cand)
    else:
        space.create_symbol("batch_size", [32, 40, 48, 56, 64])

    return space


def parse_log(unused):
    with open("log.txt") as f:
        text = f.read()

    def query(key):
        values = re.findall(key + ": +([\d\.]+)", text)
        if not values:
            return None
        return [float(v) for v in values]

    if "CUDA out of memory" in text:
        return (1, 0, text)

    batch_size = query("global batch size")
    if not batch_size:
        return (2, 0, text)
    batch_size = int(batch_size[0])

    iter_times = query("elapsed time per iteration \(ms\)")
    if not iter_times:
        return (2, 0, text)

    # 1. Every 5 steps, Megatron reports the average iteration time of the past 5 steps.
    # 2. We remove the first value (of the first 5 steps) as the warmup.
    steps = 5 * (len(iter_times) - 1)
    avg_time = sum(iter_times[1:] * 5) / steps
    return (0, batch_size / avg_time * 1e3, text)
