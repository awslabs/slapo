# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tuning configuration for OPT. Example usage (assuming you are under 'benchmark'):
python3 -m slapo.autotune.tune --config ../examples/opt/tune_cfg.py \
    --db opt-gpu8-slapo-megatron.json --error-stop symbol \
    bench_single_node.py slapo-megatron --model facebook/opt-350m --gpus 8 \
        --seq-len 1024 --batch-size batch_size --gradient-checkpoint ckpt_ratio
"""


def get_bs_range(args):
    n_gpu = int(args["gpus"])
    if "slapo-megatron" in args:
        min_bs = int((14 if n_gpu == 2 else min(6 * n_gpu, 28))) if n_gpu >= 2 else 8
    elif "slapo-deepspeed" in args:
        min_bs = int((20 if n_gpu == 2 else 12 * n_gpu))
    elif "deepspeed" in args:
        min_bs = 2 * n_gpu
    else:
        raise RuntimeError("Unknown implementation")
    max_bs = min_bs * 2 if n_gpu <= 4 else 128
    step = 4 if n_gpu <= 4 else 8
    return (min_bs, max_bs, step)
