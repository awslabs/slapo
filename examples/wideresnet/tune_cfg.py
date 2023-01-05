# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tuning configuration for WideResnet. Example usage (assuming you are under 'benchmark'):
python3 -m slapo.autotune.tune --config ../examples/wideresnet/tune_cfg.py \
    --db wideresnet-gpu8-slapo-megatron.json --error-stop symbol \
    bench_single_node.py slapo-megatron --model wideresnet-250M --gpus 8 \
        --seq-len 1 --batch-size batch_size --gradient-checkpoint ckpt_ratio
"""


def get_bs_range(args):
    n_gpu = int(args["gpus"])
    if "slapo-megatron" in args:
        min_bs = (
            int(20 * n_gpu if n_gpu <= 2 else min(12 * n_gpu, 52)) if n_gpu >= 2 else 32
        )
    elif "slapo-deepspeed" in args:
        min_bs = int((32 * n_gpu))
    elif "deepspeed" in args:
        min_bs = 32 * n_gpu
    else:
        raise RuntimeError("Unknown implementation")
    max_bs = min_bs * 2
    step = 8
    return (min_bs, max_bs, step)
