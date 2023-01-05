# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tuning configuration for GPT.
Example usage (assuming you are under 'benchmark'):
python3 -m slapo.autotune.tune --config ../examples/gpt/tune_cfg.py \
    --db gpt-gpu8-slapo-megatron.json --error-stop symbol \
    bench_single_node.py slapo-megatron --model EleutherAI/gpt-neo-1.3B --gpus 8 \
        --seq-len 1024 --batch-size batch_size --gradient-checkpoint ckpt_ratio
"""


def get_bs_range(args):
    n_gpu = int(args["gpus"])
    if "slapo-megatron" in args:
        # single-device model uses gpt-neo-125M, while single-node uses 1.3B
        min_bs = int((1 if n_gpu <= 2 else n_gpu + 3)) if n_gpu >= 2 else 12
    elif "slapo-deepspeed" in args:
        min_bs = int(n_gpu)
    elif "megatron" in args:
        min_bs = 2 if n_gpu == 2 else 5 if n_gpu == 4 else 10
    elif "deepspeed" in args:
        min_bs = n_gpu
    else:
        raise RuntimeError("Unknown implementation")
    max_bs = min_bs * 2
    step = 2
    return (min_bs, max_bs, step)
