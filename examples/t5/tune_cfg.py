# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tuning configuration for T5. Example usage (assuming you are under 'benchmark'):
python3 -m slapo.autotune.tune --config ../examples/t5/tune_cfg.py \
    --db t5-gpu8-slapo-megatron.json --error-stop symbol \
    bench_single_node.py slapo-megatron --model t5-large --gpus 8 \
        --seq-len 1024 --dec-seq-len 512 --batch-size batch_size --gradient-checkpoint ckpt_ratio
"""


def get_bs_range(args):
    n_gpu = int(args["gpus"])
    if "slapo-megatron" in args:
        # single-device uses t5-base (223M) model, while single-node use t5-large (770M)
        min_bs = int((5 if n_gpu == 2 else n_gpu + 6)) if n_gpu >= 2 else 8
    elif "slapo-deepspeed" in args:
        min_bs = int((4 if n_gpu == 2 else 3 * n_gpu))
    elif "megatron" in args:
        min_bs = 3 if n_gpu == 2 else n_gpu + 2
    elif "deepspeed" in args:
        min_bs = n_gpu
    else:
        raise RuntimeError("Unknown implementation")
    max_bs = min_bs * 2
    step = 2
    return (min_bs, max_bs, step)
