#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# This script is used to run GPT2 training with 3D parallelism enabled.
# It is tested on a single AWS p3d instance with 8*V100 GPUs.
export CUDA_LAUNCH_BLOCKING=0
deepspeed deepspeed_hf.py \
    --batch_size 32 \
    --micro_batch_size 4 \
    --model_name gpt2-xl \
    --iter_nums 100 \
    --hidden-size 2048 \
    --nlayers 24 \
    --num-attn-heads 16 \
    --dropout 0.1 \
    --activation_function gelu \
    --seq_len 2048 \
    --attn_op_name torch_sdp \
    --pmp 4 --tmp 2 --checkpoint 1.0 2>&1 | tee log.txt