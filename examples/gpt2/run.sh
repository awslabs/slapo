#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

deepspeed deepspeed_hf.py --batch_size 32 --micro_batch_size 4 --model_name gpt2-xl --iter_nums 20 --hidden-size 2048 --nlayers 24 --num-attn-heads 16 --dropout 0.1 --activation_function gelu --seq_len 2048 --pmp 2 --tmp 2 --checkpoint 1.0