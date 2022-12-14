# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

CUDA_VISIBLE_DEVICES=0 deepspeed pipeline_toy.py --world_size 1
CUDA_VISIBLE_DEVICES=0,1 deepspeed pipeline_toy.py --world_size 2