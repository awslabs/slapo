#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail

nvidia-smi -L

echo "Running unit tests..."
# -r: redirect the output of local rank 1 to None so that
# only local rank 0's output is printed to the console.
torchrun --nproc_per_node 2 -r 1:1 -m pytest tests

echo "Downloading test data..."
cd benchmark
bash download_benchmark_dataset.sh
cd ..
echo "Running end-to-end tests..."
python3 -m pytest -s tests/end2end.py
