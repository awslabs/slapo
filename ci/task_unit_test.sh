#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail

nvidia-smi -L

python3 -m pip install black==22.10.0
python3 -m pip install transformers==4.25.1 --no-deps
python3 -m pip install pylint==2.14.0 astroid==2.11.6 mock==4.0.3

echo "Running unit tests..."
# -r: redirect the output of local rank 1 to None so that
# only local rank 0's output is printed to the console.
torchrun --nproc_per_node 2 -r 1:1 -m pytest tests
