#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail

nvidia-smi -L

echo "Running unit tests..."
torchrun --nproc_per_node 2 -m pytest tests
