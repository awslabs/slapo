#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail

nvidia-smi -L

echo "Running unit tests..."
# torchrun:
#   -r: redirect the output of local rank 1 to None so that
#       only local rank 0's output is printed to the console.
# pytest:
#   -rxXs: show extra info for each test, including xfailed, xpassed, and skipped.
#   -p "no:randomly": disable randomly plugin for sharding tests.
torchrun --nproc_per_node 2 -r 1:1 -m pytest -rxXs -p "no:randomly" tests

echo "Downloading test data..."
bash benchmark/download_benchmark_dataset.sh

# Remove this path when xFormers fixes this issue.
echo "Applying xFormers path..."
XFORMER_PATH=`python3 -c "import xformers, pathlib; print(pathlib.Path(xformers.__path__[0]).parent)"`
cp scripts/xformers_patch $XFORMER_PATH
pushd $XFORMER_PATH
git config --global --add safe.directory $XFORMER_PATH
git reset --hard
git apply xformers_patch
git --no-pager diff
popd

echo "Running end-to-end tests..."
python3 -m pytest -s -p "no:randomly" tests/end2end.py
