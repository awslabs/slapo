#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail

echo "Check license header..."
python3 scripts/lint/check_license_header.py HEAD~1
python3 scripts/lint/check_license_header.py origin/main

echo "Check Python formats using black..."
bash ./scripts/lint/git-black.sh HEAD~1
bash ./scripts/lint/git-black.sh origin/main

echo "Running pylint on slapo"
python3 -m pylint slapo --rcfile=./scripts/lint/pylintrc

echo "Running pylint on tests"
python3 -m pylint tests --rcfile=./scripts/lint/pylintrc
