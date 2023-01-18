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
python3 -m pip install black==22.10.0
bash ./scripts/lint/git-black.sh HEAD~1
bash ./scripts/lint/git-black.sh origin/main

echo "Running pylint on slapo"
python3 -m pip install pylint==2.14.0 astroid==2.11.6
python3 -m pylint slapo --rcfile=./scripts/lint/pylintrc

echo "Running pylint on tests"
python3 -m pylint tests --rcfile=./scripts/lint/pylintrc
