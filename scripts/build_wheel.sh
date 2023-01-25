#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

python3 update_version.py --git-describe
rm -rf build dist
python3 setup.py bdist_wheel
