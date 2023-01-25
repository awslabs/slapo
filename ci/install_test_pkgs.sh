#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

python3 -m pip install black==22.10.0
python3 -m pip install transformers==4.25.1 --no-deps
python3 -m pip install pylint==2.14.0 astroid==2.11.6 mock==4.0.3
