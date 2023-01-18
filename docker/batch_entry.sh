#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# The entry point of AWS Batch job. This script is in charge of configuring
# the repo and executing the given command.
set -e

date

# Parse arguments
SOURCE_REF=$1
REPO=$2
COMMAND=$3

echo "Job Info"
echo "-------------------------------------"
echo "jobId: $AWS_BATCH_JOB_ID"
echo "jobQueue: $AWS_BATCH_JQ_NAME"
echo "computeEnvironment: $AWS_BATCH_CE_NAME"
echo "SOURCE_REF: $SOURCE_REF"
echo "REPO: $REPO"
echo "COMMAND: $COMMAND"
echo "-------------------------------------"

if [ -z $GITHUB_TOKEN ]; then
    echo "GITHUB_TOKEN is not set"
    exit 1
fi;

# Checkout the repo.
git clone https://$GITHUB_TOKEN:x-oauth-basic@github.com/$REPO --recursive
cd slapo

# Config the repo
git fetch origin $SOURCE_REF:working
git checkout working
git submodule update --init --recursive --force

# Execute the command
/bin/bash -o pipefail -c "$COMMAND"
COMMAND_EXIT_CODE=$?

exit $COMMAND_EXIT_CODE