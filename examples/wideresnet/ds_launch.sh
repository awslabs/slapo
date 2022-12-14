#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


TRAIN_SCRIPT=$1
TRAIN_CONFIG=$2
# HOSTFILE=$3

DATETIME=$(date +"%Y-%m-%d_%T")

JOB_NAME=${DATETIME}-wideresnet
OUTPUT_DIR="./${JOB_NAME}"

common_args="\
 \
"

# running cmd
# If you want to restrict the number of GPUs, add the following flag
# BEFORE {TRAIN_SCRIPT}: --num_gpus=1
ds_cmd="\
deepspeed ${TRAIN_SCRIPT} \
--deepspeed \
--deepspeed_config $TRAIN_CONFIG \
--job_name ${JOB_NAME}
"
#--hostfile=${HOSTFILE}

echo $ds_cmd
eval $ds_cmd
