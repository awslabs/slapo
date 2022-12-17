#! /bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

if [ $# -ne 1 ]
then
    echo "Usage: <config file>"
    exit 1
fi

RESULT_FILE="$(date +"%Y-%m-%d-%T").csv"

# Dump env

python3 bench.py env --append-to "$RESULT_FILE"

# Benchmark

echo -e "\n" >> $RESULT_FILE
echo -e "Impl\tModel\tSeq\tDecSeq\tnGPU\tBatch\tCkpt\tPerGPUParam\tThrpt\tPerGPUMem\tPerGPUTFLOPS" >> $RESULT_FILE

while IFS= read -r line || [[ -n $line ]]; do
    if [[ ${line} == \#* ]]; then
        echo "Skip ${line}"
        continue
    fi

    IFS=$'\n' line_array=($(xargs -n1 <<<"$line"))
    MODE=${line_array[0]}
    MODEL=${line_array[1]}
    GPUS=${line_array[2]}
    SEQ_LEN=${line_array[3]}
    DEC_SEQ_LEN=${line_array[4]}
    BATCH_SIZE=${line_array[5]}
    CKPT=${line_array[6]}

    echo "=== ${MODE} ${MODEL} ==="
    python3 bench.py ${MODE} --append-to "$RESULT_FILE" \
        --model ${MODEL} --gpus ${GPUS} --seq-len ${SEQ_LEN} \
        --seq-len-dec ${DEC_SEQ_LEN} \
        --batch-size ${BATCH_SIZE} --gradient-checkpoint ${CKPT} --error-stop
    sleep 1
done < $1
