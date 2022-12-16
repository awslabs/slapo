#! /bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

RESULT_FILE="$(date +"%Y-%m-%d-%T").csv"

# Dump env

python3 bench.py env --append-to "$RESULT_FILE"

# Benchmark all

echo "\n" >> $RESULT_FILE
echo -e "Impl\tModel\tSeq\tDecSeq\tnGPU\tBatch\tPerGPUParam\tThrpt\tPerGPUMem\tPerGPUTFLOPS" >> $RESULT_FILE

echo "=== Megatron BERT Large ==="
python3 bench.py megatron --append-to "$RESULT_FILE" \
    --model bert-large-uncased --gpus pow2 --seq-len 512 --batch-size "min(8*n, 48)" --error-stop

echo "=== HF BERT Large ==="
python3 bench.py hf ../examples/bert/pretrain_hf_bert.py --append-to "$RESULT_FILE" \
    --model bert-large-uncased --gpus pow2 --seq-len 512 --batch-size "min(8*n, 48)" --error-stop

echo "=== Megatron GPT 1.3B ==="
python3 bench.py megatron --append-to "$RESULT_FILE" \
    --model EleutherAI/gpt-neo-1.3B --gpus 2,4,8 --seq-len 1024 --batch-size "1 if n<=2 else n" --error-stop

echo "=== HF GPT-Neo 1.3B ==="
python3 bench.py hf ../examples/gpt/pretrain_hf_gpt.py --append-to "$RESULT_FILE" \
    --model EleutherAI/gpt-neo-1.3B --gpus 2,4,8 --seq-len 1024 --batch-size "1 if n<=2 else n" --error-stop

echo "=== Megatron T5 Large ==="
python3 bench.py megatron --append-to "$RESULT_FILE" \
    --model t5-large --gpus 2,4,8 --seq-len 1024 --seq-len-dec 512 --batch-size "n" --error-stop

echo "=== HF T5 Large ==="
 python3 bench.py hf ../examples/t5/pretrain_hf_t5.py --append-to "$RESULT_FILE" \
    --model t5-large --gpus 2,4,8 --seq-len 1024 --seq-len-dec 512 --batch-size "n" --error-stop

