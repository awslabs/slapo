<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Benchmark

This is a suite including several benchmarks with a set of models
on Megatron and DeepSpeed.

## Single Node with Tensor Parallelism on Megatron

We use a simple config file to control what to benchmark. A config file
is composed of N lines, where each line indicates a benchmark case.
The format is as follows. Note that you can add "#" in the beginning of
any line to skip that configuration.

```
MODE MODEL GPUS SEQ_LEN DEC_SEQ_LEN BATCH_SIZE CKPT
```

* MODE: Either megatron or slapo
* MODEL: HuggingFace model name (e.g., bert-large-uncased)
* GPUS: Number of GPUs (e.g., pow2, or 2,4,8)
* SEQ_LEN: Sequence length. In encoder-decoder model, this is the encoder length.
* DEC_SEQ_LEN: The decoder length. This is only used by encoder-decoder models.
* BATCH_SIZE: An expression that inputs GPU number and outputs batch size
  (e.g., "16*n" means batch size 16, 32, 64, 128 respecting to GPU number 1, 2, 4, 8).
* CKPT: Activation checkpointing. In Megatron it would be full or selective. In Slapo
  it is a floating point indicating the checkpoint ratio (e.g., 1.0 means full).

The following command runs both Megatron-LM and HuggingFace models
with Megatron framework on up to 8 V100 GPUs (16GB DRAM).
The batch size and sequence length are configured for each model.

```bash
bash run_all_single_node.sh configs/singe_node_v100.cfg
```

Similarity, the following command runs with activation checkpointing:

```bash
bash run_all_single_node.sh configs/singe_node_v100_ckpt.cfg
```

The results are logged to a .csv file and can be processed later.


## Plot Results
```bash
bash run_all_single_node.sh configs/singe_device_v100.cfg
python3 plot/single_device.py <csv_file>
```
