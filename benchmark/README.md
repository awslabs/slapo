<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Benchmark

This is a suite including several benchmarks with a set of models on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [DeepSpeed](https://github.com/microsoft/DeepSpeed).

## Preparation

### Install Dependencies

Please use the following command to install the dependencies. We recommend installing [PyTorch](https://pytorch.org/) 2.0 for better performance.

```bash
# Install PyTorch
# Please refer to https://pytorch.org/ for correct OS and CUDA version
pip3 install torch torchvision
# Install other dependencies
pip3 install transformers datasets matplotlib tabulate networkx triton pybind11
```

### Install Efficient Kernels

We leverage the [Flash Attention](https://arxiv.org/abs/2205.14135) kernel in [xformers](https://github.com/facebookresearch/xformers) library to accelerate the attention computation. Please follow the instructions below to install.

- xformers:
```bash
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git checkout 48a77cc
git submodule sync 
git submodule update --init --recursive
pip3 install -e ".[dev]"
```

Note currently we need to apply the following patch to the `xformers` library:
```bash
XFORMER_PATH=`python3 -c "import xformers, pathlib; print(pathlib.Path(xformers.__path__[0]).parent)"`
cp scripts/xformers_patch $XFORMER_PATH
pushd $XFORMER_PATH
git config --global --add safe.directory $XFORMER_PATH
git reset --hard
git apply xformers_patch
git --no-pager diff
popd
```

- flash-attention:
```bash
git clone https://github.com/jfc4050/flash-attention.git
cd flash-attention
git checkout 3676bd2
pip3 install -e ".[dev]"
```

- epoi: Currently used for T5 model
```bash
git clone https://github.com/comaniac/epoi --recursive
cd epoi
git checkout fa90fa7
pip3 install -e ".[dev]"
```

### Install Megatron-LM and DeepSpeed

We also need to install [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [Deepspeed](https://github.com/microsoft/DeepSpeed) from source in order to benchmark the performance.

- Megatron-LM:
```bash
git clone https://github.com/NVIDIA/Megatron-LM --recursive
cd Megatron-LM
git checkout 0bb597b
export PYTHONPATH=`pwd`:$PYTHONPATH
```

You will need to apply the following patch to the `Megatron-LM` library if you are using PyTorch 2.0:

```bash
MEGATRON_PATH=`python3 -c "import megatron, pathlib; print(pathlib.Path(megatron.__path__[0]).parent)"`
cp scripts/megatron_patch $MEGATRON_PATH
pushd $MEGATRON_PATH
git config --global --add safe.directory $MEGATRON_PATH
git reset --hard
git apply scripts/megatron_patch
git --no-pager diff
popd
```

- Apex (required by Megatron-LM)
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

- DeepSpeed:
```bash
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
pip install -e ".[dev]"
```

You can run those [examples](../examples/) using either Megatron-LM or DeepSpeed framework.


### Download datasets

Please download the prepared datasets for benchmarking using the following command:

```bash
bash download_benchmark_dataset.sh
```

## Single Node with Tensor Parallelism

We use a simple config file to control what to benchmark. A config file
is composed of N lines, where each line indicates a benchmark case.
The format is as follows. Note that you can add "#" in the beginning of
any line to skip that configuration.

```
MODE MODEL GPUS SEQ_LEN DEC_SEQ_LEN BATCH_SIZE CKPT
```

* MODE: megatron, slapo-megatron, deepspeed, or slapo-deepspeed
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

You can pass in the generated csv file to plot the results using the following command:

```bash
python3 plot/single_node.py <csv_file>
```
