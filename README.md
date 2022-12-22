<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Slapo: Schedule LAnguage for Progressive Optimization

A domain-specific language (DSL) for large model training with decoupled model execution from definition. It uses [torch.fx](https://pytorch.org/docs/stable/fx.html) as the IR.


## Requirements
* [PyTorch](https://pytorch.org/) >= 1.13
* [Transformers](https://github.com/huggingface/transformers) == 4.25.0.dev0
* [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)


## Installation

We currently only support installation from source. We will provide pip-wheel
in the future.

```bash
git clone https://github.com/chhzh123/model-schedule.git slapo
cd slapo
pip3 install -e ".[dev]"
```


## Quick Start
Please refer to [`examples/model-schedule-demo.ipynb`](examples/model-schedule-demo.ipynb) for more details.

| Feature | Primitive |
| :--: | :-- |
| Pattern matching | `s.find_module/function/method(mod_name_regex, func_pattern)` |
| Module replacement | `s[op].replace(new_module)` |
| Model parallelism | `s[op].shard(param, axis)` |
| Synchronization | `s[op].sync(mode="forward/backward/both")` |
| Pipeline parallelism | `s[op].cut_pipeline_stage()` |
| Forward/Backward hook | `s[op].hook(mode="fw_pre/fw_post/bw_post", func=hook)` |
| Gradient checkpointing | `s[op].checkpoint()` |


## Datasets
```bash
./benchmark/download_benchmark_dataset.sh
```


## Examples
See the [benchmark](benchmark/) folder.
