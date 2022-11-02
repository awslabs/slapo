# Model Schedule

A DSL for large model training with decoupled model execution from definition. It uses [torch.fx](https://pytorch.org/docs/stable/fx.html) as the IR.


## Requirements
* [PyTorch](https://pytorch.org/) >= 1.13
* [Transformers](https://github.com/huggingface/transformers) == 4.23.0
* [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)


## Installation
For quick development, we simply export the current project folder to the `PYTHONPATH`. We will provide `pip install` in the future.

```bash
git clone https://github.com/chhzh123/model-schedule.git
cd pt2hls
export PYTHONPATH=$(pwd):${PYTHONPATH}
```


## Quick Start
Please refer to [`model-schedule-demo.ipynb`](/model-schedule-demo.ipynb) for more details.

| Optimization | Primitive |
| :--: | :-- |
| Parameter sharding | `s[op].shard(param, axis)` |
| synchronization | `s[op].sync()` |
| Kernel Injection | `s[op].replace(OldModule, NewModule)` |


## Examples
```bash
# Change a HF model to Megatron using a few LoC
python3 benchmark/bench.py
```
