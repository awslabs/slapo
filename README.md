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
cd model-schedule
export PYTHONPATH=$(pwd):${PYTHONPATH}
```


## Quick Start
Please refer to [`examples/model-schedule-demo.ipynb`](examples/model-schedule-demo.ipynb) for more details.

| Optimization | Primitive |
| :--: | :-- |
| Parameter sharding | `s[op].shard(param, axis)` |
| synchronization | `s[op].sync()` |
| Kernel Injection | `s[op].replace(OldModule, NewModule)` |


## Examples
```bash
cd benchmark
python3 bench.py hf ../examples/bert/pretrain_hf_bert.py --model bert-large-uncased --gpus pow2 --error-stop
python3 bench.py megatron --model bert-large-uncased --gpus pow2 --error-stop
python3 bench.py hf ../examples/gpt/pretrain_hf_gpt.py --model EleutherAI/gpt-neo-1.3B --gpus 2,4,8 --seq-len 1024 --batch-size "n//2" --error-stop
python3 bench.py megatron --model EleutherAI/gpt-neo-1.3B --gpus 2,4,8 --seq-len 1024 --batch-size "n//2" --error-stop --disable-fuse-kernels
```
