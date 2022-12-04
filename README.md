# Model Schedule

A DSL for large model training with decoupled model execution from definition. It uses [torch.fx](https://pytorch.org/docs/stable/fx.html) as the IR.


## Requirements
* [PyTorch](https://pytorch.org/) >= 1.13
* [Transformers](https://github.com/huggingface/transformers) == 4.25.0.dev0
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

| Feature | Primitive |
| :--: | :-- |
| Pattern matching | `s.find_module/function/method(pattern)` |
| Parameter sharding | `s[op].shard(param, axis)` |
| synchronization | `s[op].sync(mode="forward/backward/both")` |
| Kernel Injection | `s[op].replace(OldModule, NewModule)` |
| Forward/Backward Hook | `s[op].hook("fw_pre", hook)` |
| Gradient Checkpointing | `s[op].checkpoint()` |
| Pipeline Partition | `s[op].partition()` |


## Datasets
```bash
./benchmark/download_benchmark_dataset.sh
```


## Examples
```bash
cd benchmark
python3 bench.py hf ../examples/bert/pretrain_hf_bert.py --model bert-large-uncased --gpus pow2 --error-stop --disable-flash-attn
python3 bench.py megatron --model bert-large-uncased --gpus pow2 --error-stop
python3 bench.py hf ../examples/gpt/pretrain_hf_gpt.py --model EleutherAI/gpt-neo-1.3B --gpus 2,4,8 --seq-len 1024 --batch-size "n//2" --error-stop --disable-flash-attn
python3 bench.py megatron --model EleutherAI/gpt-neo-1.3B --gpus 2,4,8 --seq-len 1024 --batch-size "n//2" --error-stop --disable-fuse-kernels
```


## Patches
We require the following changes to make our library work for Huggingface's models:
* Add `getattr` to HF fx tracer due to upstream API changes
    * https://github.com/huggingface/transformers/pull/19233
* AttributeError: ‘BertWithLMHead’ object has no attribute ‘set_input_tensor’
    * https://github.com/chhzh123/model-schedule/blob/master/benchmark/megatron_patch
* Update timing script for Huggingface's trainer
    * https://github.com/chhzh123/model-schedule/blob/master/benchmark/transfomers_patch


## `torch.fx` Limitation
* Cannot support Proxy viewed as `*args` or `**kwargs`: `x = x.view(*new_x_shape)`
* Cannot use control flow like `for`: `[torch.squeeze(t) for t in torch.split(transposed_qkv, 1, dim=-1)]`
