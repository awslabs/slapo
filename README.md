# Slapo: Schedule LAnguage for Progressive Optimization

A domain-specific language (DSL) for large model training with decoupled model execution from definition. It uses [torch.fx](https://pytorch.org/docs/stable/fx.html) as the IR.


## Requirements
* [PyTorch](https://pytorch.org/) >= 1.13
* [Transformers](https://github.com/huggingface/transformers) == 4.25.0.dev0
* [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)


## Installation
For quick development, we simply export the current project folder to the `PYTHONPATH`. We will provide `pip install` in the future.

```bash
git clone https://github.com/chhzh123/model-schedule.git slapo
cd slapo
export PYTHONPATH=$(pwd):${PYTHONPATH}
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
```bash
cd benchmark
python3 bench.py hf ../examples/bert/pretrain_hf_bert.py --model bert-large-uncased --gpus pow2 --error-stop --disable-flash-attn
python3 bench.py megatron --model bert-large-uncased --gpus pow2 --error-stop
python3 bench.py hf ../examples/gpt/pretrain_hf_gpt.py --model EleutherAI/gpt-neo-1.3B --gpus 2,4,8 --seq-len 1024 --batch-size "n//2" --error-stop --disable-flash-attn
python3 bench.py megatron --model EleutherAI/gpt-neo-1.3B --gpus 2,4,8 --seq-len 1024 --batch-size "n//2" --error-stop --disable-fuse-kernels
```
