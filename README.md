<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Slapo: Schedule Language for Large Model Training

Slapo is a schedule language for progressive optimization of large deep learning model training. We aim to address the tension between usability and training efficieny through separtion of concerns. Slapo decouples model execution from definition, enabling users to work on a PyTorch model and use a set of schedule primitives to convert it for common model training optimizations such as high-performance kernels, effective 3D parallelism, and efficient activation checkpointing. Slapo progressively optimizes the model "as-needed" through high-level primitives, and thus preserving programmability and debuggability for users to a large extent.


## Getting Started

### Requirements
* [PyTorch](https://pytorch.org/) >= 1.13
* [HuggingFace Transformers](https://github.com/huggingface/transformers) >= 4.25.1
* [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) >= 3.0.2
* [DeepSpeed](https://github.com/microsoft/DeepSpeed) >= 0.7.7


### Installation

We currently only support installation from source. We will provide pip-wheel in the future.

```bash
git clone https://github.com/awslabs/slapo.git slapo
cd slapo
pip3 install -e ".[dev]"
```

### Usage
Please see the [example](example/) folder for more details.
```python
import slapo

# load a PyTorch model from HuggingFace Hub, TorchVision, etc.
from transformers import BertLMHeadModel, AutoConfig
config = AutoConfig.from_pretrained("bert-large-uncased")
bert = BertLMHeadModel(config)

# Create a default schedule
sch = slapo.create_schedule(bert)

# Conduct optimizations
# Please refer to examples/bert/schedule.py for how to apply our primitives
sch["bert.xxx"].primitve(...)

# Build an optimized model
opt_model = slapo.build(sch)

# Run the optimized model
inputs = ...
outputs = opt_model(inputs)
```


## Supported Primitives
To maximally reduce the risk introduced by tracers and compilers, we leverage **progressive optimization** to gradually apply primitives to a part of the model. We classify the primitives into two catagories. The first type of primitives does *not* require tracing and can be directly applied to modules and parameters; the second type of primitives requires a static graph, and thus needs to apply the `.trace()` primitive first.

We provide the following primitives for dynamic graph optimizations:
| Feature | Primitive |
| :--: | :-- |
| Module replacement | `s[op].replace(new_module)` |
| Tensor parallelism | `s[op].shard(param_name, axis)` |
| Synchronization | `s[op].sync(mode="forward/backward/both")` |
| Checkpointing | `s[op].checkpoint()` |
| Forward/Backward hook | `s[op].hook(mode="fw_pre/fw_post/bw_post", func=hook)` |

And the following primitives for static graph optimizations:
| Feature | Primitive |
| :--: | :-- |
| Module Tracing | `s.trace(leaves, flatten)` |
| Pattern matching | `s.find(mod_name_regex, func_pattern)` |
| Operator fusion | `s[op].fuse(compiler, subgraph)` |
| Partial module replacement | `s[op].replace(new_module, subgraph)` |
| Partial gradient checkpointing | `s[op].checkpoint()` |
| Pipeline parallelism | `s[op].pipeline_split()` |


### Auto-Tuning
We also provide a light-weight interface for auto-tuning, so the users can (1) construct a polyhedral search space using our APIs, and (2) leverage our auto-tuner for automatically search for the best configuration.

```bash
cd benchmark
python3 tun_single_node.py
```


## Benchmarking
We provide scripts to reproduce our results on a single node with 8 * V100 GPUs.
```bash
cd benchmark
# Download test datasets
bash download_benchmark_dataset.sh
# Benchmark
bash run_all_single_node.sh config/single_node_v100.cfg
```


## Publication
If you use Slapo in your project, please feel free to cite our [paper](https://arxiv.org/):
- **Slapo: A Schedule Language for Progressive Optimization of Large Deep Learning Model Training**
  Hongzheng Chen, Cody Hao Yu, Shuai Zheng, Zhen Zhang, Zhiru Zhang, and Yida Wang.


## License
Slapo is released under the [Apache 2.0 license](LICENSE).
