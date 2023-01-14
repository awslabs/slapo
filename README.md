<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Slapo: Schedule Language for Large Model Training

Slapo is a schedule language for progressive optimization of large deep learning model training.

Large deep learning models demonstrate dominating model accuracy on a range of tasks in NLP and CV, but it is hard to train the model efficiently while preserving the usability. Slapo aims to address this tension through separation of concerns. Slapo decouples model execution from definition, enabling developers to use a set of schedule primitives to convert a PyTorch model for common model training optimizations without directly changing the model itself.

Slapo highlights the following features:

:rocket: **Progressive optimization**. Slapo incorporates a "trace by need" approach that only traces a desired module to be a static graph for compiler-based aggressive optimizations.

:building_construction: **Structure-preserving scheduling**. Slapo preserves the module hierarchy when constructing the schedule, so developers can easily locate the module and apply scheduling, which also facilitates the users to debug any performance and convergence issue.

:gear: **Auto-tuning**. Slapo provides a programming interface that allows developers to specify a set of tuneable knobs to form an efficient tuning space, which can then be explored by Slapo auto-tuner to realize the optimal configuration.


## Getting Started

### Installation

We currently only support installation from source, and will provide pip-wheel in the future. Please make sure you have installed [PyTorch](https://pytorch.org/) (>= v1.13) in advance.

```bash
git clone https://github.com/awslabs/slapo.git slapo
cd slapo
pip3 install -e ".[dev]"
```

You can optionally install [HuggingFace Transformers](https://github.com/huggingface/transformers) (>= v4.25.1) to retrieve models. Also, we support the following frameworks. You can run the scheduled models on these frameworks if needed.
* [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) >= 3.0.2
* [DeepSpeed](https://github.com/microsoft/DeepSpeed) >= 0.7.7


### Usage
Please see the [examples](examples/) folder for more details. Documentations will be released soon.
```python
import slapo

# Load a PyTorch model from HuggingFace Hub, TorchVision, etc.
from transformers import BertLMHeadModel, AutoConfig
config = AutoConfig.from_pretrained("bert-large-uncased")
bert = BertLMHeadModel(config)

# Create a default schedule
sch = slapo.create_schedule(bert)

# Apply primitives to optimize the model
# Please refer to examples/bert/schedule.py for details
sch["bert.encoder.layer.0"].primitve(...)

# Build an optimized model
opt_model = slapo.build(sch)

# Run the optimized model
inputs = ...
outputs = opt_model(inputs)
```


## Supported Primitives
To maximally reduce the risk introduced by tracers and compilers, we leverage **progressive optimization** to gradually apply primitives to a part of the model. We classify the primitives into two categories. The first type of primitives does *not* require tracing and can be directly applied to modules and parameters; the second type of primitives requires a static graph, and thus needs to apply the `.trace()` primitive first.

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
We also provide a light-weight interface for auto-tuning, so the developers can (1) construct a polyhedral search space using our APIs, and (2) leverage Slapo auto-tuner to automatically search for the best training configuration.

```bash
cd benchmark
# Single device
# To following script will trigger the tuning jobs for all the models
python3 tune_single_device.py
# Single node
python3 tune_single_node.py
```


## Benchmarking
We provide scripts to reproduce our results on a single AWS EC2 p3.16xlarge node with 8 * V100 GPUs.

```bash
cd benchmark
# Download datasets
bash download_benchmark_dataset.sh
# Run benchmarking
# Megatron-LM and Deepspeed are required for executing the experiments
bash run_all_single_node.sh config/single_node_v100.cfg
```


## Publication
If you use Slapo in your project, please feel free to cite our ArXiv [paper](https://arxiv.org/):
- **Slapo: A Schedule Language for Progressive Optimization of Large Deep Learning Model Training**
  Hongzheng Chen, Cody Hao Yu, Shuai Zheng, Zhen Zhang, Zhiru Zhang, and Yida Wang.


## License
Slapo is released under the [Apache 2.0 license](LICENSE).
