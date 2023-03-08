<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Examples

This folder includes some examples of using model schedule. 
Before running examples, you need to install the following Python packages:

- transformers:
```
pip3 install transformers
```

- megatron-lm:
```
git clone https://github.com/NVIDIA/Megatron-LM --recursive
cd Megatron-LM
git checkout 0bb597b
export PYTHONPATH=`pwd`:$PYTHONPATH
```

- xformers:
```
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git checkout 48a77cc
git submodule sync 
git submodule update --init --recursive
pip3 install -e ".[dev]"
```

Note currently we need to apply the following patch to the `xformers` library:
```
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
```
git clone https://github.com/jfc4050/flash-attention.git
cd flash-attention
git checkout 3676bd2
pip3 install -e ".[dev]"
```

- epoi: Currently used for T5 model
```
git clone https://github.com/comaniac/epoi --recursive
cd epoi
git checkout fa90fa7
pip3 install -e ".[dev]"
```

- This repo:
```
git clone https://github.com/awslabs/slapo.git slapo
cd slapo
pip3 install -e ".[dev]"
```

You can run these examples using Megatron-LM framework.
Please refer to `../benchmark/bench_single_node.py`, and make sure you have installed the following packages before running this script.

```
pip3 install matplotlib tabulate datasets networkx triton
```

Meanwhile, if you attempt to run the scheduled model on other frameworks, you can invoke the scheduled model as follows:

```python
# my_script.py
from transformers import AutoConfig
from transformers.models.modeling_bert import BertLMHeadModel
from bert_model import get_scheduled_bert

model_name = "bert-large-uncased"
config = AutoConfig.from_pretrained(model_name)
model = BertLMHeadModel(config)

def apply_and_build_schedule(model, config):
    from slapo.model_schedule import apply_schedule

    sch = apply_schedule(
        model, "bert", model_config=config, prefix="bert", fp16=True, ckpt_ratio=0
    )
    opt_model, _ = slapo.build(sch, init_weights=model._init_weights)
    return opt_model

opt_model = apply_and_build_schedule(model, config)
# ... training logic
```
