<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Examples

This folder includes some examples of using model schedule. 
Before running examples, you need to install the following Python packages:

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

- flash-attention:
```
git clone https://github.com/jfc4050/flash-attention.git
cd flash-attention
git checkout 3676bd2
pip3 install -e ".[dev]"
```

- epoi:
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

You can run these examples using Megatron-LM framework (referring to `../benchmark/bench_single_node.py`).
Meanwhile, if you attempt to run the scheduled model on other frameworks, you can invoke
the scheduled model as follows:

```python
# my_script.py
from transformers import AutoConfig
from transformers.models.modeling_bert import BertLMHeadModel
from bert_model import get_scheduled_bert

model_name = "bert-large-uncased"
config = AutoConfig.from_pretrained(model_name)
model = BertLMHeadModel(config)

model.bert = get_scheduled_bert(
    model_name,
    padded_vocab_size=None,
    binary_head=False,
    add_pooling_layer=True,
    disable_flash_attn=False,
    fp16=True,
    ckpt_ratio=0.0,
)

# ... training logic
```
