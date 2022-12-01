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
git clone https://github.com/jfc4050/xformers --recursive
cd xformers
git checkout -b attn-bias origin/attn-bias
git checkout c4a27be
pip3 install -e ".[dev]"
```

- epoi:
```
git clone https://github.com/comaniac/epoi --recursive
cd epoi
git checkout 1980c3a
pip3 install -e ".[dev]"
```

- This repo:
```
git clone https://github.com/chhzh123/model-schedule
cd model-schedule
export PYTHONPATH=`pwd`:$PYTHONPATH
```

You can run these examples using Megatron-LM framework (referring to `../benchmark/bench.py`).
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
