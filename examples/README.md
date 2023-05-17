<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Examples

This folder includes some examples of using model schedule. 
Before running examples, please follow the guidance in the [benchmark](../benchmark/README.md) folder to install the required Python packages.

These examples can be run on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [DeepSpeed](https://github.com/microsoft/DeepSpeed).
If you attempt to run the scheduled model on other frameworks, you can invoke the scheduled model as follows:

```python
# my_script.py
from transformers import AutoConfig
from transformers.models.modeling_bert import BertLMHeadModel

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
