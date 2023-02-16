"""
.. currentmodule:: slapo

Quick Start
===========

This guide walks through the key functionality of Slapo.
We will use the BERT model in `HuggingFace Hub <https://github.com/huggingface/transformers>`_ as an example and leverage Slapo to optimize its performance.
"""
# %%
# Optimize PyTorch model with Slapo
# ---------------------------------
# We first import the Slapo package. Make sure you have already installed the PyTorch package.

import slapo
import torch

# %%
# We then load a BERT model implemented in PyTorch from HuggingFace Hub.
# This is the model definition part.

from transformers import BertLMHeadModel, AutoConfig

config = AutoConfig.from_pretrained("bert-large-uncased")
model = BertLMHeadModel(config)
print(model)

# %%
# After we have the model defintion, we can create a default schedule `sch`.
# Later on, all the optimizations will be conducted on this schedule,
# and we do not need to directly modify the original model.
# The original module is stored in the :class:`~slapo.Schedule`, and can be accessed by `sch.mod`.

from slapo.model_schedule import apply_schedule

sch = apply_schedule(
    model, "bert", model_config=config, prefix="bert", fp16=True, ckpt_ratio=0
)
print(sch.mod)

# %%
opt_model, _ = slapo.build(sch, init_weights=model._init_weights)
device = "cuda"
bs = 8
seq_length = 512
input_ids = torch.ones(bs, seq_length, dtype=torch.long, device=device)
attention_mask = torch.ones(bs, seq_length, dtype=torch.float16, device=device)
token_type_ids = torch.ones(bs, seq_length, dtype=torch.long, device=device)
labels = input_ids.clone()
opt_model.to(device)


# %%
optimizer = torch.optim.AdamW(opt_model.parameters(), lr=0.001)
for step in range(10):
    inputs = (input_ids, attention_mask, token_type_ids)
    loss = opt_model(*inputs, labels=labels).loss
    loss.backward()
    optimizer.step()

    if step % 1 == 0:
        print(f"step {step} loss: {loss.item()}")