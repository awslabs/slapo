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
# We load a BERT model implemented in PyTorch from HuggingFace Hub. 

from transformers import BertLMHeadModel, AutoConfig

config = AutoConfig.from_pretrained("bert-large-uncased")
model = BertLMHeadModel(config)
print(model)

# %%
# After we have the model defintion, we can create a schedule and optimize it.
# Slapo provides an ``apply_schedule`` API for users to directly apply a predefined
# schedule to the model. By default, the schedule will inject the
# `Flash Attention <https://arxiv.org/abs/2205.14135>`_ kernel, conduct tensor
# parallelism, and fuse the operators. Users can also customize the schedule by
# passing in the schedule configurations like data type (fp16/bf16) or checkpoint ratio.
# Detailed schedule configurations can be found in ``slapo.model_schedule``.
# 
# After applying the schedule, we can build the optimized model by calling ``slapo.build``.
# Here we explicitly pass in the ``_init_weights`` function of HuggingFace models to
# initialize the parameters of the optimized model.

def apply_and_build_schedule(model, config):
    from slapo.model_schedule import apply_schedule

    sch = apply_schedule(
        model, "bert", model_config=config, prefix="bert", fp16=True, ckpt_ratio=0
    )
    opt_model, _ = slapo.build(sch, init_weights=model._init_weights)
    return opt_model


# %%
# The optimized model is still a PyTorch ``nn.Module``, so we can pass it to the
# PyTorch training loop as usual.

def train(model, device="cuda", bs=8, seq_length=512):
    input_ids = torch.ones(bs, seq_length, dtype=torch.long, device=device)
    attention_mask = torch.ones(bs, seq_length, dtype=torch.float16, device=device)
    token_type_ids = torch.ones(bs, seq_length, dtype=torch.long, device=device)
    labels = input_ids.clone()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for step in range(100):
        inputs = (input_ids, attention_mask, token_type_ids)
        loss = model(*inputs, labels=labels).loss
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step {step} loss: {loss.item()}")
