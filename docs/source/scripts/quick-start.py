"""
.. currentmodule:: slapo

Quick Start
===========

This guide walks through the key functionality of Slapo.
"""
# %%
# Optimize PyTorch model with Slapo
# ---------------------------------
# We first import the Slapo package.

import slapo

# Load a PyTorch model from HuggingFace Hub, TorchVision, etc.
from transformers import BertLMHeadModel, AutoConfig
config = AutoConfig.from_pretrained("bert-large-uncased")
bert = BertLMHeadModel(config)

# Create a default schedule
sch = slapo.create_schedule(bert)
print(sch.mod)