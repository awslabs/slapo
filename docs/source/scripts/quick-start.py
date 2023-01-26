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

# %%
# We then load a BERT model implemented in PyTorch from HuggingFace Hub.
# This is the model definition part.

from transformers import BertLMHeadModel, AutoConfig

config = AutoConfig.from_pretrained("bert-large-uncased")
bert = BertLMHeadModel(config)

# %%
# After we have the model defintion, we can create a default schedule `sch`.
# Later on, all the optimizations will be conducted on this schedule,
# and we do not need to directly modify the original model.
# The original module is stored in the :class:`~slapo.Schedule`, and can be accessed by `sch.mod`.

sch = slapo.create_schedule(bert)
print(sch.mod)

# %%
# From the above output, we can see that the original hierachical structure
# is preserved in Slapo schedule. Actually, since we have not added any
# optimizations, the model should be exactly the same as the vanilla PyTorch one.
# Users can leverage this structure to access the inner modules to
# conduct optimizations. For example, we can use the following code to access
# the first attention layer.

subsch = sch["bert.encoder.layer.0.attention"]
print(subsch.mod)

# %%
# The output submodule is just a part of the original one, but this helps users
# to quickly locate the submodule they need.
