# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLP module using with fused kernels."""
from __future__ import annotations

import torch
from .linear import LinearWithAct, LinearWithDropout


class FusedMLP(torch.nn.Module):
    """A wrapper MLP to make use of fused bias+gelu and bias+dropout.
    Note that both linear modules in this MLP have bias, so users should
    not replace the original MLP with this module if the original MLP
    does not have bias.
    """

    def __init__(self, hidden_size, intermediate_size, orig_act, resid_pdrop):
        super().__init__()
        self.fc_in = LinearWithAct(hidden_size, intermediate_size, orig_act)
        self.fc_out = LinearWithDropout(intermediate_size, hidden_size, p=resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        return hidden_states
