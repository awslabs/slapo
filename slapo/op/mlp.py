# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLP module using with fused kernels."""
from __future__ import annotations

import torch
from .fused_bias import FusedBiasGELU, FusedBiasNewGELU, FusedBiasDropout


class FusedMLP(torch.nn.Module):
    """A wrapper MLP to make use of fused bias+gelu and bias+dropout.
    Note that both linear modules in this MLP have bias, so users should
    not replace the original MLP with this module if the original MLP
    does not have bias.
    """

    def __init__(self, hidden_size, intermediate_size, orig_act, resid_pdrop):
        super().__init__()
        if orig_act == "gelu":
            self.fc_in = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
            self.act = FusedBiasGELU(intermediate_size, prev_weight=self.fc_in.weight)
        elif orig_act == "gelu_new":
            self.fc_in = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
            self.act = FusedBiasNewGELU(
                intermediate_size, prev_weight=self.fc_in.weight
            )
        else:
            raise NotImplementedError(f"Unsupported activation: {orig_act}")

        self.fc_out = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        self.bias_dropout = FusedBiasDropout(
            hidden_size, resid_pdrop, prev_weight=self.fc_out.weight
        )

    def forward(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.bias_dropout(hidden_states)
        return hidden_states
