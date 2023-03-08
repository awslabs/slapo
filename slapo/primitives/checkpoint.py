# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint primitive."""
# pylint: disable=arguments-differ

from torch import nn

from .base import register_primitive, Primitive
from ..checkpoint import checkpoint as checkpoint_module


@register_primitive()
class CheckpointPrimitive(Primitive):
    """Add activation checkpointing to the entire module or a subgraph.

    Parameters
    ----------
    subgraph : Optional[List[List[torch.fx.Node]]]
        The subgraph to be checkpointed. If None, checkpoint the entire module.
    order_args_fn : Optional[Callable]
        A function to order the position and keyword arguments of the module.
        The function should take the same arguments as the module forward function,
        and return a list or tuple of the ordered arguments.
        If None, the arguments will be ordered by the order of the position arguments,
        followed by the key order of the keyword arguments.
    """

    @staticmethod
    def name():
        return "checkpoint"

    @staticmethod
    def apply(sch, subgraph=None, order_args_fn=None):
        class CheckPointWrapper(nn.Module):
            def __init__(self, mod) -> None:
                super().__init__()
                self.traceable = False
                self.mod = mod

            def forward(self, *args, **kwargs):
                ordered_args = []
                if order_args_fn is None:
                    ordered_args = list(args)
                    for value in kwargs.values():
                        ordered_args += [value]
                else:
                    ordered_args = order_args_fn(*args, **kwargs)

                # Note: checkpoint cannot accept kwargs
                return checkpoint_module(self.mod, *ordered_args)

        if subgraph is None:
            # Checkpoint the entire module.
            sch.replace(CheckPointWrapper(sch.mod))
        else:
            # Checkpoint the subgraph
            new_gm = sch._construct_fx_graph(subgraph[0])
            sch.replace(CheckPointWrapper(new_gm), subgraph)
