# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fusion related primitives."""
# pylint: disable=arguments-differ

import types
import torch
from torch import nn

from .base import register_primitive, Primitive
from ..initialization import init_empty_weights
from ..op import LinearWithSeparateBias


@register_primitive()
class FusePrimitive(Primitive):
    """Fuse a subgraph into a single module using a backend compiler.

    Parameters
    ----------
    subgraph : List[List[torch.fx.Node]]
        The subgraph to be fused.
    compiler : Union[None, str]
        The backend compiler to be used. Currently only support "TorchScript".
    name : str
        The name of the fused module.
    """

    @staticmethod
    def name():
        return "fuse"

    @staticmethod
    def apply(sch, subgraph, compiler="TorchScript", name="FusedModule"):
        assert (
            len(subgraph) == 1 and len(subgraph[0]) > 1
        ), f"Only vertical fusion is supported. Got subgraph: {subgraph}"
        if compiler == "TorchScript":
            new_gm = sch._construct_fx_graph(subgraph[0])
            new_mod = torch.jit.script(new_gm)
            sch.replace(new_mod, subgraph, name)
        elif compiler is None:
            mod_list = []
            for _, node in subgraph[0]:
                mod_list.append(sch.get_module(node.target))
            fused_class = types.new_class(name, (nn.Sequential,))
            fused_mod = fused_class(*mod_list)
            sch.replace(fused_mod, subgraph, name)
        else:
            raise ValueError(
                f"Unsupported compiler: {compiler}. Only support TorchScript as the backend compiler for now"
            )


@register_primitive()
class DecomposePrimitive(Primitive):
    """Decompose a module. Currently only support decomposing a linear layer.
    Specifically, this primitive replaces a linear layer with LinearWithSeparateBias,
    which computes the bias separately from the weight. This is useful when we want
    to fuse the bias addition into the following logic (e.g., activation function).

    Parameters
    ----------
    mod : torch.nn.Module
        The module to be decomposed.
    """

    @staticmethod
    def name():
        return "decompose"

    @staticmethod
    def apply(sch):
        if not isinstance(sch.mod, nn.Linear):
            raise RuntimeError(
                "Can only support decomposing a `nn.Linear` layer for now"
            )
        if (
            sch.mod.weight.shape[1] != sch.mod.in_features
            or sch.mod.weight.shape[0] != sch.mod.out_features
        ):
            raise RuntimeError(".shard() should be applied after .decompose()")
        # Replace the linear module
        with init_empty_weights(enable=(sch.mod.weight.device == torch.device("meta"))):
            new_mod = LinearWithSeparateBias(
                sch.mod.weight.shape[1],
                sch.mod.weight.shape[0],
                device=sch.mod.weight.device,
                dtype=sch.mod.weight.dtype,
            )
            # Use original value
            new_mod.weight = sch.mod.weight
            new_mod.bias = sch.mod.bias
            sch.replace(new_mod)
