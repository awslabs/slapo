# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test trace primitives."""

import inspect

import pytest
import torch
from torch import fx

import slapo


def generate_concrete_args(model, input_names):
    """Generate concrete args for tracing."""
    sig = inspect.signature(model.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    return concrete_args


def test_hf_bert():
    """Test tracing HF bert model."""
    from transformers import AutoConfig, BertModel

    config = AutoConfig.from_pretrained("bert-base-uncased")
    model = BertModel(config)
    sch = slapo.create_schedule(model)

    # The original module list.
    assert isinstance(sch["encoder"].mod, torch.nn.Module)
    assert isinstance(sch["encoder.layer.0"].mod, torch.nn.Module)
    assert isinstance(sch["encoder.layer.0.attention"].mod, torch.nn.Module)

    sub_sch = sch["encoder.layer.0.attention"]
    input_names = ["hidden_states", "attention_mask"]
    sig = inspect.signature(sub_sch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    sub_sch.trace(tracer="pytorch", concrete_args=concrete_args)

    # Only the traced submodules are graph modules.
    assert isinstance(sch["encoder.layer.0.attention"].mod, fx.GraphModule)
    assert isinstance(sch["encoder.layer.0.attention.self"].mod, fx.GraphModule)
    assert isinstance(sch["encoder.layer.0.attention.output"].mod, fx.GraphModule)

    # Other modules remain the same.
    assert isinstance(sch["encoder.layer.0.intermediate"].mod, torch.nn.Module)
    assert isinstance(sch["encoder.layer.0.output"].mod, torch.nn.Module)


def test_hf_gpt_neo():
    """Test tracing HF gpt-neo model."""
    from transformers import AutoConfig, GPTNeoModel

    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
    model = GPTNeoModel(config)

    sch = slapo.create_schedule(model)

    # The original module list.
    assert isinstance(sch["h.0"].mod, torch.nn.Module)

    # Traced layers.
    sub_sch = sch["h.0"]
    input_names = ["hidden_states", "attention_mask"]
    sig = inspect.signature(sub_sch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    sub_sch.trace(tracer="pytorch", concrete_args=concrete_args)
    assert isinstance(sch["h.0"].mod, fx.GraphModule)
    assert isinstance(sch["h.0.attn"].mod, fx.GraphModule)
    assert isinstance(sch["h.0.mlp"].mod, fx.GraphModule)

    # Attention submodule cannot be traced.
    assert isinstance(sch["h.0.attn.attention"].mod, torch.nn.Module)


def test_torchvision_wideresnet():
    """Test tracing torchvision wideresnet model."""
    from torchvision.models.resnet import Bottleneck, ResNet

    model = ResNet(Bottleneck, [6, 8, 4, 6], width_per_group=128)
    concrete_args = generate_concrete_args(model, ["x"])
    sch = slapo.create_schedule(model)
    sch.trace(tracer="pytorch", concrete_args=concrete_args)

    assert isinstance(sch["layer1"].mod, fx.GraphModule)
    for idx in range(6):
        assert isinstance(sch.get_module(f"layer1.{idx}"), fx.GraphModule)

        # Should not trace leaf.
        assert not isinstance(sch[f"layer1.{idx}.conv1"].mod, fx.GraphModule)

    # Should have "layer1.0.downsample" instead of "layer1.0.downsample.0"
    assert isinstance(sch["layer1.0.downsample"].mod, fx.GraphModule)

    # Should not trace leaf.
    assert not isinstance(sch["layer1.0.downsample.0"].mod, fx.GraphModule)


def find_module_in_graph(graph, target):
    return any(
        node.op == "call_module" and node.target == target for node in graph.nodes
    )


def test_flattened_hf_bert():
    """Test tracing HF bert model using flattened mode"""
    from transformers import AutoConfig, BertModel

    config = AutoConfig.from_pretrained("bert-base-uncased")
    model = BertModel(config)
    sch = slapo.create_schedule(model)

    sub_sch = sch["encoder.layer.0.attention"]
    # Before tracing into a flattened graph, the inner submodules can still be accessed
    assert isinstance(sch["encoder.layer.0.attention.self.query"].mod, torch.nn.Module)

    input_names = ["hidden_states", "attention_mask"]
    sig = inspect.signature(sub_sch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    sub_sch.trace(tracer="pytorch", flatten=True, concrete_args=concrete_args)
    assert isinstance(sch["encoder.layer.0.attention"].mod, fx.GraphModule)
    assert isinstance(sch["encoder.layer.0.attention.self"].mod, torch.nn.Module)
    # After tracing, the submodules are no longer encapsulated in a schedule
    with pytest.raises(Exception):
        print(sch["encoder.layer.0.attention.self.query"].mod)
    assert isinstance(
        sch["encoder.layer.0.attention.self"].get_module("query"), torch.nn.Module
    )
    assert find_module_in_graph(sub_sch.mod.graph, "self.query")
    assert find_module_in_graph(sub_sch.mod.graph, "output.dense")


def test_two_level_flattened_hf_bert():
    """Test tracing HF bert model using flattened mode"""
    from transformers import AutoConfig, BertModel

    config = AutoConfig.from_pretrained("bert-base-uncased")
    model = BertModel(config)
    sch = slapo.create_schedule(model)

    sub_sch = sch["encoder.layer.0"]
    # Before tracing into a flattened graph, the inner submodules can still be accessed
    assert isinstance(sch["encoder.layer.0.attention.self.query"].mod, torch.nn.Module)

    input_names = ["hidden_states", "attention_mask"]
    sig = inspect.signature(sub_sch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    sub_sch.trace(
        tracer="pytorch",
        flatten=True,
        concrete_args=concrete_args,
        leaf_modules=["BertSelfAttention", "BertSelfOutput"],
    )
    assert isinstance(sch["encoder.layer.0.attention"].mod, torch.nn.Module)
    # After tracing, the submodules are no longer encapsulated in a schedule
    with pytest.raises(Exception):
        assert isinstance(sch["encoder.layer.0.attention.self"].mod, torch.nn.Module)
    assert find_module_in_graph(sub_sch.mod.graph, "attention.self")
    assert find_module_in_graph(sub_sch.mod.graph, "attention.output")
    # Only two levels are flattened, and other submodules are specified as leaf
    assert not find_module_in_graph(sub_sch.mod.graph, "attention.self.query")


if __name__ == "__main__":
    pytest.main([__file__])
