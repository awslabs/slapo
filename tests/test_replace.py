# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test replace primitives."""
# pylint: disable=comparison-with-callable, unused-argument

import math
import operator
import pytest

import torch
from torch import nn
import torch.nn.functional as F

import slapo
from slapo.utils.common import get_hooks
from slapo.pattern import call_module


def test_replace_single_module():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1024, 1024)
            self.activation = nn.ReLU()

        def forward(self, x):
            x = self.linear(x)
            x = self.activation(x)
            return x

    model = Model()
    sch = slapo.create_schedule(model)
    new_act = nn.GELU()
    sch["activation"].replace(new_act)
    assert isinstance(sch["activation"].mod, nn.GELU)


def test_replace_all_module():
    class SubMod(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1024, 1024)
            self.activation = nn.ReLU()

        def forward(self, x):
            x = self.linear(x)
            x = self.activation(x)
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1024, 1024)
            self.act1 = nn.ReLU()
            self.fc2 = nn.Linear(1024, 1024)
            self.act2 = nn.ReLU()
            self.submod = SubMod()

        def forward(self, x):
            x = self.fc1(x)
            x = self.act1(x)
            x = self.fc2(x)
            x = self.act2(x)
            x = self.submod(x)
            return x

    model = Model()
    sch = slapo.create_schedule(model)

    def make_gelu(name, mod):
        return nn.GELU()

    sch.replace_all(nn.ReLU, make_gelu)
    assert isinstance(sch["act1"].mod, nn.GELU)
    assert isinstance(sch["act2"].mod, nn.GELU)
    assert isinstance(sch["submod.activation"].mod, nn.GELU)

    # test giving different shape of parameters
    def make_linear(name, mod):
        if name == "fc1":
            in_feat, out_feat = 1024, 1025
        elif name == "fc2":
            in_feat, out_feat = 1025, 1026
        else:
            in_feat, out_feat = 1026, 1027
        return nn.Linear(in_feat, out_feat)

    sch.replace_all(nn.Linear, make_linear)
    assert sch["fc1"].mod.out_features == 1025
    assert sch["fc2"].mod.out_features == 1026
    assert sch["submod.linear"].mod.out_features == 1027


def test_replace_all_with_seq():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1024, 1024)
            self.fc2 = nn.Linear(1024, 1024)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    model = Model()
    sch = slapo.create_schedule(model)

    # Test if Slapo can
    # 1. avoid recursive replacement
    # 2. accept a kwarg
    def make_seq(name, mod, is_relu=False):
        if is_relu:
            seq = nn.Sequential(mod, nn.ReLU())
        else:
            raise NotImplementedError
        return seq

    sch.replace_all(nn.Linear, make_seq, is_relu=True)
    assert getattr(sch["fc1"].mod, "0").out_features == 1024
    assert isinstance(getattr(sch["fc1"].mod, "1"), nn.ReLU)
    assert getattr(sch["fc2"].mod, "0").out_features == 1024


def test_replace_all_module_list():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.mod = nn.ModuleList(
                [nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU()]
            )
            self.seq = nn.Sequential(
                nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU()
            )

        def forward(self, x):
            for mod in self.mod:
                x = mod(x)
            x = self.seq(x)
            return x

    model = Model()
    sch = slapo.create_schedule(model)

    def make_linear(name, mod):
        return nn.Linear(512, 512)

    sch.replace_all(nn.Linear, make_linear)
    assert sch["mod.0"].mod.out_features == 512
    assert sch["mod.2"].mod.out_features == 512
    assert sch["seq.0"].mod.out_features == 512
    assert sch["seq.2"].mod.out_features == 512


def test_vertical_replacement():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1024, 1024)
            self.bn = nn.BatchNorm1d(1024)
            self.activation = nn.ReLU()

        def forward(self, x):
            x = self.linear(x)
            x = self.bn(x)
            x = self.activation(x)
            return x

    model = Model()
    sch = slapo.create_schedule(model)

    def fwd_pre_hook(mod, inp):
        return inp

    def fwd_post_hook(mod, inp, out):
        return out

    def bwd_post_hook(mod, grad_inp, grad_out):
        return grad_inp

    # Directly register hooks instead of using .sync, because
    # we do not want to test .sync in this test.
    sch.mod.linear.register_forward_pre_hook(fwd_pre_hook)
    sch.mod.linear.register_backward_hook(bwd_post_hook)

    def pattern(x):
        x = call_module("linear", x)
        x = call_module("bn", x)
        return x

    subgraph = sch.find(pattern)
    assert len(subgraph) == 1
    assert len(subgraph[0]) == 2

    class Identity(nn.Module):
        def forward(self, x):
            return x

    mod = Identity()
    sch.replace(mod, subgraph)
    assert isinstance(sch["Identity_0"].mod, Identity)

    # test valid hooks
    all_hooks = get_hooks(sch["Identity_0"].mod)
    assert len(all_hooks["fwd_pre"]) == 1
    assert len(all_hooks["bwd_post"]) == 1

    # test naming
    model = Model()
    sch = slapo.create_schedule(model)
    subgraph = sch.find(pattern)
    sch.replace(mod, subgraph, name="test")
    assert isinstance(sch["test_0"].mod, Identity)

    # test invalid hooks
    model = Model()
    sch = slapo.create_schedule(model)
    sch.mod.linear.register_forward_hook(fwd_post_hook)
    subgraph = sch.find(pattern)
    with pytest.raises(Exception):
        sch.replace(mod, subgraph)


def test_horizontal_replacement():
    class CoreAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 32)
            self.fc2 = nn.Linear(32, 32)

        def permute(self, x):
            return x.permute(0, 2, 1, 3)

        def forward(self, x):
            y1 = self.permute(self.fc1(x))
            y2 = self.permute(self.fc2(x))
            return y1 + y2

    attn = CoreAttention()
    sch = slapo.create_schedule(attn)

    def pattern(x):
        x = call_module(r"fc1|fc2", x)
        return x.permute(0, 2, 1, 3)

    subgraph = sch.find(pattern)
    assert len(subgraph) == 2
    assert len(subgraph[-1]) == 2
    assert subgraph[0][0][1].target == "fc1"
    assert subgraph[1][0][1].target == "fc2"
    assert subgraph[0][1][1].target == "permute"

    class Identity(nn.Module):
        def forward(self, x):
            return (x, x)

    mod = Identity()
    sch.replace(mod, subgraph, name="test")
    assert isinstance(sch["test_0"].mod, Identity)
    cnt = 0
    for node in sch.mod.graph.nodes:
        if node.target == operator.getitem and node.args[0].target == "test_0":
            cnt += 1
    assert cnt == 2

    # test invalid hooks
    attn = CoreAttention()
    sch = slapo.create_schedule(attn)

    def fwd_pre_hook(mod, inp):
        return inp

    # Directly register hooks instead of using .sync, because
    # we do not want to test .sync in this test.
    sch.mod.fc1.register_forward_pre_hook(fwd_pre_hook)
    subgraph = sch.find(pattern)
    with pytest.raises(Exception):
        sch.replace(mod, subgraph)


def test_mismatched_arguments():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1024, 1024)
            self.bn = nn.BatchNorm1d(1024)
            self.activation = nn.ReLU()

        def forward(self, x):
            y = self.linear(x)
            z = self.activation(y) + x
            return z

    model = Model()
    sch = slapo.create_schedule(model)

    def pattern(x):
        return F.relu(x) + x

    subgraphs = sch.find(pattern)
    assert len(subgraphs) == 1
    assert len(subgraphs[0]) == 2

    class NewMod(nn.Module):
        def forward(self, x, y, z):
            return x + y + z

    mod = NewMod()
    with pytest.raises(Exception):
        sch.replace(mod, subgraphs)
    sch.replace(mod, subgraphs, concrete_args={"z": 0})
    assert isinstance(sch["NewMod_1"].mod, nn.Module)


def test_replace_function():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1024, 1024)

        def forward(self, x):
            x = self.linear(x)
            x = F.relu(x)
            return x

    def identity(x):
        return x

    model = Model()
    sch = slapo.create_schedule(model)
    nodes = sch.find_node(
        lambda node: node.op == "call_function" and node.target == F.relu
    )
    assert len(nodes) == 1
    sch.replace(identity, nodes[0])
    cnt = 0
    for node in sch.mod.graph.nodes:
        if node.op == "call_function" and node.target == identity:
            cnt += 1
    assert cnt == 1


def test_transfer_hook():
    """Test whether the hooks are transferred to the new replaced module."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            return self.linear(x)

    model = Model()

    def fwd_pre_hook(mod, inp):
        return inp

    def fwd_post_hook(mod, inp, out):
        return out

    def bwd_post_hook(mod, grad_inp, grad_out):
        return grad_inp

    sch = slapo.create_schedule(model)

    # Directly register hooks instead of using .sync, because
    # we do not want to test .sync in this test.
    sch.mod.register_forward_pre_hook(fwd_pre_hook)
    sch.mod.register_forward_hook(fwd_post_hook)
    sch.mod.register_backward_hook(bwd_post_hook)
    all_hooks = get_hooks(sch.mod)
    assert len(all_hooks["fwd_pre"]) == 1
    assert len(all_hooks["fwd_post"]) == 1
    assert len(all_hooks["bwd_post"]) == 1

    sch.trace()

    all_hooks = get_hooks(sch.mod)
    assert len(all_hooks["fwd_pre"]) == 1
    assert len(all_hooks["fwd_post"]) == 1
    assert len(all_hooks["bwd_post"]) == 1


def get_traced_bert():
    from transformers import BertLMHeadModel, AutoConfig
    import inspect
    from torch import fx

    config = AutoConfig.from_pretrained("bert-large-uncased")
    with slapo.init_empty_weights():
        model = BertLMHeadModel(config)

    sch = slapo.create_schedule(model)
    input_names = ["hidden_states"]
    subsch = sch["bert.encoder.layer.0.attention.self"]
    sig = inspect.signature(subsch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    subsch.trace(
        recursive=False, flatten=True, tracer="pytorch", concrete_args=concrete_args
    )
    assert isinstance(subsch.mod, fx.GraphModule)
    return subsch, config


def test_qkv():
    subsch, _ = get_traced_bert()

    def pattern(x):
        x = call_module(r"(query|key|value)", x)
        new_shape = x.size()[:-1] + (16, -1)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    qkv_subgraphs = subsch.find(pattern)
    assert len(qkv_subgraphs) == 3
    for subgraph in qkv_subgraphs:
        assert len(subgraph) == 6
    # make sure matching different nodes
    assert qkv_subgraphs[0][0][1].target == "query"
    assert qkv_subgraphs[0][1][1].name == "size"
    assert qkv_subgraphs[1][0][1].target == "key"
    assert qkv_subgraphs[1][1][1].name == "size_1"
    assert qkv_subgraphs[2][0][1].target == "value"
    assert qkv_subgraphs[2][1][1].name == "size_2"

    def qkv(x):
        return (x, x, x)

    subsch.replace(qkv, qkv_subgraphs)
    cnt = 0
    for node in subsch.mod.graph.nodes:
        if node.target == operator.getitem and node.args[0].target == qkv:
            cnt += 1
    assert cnt == 3

    subsch_1, _ = get_traced_bert()
    qkv_subgraphs_1 = subsch_1.find(pattern)

    class Identity(nn.Module):
        def forward(self, x):
            return (x, x, x)

    subsch_1.replace(Identity(), qkv_subgraphs_1)
    cnt = 0
    for node in subsch_1.mod.graph.nodes:
        if node.target == operator.getitem and node.args[0].target == "Identity_0":
            cnt += 1
    assert cnt == 3


def test_efficient_attn():
    subsch, config = get_traced_bert()

    # https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html
    def scaled_dot_product(query_layer, key_layer, value_layer):
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            config.hidden_size // config.num_attention_heads
        )
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = F.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        return context_layer

    subgraphs = subsch.find(scaled_dot_product)
    assert len(subgraphs[0]) == 6
    inp = torch.randn(1, 512, 1024)
    # Test replacing with a function
    # query, key, value argument orders are different
    # should be able to replace, but the results are incorrect
    with pytest.raises(AssertionError):
        with slapo.Verify(subsch, [inp]):
            subsch.replace(F.scaled_dot_product_attention, subgraphs)

    subsch_1, _ = get_traced_bert()
    subgraphs_1 = subsch_1.find(scaled_dot_product)
    assert len(subgraphs_1[0]) == 6

    class EfficientAttention(torch.nn.Module):
        def forward(self, key_layer, query_layer, value_layer):
            return F.scaled_dot_product_attention(query_layer, key_layer, value_layer)

    # Test replacing with a module
    with slapo.Verify(subsch_1, [inp]):
        subsch_1.replace(EfficientAttention(), subgraphs_1)


if __name__ == "__main__":
    pytest.main([__file__])
