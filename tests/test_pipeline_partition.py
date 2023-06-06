# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test pipeline partition related logic."""
# pylint: disable=duplicate-code
import inspect
import pytest
from mock import MagicMock

from torch import nn
from torch import fx

import slapo
from slapo.pipeline import generate_pipeline_partition


class Pre(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        # delibarately add control flow to avoid tracing
        if x.size() > 0:
            return self.linear(x)
        return x


class Post(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class SeqModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = Pre()
        self.layers = nn.Sequential(
            Inner(),
            Inner(),
        )
        self.post = Post()

    def forward(self, x):
        return self.post(self.layers(self.pre(x)))


class ModListModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = Pre()
        self.layers = nn.ModuleList([Inner(), Inner()])
        self.post = Post()

    def forward(self, x):
        x = self.pre(x)
        for layer in self.layers:
            x = layer(x)
        x = self.post(x)
        return x


def test_pipeline_partition():
    def test_model(model, is_module_list=False):
        sch = slapo.create_schedule(model)
        if not is_module_list:
            sch.trace_until("layers", tracer="pytorch")
        else:
            sch.trace_until("", tracer="pytorch")
        assert isinstance(sch.mod, fx.GraphModule)
        assert isinstance(sch["pre"].mod, nn.Module)
        if not is_module_list:
            assert isinstance(sch["layers"].mod, fx.GraphModule)
        assert isinstance(sch["layers.0"].mod, nn.Module)
        assert isinstance(sch["layers.1"].mod, nn.Module)
        if not is_module_list:
            assert isinstance(sch["post"].mod, fx.GraphModule)

        sch["layers.0"].cut_pipeline_stage()

        sch = generate_pipeline_partition(sch)
        # The traced pipeline module should include the pre and post modules.
        if not is_module_list:
            assert isinstance(sch["submod_0.pre"].mod, Pre)
            assert isinstance(sch["submod_0.submod_0"].mod, fx.GraphModule)
            assert isinstance(sch["submod_0.submod_0.0"].mod, Inner)
            assert isinstance(sch["submod_1.submod_1.1"].mod, Inner)
            assert isinstance(sch["submod_1.submod_1"].mod, fx.GraphModule)
            assert isinstance(sch["submod_1.post"].mod, fx.GraphModule)
        else:
            assert isinstance(sch["submod_0"].mod, fx.GraphModule)
            assert isinstance(sch["submod_0.pre"].mod, Pre)
            assert isinstance(sch["submod_0.layers_0"].mod, Inner)
            assert isinstance(sch["submod_1.layers_1"].mod, Inner)
            assert isinstance(sch["submod_1.post"].mod, Post)
            assert isinstance(sch["submod_1"].mod, fx.GraphModule)

    test_model(SeqModel(), is_module_list=False)
    test_model(ModListModel(), is_module_list=True)


def test_analyze_tie_weights():
    class Stage0(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(10, 10)
            self.linear = nn.Linear(10, 10, bias=False)

        def forward(self, x):
            return self.linear(self.wte(x))

    class StageN(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10, bias=False)

        def forward(self, x):
            return self.linear(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.stage0 = Stage0()
            self.stage1 = StageN()
            self.stage2 = StageN()

        def forward(self, x):
            return self.stage2(self.stage1(self.stage0(x)))

    with slapo.init_empty_weights():
        model = Model()
        # Tie weights
        model.stage1.linear.weight = model.stage0.wte.weight
        model.stage2.linear.weight = model.stage0.wte.weight

    # Analyze tie weights for a normal model.
    tie_weights = slapo.pipeline.analyze_tie_weights(model, False)

    assert len(tie_weights) == 1
    val = list(tie_weights.values())[0]
    assert len(val) == 3
    assert ("stage0.wte.weight", 0) in val
    assert ("stage1.linear.weight", 0) in val
    assert ("stage2.linear.weight", 0) in val

    # Analyze tie weights for a pipeline model. In this case,
    # the forward in top module only runs each pipeline stage sequentially.
    tie_weights = slapo.pipeline.analyze_tie_weights(model, True)

    assert len(tie_weights) == 1
    val = list(tie_weights.values())[0]
    assert len(val) == 3
    assert ("wte.weight", 0) in val
    assert ("linear.weight", 1) in val
    assert ("linear.weight", 2) in val


def test_deepspeed_analyze_tie_ranks():
    # Mock deepspeed.runtime.pipe.topology.PipeModelDataParallelTopology
    # This mocked topology assumes pp=4, tp=2, dp=1.
    topology = MagicMock()
    topology.filter_match = lambda pipe: [pipe * 2, pipe * 2 + 1]

    class Stage0(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(10, 10)
            self.linear = nn.Linear(10, 10, bias=False)

        def forward(self, x):
            return self.linear(self.wte(x))

    class StageN(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10, bias=False)

        def forward(self, x):
            return self.linear(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.stage0 = Stage0()
            self.stage1 = StageN()
            self.stage2 = StageN()
            self.stage3 = StageN()

        def forward(self, x):
            return self.stage3(self.stage2(self.stage1(self.stage0(x))))

    with slapo.init_empty_weights():
        model = Model()
        # Tie weights
        model.stage3.linear.weight = model.stage0.wte.weight
        model.stage2.linear.weight = model.stage1.linear.weight

    tie_weights = list(slapo.pipeline.analyze_tie_weights(model, True).values())
    (
        tie_ranks,
        tie_stages,
    ) = slapo.framework_dialect.deepspeed.pipeline.analyze_tie_ranks(
        tie_weights, topology
    )

    # Expected tie_ranks (order may vary): [[[0, 6], [1, 7]], [[2, 4], [3, 5]]]
    assert len(tie_ranks) == 2
    assert len(tie_ranks[0]) == 2
    assert len(tie_ranks[0][0]) == 2
    assert [[0, 6], [1, 7]] in tie_ranks
    assert [[2, 4], [3, 5]] in tie_ranks

    # Expected tie_stages (order should be the same as tie_ranks): [[0, 3], [1, 2]]
    assert len(tie_stages) == 2
    assert len(tie_stages[0]) == 2
    assert tie_stages[tie_ranks.index([[0, 6], [1, 7]])] == [0, 3]
    assert tie_stages[tie_ranks.index([[2, 4], [3, 5]])] == [1, 2]


def get_traced_gpt_neo_model():
    from transformers import GPTNeoModel, AutoConfig

    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-1.3B")
    config.use_cache = False
    with slapo.init_empty_weights():
        model = GPTNeoModel(config)
    sch = slapo.create_schedule(model)
    input_names = ["input_ids", "attention_mask", "position_ids"]
    sig = inspect.signature(sch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    sch.trace_until("", tracer="huggingface", concrete_args=concrete_args)
    return sch


def get_all_placeholders(gm):
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            yield node.name


def test_gpt_neo_two_stages():
    """
    Expected output:

    def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor, position_ids : torch.Tensor):
        # Input: (input_ids, position_ids, attention_mask)
        #        Notice the order of attention_mask and position_ids is swapped.
        #        Also, position_ids is only used in the first stage.
        #        https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L579
        submod_0 = self.submod_0(input_ids, position_ids, attention_mask);  input_ids = position_ids = attention_mask = None
        getitem = submod_0[0]
        getitem_1 = submod_0[1]
        getitem_2 = submod_0[2];  submod_0 = None

        # Input: (h_11, mul, add_1)
        #        (hidden_states, attention_mask, output_shape)
        # output_shape defined here: (before calling the Modulelist)
        #     https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L588
        #     and used here: (in the last stage)
        #     https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L639
        submod_1 = self.submod_1(getitem, getitem_1, getitem_2);  getitem = getitem_1 = getitem_2 = None
        return {'last_hidden_state': submod_1}
    """
    sch = get_traced_gpt_neo_model()
    sch["h.11"].cut_pipeline_stage()
    sch = generate_pipeline_partition(sch)
    assert list(get_all_placeholders(sch.mod.submod_0)) == [
        "input_ids",
        "position_ids",
        "attention_mask",
    ]
    assert list(get_all_placeholders(sch.mod.submod_1)) == ["h_11", "mul", "add_1"]


def test_gpt_neo_four_stages():
    """
    Expected output:

    def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor, position_ids : torch.Tensor):
        # Input: (input_ids, position_ids, attention_mask)
        #        Notice the order of attention_mask and position_ids is swapped.
        #        Also, position_ids is only used in the first stage.
        #        https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L579
        submod_0 = self.submod_0(input_ids, position_ids, attention_mask);  input_ids = position_ids = attention_mask = None
        getitem = submod_0[0]
        getitem_1 = submod_0[1]
        getitem_2 = submod_0[2];  submod_0 = None

        # Input: (h_5, mul)
        #        (hidden_states, attention_mask)
        submod_1 = self.submod_1(getitem, getitem_1);  getitem = None

        # Input: (h_11, mul)
        #        (hidden_states, attention_mask)
        submod_2 = self.submod_2(submod_1, getitem_1);  submod_1 = None

        # Input: (h_17, mul, add_1)
        #        (hidden_states, attention_mask, output_shape)
        # output_shape defined here: (before calling the Modulelist)
        #     https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L588
        #     and used here: (in the last stage)
        #     https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L639
        submod_3 = self.submod_3(submod_2, getitem_1, getitem_2);  submod_2 = getitem_1 = getitem_2 = None
        return {'last_hidden_state': submod_1}
    """
    sch = get_traced_gpt_neo_model()
    sch["h.5"].cut_pipeline_stage()
    sch["h.11"].cut_pipeline_stage()
    sch["h.17"].cut_pipeline_stage()
    sch = generate_pipeline_partition(sch)
    assert list(get_all_placeholders(sch.mod.submod_0)) == [
        "input_ids",
        "position_ids",
        "attention_mask",
    ]
    assert list(get_all_placeholders(sch.mod.submod_1)) == ["h_5", "mul"]
    assert list(get_all_placeholders(sch.mod.submod_2)) == ["h_11", "mul"]
    assert list(get_all_placeholders(sch.mod.submod_3)) == ["h_17", "mul", "add_1"]


if __name__ == "__main__":
    pytest.main([__file__])
