import inspect
import pytest

import torch
import torch.fx as fx
import ms


def generate_concrete_args(model, input_names):
    sig = inspect.signature(model.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    return concrete_args


def test_hf_bert():
    from transformers import AutoConfig, BertModel

    config = AutoConfig.from_pretrained("bert-base-uncased")
    model = BertModel(config)
    input_names = list(model.dummy_inputs.keys())  # only has "input_ids"
    input_names += ["attention_mask", "token_type_ids"]  # "position_ids"
    concrete_args = generate_concrete_args(model, input_names)
    sch = ms.create_schedule(
        model,
        world_size=1,
        rank=0,
        tracer="huggingface",
        concrete_args=concrete_args,
    )

    # The original module list.
    assert isinstance(sch.get_module("encoder"), torch.nn.Module)

    # Traced layers.
    assert isinstance(sch.get_module("encoder.layer.0"), fx.GraphModule)
    assert isinstance(sch.get_module("encoder.layer.0.attention"), fx.GraphModule)
    # self will be renamed to self_m because it is a Python preserved keyword.
    assert isinstance(
        sch.get_module("encoder.layer.0.attention.self_m"), fx.GraphModule
    )
    assert isinstance(sch.get_module("encoder.layer.0.intermediate"), fx.GraphModule)
    assert isinstance(sch.get_module("encoder.layer.0.output"), fx.GraphModule)


def test_hf_gpt_neo():
    from transformers import AutoConfig, GPTNeoModel

    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
    model = GPTNeoModel(config)

    input_names = list(model.dummy_inputs.keys())
    input_names += ["attention_mask", "position_ids"]
    concrete_args = generate_concrete_args(model, input_names)
    sch = ms.create_schedule(
        model,
        world_size=1,
        rank=0,
        tracer="huggingface",
        concrete_args=concrete_args,
    )

    # The original module list.
    assert isinstance(sch.get_module("h"), torch.nn.Module)

    # Traced layers.
    assert isinstance(sch.get_module("h.0"), fx.GraphModule)
    assert isinstance(sch.get_module("h.0.attn"), fx.GraphModule)
    assert isinstance(sch.get_module("h.0.attn.attention"), fx.GraphModule)
    assert isinstance(sch.get_module("h.0.mlp"), fx.GraphModule)


def test_torchvision_wideresnet():
    from torchvision.models.resnet import ResNet, Bottleneck

    model = ResNet(Bottleneck, [6, 8, 4, 6], width_per_group=128)
    concrete_args = generate_concrete_args(model, ["x"])
    sch = ms.create_schedule(
        model,
        world_size=1,
        rank=0,
        tracer="pytorch",
        concrete_args=concrete_args,
    )

    assert isinstance(sch.get_module("layer1"), fx.GraphModule)
    for idx in range(6):
        assert isinstance(sch.get_module(f"layer1.{idx}"), fx.GraphModule)

    # Should not trace leaf.
    assert not isinstance(sch.get_module("layer1.0.conv1"), fx.GraphModule)

    # Should have "layer1.0.downsample" instead of "layer1.0.downsample.0"
    assert isinstance(sch.get_module("layer1.0.downsample"), fx.GraphModule)

    # Should not trace leaf.
    assert not isinstance(sch.get_module("layer1.0.downsample.0"), fx.GraphModule)


if __name__ == "__main__":
    pytest.main([__file__])
