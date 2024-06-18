# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test DeepSpeed Pipeline."""
import os
import inspect

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import deepspeed

import slapo
from slapo.framework_dialect.deepspeed.pipeline import (
    get_ds_config,
    create_dist_group_for_pipeline,
)
from slapo.logger import get_logger

logger = get_logger()


class LinearReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        return x


class Model(nn.Module):
    def __init__(self, num_layers=12, has_relu=False):
        super().__init__()
        if not has_relu:
            self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(num_layers)])
        else:
            self.layers = nn.ModuleList([LinearReLU() for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_pipeline_2stages_pp_dp():
    deepspeed.init_distributed(dist_backend="nccl")
    if dist.get_world_size() != 4:
        logger.info("This test requires 4 GPUs.")
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    with slapo.init_empty_weights():
        model = Model(12)
    # If total number of devices is 4, then num_dp = 2
    num_pp = 2
    num_mp = 1
    num_dp = dist.get_world_size() // (num_pp * num_mp)
    topology, group = create_dist_group_for_pipeline(num_pp=num_pp, num_mp=num_mp)
    sch = slapo.create_schedule(model, group=group)
    sch.trace_until("")

    bs = 8
    ds_config_dict = get_ds_config(
        batch_size=bs,
        micro_batch_size_per_gpu=bs // num_dp,
        fp16=False,
    )
    inp = torch.randn(bs, 10, device=dist.get_rank())
    label = torch.randint(0, 10, (bs,), dtype=torch.long, device=dist.get_rank())
    with slapo.Verify(
        sch,
        example_inputs=[inp],
        example_outputs=label,
        loss_fn=F.cross_entropy,
        topology=topology,
        config=ds_config_dict,
    ):
        sch["layers.5"].cut_pipeline_stage()


def test_pipeline_2stages_pp_tp():
    deepspeed.init_distributed(dist_backend="nccl")
    if dist.get_world_size() != 4:
        logger.info("This test requires 4 GPUs.")
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    with slapo.init_empty_weights():
        model = Model(2, has_relu=True)
    num_pp = 2
    num_mp = 2
    num_dp = dist.get_world_size() // (num_pp * num_mp)
    topology, group = create_dist_group_for_pipeline(num_pp=num_pp, num_mp=num_mp)
    sch = slapo.create_schedule(model, group=group)
    sch.trace_until("")

    bs = 8
    ds_config_dict = get_ds_config(
        batch_size=bs,
        micro_batch_size_per_gpu=bs // num_dp,
        fp16=False,
    )
    inp = torch.randn(bs, 10, device=dist.get_rank())
    label = torch.randint(0, 10, (bs,), dtype=torch.long, device=dist.get_rank())
    with slapo.Verify(
        sch,
        example_inputs=[inp],
        example_outputs=label,
        loss_fn=F.cross_entropy,
        topology=topology,
        config=ds_config_dict,
    ):
        # TODO: Fix layers.0/1 indexing bug
        sch["layers.0.linear"].shard("weight", axis=0)
        sch["layers.0.linear"].shard("bias", axis=0)
        sch["layers.1.linear"].shard("weight", axis=1)
        sch["layers.1.linear"].sync("fwd_post", sync_op_or_fn="all_reduce")
        sch["layers.0.linear"].sync("bwd_post", sync_op_or_fn="all_reduce")
        sch["layers.0"].cut_pipeline_stage()


def test_pipeline_4stages_pp():
    deepspeed.init_distributed(dist_backend="nccl")
    if dist.get_world_size() != 4:
        logger.info("This test requires 4 GPUs.")
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    with slapo.init_empty_weights():
        model = Model(12)
    topology, group = create_dist_group_for_pipeline(num_pp=4, num_mp=1)
    sch = slapo.create_schedule(model, group=group)
    sch.trace_until("")

    bs = 8
    ds_config_dict = get_ds_config(
        batch_size=bs,
        micro_batch_size_per_gpu=bs,
        fp16=False,
    )
    inp = torch.randn(bs, 10, device=dist.get_rank())
    label = torch.randint(0, 10, (bs,), dtype=torch.long, device=dist.get_rank())
    with slapo.Verify(
        sch,
        example_inputs=[inp],
        example_outputs=label,
        loss_fn=F.cross_entropy,
        topology=topology,
        config=ds_config_dict,
    ):
        sch["layers.2"].cut_pipeline_stage()
        sch["layers.5"].cut_pipeline_stage()
        sch["layers.8"].cut_pipeline_stage()


def test_pipeline_2stages_pp_tp_dp():
    deepspeed.init_distributed(dist_backend="nccl")
    if dist.get_world_size() != 8:
        logger.info("This test requires 8 GPUs.")
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    with slapo.init_empty_weights():
        model = Model(2, has_relu=True)
    num_pp = 2
    num_mp = 2
    num_dp = dist.get_world_size() // (num_pp * num_mp)
    topology, group = create_dist_group_for_pipeline(num_pp=num_pp, num_mp=num_mp)
    sch = slapo.create_schedule(model, group=group)
    sch.trace_until("")

    bs = 8
    ds_config_dict = get_ds_config(
        batch_size=bs,
        micro_batch_size_per_gpu=bs // num_dp,
        fp16=False,
    )
    inp = torch.randn(bs, 10, device=dist.get_rank())
    label = torch.randint(0, 10, (bs,), dtype=torch.long, device=dist.get_rank())
    with slapo.Verify(
        sch,
        example_inputs=[inp],
        example_outputs=label,
        loss_fn=F.cross_entropy,
        topology=topology,
        config=ds_config_dict,
    ):
        # TODO: Fix layers.0/1 indexing bug
        sch["layers.0.linear"].shard("weight", axis=0)
        sch["layers.0.linear"].shard("bias", axis=0)
        sch["layers.1.linear"].shard("weight", axis=1)
        sch["layers.1.linear"].sync("fwd_post", sync_op_or_fn="all_reduce")
        sch["layers.0.linear"].sync("bwd_post", sync_op_or_fn="all_reduce")
        sch["layers.0"].cut_pipeline_stage()


def test_bert_2stages_pp():
    deepspeed.init_distributed(dist_backend="nccl")
    if dist.get_world_size() != 2:
        logger.info("This test requires 2 GPUs.")
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    from transformers import BertLMHeadModel, AutoConfig

    config = AutoConfig.from_pretrained("bert-large-uncased")
    # config.tie_word_embeddings = False
    with slapo.init_empty_weights():
        model = BertLMHeadModel(config)

    num_pp = 2
    num_mp = 1
    num_dp = dist.get_world_size() // (num_pp * num_mp)
    topology, group = create_dist_group_for_pipeline(num_pp=num_pp, num_mp=num_mp)
    sch = slapo.create_schedule(model, group=group)
    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    sig = inspect.signature(sch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }

    loss_fct = nn.CrossEntropyLoss()

    def loss_fn(outputs, labels):
        # (bs, seq, vocab)
        if isinstance(outputs, torch.Tensor):
            # DS PP output has already removed the data structure
            prediction_scores = outputs
        else:
            prediction_scores = outputs["logits"]
        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        lm_loss = loss_fct(
            shifted_prediction_scores.view(-1, config.vocab_size), labels.view(-1)
        )
        return lm_loss

    bs = 2
    seq_len = 512
    micro_bs = bs // num_dp
    ds_config_dict = get_ds_config(
        batch_size=bs,
        micro_batch_size_per_gpu=micro_bs,
        fp16=False,
    )
    device = "cuda"
    input_ids = torch.ones(micro_bs, seq_len, dtype=torch.long, device=device)
    attention_mask = torch.ones(
        micro_bs, seq_len, dtype=torch.float32, requires_grad=False, device=device
    )
    token_type_ids = torch.ones(
        micro_bs, seq_len, dtype=torch.long, requires_grad=False, device=device
    )
    labels = torch.randint(
        0, 10, (micro_bs, seq_len), dtype=torch.long, device=sch.rank
    )
    with slapo.Verify(
        sch,
        example_inputs=[input_ids, attention_mask, token_type_ids],
        example_outputs=labels,
        loss_fn=loss_fn,
        topology=topology,
        config=ds_config_dict,
    ):
        sch.trace_until(
            "bert.encoder", tracer="huggingface", concrete_args=concrete_args
        )
        sch["bert.encoder.layer.11"].cut_pipeline_stage()


def test_gpt2_2stages_pp():
    deepspeed.init_distributed(dist_backend="nccl")
    if dist.get_world_size() != 2:
        logger.info("This test requires 2 GPUs.")
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    from transformers import GPT2LMHeadModel, AutoConfig

    config = AutoConfig.from_pretrained("gpt2-xl")
    config.use_cache = False
    with slapo.init_empty_weights():
        model = GPT2LMHeadModel(config)

    num_pp = 2
    num_mp = 1
    num_dp = dist.get_world_size() // (num_pp * num_mp)
    topology, group = create_dist_group_for_pipeline(num_pp=num_pp, num_mp=num_mp)
    sch = slapo.create_schedule(model, group=group)
    input_names = ["input_ids", "attention_mask", "position_ids"]
    sig = inspect.signature(sch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }

    loss_fct = nn.CrossEntropyLoss()

    def loss_fn(outputs, labels):
        # (bs, seq, vocab)
        if isinstance(outputs, torch.Tensor):
            # DS PP output has already removed the data structure
            prediction_scores = outputs
        else:
            prediction_scores = outputs["logits"]
        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        lm_loss = loss_fct(
            shifted_prediction_scores.view(-1, config.vocab_size), labels.view(-1)
        )
        return lm_loss

    # avoid OOM
    bs = 2
    seq_len = 512
    micro_bs = bs // num_dp
    ds_config_dict = get_ds_config(
        batch_size=bs,
        micro_batch_size_per_gpu=micro_bs,
        fp16=True,
    )
    device = "cuda"
    input_ids = torch.ones(micro_bs, seq_len, dtype=torch.long, device=device)
    attention_mask = torch.ones(
        micro_bs, seq_len, dtype=torch.float16, requires_grad=False, device=device
    )
    position_ids = torch.ones(
        micro_bs, seq_len, dtype=torch.long, requires_grad=False, device=device
    )
    labels = torch.randint(
        0, 10, (micro_bs, seq_len), dtype=torch.long, device=sch.rank
    )
    with slapo.Verify(
        sch,
        # (input_ids, past_key_values, attention_mask, token_type_ids, position_ids)
        example_inputs=[input_ids, None, attention_mask, None, position_ids],
        example_outputs=labels,
        loss_fn=loss_fn,
        topology=topology,
        config=ds_config_dict,
        init_weights=model._init_weights,
        dtype=torch.float16,
    ):
        sch.trace_until(
            "transformer", tracer="huggingface", concrete_args=concrete_args
        )
        sch["transformer.h.23"].cut_pipeline_stage()


def test_gpt2_4stages_pp():
    """
    def forward(self, input_ids : typing_Union[torch.LongTensor,NoneType] = None, past_key_values = None, attention_mask : typing_Union[torch.FloatTensor,NoneType] = None, token_type_ids = None, position_ids : typing_Union[torch.LongTensor,NoneType] = None, head_mask = None, inputs_embeds = None, encoder_hidden_states = None, encoder_attention_mask = None, labels = None, use_cache = None, output_attentions = None, output_hidden_states = None, return_dict = None):
        submod_0 = self.submod_0(input_ids, position_ids, attention_mask);  input_ids = position_ids = attention_mask = None
        submod_1 = self.submod_1(submod_0);  submod_0 = None
        getitem = submod_1[0]
        getitem_1 = submod_1[1]
        getitem_2 = submod_1[2];  submod_1 = None
        submod_2 = self.submod_2(getitem, getitem_1);  getitem = None
        submod_3 = self.submod_3(submod_2, getitem_1, getitem_2);  submod_2 = getitem_1 = getitem_2 = None
        return {'logits': submod_3, 'past_key_values': None, 'hidden_states': None, 'attentions': None, 'cross_attentions': None}
    """
    deepspeed.init_distributed(dist_backend="nccl")
    mem = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
    if dist.get_world_size() != 4 or mem < 30:
        logger.info("This test requires 4 GPUs with large memory (~32GB).")
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    from transformers import GPT2LMHeadModel, AutoConfig

    config = AutoConfig.from_pretrained("gpt2-xl")
    config.use_cache = False
    with slapo.init_empty_weights():
        model = GPT2LMHeadModel(config)

    num_pp = 4
    num_mp = 1
    num_dp = dist.get_world_size() // (num_pp * num_mp)
    topology, group = create_dist_group_for_pipeline(num_pp=num_pp, num_mp=num_mp)
    sch = slapo.create_schedule(model, group=group)
    input_names = ["input_ids", "attention_mask", "position_ids"]
    sig = inspect.signature(sch.mod.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }

    loss_fct = nn.CrossEntropyLoss()

    def loss_fn(outputs, labels):
        # (bs, seq, vocab)
        if isinstance(outputs, torch.Tensor):
            # DS PP output has already removed the data structure
            prediction_scores = outputs
        else:
            prediction_scores = outputs["logits"]
        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        lm_loss = loss_fct(
            shifted_prediction_scores.view(-1, config.vocab_size), labels.view(-1)
        )
        return lm_loss

    # avoid OOM
    bs = 2
    seq_len = 512
    micro_bs = bs // num_dp
    ds_config_dict = get_ds_config(
        batch_size=bs,
        micro_batch_size_per_gpu=micro_bs,
        fp16=True,
    )
    device = "cuda"
    input_ids = torch.ones(micro_bs, seq_len, dtype=torch.long, device=device)
    attention_mask = torch.ones(
        micro_bs, seq_len, dtype=torch.float16, requires_grad=False, device=device
    )
    position_ids = torch.ones(
        micro_bs, seq_len, dtype=torch.long, requires_grad=False, device=device
    )
    labels = torch.randint(
        0, 10, (micro_bs, seq_len), dtype=torch.long, device=sch.rank
    )
    with slapo.Verify(
        sch,
        # (input_ids, past_key_values, attention_mask, token_type_ids, position_ids)
        example_inputs=[input_ids, None, attention_mask, None, position_ids],
        example_outputs=labels,
        loss_fn=loss_fn,
        topology=topology,
        config=ds_config_dict,
        init_weights=model._init_weights,
        dtype=torch.float16,
    ):
        sch.trace_until(
            "transformer", tracer="huggingface", concrete_args=concrete_args
        )
        sch["transformer.h.11"].cut_pipeline_stage()
        sch["transformer.h.23"].cut_pipeline_stage()
        sch["transformer.h.35"].cut_pipeline_stage()


if __name__ == "__main__":
    test_pipeline_2stages_pp_dp()
    test_pipeline_2stages_pp_tp()
    test_pipeline_4stages_pp()
    test_pipeline_2stages_pp_tp_dp()
    test_bert_2stages_pp()
    test_gpt2_2stages_pp()
    test_gpt2_4stages_pp()