# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test checkpoints. Note that this test has to be invoked by torchrun.
See ci/task_unit_tests.sh for an example.
"""
# pylint: disable=unused-argument

import os
import copy

import pytest
import torch
from torch import distributed as dist
from torch.autograd import Variable
import torch.nn.functional as F

import slapo
from slapo import checkpoint, get_cuda_rng_tracker, set_random_seed
from slapo.sharding import reduce_backward_grad, reduce_forward_output
from slapo.pattern import call_module


def test_checkpoint_function():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(32, 32)
            self.layer2 = torch.nn.Linear(32, 32)

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = self.layer2(x)
            return x

    model = Model()
    sch = slapo.create_schedule(copy.deepcopy(model))
    # Only checkpoint the first submodule.

    def pattern(x):
        x = F.relu(call_module("layer1", x))
        return x

    subgraphs = sch.find(pattern)
    assert len(subgraphs) == 1
    assert len(subgraphs[0]) == 2
    sch.checkpoint(subgraphs)
    assert isinstance(sch["CheckPointWrapper_0"].mod.mod.layer1, torch.nn.Linear)
    sch_model, _ = slapo.build(sch, init_weights=False)
    data = torch.randn((32, 32), requires_grad=True).cuda()
    model.cuda()
    sch_model.cuda()
    # 1. Run the model forward.
    ref_data = Variable(data, requires_grad=True)
    ref_out = model(ref_data)
    sch_data = Variable(data, requires_grad=True)
    sch_out = sch_model(sch_data)
    torch.testing.assert_close(ref_out, sch_out)
    # 2. Run the model backward.
    ref_out.mean().backward()
    sch_out.mean().backward()
    linear1_weight_grad = model.layer1.weight.grad.clone()
    linear2_weight_grad = model.layer2.weight.grad.clone()
    sch_linear1_weight_grad = (
        sch_model.CheckPointWrapper_0.mod.layer1.weight.grad.clone()
    )
    sch_linear2_weight_grad = sch_model.layer2.weight.grad.clone()
    torch.testing.assert_close(linear1_weight_grad, sch_linear1_weight_grad)
    torch.testing.assert_close(linear2_weight_grad, sch_linear2_weight_grad)
    input_grad = ref_data.grad.clone()
    sch_input_grad = sch_data.grad.clone()
    torch.testing.assert_close(input_grad, sch_input_grad)


def test_checkpoint_module(init_dist):
    world_size = dist.get_world_size()
    full_size = 5 * world_size

    tp_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(tp_rank)

    class LinearAct(torch.nn.Module):
        def __init__(self, inp, oup, bias) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(inp, oup, bias=bias)
            self.act = torch.nn.ReLU()

        def forward(self, x):
            x = self.linear(x)
            x = self.act(x)
            return x

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = LinearAct(full_size, 5, bias=False)
            self.layer2 = LinearAct(5, full_size, bias=False)

        def forward(self, x):
            x = reduce_backward_grad(x, None)
            x = self.layer1(x)
            x = self.layer2(x)
            x = reduce_forward_output(x, None)
            return x

    data = torch.randn((full_size, full_size), requires_grad=True).cuda(tp_rank)
    dist.broadcast(data, src=0)

    model = Model()
    sch = slapo.create_schedule(copy.deepcopy(model))
    # Only checkpoint the first submodule.
    sch["layer1"].checkpoint()
    assert sch["layer1"].mod.__class__.__name__ == "CheckPointWrapper"
    sch_model, _ = slapo.build(sch, init_weights=False)
    model.cuda(tp_rank)
    sch_model.cuda(tp_rank)

    # 1. Run the model forward.
    ref_data = Variable(data, requires_grad=True)
    ref_out = model(ref_data)
    sch_data = Variable(data, requires_grad=True)
    sch_out = sch_model(sch_data)
    torch.testing.assert_close(ref_out, sch_out)
    # 2. Run the model backward.
    ref_out.mean().backward()
    sch_out.mean().backward()
    linear1_weight_grad = model.layer1.linear.weight.grad.clone()
    linear2_weight_grad = model.layer2.linear.weight.grad.clone()
    sch_linear1_weight_grad = sch_model.layer1.mod.linear.weight.grad.clone()
    sch_linear2_weight_grad = sch_model.layer2.linear.weight.grad.clone()
    torch.testing.assert_close(linear1_weight_grad, sch_linear1_weight_grad)
    torch.testing.assert_close(linear2_weight_grad, sch_linear2_weight_grad)
    input_grad = ref_data.grad.clone()
    sch_input_grad = sch_data.grad.clone()
    torch.testing.assert_close(input_grad, sch_input_grad)


def test_activation_checkpoint_with_rng_states(init_dist):
    world_size = dist.get_world_size()
    full_size = 5 * world_size

    tp_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(tp_rank)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(full_size, 5, bias=False)
            self.dropout1 = torch.nn.Dropout(0.5)
            self.linear2 = torch.nn.Linear(5, full_size, bias=False)
            self.dropout2 = torch.nn.Dropout(0.5)

        def orig_forward(self, x):
            x = reduce_backward_grad(x, None)
            x = self.linear1(x)
            # The output of linear1 is partitioned, so we use different seeds.
            with get_cuda_rng_tracker().fork():
                x = self.dropout1(x)
            x = self.linear2(x)
            # The output of linear2 is partial sum, so we use the same seed.
            x = self.dropout2(x)
            x = reduce_forward_output(x, None)
            return x

        def forward(self, x, enable_checkpoint):
            if enable_checkpoint:
                return checkpoint(self.orig_forward, x)
            return self.orig_forward(x)

    data = torch.randn((full_size, full_size), requires_grad=True).cuda(tp_rank)
    dist.broadcast(data, src=0)
    data = Variable(data, requires_grad=True)

    model = Model().cuda(tp_rank)

    def run(model, data, enable_checkpoint):
        # 1. Run the model forward and backward.
        out = model(data, enable_checkpoint)
        out.mean().backward()
        # 2. Retrieve gradients.
        linear1_weight_grad = model.linear1.weight.grad.clone()
        linear2_weight_grad = model.linear2.weight.grad.clone()
        input_grad = data.grad.clone()
        # 3. Clear gradients.
        model.linear1.weight.grad = None
        model.linear2.weight.grad = None
        data.grad = None
        return out, linear1_weight_grad, linear2_weight_grad, input_grad

    # Run the model without activation checkpointing for reference.
    set_random_seed(123, None, None, tp_rank)
    refs = run(model, data, False)
    # Run the model with activation checkpointing.
    set_random_seed(123, None, None, tp_rank)
    outs = run(model, data, True)

    for ref, out in zip(refs, outs):
        torch.testing.assert_close(out, ref)


if __name__ == "__main__":
    pytest.main([__file__])
