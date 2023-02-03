# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test sync ops for sharding. Note that this test has to be invoked by torchrun.
See ci/task_unit_tests.sh for an example.
"""
# pylint: disable=unused-argument
import copy
import os
import pytest

import torch
import torch.distributed as dist
from torch import nn
from torch.autograd import Variable

from slapo.sharding import sync_ops


def init_model_and_data(local_rank):
    model = nn.Linear(10, 10).cuda(local_rank)
    ref_model = copy.deepcopy(model)

    data = torch.randn((10, 10), requires_grad=True).cuda(local_rank)
    # Make sure all devices have the same data.
    dist.broadcast(data, src=0)

    # Make data.grad avaiable for verifying
    data = Variable(data, requires_grad=True)

    return model, ref_model, data


def verify_out_and_grad(out, out_ref, grad, grad_ref):
    torch.testing.assert_close(
        out,
        out_ref,
        msg=lambda msg: f"output mismatch\n{msg}",
        atol=1e-5,
        rtol=1e-5,
    )
    if grad is not None:
        torch.testing.assert_close(
            grad,
            grad_ref,
            msg=lambda msg: f"input grad mismatch\n{msg}",
            atol=1e-5,
            rtol=1e-5,
        )


def test_all_gather_forward_output(init_dist):
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model, ref_model, data = init_model_and_data(local_rank)

    def hook_fn(_module, _input, output):
        output = sync_ops.all_gather_forward_output(
            output, dim=1, group=None, tensor_parallel_output_grad=False
        )
        return output

    model.register_forward_hook(hook_fn)

    out = model(data)
    # To align the reference output.
    out = sync_ops.scatter_forward_output(out, dim=1, group=None)
    out.mean().backward()
    grad = data.grad
    data.grad = None

    if rank == 0:
        ref_model.cuda(local_rank)
        out_ref = ref_model(data)
        out_ref.mean().backward()
        verify_out_and_grad(out, out_ref, grad, data.grad)


def test_reduce_scatter_forward_output(init_dist):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model, ref_model, data = init_model_and_data(local_rank)

    def hook_fn(_module, _input, output):
        output = sync_ops.reduce_scatter_forward_output(output, dim=1, group=None)
        return output

    model.register_forward_hook(hook_fn)

    out = model(data)
    # To align the reference output.
    out = sync_ops.all_gather_forward_output(
        out, dim=1, group=None, tensor_parallel_output_grad=False
    )
    out.mean().backward()
    grad = data.grad
    data.grad = None

    if rank == 0:
        ref_model.cuda(local_rank)
        out_ref = ref_model(data)
        out_ref.mean().backward()
        verify_out_and_grad(out, out_ref * world_size, grad, data.grad)


def test_scatter_forward_output(init_dist):
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model, ref_model, data = init_model_and_data(local_rank)

    def hook_fn(_module, _input, output):
        output = sync_ops.scatter_forward_output(output, dim=1, group=None)
        return output

    model.register_forward_hook(hook_fn)

    out = model(data)
    # To align the reference output.
    out = sync_ops.all_gather_forward_output(
        out, dim=1, group=None, tensor_parallel_output_grad=False
    )
    out.mean().backward()
    grad = data.grad
    data.grad = None

    if rank == 0:
        ref_model.cuda(local_rank)
        out_ref = ref_model(data)
        out_ref.mean().backward()
        verify_out_and_grad(out, out_ref, grad, data.grad)


def test_reduce_forward_output(init_dist):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model, ref_model, data = init_model_and_data(local_rank)

    def hook_fn(_module, _input, output):
        output = sync_ops.reduce_forward_output(output, None)
        return output

    model.register_forward_hook(hook_fn)

    out = model(data)
    out.mean().backward()

    if rank == 0:
        ref_model.cuda(local_rank)
        out_ref = ref_model(data)
        out_ref.mean().backward()
        verify_out_and_grad(out, out_ref * world_size, None, None)


def test_reduce_backward_grad(init_dist):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model, ref_model, data = init_model_and_data(local_rank)

    def hook_fn(_module, _input):
        _input = sync_ops.reduce_backward_grad(_input[0], None)
        return _input

    model.register_forward_pre_hook(hook_fn)

    out = model(data)
    out.mean().backward()
    grad = data.grad
    data.grad = None

    if rank == 0:
        ref_model.cuda(local_rank)
        out_ref = ref_model(data)
        out_ref.mean().backward()
        verify_out_and_grad(out, out_ref, grad, data.grad * world_size)


if __name__ == "__main__":
    pytest.main([__file__])
