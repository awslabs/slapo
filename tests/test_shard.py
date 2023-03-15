# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test sharding primitive. Note that this test has to be invoked by torchrun.
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
from torch.nn import functional as F

import slapo
from slapo import op
from slapo.model_schedule.gpt2 import gen_embedding_hooks

from .utils import reset_random_seeds


def gather_grad(model, param_path_and_gather_axis):
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    def _allreduce_grad(grad):
        dist.all_reduce(grad)
        return grad

    def _gather_grad(part_grad, axis=0):
        if axis < 0:
            return part_grad

        parts = [
            torch.zeros(part_grad.shape, dtype=part_grad.dtype).cuda(local_rank)
            for _ in range(world_size)
        ]
        dist.all_gather(parts, part_grad)
        return torch.cat(parts, dim=axis)

    ret = {}
    for path, axis in param_path_and_gather_axis.items():
        param = model
        for token in path.split("."):
            param = getattr(param, token)
        if axis == "allreduce":
            grad = _allreduce_grad(param.grad)
        elif axis == "none":
            grad = param.grad
        else:
            grad = _gather_grad(param.grad, axis)
        ret[path] = grad
    return ret


def gather_and_copy_model(src_model, dest_model, param_path_and_gather_axis):
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    def _gather_param(part_param, axis=0):
        if not isinstance(axis, int) or axis < 0:
            return part_param

        parts = [
            torch.zeros(part_param.shape, dtype=part_param.dtype).cuda(local_rank)
            for _ in range(world_size)
        ]
        dist.all_gather(parts, part_param.contiguous())
        return torch.cat(parts, dim=axis)

    for path, axis in param_path_and_gather_axis.items():
        part_param = src_model
        dest_param = dest_model
        for token in path.split("."):
            part_param = getattr(part_param, token)
            dest_param = getattr(dest_param, token)
        dest_param.data = _gather_param(part_param, axis)


def verify_grads(ref_model, path_and_grads, tol=1e-5):
    for path, grad in path_and_grads.items():
        param = ref_model
        for token in path.split("."):
            param = getattr(param, token)
        torch.testing.assert_close(
            grad,
            param.grad,
            # pylint: disable=cell-var-from-loop
            msg=lambda msg: f"{path}.grad mismatch\n{msg}",
            atol=tol,
            rtol=tol,
        )
        print(f"{path}.grad verified")


def test_mlp(init_dist):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = op.FusedMLP(10, 10, "gelu", 0.1, use_torchscript=False)

        def forward(self, data):
            return self.mlp(data)

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    model = Model()

    sch = slapo.create_schedule(copy.deepcopy(model))
    sch["mlp.fc_in"].shard("weight", axis=0)
    sch["mlp.fc_in"].shard("bias", axis=0)
    sch["mlp.fc_in"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
    sch["mlp.fc_out"].shard("weight", axis=1)
    sch["mlp.fc_out"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")
    sch_model, _ = slapo.build(sch, init_weights=False)

    sch_model.cuda(local_rank)
    data = torch.randn((10, 10), requires_grad=True).cuda(local_rank)
    dist.broadcast(data, src=0)
    reset_random_seeds()
    out = sch_model(data)
    out.mean().backward()

    param_path_and_gather_axis = {
        "mlp.fc_in.weight": 0,
        "mlp.fc_in.bias": 0,
        "mlp.fc_out.weight": 1,
        "mlp.fc_out.bias": "none",
    }
    path_and_grads = gather_grad(sch_model, param_path_and_gather_axis)

    gather_and_copy_model(sch_model, model, param_path_and_gather_axis)

    reset_random_seeds()
    if rank == 0:
        model.cuda(local_rank)
        out_ref = model(data)
        out_ref.mean().backward()

        torch.testing.assert_close(out, out_ref)
        verify_grads(model, path_and_grads)


def test_linear(init_dist):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(20, 30)
            self.linear2 = torch.nn.Linear(30, 40)

        def forward(self, data):
            out = self.linear1(data)
            out = self.linear2(out)
            return out

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    model = Model()

    sch = slapo.create_schedule(copy.deepcopy(model))
    sch["linear1"].shard("weight", axis=0)
    sch["linear1"].shard("bias", axis=0)
    sch["linear1"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
    sch["linear2"].shard("weight", axis=1)
    sch["linear2"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")
    sch_model, _ = slapo.build(sch, init_weights=False)

    sch_model.cuda(local_rank)
    data = torch.randn((10, 20), requires_grad=True).cuda(local_rank)
    dist.broadcast(data, src=0)
    out = sch_model(data)
    out.mean().backward()

    param_path_and_gather_axis = {
        "linear1.weight": 0,
        "linear1.bias": 0,
        "linear2.weight": 1,
        "linear2.bias": "none",
    }
    path_and_grads = gather_grad(sch_model, param_path_and_gather_axis)

    gather_and_copy_model(sch_model, model, param_path_and_gather_axis)

    if rank == 0:
        model.cuda(local_rank)
        out_ref = model(data)
        out_ref.mean().backward()

        torch.testing.assert_close(out, out_ref)
        verify_grads(model, path_and_grads)


def test_conv1d(init_dist):
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:
        pytest.skip(reason="transformers not installed")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = Conv1D(30, 20)
            self.linear2 = Conv1D(40, 30)

        def forward(self, data):
            out = self.linear1(data)
            out = self.linear2(out)
            return out

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    model = Model()

    sch = slapo.create_schedule(copy.deepcopy(model))
    sch["linear1"].shard("weight", axis=1)
    sch["linear1"].shard("bias", axis=0)
    sch["linear1"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
    sch["linear2"].shard("weight", axis=0)
    sch["linear2"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")
    sch_model, _ = slapo.build(sch, init_weights=False)

    sch_model.cuda(local_rank)
    data = torch.randn((10, 20), requires_grad=True).cuda(local_rank)
    dist.broadcast(data, src=0)
    out = sch_model(data)
    out.mean().backward()

    param_path_and_gather_axis = {
        "linear1.weight": 1,
        "linear1.bias": 0,
        "linear2.weight": 0,
        "linear2.bias": "none",
    }
    path_and_grads = gather_grad(sch_model, param_path_and_gather_axis)

    gather_and_copy_model(sch_model, model, param_path_and_gather_axis)

    if rank == 0:
        model.cuda(local_rank)
        out_ref = model(data)
        out_ref.mean().backward()

        torch.testing.assert_close(out, out_ref)
        verify_grads(model, path_and_grads)


def test_seq_para(init_dist):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(30, 30)
            self.linear2 = torch.nn.Linear(30, 30)

        def forward(self, data):
            out = self.linear1(data)
            out = self.linear2(out)
            out = F.relu(out)
            return out

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    model = Model()

    def sequence_partition(module, _input):
        world_size = dist.get_world_size()
        sharded_seq_len = _input[0].shape[1] // world_size
        return _input[0].split(sharded_seq_len, dim=1)[rank].contiguous()

    sch = slapo.create_schedule(copy.deepcopy(model))
    # The following two lines are used to test all_gather feature for pre-forward
    sch["linear1"].sync(mode="fwd_pre", sync_op_or_fn=sequence_partition)
    sch["linear1"].sync(mode="fwd_pre", sync_op_or_fn="all_gather", axis=1)
    sch["linear1"].shard("weight", axis=0)
    sch["linear1"].shard("bias", axis=0)
    sch["linear1"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
    sch["linear2"].shard("weight", axis=1)

    # forward reduce_scatter, and allgather at the end of the top module.
    sch["linear2"].sync(mode="fwd_post", sync_op_or_fn="reduce_scatter", axis=1)
    sch.sync(
        mode="fwd_post",
        sync_op_or_fn="all_gather",
        axis=1,
        tensor_parallel_output_grad=False,
    )

    sch_model, _ = slapo.build(sch, init_weights=False)
    sch_model.cuda(local_rank)

    data = torch.randn((3, 16, 30), requires_grad=True).cuda(local_rank)
    dist.broadcast(data, src=0)
    out = sch_model(data)
    out.mean().backward()

    param_path_and_gather_axis = {
        "linear1.weight": 0,
        "linear1.bias": 0,
        "linear2.weight": 1,
        "linear2.bias": "allreduce",
    }
    path_and_grads = gather_grad(sch_model, param_path_and_gather_axis)

    gather_and_copy_model(sch_model, model, param_path_and_gather_axis)

    if rank == 0:
        model.cuda(local_rank)
        out_ref = model(data)
        out_ref.mean().backward()

        torch.testing.assert_close(out, out_ref)
        verify_grads(model, path_and_grads)


@pytest.mark.skip(reason="Flaky test")
def test_conv(init_dist):
    """Test conv2d sharding. The workload is from WideResNet from torchvision."""
    expansion = 4
    inplanes = planes = 64
    base_width = 128
    groups = 1
    rank = dist.get_rank()

    def conv3x3(
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
    ):
        return torch.nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        )

    def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
        return torch.nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, bias=False
        )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            width = int(planes * (base_width / 64.0)) * groups
            self.conv1 = conv1x1(inplanes, width)
            self.bn1 = torch.nn.BatchNorm2d(width)
            self.conv2 = conv3x3(width, width, 1, groups, 1)
            self.bn2 = torch.nn.BatchNorm2d(width)
            self.conv3 = conv1x1(width, planes * expansion)
            self.bn3 = torch.nn.BatchNorm2d(planes * expansion)

        def forward(self, data):
            out = self.conv1(data)
            out = self.bn1(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.conv3(out)
            out = self.bn3(out)
            return out

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    model = Model()

    sch = slapo.create_schedule(copy.deepcopy(model))
    # Layout of input/weight: (N, C, H, W), (O, I, H, W)

    # Forward: partitioned output (optional allgather).
    # Backward: allreduce.
    sch["conv1"].shard("weight", axis=0)
    sch["conv1"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")

    # We choose not allgather, so we need to shard bn as well.
    sch["bn1"].shard("weight", axis=0)
    sch["bn1"].shard("bias", axis=0)
    sch["bn1"].shard("running_mean", axis=0)
    sch["bn1"].shard("running_var", axis=0)

    # Forward: partial output (need allreduce)
    # Backward: do nothing.
    sch["conv2"].shard("weight", axis=1)
    sch["conv2"].sync(
        mode="fwd_post", sync_op_or_fn="all_reduce"
    )  # forward allreduce only

    # Forward: partitioned output (optional allgather).
    # Backward: allreduce.
    sch["conv3"].shard("weight", axis=0)
    sch["conv3"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")

    # We choose not allgather, so we need to shard bn as well.
    # If we choose allgather, then we don't need to
    # worry about bn.
    sch["bn3"].shard("weight", axis=0)
    sch["bn3"].shard("bias", axis=0)
    sch["bn3"].shard("running_mean", axis=0)
    sch["bn3"].shard("running_var", axis=0)
    sch["bn3"].sync(
        mode="fwd_post",
        sync_op_or_fn="all_gather",
        axis=1,
        tensor_parallel_output_grad=False,
    )
    sch["bn3"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
    sch_model, _ = slapo.build(sch, init_weights=False)

    sch_model.cuda(local_rank)
    data = torch.randn((4, 64, 56, 56), requires_grad=True).cuda(local_rank)
    dist.broadcast(data, src=0)
    data = Variable(data, requires_grad=True)  # Make data.grad avaiable for verifying
    out = sch_model(data)
    out.mean().backward()
    data_grad = data.grad
    data.grad = None

    grad_path_and_gather_axis = {
        "conv3.weight": 0,
        "conv2.weight": 1,
        "conv1.weight": 0,
        "bn1.weight": 0,
        "bn1.bias": 0,
        "bn2.weight": "none",
        "bn2.bias": "none",
        "bn3.weight": 0,
        "bn3.bias": 0,
    }
    path_and_grads = gather_grad(sch_model, grad_path_and_gather_axis)

    if rank == 0:
        model.cuda(local_rank)
        out_ref = model(data)
        out_ref.mean().backward()

        torch.testing.assert_close(out, out_ref)
        torch.testing.assert_allclose(data_grad, data.grad)
        verify_grads(model, path_and_grads)


def test_word_embedding(init_dist):
    """Test whether sharding embedding using custom hooks is numerically correct."""
    vocab_size = 100

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 10)  # vocab_size, hidden_size

        def forward(self, x):
            return self.embedding(x)

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    model = Model()

    sch = slapo.create_schedule(copy.deepcopy(model))
    sch["embedding"].shard("weight", axis=0)
    fwd_pre_hook, fwd_post_hook = gen_embedding_hooks(sch, vocab_size)
    sch["embedding"].sync(mode="fwd_pre", sync_op_or_fn=fwd_pre_hook)
    sch["embedding"].sync(mode="fwd_post", sync_op_or_fn=fwd_post_hook)
    sch_model, _ = slapo.build(sch, init_weights=False)

    sch_model.cuda(local_rank)
    data = torch.randint(1, vocab_size - 1, (1, 10)).cuda(local_rank)
    dist.broadcast(data, src=0)
    reset_random_seeds()
    out = sch_model(data)
    out.mean().backward()

    param_path_and_gather_axis = {
        "embedding.weight": 0,
    }
    path_and_grads = gather_grad(sch_model, param_path_and_gather_axis)

    gather_and_copy_model(sch_model, model, param_path_and_gather_axis)

    reset_random_seeds()
    if rank == 0:
        model.cuda(local_rank)
        out_ref = model(data)
        out_ref.mean().backward()

        torch.testing.assert_close(out, out_ref)
        verify_grads(model, path_and_grads)


def test_tie_weights(init_dist):
    """Test whether the tie weights are preserved after sharding."""

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

    sch = slapo.create_schedule(model)
    print(sch.metadata.tie_weights)

    assert id(sch.mod.stage0.wte.weight) == id(sch.mod.stage1.linear.weight)
    assert id(sch.mod.stage0.wte.weight) == id(sch.mod.stage2.linear.weight)

    sch["stage0.wte"].shard("weight", axis=0)
    sch["stage0.wte"].sync(
        mode="fwd_post",
        sync_op_or_fn="all_gather",
        axis=0,
        tensor_parallel_output_grad=False,
    )
    sch["stage0.wte"].sync(mode="bwd_post", sync_op_or_fn="all_reduce")
    sch["stage1.linear"].shard("weight", axis=0)
    sch["stage2.linear"].shard("weight", axis=0)

    assert id(sch.mod.stage0.wte.weight) == id(sch.mod.stage1.linear.weight)
    assert id(sch.mod.stage0.wte.weight) == id(sch.mod.stage2.linear.weight)


if __name__ == "__main__":
    pytest.main([__file__])
