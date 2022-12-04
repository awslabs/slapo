"""
Test sharding primitive. Note that this test has to be invoked by torchrun. For example:
torchrun --nproc_per_node 2 -m pytest test_shard.py
"""
import copy
import pytest

import torch
import torch.distributed as dist
from torch.autograd import Variable
import ms


@pytest.fixture(scope="session", autouse=True)
def init_dist(request):
    torch.manual_seed(9999)
    try:
        dist.init_process_group(backend="nccl")
    except:
        pytest.skip(f"Skip {__file__} because torch.distributed is not initialized")

    def destory_dist():
        dist.destroy_process_group()

    request.addfinalizer(destory_dist)


def gather_grad(model, param_path_and_gather_axis):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    def _gather_grad(part_grad, axis=0):
        if axis < 0:
            return part_grad

        parts = [
            torch.zeros(part_grad.shape, dtype=part_grad.dtype).cuda(rank)
            for _ in range(world_size)
        ]
        dist.all_gather(parts, part_grad)
        return torch.cat(parts, dim=axis)

    ret = {}
    for path, axis in param_path_and_gather_axis.items():
        param = model
        for token in path.split("."):
            param = getattr(param, token)
        ret[path] = _gather_grad(param.grad, axis)
    return ret


def verify_grads(ref_model, path_and_grads, tol=1e-5):
    for path, grad in path_and_grads.items():
        param = ref_model
        for token in path.split("."):
            param = getattr(param, token)
        torch.testing.assert_close(
            grad,
            param.grad,
            msg=lambda msg: f"{path}.grad mismatch\n{msg}",
            atol=tol,
            rtol=tol,
        )
        print(f"{path}.grad verified")


def test_linear():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(20, 30)
            # FIXME: Enable bias results in incorrect results with sharding,
            # because when sharding the input dimension, bias should also
            # be scaled by world size,
            self.linear2 = torch.nn.Linear(30, 40, bias=False)

        def forward(self, data):
            out = self.linear1(data)
            out = self.linear2(out)
            return out

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    model = Model()

    sch = ms.create_schedule(
        copy.deepcopy(model), world_size=world_size, rank=rank, tracer="pytorch"
    )
    sch["linear1"].shard("weight", axis=0)
    sch["linear1"].shard("bias", axis=0)
    sch["linear1"].sync(mode="backward") # backward allreduce only
    sch["linear2"].shard("weight", axis=1)
    sch["linear2"].sync(mode="forward") # forward allreduce only
    sch_model, _ = ms.build(sch)

    sch_model.cuda(rank)
    data = torch.randn((10, 20), requires_grad=True).cuda(rank)
    out = sch_model(data)
    out.mean().backward()

    param_path_and_gather_axis = {
        "linear1.weight": 0,
        "linear1.bias": 0,
        "linear2.weight": 1,
    }
    path_and_grads = gather_grad(sch_model, param_path_and_gather_axis)

    if rank == 0:
        model.cuda(rank)
        out_ref = model(data)
        out_ref.mean().backward()

        torch.testing.assert_close(out, out_ref)
        verify_grads(model, path_and_grads)


def test_conv():
    """Test conv2d sharding. The workload is from WideResNet from torchvision."""
    expansion = 4
    inplanes = planes = 64
    base_width = 128
    groups = 1

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
            self.conv2 = conv3x3(width, width, 1, groups, 1)
            self.conv3 = conv1x1(width, planes * expansion)

        def forward(self, data):
            out = self.conv1(data)
            out = self.conv2(out)
            out = self.conv3(out)
            return out

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    model = Model()

    sch = ms.create_schedule(
        copy.deepcopy(model), world_size=world_size, rank=rank, tracer="pytorch"
    )
    # Layout of input/weight: (N, C, H, W), (O, I, H, W)
    sch["conv1"].shard("weight", axis=0)
    sch["conv1"].sync(mode="backward") # backward allreduce only
    sch["conv2"].shard("weight", axis=1)
    sch["conv2"].sync(mode="forward") # forward allreduce only
    sch["conv3"].shard("weight", axis=0)
    sch["conv3"].sync(mode="both") # forward allgather + backward split/allreduce
    sch_model, _ = ms.build(sch)

    sch_model.cuda(rank)
    data = torch.randn((4, 64, 56, 56), requires_grad=True).cuda(rank)
    data = Variable(data, requires_grad=True)  # Make data.grad avaiable for verifying
    out = sch_model(data)
    out.mean().backward()
    data_grad = data.grad
    data.grad = None

    param_path_and_gather_axis = {
        "conv3.weight": 0,
        "conv2.weight": 1,
        "conv1.weight": 0,
    }
    path_and_grads = gather_grad(sch_model, param_path_and_gather_axis)

    if rank == 0:
        model.cuda(rank)
        out_ref = model(data)
        out_ref.mean().backward()

        torch.testing.assert_close(out, out_ref)
        torch.testing.assert_allclose(data_grad, data.grad)
        verify_grads(model, path_and_grads)


if __name__ == "__main__":
    pytest.main([__file__])
