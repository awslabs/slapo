# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import torch.distributed as dist

from ..logger import get_logger

logger = get_logger()


def all_gather_along_dim(inp, dim, world_size, group):
    """all-gather along the given dimension. Will use all_gather_into_tensor
    if available as it is more efficient; otherwise fallback to all_gather.

    Paramters
    ---------
    inp: torch.Tensor
        The input tensor to all-gather.
    dim: int
        The dimension to all-gather along.
    world_size: int
        The number of processes in the group.
    group: torch.distributed.ProcessGroup
        The process group to all-gather.

    Returns
    -------
    torch.Tensor
        The gathered tensor.
    """
    if hasattr(dist, "all_gather_into_tensor"):
        temp = inp.transpose(0, dim) if dim != 0 else inp
        temp = temp.contiguous()
        gather_shape = list(temp.shape)
        gather_shape[0] = world_size * gather_shape[0]
        ret = torch.empty(gather_shape, dtype=temp.dtype, device=inp.device)
        dist.all_gather_into_tensor(ret, temp, group=group)
        ret = ret.transpose(0, dim).contiguous() if dim != 0 else ret
    else:
        # Fallback to all_gather. This may lead to suboptimal performance.
        parts = [
            torch.empty(inp.shape, dtype=inp.dtype, device=inp.device)
            for _ in range(world_size)
        ]
        dist.all_gather(parts, inp, group=group)
        ret = torch.cat(parts, dim=dim)
    return ret


def reduce_scatter_along_dim(inp, dim, world_size, group):
    """reduce-scatter along the given dimension.

    Paramters
    ---------
    inp: torch.Tensor
        The input tensor to reduce-scatter.
    dim: int
        The dimension to all-gather along.
    world_size: int
        The number of processes in the group.
    group: torch.distributed.ProcessGroup
        The process group to all-gather.

    Returns
    -------
    torch.Tensor
        The reduce-scattered tensor.
    """
    # reduce_scatter always targets dim 0, so we transpose the target dim
    # to dim 0, and transpose the result back.
    assert inp.shape[dim] % world_size == 0, (
        f"Reduce scatter dimension {dim} size {inp.shape} "
        f"should be divisible by world size {world_size}"
    )

    temp = inp.transpose(0, dim) if dim != 0 else inp
    temp = temp.contiguous()
    scatter_shape = list(temp.shape)
    scatter_shape[0] //= world_size
    ret = torch.empty(scatter_shape, dtype=inp.dtype, device=inp.device)

    dist.reduce_scatter_tensor(ret, temp, group=group)
    if dim != 0:
        ret = ret.transpose(0, dim).contiguous()
    return ret


def scatter_along_dim(inp, dim, world_size, group):
    """scatter along the given dimension.

    Paramters
    ---------
    inp: torch.Tensor
        The input tensor to reduce-scatter.
    dim: int
        The dimension to all-gather along.
    world_size: int
        The number of processes in the group.
    group: torch.distributed.ProcessGroup
        The process group to all-gather.

    Returns
    -------
    torch.Tensor
        The scattered tensor.
    """
    assert inp.shape[dim] % world_size == 0, (
        f"Scatter dimension {dim} size {inp.shape} "
        f"should be divisible by world size {world_size}"
    )

    rank = dist.get_rank(group)
    sharded_size = inp.shape[dim] // world_size
    ret = inp.split(sharded_size, dim=dim)[rank].contiguous()

    return ret


class _AllGatherForwardOutput(torch.autograd.Function):
    """The custom sync op (F: all-gather, B: split or reduce-scatter) used for
    forward hook."""

    # pylint: disable=abstract-method, arguments-differ
    @staticmethod
    def forward(ctx, inp, dim, group, tensor_parallel_output_grad=True):
        ctx.dim = dim
        ctx.group = group
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        world_size = dist.get_world_size(group)
        return all_gather_along_dim(inp, dim, world_size, group)

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        group = ctx.group
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad
        world_size = dist.get_world_size(group)

        if tensor_parallel_output_grad:
            ret = reduce_scatter_along_dim(grad_output, dim, world_size, group)
        else:
            ret = scatter_along_dim(grad_output, dim, world_size, group)

        return (ret, None, None, None)


class _ReduceScatterForwardOutput(torch.autograd.Function):
    """The custom sync op (F: reduce-scatter, B: all-gather) used for forward hook."""

    # pylint: disable=abstract-method, arguments-differ
    @staticmethod
    def forward(ctx, inp, dim, group):
        ctx.dim = dim
        ctx.group = group
        world_size = dist.get_world_size(group)

        return reduce_scatter_along_dim(inp, dim, world_size, group)

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        group = ctx.group
        world_size = dist.get_world_size(group)
        return (
            all_gather_along_dim(grad_output, dim, world_size, group),
            None,
            None,
        )


class _ScatterForwardOutput(torch.autograd.Function):
    """The custom sync op (sync op (F: scatter, B: all-gather) used for forward hook."""

    # pylint: disable=abstract-method, arguments-differ
    @staticmethod
    def forward(ctx, inp, dim, group):
        ctx.dim = dim
        ctx.group = group
        world_size = dist.get_world_size(group)

        return scatter_along_dim(inp, dim, world_size, group)

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        group = ctx.group
        world_size = dist.get_world_size(group)
        return (
            all_gather_along_dim(grad_output, dim, world_size, group),
            None,
            None,
        )


class _ReduceForwardOutput(torch.autograd.Function):
    """The custom sync op sync op (F: all-reduce, B: no-op) used for forward hook."""

    # pylint: disable=abstract-method, arguments-differ
    @staticmethod
    def forward(ctx, inp, group):
        dist.all_reduce(inp, group=group)
        return inp

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output,
            None,
        )


class _ReduceBackwardGradient(torch.autograd.Function):
    """The custom sync op (F: no-op, B: all-reduce) used for forward hook."""

    # pylint: disable=abstract-method, arguments-differ
    @staticmethod
    def forward(ctx, inp, group):
        ctx.group = group
        return inp

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        dist.all_reduce(grad_output, group=group)
        return (
            grad_output,
            None,
        )


def all_gather_forward_output(inp, dim, group, tensor_parallel_output_grad=True):
    """The custom sync op (F: all-gather, B: split or reduce-scatter) used for
    forward hook.

    Parameters
    ----------
    inp: torch.Tensor
        The input tensor to all-gather.
    dim: int
        The dimension to all-gather along.
    group: torch.distributed.ProcessGroup
        The process group to all-gather.
    tensor_parallel_output_grad: bool
        If the operator following the gather operation is sharded,
        output gradients need to be reduced and scattered; whereas
        if the operator is duplicated, output gradients only need to be scattered.

    Returns
    -------
    torch.Tensor
        The gathered tensor.
    """
    return _AllGatherForwardOutput.apply(inp, dim, group, tensor_parallel_output_grad)


def reduce_scatter_forward_output(inp, dim, group):
    """The custom sync op (F: reduce-scatter, B: all-gather) used for forward hook.
    Parameters
    ----------
    inp: torch.Tensor
        The input tensor to reduce-scatter.
    dim: int
        The dimension to reduce-scatter along.
    group: torch.distributed.ProcessGroup
        The process group to reduce-scatter.

    Returns
    -------
    torch.Tensor
        The reduced tensor.
    """
    return _ReduceScatterForwardOutput.apply(inp, dim, group)


def scatter_forward_output(inp, dim, group):
    """The custom sync op (sync op (F: scatter, B: all-gather) used for forward hook.
    Parameters
    ----------
    inp: torch.Tensor
        The input tensor to reduce-scatter.
    dim: int
        The dimension to reduce-scatter along.
    group: torch.distributed.ProcessGroup
        The process group to reduce-scatter.

    Returns
    -------
    torch.Tensor
        The scattered tensor.
    """
    return _ScatterForwardOutput.apply(inp, dim, group)


def reduce_forward_output(inp, group):
    """The custom sync op sync op (F: all-reduce, B: no-op) used for forward hook.
    Parameters
    ----------
    inp: torch.Tensor
        The input tensor to reduce.
    group: torch.distributed.ProcessGroup
        The process group to reduce.

    Returns
    -------
    torch.Tensor
        The reduced tensor.
    """
    return _ReduceForwardOutput.apply(inp, group)


def reduce_backward_grad(inp, group):
    """The custom sync op (F: no-op, B: all-reduce) used for forward hook.
    Parameters
    ----------
    inp: torch.Tensor
        The input tensor.
    group: torch.distributed.ProcessGroup
        The process group to reduce.

    Returns
    -------
    torch.Tensor
        The original input tensor. However, its gradient will be reduced.
    """
    return _ReduceBackwardGradient.apply(inp, group)
