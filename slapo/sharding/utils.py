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
        ret = torch.empty(gather_shape, dtype=temp.dtype).to(inp.device)
        dist.all_gather_into_tensor(ret, temp, group=group)
        ret = ret.transpose(0, dim).contiguous() if dim != 0 else ret
    else:
        # Fallback to all_gather. This may lead to suboptimal performance.
        parts = [
            torch.empty(inp.shape, dtype=inp.dtype).to(inp.device)
            for _ in range(world_size)
        ]
        dist.all_gather(parts, inp, group=group)
        ret = torch.cat(parts, dim=dim)
    return ret


class _AllGatherForwardOutput(torch.autograd.Function):
    """The cusom all gather op used for forward hook."""

    # pylint: disable=abstract-method, arguments-differ
    @staticmethod
    def forward(ctx, inp, dim, group):
        ctx.dim = dim
        ctx.group = group
        world_size = dist.get_world_size(group)
        return all_gather_along_dim(inp, dim, world_size, group)

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        group = ctx.group
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        sharded_size = grad_output.shape[dim] // world_size
        ret = grad_output.split(sharded_size, dim=dim)[rank].contiguous()
        return ret, None, None


class _ReduceScatterForwardOutput(torch.autograd.Function):
    """The cusom reduce scatter op used for forward hook."""

    # pylint: disable=abstract-method, arguments-differ
    @staticmethod
    def forward(ctx, inp, dim, group):
        ctx.dim = dim
        ctx.group = group
        world_size = dist.get_world_size(group)

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
        ret = torch.zeros(scatter_shape, dtype=inp.dtype).to(inp.device)

        dist.reduce_scatter_tensor(ret, temp, group=group)
        if dim != 0:
            ret = ret.transpose(0, dim).contiguous()
        return ret

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


def all_gather_forward_output(inp, dim, group):
    """The custom all gather op used for forward hook.

    Parameters
    ----------
    inp: torch.Tensor
        The input tensor to all-gather.
    dim: int
        The dimension to all-gather along.
    group: torch.distributed.ProcessGroup
        The process group to all-gather.

    Returns
    -------
    torch.Tensor
        The gathered tensor.
    """
    return _AllGatherForwardOutput.apply(inp, dim, group)


def reduce_scatter_forward_output(inp, dim, group):
    """The custom reduce scatter op used for forward hook.
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
