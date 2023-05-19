# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=anomalous-backslash-in-string
"""
Implementation of different resharding schemes.

We follow the R/S notation that `R` means replicated and `S` means sharded along the specified dimension.
For example, `RS` means the tensor is replicated along the first dimension and sharded along the second dimension.

There are in total 9 combination of the resharding schemes, and we can calculate the communication volume
for each combination in the following table:

| src\dst |  RR  |      RS        |       SR        |
|   RR    |  0   |       0        |        0        |
|   RS    | 1/pV |       0        |  (1/p - 1/p^2)V |
|   SR    | 1/pV | (1/p - 1/p^2)V |        0        |

* p is the number of devices
* V is the size of the tensor
"""
from functools import partial

import torch
import torch.distributed as dist


def parse_reshard(scheme, group):
    if scheme == "RR->RR":
        return identity
    if scheme == "RR->RS":
        return partial(reshard_RR_to_RS, group=group)
    if scheme == "RR->SR":
        return partial(reshard_RR_to_SR, group=group)
    if scheme == "RS->RR":
        return partial(reshard_RS_to_RR, group=group)
    if scheme == "RS->RS":
        return identity
    if scheme == "RS->SR":
        return partial(reshard_RS_to_SR, group=group)
    if scheme == "SR->RR":
        return partial(reshard_SR_to_RR, group=group)
    if scheme == "SR->RS":
        return partial(reshard_SR_to_RS, group=group)
    if scheme == "SR->SR":
        return identity
    raise ValueError(f"Unknown resharding scheme: {scheme}")


def identity(in_tensor):
    return in_tensor


# ==================== src: RR ====================
# 1. RR -> RR
#    Omitted


# 2. RR -> RS
def reshard_RR_to_RS(in_tensor, group):
    # Get the current rank's tensor. Slice across the last dimension
    shard_dim_size = in_tensor.shape[-1] // dist.get_world_size(group)
    start_idx = (int)(dist.get_rank(group) * shard_dim_size)
    end_idx = (int)((dist.get_rank(group) + 1) * shard_dim_size)

    slices = [slice(None)] * len(in_tensor.shape)
    slices[-1] = slice(start_idx, end_idx)

    # Slice the tensor
    ret = in_tensor[slices]
    return ret


# 3. RR -> SR
def reshard_RR_to_SR(in_tensor, group):
    # get the current rank's tensor. Slice across the 2nd last dimension
    shard_dim_size = in_tensor.shape[-2] // dist.get_world_size(group)
    start_idx = (int)(dist.get_rank(group) * shard_dim_size)
    end_idx = (int)((dist.get_rank(group) + 1) * shard_dim_size)

    slices = [slice(None)] * len(in_tensor.shape)
    slices[-2] = slice(start_idx, end_idx)

    # Slice the tensor
    ret = in_tensor[slices]
    return ret


# ==================== src: RS ====================
# 4. RS -> RR
def reshard_RS_to_RR(in_tensor, group):
    temp = in_tensor.transpose(0, -1).contiguous()
    gather_shape = list(temp.shape)

    gather_shape[0] = dist.get_world_size(group) * gather_shape[0]

    ret = torch.empty(gather_shape, dtype=temp.dtype, device=in_tensor.device)
    dist.all_gather_into_tensor(ret, temp, group=group)
    ret = ret.transpose(0, -1).contiguous()
    return ret


# 5. RS -> RS
#    Omitted


# 6. RS -> SR
def reshard_RS_to_SR(in_tensor, group):
    # TODO: Need a more efficient implementation
    in_shape = in_tensor.shape
    chunk_shape = list(in_shape[:-2]) + [
        in_shape[-2] // dist.get_world_size(group),
        in_shape[-1],
    ]

    splitted_tensor = torch.split(
        in_tensor, in_shape[-2] // dist.get_world_size(group), dim=-2
    )

    for i in range(dist.get_world_size(group)):
        send_tensor = splitted_tensor[i].contiguous()

        if dist.get_rank() != i:
            dist.gather(send_tensor, dst=i, async_op=True, group=group)  # send
        else:
            gather_list = [
                torch.empty(chunk_shape, dtype=in_tensor.dtype, device=in_tensor.device)
                for _ in range(dist.get_world_size(group))
            ]
            handle = dist.gather(
                send_tensor, gather_list, dst=i, async_op=True, group=group
            )  # recv
    handle.wait()

    ret = torch.cat(gather_list, dim=-1)
    return ret


# ==================== src: SR ====================
# 7. SR -> RR
def reshard_SR_to_RR(in_tensor, group):
    temp = in_tensor.transpose(0, -2)
    temp = temp.contiguous()
    gather_shape = list(temp.shape)

    gather_shape[0] = dist.get_world_size(group) * gather_shape[0]

    ret = torch.empty(gather_shape, dtype=temp.dtype, device=in_tensor.device)
    dist.all_gather_into_tensor(ret, temp, group=group)

    ret = ret.transpose(0, -2).contiguous()
    return ret


# 8. SR -> RS
def reshard_SR_to_RS(in_tensor, group):
    in_shape = in_tensor.shape
    chunk_shape = list(in_shape[:-1]) + [in_shape[-1] // dist.get_world_size(group)]

    splitted_tensor = torch.split(
        in_tensor, in_shape[-1] // dist.get_world_size(group), dim=-1
    )

    for i in range(dist.get_world_size(group)):
        send_tensor = splitted_tensor[i].contiguous()

        if dist.get_rank() != i:
            dist.gather(send_tensor, dst=i, async_op=True, group=group)  # send
        else:
            gather_list = [
                torch.empty(chunk_shape, dtype=in_tensor.dtype, device=in_tensor.device)
                for _ in range(dist.get_world_size(group))
            ]
            handle = dist.gather(
                send_tensor, gather_list, dst=i, async_op=True, group=group
            )  # recv
    handle.wait()

    ret = torch.cat(gather_list, dim=-2)
    return ret


# 9. SR -> SR
#    Omitted
