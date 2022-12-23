# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
from argparse import ArgumentParser

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from slapo.logger import get_logger
from model import get_model

NO_DEEPSPEED = bool(int(os.environ.get("NO_DEEPSPEED", "0")))

logger = get_logger("WideResNet")

def count_parameters(model):
    try:
        return sum(p.ds_numel for p in model.parameters())
    except:
        return sum(p.numel() for p in model.parameters())

def get_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--num-epochs", help="epochs", default=10, type=int)
    # local_rank and job_name are only used by deepspeed
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--job_name", type=str)
    deepspeed.add_config_arguments(parser)
    args = parser.parse_args(sys.argv[1:])

    config = json.load(open(args.deepspeed_config, "r", encoding="utf-8"))
    args.config = config

    return args


class FakeDataset(Dataset):
    """temporarily using this to avoid having to load a real dataset"""

    def __init__(self, epoch_sz: int, W=224, H=224) -> None:
        self.__epoch_sz = epoch_sz
        self.W = W
        self.H = H

    def __getitem__(self, _: int) -> dict:
        return torch.rand((3, self.H, self.W)), torch.randint(0, 1000, (1,))

    def __len__(self) -> int:
        return self.__epoch_sz


def get_dataloader(args):
    train_dataset = FakeDataset(8192 * 100)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    micro_bs = args.config["train_micro_batch_size_per_gpu"]
    dtype = torch.half if args.config["fp16"]["enabled"] else torch.float

    def _collate_fn(batch):
        data = torch.vstack([e[0].unsqueeze(0) for e in batch]).to(
            args.local_rank, dtype=dtype
        )
        labels = torch.vstack([e[1].unsqueeze(0) for e in batch]).to(args.local_rank)
        return data, labels

    training_loader = DataLoader(
        train_dataset,
        batch_size=micro_bs,
        collate_fn=_collate_fn,
        sampler=DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            seed=1234,
        ),
    )

    return training_loader


def main():
    args = get_args()
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    if NO_DEEPSPEED:
        dist.init_process_group(backend="nccl")
    else:
        deepspeed.init_distributed(dist_backend="nccl")

    # get data_loader
    training_dataloader = get_dataloader(args)

    model_config = args.config["wideresnet_config"]
    model = get_model(model_config["width_per_group"], model_config["layers"])
    logger.info(model, ranks=0)
    logger.info(f"model param size {count_parameters(model)/1e9} B", ranks=0)
    loss_fn = torch.nn.CrossEntropyLoss()

    if NO_DEEPSPEED:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    else:
        model, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=args.config,
        )
    for e in range(args.num_epochs):
        for n, inputs in enumerate(training_dataloader):
            if n < 5 and e < 1:
                logger.info(
                    f"inputs sizes {[e.size() for e in inputs]}, "
                    f"device {[e.device for e in inputs]}",
                    ranks=0,
                )
            outputs = model(inputs[0])
            loss = loss_fn(outputs, inputs[1].squeeze())
            if NO_DEEPSPEED:
                loss.backward()
                optimizer.step()
            else:
                model.backward(loss)
                model.step()

            if (
                not hasattr(model, "is_gradient_accumulation_boundary")
                or model.is_gradient_accumulation_boundary()
            ):
                logger.info(f"{e} {n}, LOSS: {loss.item()}", ranks=0)

            loss = None


if __name__ == "__main__":
    main()
