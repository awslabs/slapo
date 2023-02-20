# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train with DeepSpeed ZeRO-3 or pipeline.
Note that this model is not from HuggingFace, but we just use a unified
file name for easy benchmarking.
"""
import os
import argparse

import deepspeed
import torch
import torch.distributed as dist

import slapo
from slapo.logger import get_logger
from slapo.op.cross_entropy import ParallelCrossEntropy
from slapo.utils.report import report_memory
from slapo.model_schedule import apply_schedule

from model import get_model_config, get_model
from utils import count_parameters, get_data_loader
from examples.utils import (
    train_with_deepspeed_engine,
    get_ds_config,
    create_dist_group_for_pipeline,
)

SINGLE_DEVICE_FOR_DEBUG = False

logger = get_logger()


def train(args):
    batch_size = args.batch_size
    micro_batch_size = args.micro_batch_size
    if micro_batch_size is None:
        micro_batch_size = 8

    num_pp, num_mp = 1, 1
    rank = args.local_rank
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    # Configurations.
    enable_pipeline = not SINGLE_DEVICE_FOR_DEBUG and not args.disable_pipeline
    if args.disable_schedule and args.checkpoint not in [0.0, 1.0]:
        raise ValueError("checkpoint must be 0.0 or 1.0 with disable_schedule")
    use_default_ckpt = args.checkpoint == 1.0 and args.disable_schedule

    topology, group = None, None
    if not SINGLE_DEVICE_FOR_DEBUG:
        deepspeed.init_distributed(dist_backend="nccl")
        logger.info("Use deepspeed to initialize", ranks=0)
        if enable_pipeline:
            num_pp, num_mp = 4, 1  # FIXME: May need to change for multi-node.
        else:
            logger.info("Pipeline disabled", ranks=0)
        topology, group = create_dist_group_for_pipeline(num_pp, num_mp)

        # FIXME: Pytorch _coalescing_manager requires all the ranks to join
        # if that is the first collective call in the given group.
        # We use the following broadcast as the first call for workaround,
        # and it will be removed once we implement the features to synchonrize
        # the model parameters during initialization.
        x = torch.tensor(0, device=torch.cuda.current_device())
        dist.broadcast(x, src=0)

    model_config = get_model_config(args.model_name)

    report_memory(msg="Before creating model")
    with slapo.init_empty_weights(enable=enable_pipeline):
        model = get_model(*model_config["block_size"])
    report_memory(msg="After creating model")
    logger.info(f"Param size {count_parameters(model)/1e9}B", ranks=0)

    # Partition layers (6, 8, 46, 6) for pipelining.
    pipeline_cuts = []
    if enable_pipeline:
        pipeline_cuts = [[], [], [1, 16, 33], []]
    logger.info(f"Pipeline cuts: {pipeline_cuts}", ranks=0)

    if args.disable_schedule:
        assert not enable_pipeline
        sch = slapo.create_schedule(model, group=group)
        if args.fp16:
            sch.mod = sch.mod.half()
    else:
        sch = apply_schedule(
            model,
            "wideresnet",
            model_config=model_config,
            prefix="model",
            ckpt_ratio=args.checkpoint,
            bcast_input=True,
            group=group,
            fp16=args.fp16,
            pipeline_cuts=pipeline_cuts,
            fuse_conv=(dist.get_world_size() == 1),
        )

    if enable_pipeline:
        # FIXME: is mbs=1 correct?
        batch_size = 32 if batch_size is None else batch_size
        ds_config_dict = get_ds_config(batch_size, 1, args.fp16, 0, "Pipeline")
        loss_fct = ParallelCrossEntropy(group=group)

        def loss_fn(outputs, labels):
            loss = loss_fct(outputs.contiguous(), labels.squeeze()).contiguous().mean()
            return loss

        model, _ = slapo.build(
            sch,
            topology=topology,
            target="deepspeed",
            config=ds_config_dict,
            loss_fn=loss_fn,
        )

    else:
        if batch_size is not None:
            micro_batch_size = batch_size // args.world_size
        else:
            assert micro_batch_size is not None
            batch_size = micro_batch_size * args.world_size

        logger.info(f"BS={batch_size}, MBS={micro_batch_size}", ranks=0)
        ds_config_dict = get_ds_config(
            batch_size, micro_batch_size, args.fp16, 3, "ZeRO-3"
        )
        model, _ = slapo.build(
            sch,
            topology=topology,
            target="deepspeed",
            config=ds_config_dict,
        )
        model = model.to(device)
    report_memory(msg="After building model")

    loader = get_data_loader(
        micro_batch_size, device, dtype=torch.float16 if args.fp16 else torch.float
    )

    num_iters = args.iter_nums
    if enable_pipeline:
        data_iter = iter(loader)
        for _ in range(num_iters):
            model.train_batch(data_iter=data_iter)
    else:
        train_with_deepspeed_engine(
            model,
            loader,
            steps=num_iters,
        )


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    # This is passed in via cmd
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=n_gpus)
    parser.add_argument("--iter_nums", type=int, default=40)
    parser.add_argument(
        "--model_name",
        type=str,
        default="wideresnet-250M",
        help="Model name",
    )
    parser.add_argument(
        "--checkpoint",
        type=float,
        default=0.0,
        help="Activation checkpointing ratio. 1.0 means all",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Total batch size",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=None,
        help="Micro batch size per GPU",
    )
    parser.add_argument(
        "--disable_pipeline",
        action="store_true",
        help="Disable pipeline and only use ZeRO-3",
    )
    parser.add_argument(
        "--disable_schedule",
        action="store_true",
        help="Disable Slapo schedule (only applicable with --disable-pipeline)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        help="Unused. Only for compatibility with other scripts.",
    )
    args = parser.parse_args()
    if os.environ.get("LOCAL_RANK"):
        args.local_rank = int(os.environ["LOCAL_RANK"])
    # The main entry point is called directly without using subprocess
    train(args)
