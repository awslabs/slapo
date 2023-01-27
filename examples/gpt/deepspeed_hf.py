# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.utils import RepeatingLoader
from transformers import GPTNeoForCausalLM, AutoConfig

import slapo
from slapo.logger import get_logger
from slapo.op.cross_entropy import ParallelCrossEntropy
from slapo.utils.report import report_memory

from model import schedule_model
from examples.utils import (
    train_with_torch,
    get_ds_config,
    create_dist_group_for_pipeline,
    generate_pipeline_cuts,
)

from examples.data_util import get_dataloader

SINGLE_DEVICE_FOR_DEBUG = False

logger = get_logger()


def reconfig_model(args, model_config):
    if args.hidden_size > 0:
        model_config.hidden_size = args.hidden_size
        model_config.num_layers = args.nlayers
        model_config.num_heads = args.num_attn_heads

        assert args.nlayers % 2 == 0, "number of layers must be even"
        # config "attention_types"
        model_config.attention_types = [[["global"], model_config.num_layers]]
        model_config.attention_layers = ["global"] * model_config.num_layers

    return model_config


def train(args):
    batch_size = args.batch_size
    micro_batch_size = args.micro_batch_size

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
            # num_pp, num_mp = 4, 2 # For single node testing.
            num_pp = args.pmp
            num_mp = args.tmp
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

    logger.info(f"TMP {num_mp}, PMP {num_pp}", ranks=[0])
    # https://huggingface.co/EleutherAI/gpt-neo-2.7B/blob/main/config.json
    config = AutoConfig.from_pretrained(args.model_name)
    # FIXME: This model has vocab size 50257 that cannot be sharded by 2,
    # so we pad it to 50258 in this example. In practice, the tokenizer
    # should be used to pad the vocab size to a multiple of 2.
    config.vocab_size = (config.vocab_size // 8 + 1) * 8
    config.use_cache = False
    config.gradient_checkpointing = use_default_ckpt
    config = reconfig_model(args, config)
    logger.info(config, ranks=[0])

    report_memory(msg="Before creating model")
    with slapo.init_empty_weights(enable=enable_pipeline):
        model = GPTNeoForCausalLM(config)
    report_memory(msg="After creating model")

    # Evenly partition layers for pipelining.
    if enable_pipeline:
        pipeline_cuts = generate_pipeline_cuts(config.num_layers, num_pp)
    elif SINGLE_DEVICE_FOR_DEBUG:
        pipeline_cuts = generate_pipeline_cuts(config.num_layers, 4)
    else:
        pipeline_cuts = []
    logger.info(f"Pipeline cuts: {pipeline_cuts}", ranks=0)

    if args.disable_schedule:
        assert not enable_pipeline
        sch = slapo.create_schedule(model, group=group)
    else:
        sch = schedule_model(
            model,
            config,
            prefix="transformer",
            ckpt_ratio=args.checkpoint,
            bcast_input=True,
            group=group,
            pipeline_cuts=pipeline_cuts,
            delay_init=enable_pipeline,
            sequence_parallel=args.sequence_parallel,
        )

    if enable_pipeline:
        # FIXME: is mbs=1 correct?
        batch_size = 16 if batch_size is None else batch_size
        micro_batch_size = 4 if micro_batch_size is None else micro_batch_size
        ds_config_dict = get_ds_config(
            batch_size, micro_batch_size, True, False, "Pipeline", False
        )
        loss_fct = ParallelCrossEntropy(group=group)

        def loss_fn(outputs, labels):
            prediction_scores = outputs
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            lm_loss = loss_fct(shifted_prediction_scores, labels)
            lm_loss = lm_loss.contiguous().mean()
            return lm_loss

        model, _ = slapo.build(
            sch,
            topology=topology,
            target="deepspeed",
            config=ds_config_dict,
            loss_fn=loss_fn,
            init_weights=model._init_weights,
        )
    else:
        if batch_size is not None and micro_batch_size is None:
            micro_batch_size = batch_size // args.world_size
        if batch_size is None and micro_batch_size is not None:
            batch_size = micro_batch_size * args.world_size

        logger.info(f"BS={batch_size}, MBS={micro_batch_size}", ranks=0)
        ds_config_dict = get_ds_config(
            batch_size, micro_batch_size, True, True, "ZeRO-3"
        )
        model, _ = slapo.build(
            sch,
            topology=topology,
            target="deepspeed",
            config=ds_config_dict,
            init_weights=model._init_weights,
        )
        model = model.to(device)
    report_memory(msg="After building model")

    random_seed = 2000 + dist.get_rank()
    torch.manual_seed(random_seed)

    # for now always use seq_length 1024
    # TODO: make the dataloader generic to different sequence length
    train_loader, _ = get_dataloader(args.model_name, micro_batch_size, enable_pipeline)

    loader = RepeatingLoader(train_loader)

    num_iters = args.iter_nums
    if enable_pipeline:
        data_iter = iter(loader)
        for _ in range(num_iters):
            model.train_batch(data_iter=data_iter)
    else:
        train_with_torch(model, loader, steps=num_iters)


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
        default="EleutherAI/gpt-neo-2.7B",
        help="Model name",
    )
    parser.add_argument(
        "--checkpoint",
        type=float,
        default=0.0,
        help="Activation checkpointing ratio. 1.0 means all",
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
        "--seq_len",
        type=int,
        default=1024,
        help="Sequence length",
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
        "--hidden-size",
        type=int,
        default=-1,
        help="Config hidden size of the model, if it is negative value,"
        " it uses default value associated with the model name",
    )
    parser.add_argument(
        "--nlayers", type=int, default=-1, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num-attn-heads", type=int, default=-1, help="Number of attention heads"
    )
    parser.add_argument(
        "--pmp", type=int, default=2, help="Pipeline model parallel size"
    )
    parser.add_argument("--tmp", type=int, default=8, help="Tensor parallel size")
    parser.add_argument(
        "--sequence_parallel",
        action="store_true",
        help="Sequence parallelism is enabled",
    )
    args = parser.parse_args()

    if args.hidden_size > 0:
        assert args.nlayers > 0, "must have nlayers > 0"
        assert args.num_attn_heads > 0, "must have num_attn_heads > 0"

    args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    train(args)
