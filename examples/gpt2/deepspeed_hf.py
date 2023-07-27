# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.utils import RepeatingLoader
from transformers import GPT2LMHeadModel, AutoConfig

import slapo
from slapo import set_random_seed
from slapo.logger import get_logger
from slapo.op import ParallelCrossEntropy
from slapo.utils.report import report_memory

from slapo.model_schedule import apply_schedule
from examples.utils import (
    train_with_deepspeed_engine,
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
        model_config.num_hidden_layers = args.nlayers
        model_config.num_attention_heads = args.num_attn_heads

    model_config.attn_pdrop = args.dropout
    model_config.resid_pdrop = args.dropout
    model_config.embd_pdrop = args.dropout

    model_config.activation_function = args.activation_function
    model_config.max_position_embeddings = args.seq_len

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
        logger.info("Use deepspeed to initialize")
        if enable_pipeline:
            # num_pp, num_mp = 4, 2 # For single node testing.
            num_pp = args.pmp
            num_mp = args.tmp
        else:
            logger.info("Pipeline disabled", ranks=0)
            num_pp = 1
            num_mp = args.tmp

        topology, group = create_dist_group_for_pipeline(num_pp, num_mp)

        # FIXME: Pytorch _coalescing_manager requires all the ranks to join
        # if that is the first collective call in the given group.
        # We use the following broadcast as the first call for workaround,
        # and it will be removed once we implement the features to synchonrize
        # the model parameters during initialization.
        dist.barrier()

    logger.info(f"TMP {num_mp}, PMP {num_pp}")
    # https://huggingface.co/EleutherAI/gpt-neo-2.7B/blob/main/config.json
    config = AutoConfig.from_pretrained(args.model_name)
    # FIXME: This model has vocab size 50257 that cannot be sharded by 2,
    # so we pad it to 50258 in this example. In practice, the tokenizer
    # should be used to pad the vocab size to a multiple of 2.
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - config.vocab_size % 8
    config.use_cache = False
    config.gradient_checkpointing = use_default_ckpt
    config = reconfig_model(args, config)
    logger.info(config, ranks=[0])

    report_memory(msg="Before creating model")
    with slapo.init_empty_weights():
        model = GPT2LMHeadModel(config)
    report_memory(msg="After creating model")

    # Evenly partition layers for pipelining.
    if enable_pipeline:
        pipeline_cuts = generate_pipeline_cuts(config.num_hidden_layers, num_pp)
    elif SINGLE_DEVICE_FOR_DEBUG:
        pipeline_cuts = generate_pipeline_cuts(config.num_hidden_layers, 4)
    else:
        pipeline_cuts = []
    logger.info(f"Pipeline cuts: {pipeline_cuts}", ranks=0)

    if args.disable_schedule:
        assert not enable_pipeline
        sch = slapo.create_schedule(model, group=group)
    else:
        sch = apply_schedule(
            model,
            "gpt2",
            model_config=config,
            prefix="transformer",
            attn_op_name=args.attn_op_name,
            ckpt_ratio=args.checkpoint,
            bcast_input=True,
            fp16=args.fp16,
            bf16=args.bf16,
            group=group,
            pipeline_cuts=pipeline_cuts,
            delay_init=enable_pipeline,
            sequence_parallel=args.sequence_parallel,
            checkpoint_method=args.checkpoint_method,
        )
    tp_rank = sch.rank

    loss_fct = ParallelCrossEntropy(group=group)

    def loss_fn(outputs, labels):
        prediction_scores = outputs
        shifted_prediction_scores = prediction_scores[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        lm_loss = loss_fct(shifted_prediction_scores, labels)
        lm_loss = lm_loss.contiguous().mean()
        return lm_loss

    # After scheduling, we check again whether the pipeline is really enabled.
    # If users specified to enable pipeline but the number of pipeline stage is 1,
    # then we set enable_pipeline=False for the rest process to propertly setup
    # DeepSpeed config and runtime engine.
    enable_pipeline = enable_pipeline and pipeline_cuts

    if enable_pipeline:
        batch_size = 16 if batch_size is None else batch_size
        micro_batch_size = 4 if micro_batch_size is None else micro_batch_size
        zero_opt_stage = 0
        ds_config_dict = get_ds_config(
            batch_size,
            micro_batch_size,
            args.fp16,
            zero_opt_stage,
            "Pipeline",
            args.bf16,
            args.sequence_parallel,
        )

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

        # if the TP == 1 use zero 3, otherwise disable ZeRO.
        zero_opt_stage = 3 if args.tmp == 1 else 0
        logger.info(f"BS={batch_size}, MBS={micro_batch_size}", ranks=0)
        ds_config_dict = get_ds_config(
            batch_size,
            micro_batch_size,
            args.fp16,
            zero_opt_stage,
            f"ZeRO-{zero_opt_stage}",
            args.bf16,
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

    pp_rank = None if args.disable_pipeline else model.mpu.get_pipe_parallel_rank()
    set_random_seed(
        2013,
        model.mpu.get_data_parallel_rank(),
        pp_rank,
        tp_rank,
        always_enable_tp_seed=args.sequence_parallel,
    )

    def getitem_fn(entry):
        ret = [
            entry["input_ids"],
            entry["attention_mask"],
            # position_ids
            torch.arange(len(entry["input_ids"]), requires_grad=False),
            entry["labels"],
        ]
        return ret

    def collate_fn(batch, enable_pipeline=True):
        input_ids = torch.tensor([x[0] for x in batch], dtype=torch.long)
        attention_mask = torch.tensor(
            [x[1] for x in batch], dtype=torch.float16, requires_grad=False
        )
        position_ids = torch.stack([x[2] for x in batch])
        position_ids.requires_grad = False
        labels = torch.tensor([x[3] for x in batch], dtype=torch.long)

        ret = [input_ids, attention_mask, position_ids, labels]
        if not enable_pipeline:
            # insert None in second and fourth position
            ret.insert(1, None)  # past_key_values
            ret.insert(3, None)  # token_type_ids
        else:
            # DeepSpeed pipeline does not accept None as input.
            pass

        # group first inputs
        return [ret[:-1], ret[-1]]

    train_loader, _ = get_dataloader(
        args.model_name,
        "wikitext-2-v1",
        micro_batch_size,
        enable_pipeline,
        collate_fn=collate_fn,
        getitem_fn=getitem_fn,
        mpu=model.mpu,
        max_seq_length=args.seq_len,
    )

    loader = RepeatingLoader(train_loader)

    num_iters = args.iter_nums
    if enable_pipeline:
        data_iter = iter(loader)
        for _ in range(num_iters):
            model.train_batch(data_iter=data_iter)
    else:
        # use the Parallel Loss
        loss_fn = None if args.tmp == 1 else loss_fn
        train_with_deepspeed_engine(
            model,
            loader,
            steps=num_iters,
            loss_fn=loss_fn,
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
        default="gpt2-xl",  # 1.5B
        help="Model name",
    )
    parser.add_argument(
        "--checkpoint",
        type=float,
        default=0.0,
        help="Activation checkpointing ratio. 1.0 means all",
    )
    parser.add_argument(
        "--checkpoint_method",
        type=str,
        default="head",
        help="Activation checkpointing method {'head', 'uniform'}",
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
        "--activation_function",
        type=str,
        default="gelu_new",
        help="Activation function",
    )
    parser.add_argument(
        "--attn_op_name",
        type=str,
        default="cuda",
        help="Attention op name {'native_xformers', 'cutlass', 'triton', 'cuda'}. "
        "'cuda' and 'triton' only support sm_80+, and other archs will "
        "fallback to 'cutlas'",
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
        "--dropout", type=float, default=0.1, help="Dropout probability"
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
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="fp16 is enabled. fp16 is enabled by default",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="bf16 is enabled",
    )
    args = parser.parse_args()
    if os.environ.get("LOCAL_RANK"):
        args.local_rank = int(os.environ["LOCAL_RANK"])

    if args.fp16 and args.bf16:
        raise ValueError(
            f"fp16={args.fp16} and bf16={args.bf16} cannot be enabled at the same time"
        )
    elif not args.fp16 and not args.bf16:
        args.fp16 = True
        logger.info("fp16 is enabled by default", ranks=0)

    if args.hidden_size > 0:
        assert args.nlayers > 0, "must have nlayers > 0"
        assert args.num_attn_heads > 0, "must have num_attn_heads > 0"

    # The main entry point is called directly without using subprocess
    train(args)
