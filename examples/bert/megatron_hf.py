# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# https://github.com/NVIDIA/Megatron-LM/blob/52e6368/pretrain_bert.py
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""Pretrain BERT"""

from functools import partial
import os

import torch
import torch.nn.functional as F

from transformers import AutoConfig, BertModel

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import mpu
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.model import ModelType
from megatron.model.bert_model import post_language_model_processing, BertLMHead
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from megatron.model.utils import init_method_normal


def get_model(
    model_name,
    padded_vocab_size=None,
    binary_head=False,
    add_pooling_layer=True,
    disable_flash_attn=False,
    fp16=True,
    ckpt_ratio=0.0,
    impl="slapo",
    delay_init=True,
):
    config = AutoConfig.from_pretrained(model_name)
    if padded_vocab_size is not None:
        config.vocab_size = padded_vocab_size
    config.type_vocab_size = 2 if binary_head else 0

    if "slapo" in impl:
        import slapo
        from slapo.utils.report import report_memory
        from model import schedule_model

        report_memory()
        with slapo.init_empty_weights(enable=delay_init):
            model = BertModel(config, add_pooling_layer=add_pooling_layer)
        report_memory()
        sch = schedule_model(
            model,
            config,
            attn_op_name="native_xformers" if disable_flash_attn else "cuda",
            fp16=fp16,
            ckpt_ratio=ckpt_ratio,
            disable_fuse_bias_gelu=False,
            delay_init=delay_init,
        )
        model, _ = slapo.build(sch, init_weights=model._init_weights)
        report_memory()

    elif impl == "torchscript":
        if ckpt_ratio > 0:
            raise RuntimeError("TorchScript cannot support ckpt")

        # https://huggingface.co/docs/transformers/torchscript
        config.torchscript = True
        model = BertModel(config=config, add_pooling_layer=add_pooling_layer)
        if fp16:
            model.half()
        model.cuda()

        bs = 8
        seq_length = 512
        device = "cuda"
        input_ids = torch.ones(bs, seq_length, dtype=torch.long, device=device)
        attention_mask = torch.ones(bs, seq_length, dtype=torch.long, device=device)
        token_type_ids = torch.ones(bs, seq_length, dtype=torch.long, device=device)
        model = torch.jit.trace(model, [input_ids, attention_mask, token_type_ids])

    elif impl == "eager":
        model = BertModel(config, add_pooling_layer=add_pooling_layer)
        if ckpt_ratio > 0:
            model.gradient_checkpointing_enable()

    else:
        raise RuntimeError(f"Unrecognized impl `{impl}`")

    if fp16:
        model.half()
    model.cuda()
    return model


def model_provider(pre_process=True, post_process=True):
    args = get_args()
    model_name = os.environ.get("MODEL_NAME", None)
    if model_name is None:
        raise RuntimeError("'MODEL_NAME' not found in environment")
    disable_flash_attn = bool(int(os.environ.get("DISABLE_FLASH_ATTN", "0")))
    ckpt_ratio = 0.0
    if args.recompute_granularity is not None:
        ckpt_ratio = os.environ.get("ckpt_ratio", 1.0)
        if ckpt_ratio == "selective":
            raise NotImplementedError
        ckpt_ratio = 1.0 if ckpt_ratio == "full" else float(ckpt_ratio)

    impl = os.environ.get("IMPL", None)
    if impl is None:
        raise RuntimeError("'IMPL' not found in environment")

    class BertWithLMHead(torch.nn.Module):
        def __init__(self, add_pooling_layer):
            super().__init__()
            self.bert = get_model(
                model_name,
                args.padded_vocab_size,
                args.bert_binary_head,
                add_pooling_layer,
                disable_flash_attn,
                args.fp16,
                ckpt_ratio,
                impl,
            )
            init_method = init_method_normal(args.init_method_std)
            self.binary_head = torch.nn.Linear(args.hidden_size, 2)
            self.lm_head = BertLMHead(
                self.bert.embeddings.word_embeddings.weight.size(0),
                args.hidden_size,
                init_method,
                args.layernorm_epsilon,
                parallel_output=True,
            )

        def set_input_tensor(self, input_tensor):
            # We don't support Megatron pipeline so this has no effect.
            pass

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            labels=None,
        ):
            # Note: other arguments (e.g., head_mask) are not supported yet.
            assert attention_mask is not None
            assert token_type_ids is not None
            output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            if isinstance(output, dict):
                lm_output = output["last_hidden_state"].transpose(0, 1).contiguous()
                pooled_output = output["pooler_output"]
            elif isinstance(output, tuple):
                lm_output = output[0].transpose(0, 1).contiguous()
                pooled_output = output[1]
            else:
                raise RuntimeError
            output_tensor = post_language_model_processing(
                lm_output,
                pooled_output,
                self.lm_head,
                self.binary_head,
                labels,
                self.bert.embeddings.word_embeddings.weight,
                args.fp16_lm_cross_entropy,
            )
            return output_tensor

    model = BertWithLMHead(add_pooling_layer=args.bert_binary_head)
    return model


def get_batch(data_iterator):
    """Build the batch."""

    # Items and their type.
    keys = ["text", "types", "labels", "is_random", "loss_mask", "padding_mask"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b["text"].long()
    types = data_b["types"].long()
    sentence_order = data_b["is_random"].long()
    loss_mask = data_b["loss_mask"].float()
    lm_labels = data_b["labels"].long()
    padding_mask = data_b["padding_mask"].long()

    return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask


def loss_func(loss_mask, sentence_order, output_tensor):
    lm_loss_, sop_logits = output_tensor

    lm_loss_ = lm_loss_.float()
    loss_mask = loss_mask.float()
    lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    if sop_logits is not None:
        sop_loss = F.cross_entropy(
            sop_logits.view(-1, 2).float(), sentence_order.view(-1), ignore_index=-1
        )
        sop_loss = sop_loss.float()
        loss = lm_loss + sop_loss
        averaged_losses = average_losses_across_data_parallel_group([lm_loss, sop_loss])
        return loss, {"lm loss": averaged_losses[0], "sop loss": averaged_losses[1]}

    else:
        loss = lm_loss
        averaged_losses = average_losses_across_data_parallel_group([lm_loss])
        return loss, {"lm loss": averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator").start()
    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = get_batch(
        data_iterator
    )
    timers("batch-generator").stop()

    if not args.bert_binary_head:
        types = None

    # Forward pass through the model.
    output = model(tokens, padding_mask, labels=lm_labels, token_type_ids=types)

    return output, partial(loss_func, loss_mask, sentence_order)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0("> building train, validation, and test datasets " "for BERT ...")
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=args.seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        binary_head=args.bert_binary_head,
    )
    print_rank_0("> finished creating BERT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "BertWordPieceLowerCase"},
    )
