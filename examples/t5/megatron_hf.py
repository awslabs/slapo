# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# https://github.com/NVIDIA/Megatron-LM/blob/52e6368/pretrain_t5.py
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""Pretrain T5"""
from functools import partial
import os

import torch

from transformers import AutoConfig, T5Model

from megatron import get_args, get_timers, mpu, print_rank_0
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.model import ModelType
from megatron.model.t5_model import T5LMHead
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group


"""
Pipeline parallelism for T5
===========================
T5 is a model architecture with both encoder and decoder blocks.
Consequently, pipeline parallelism is implemented slightly differently
compared to architectures like GPT and BERT.
In particular, when pipeline_model_parallel_world_size > 1, each stage
either executes an encoder block or a decoder block. The
--pipeline-model-parallel-split-rank argument controls the rank at which
the split happens: all ranks lower than this argument execute the
encoder block, and all ranks equal to or higher than this argument value
execute the decoder block.
In the encoder section of the model, only one tensor is sent downstream:
the intermediate encoder_hidden_state. In the decoder section of the
model, two tensors are sent downstream in the forward pass: the fully
computed encoder_hidden_state, and the intermediate decoder_hidden_state.
In particular, these are the shapes of the tensors sent between
different workers:
    If rank is in decoder section:
        intermediate decoder_hidden_state (pre-transpose),
        complete encoder_hidden_state (post-transpose).
    If rank is at boundary between encoder and decoder sections:
        complete encoder_hidden_state (post-transpose).
    If rank is in encoder section:
        intermediate encoder_hidden_state (pre-transpose).
Additionally, we have code in the backward_step function in schedules.py
to accumulate the encoder_hidden_state gradient across skip connections
(encoder_hidden_state fed in as input to each layer in the decoder).
"""


def get_model(
    model_name,
    padded_vocab_size=None,
    disable_flash_attn=False,
    fp16=True,
    ckpt_ratio=0.0,
    impl="slapo",
    delay_init=True,
):
    config = AutoConfig.from_pretrained(model_name)
    config.vocab_size = padded_vocab_size
    config.use_cache = False

    if "slapo" in impl:
        import slapo
        from slapo.utils.report import report_memory
        from slapo.model_schedule.t5 import schedule_model

        report_memory()
        with slapo.init_empty_weights(enable=delay_init):
            model = T5Model(config)
        report_memory()
        sch = schedule_model(
            model,
            config,
            disable_flash_attn=disable_flash_attn,
            fp16=fp16,
            ckpt_ratio=ckpt_ratio,
            delay_init=delay_init,
        )
        model, _ = slapo.build(sch, init_weights=model._init_weights)
        report_memory()

    elif impl == "torchscript":
        if ckpt_ratio > 0:
            raise RuntimeError("TorchScript cannot support ckpt")

        config.torchscript = True
        model = T5Model(config=config)
        if fp16:
            model.half()
        model.cuda()

        bs = 8
        seq_length = 512
        device = "cuda"
        encoder_input_ids = torch.ones(bs, seq_length, dtype=torch.long, device=device)
        decoder_input_ids = torch.ones(bs, seq_length, dtype=torch.long, device=device)
        encoder_attn_mask = torch.ones(bs, seq_length, dtype=torch.long, device=device)
        decoder_attn_mask = torch.ones(bs, seq_length, dtype=torch.long, device=device)
        model = torch.jit.trace(
            model,
            [
                encoder_input_ids,
                decoder_input_ids,
                encoder_attn_mask,
                decoder_attn_mask,
            ],
        )

    elif impl == "eager":
        model = T5Model(config)
        if ckpt_ratio > 0:
            model.gradient_checkpointing_enable()

    else:
        raise RuntimeError(f"Unrecognized impl `{impl}`")

    if fp16:
        model.half()
    model.cuda()
    return model


def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True
):
    """Build the model."""

    args = get_args()
    model_name = os.environ.get("MODEL_NAME", None)
    if model_name is None:
        raise RuntimeError(f"'MODEL_NAME' not found in environment")
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

    class T5WithLMHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = get_model(
                model_name,
                args.padded_vocab_size,
                disable_flash_attn,
                args.fp16,
                ckpt_ratio,
                impl,
            )

            if post_process and add_decoder:
                self.lm_head = T5LMHead(
                    self.model.encoder.embed_tokens.weight.size(0), True
                )

        def post_model_processing(self, lm_output, lm_labels=None):
            # Output. [s, b, h]
            lm_logits = self.lm_head(lm_output, self.model.encoder.embed_tokens.weight)

            if lm_labels is None:
                # [s b h] => [b s h]
                return lm_logits.transpose(0, 1).contiguous()
            else:
                # [b s] => [s b]
                lm_labels = lm_labels.transpose(0, 1).contiguous()
                if args.fp16_lm_cross_entropy:
                    assert lm_logits.dtype == torch.half
                    lm_loss = mpu.vocab_parallel_cross_entropy(lm_logits, lm_labels)
                else:
                    lm_loss = mpu.vocab_parallel_cross_entropy(
                        lm_logits.float(), lm_labels
                    )
                # [s b] => [b s]
                lm_loss = lm_loss.transpose(0, 1).contiguous()
            return lm_loss

        def set_input_tensor(self, input_tensor):
            # We don't support Megatron pipeline so this has no effect.
            pass

        def forward(
            self,
            encoder_input_ids,
            decoder_input_ids,
            encoder_attn_mask,
            decoder_attn_mask,
            encoder_decoder_attn_mask,
            tokentype_ids=None,
            lm_labels=None,
            enc_hidden_states=None,
        ):
            assert tokentype_ids is None, "Not traced"
            assert enc_hidden_states is None, "Not traced"
            # HF model uses decoder_attn_mask for all attention layers in decoder,
            # so encoder_decoder_attn_mask is not used.

            output = self.model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attn_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attn_mask,
            )
            if isinstance(output, dict):
                lm_output = output["last_hidden_state"].transpose(0, 1).contiguous()
            elif isinstance(output, tuple):
                lm_output = output[0].transpose(0, 1).contiguous()
            else:
                raise RuntimeError

            output_tensor = self.post_model_processing(lm_output, lm_labels)
            return output_tensor

    model = T5WithLMHead()
    return model


def get_batch(data_iterator):
    """Build the batch."""

    keys = [
        "text_enc",
        "text_dec",
        "labels",
        "loss_mask",
        "enc_mask",
        "dec_mask",
        "enc_dec_mask",
    ]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_enc = data_b["text_enc"].long()
    tokens_dec = data_b["text_dec"].long()
    labels = data_b["labels"].long()
    loss_mask = data_b["loss_mask"].float()

    enc_mask = data_b["enc_mask"] < 0.5
    dec_mask = data_b["dec_mask"] < 0.5
    enc_dec_mask = data_b["enc_dec_mask"] < 0.5

    def post_process_mask(mask):
        # The shape of attention mask is (batch, seq_length, seq_length) while
        # HF model expects (batch, 1, 1, seq_length).
        # (B, 1, S, S) -> (B, 1, S, 1)
        mask = mask[..., -1:]
        # (B, 1, S, 1) -> (B, 1, S)
        mask = mask.squeeze(-1)
        return mask

    enc_mask = post_process_mask(enc_mask)
    dec_mask = post_process_mask(dec_mask)

    return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask


def loss_func(loss_mask, output_tensor):
    lm_loss_ = output_tensor.float()
    lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    loss = lm_loss
    averaged_losses = average_losses_across_data_parallel_group([lm_loss])

    return loss, {"lm loss": averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch generator").start()
    (
        tokens_enc,
        tokens_dec,
        loss_mask,
        lm_labels,
        enc_mask,
        dec_mask,
        enc_dec_mask,
    ) = get_batch(data_iterator)
    timers("batch generator").stop()

    # Forward model lm_labels
    output_tensor = model(
        tokens_enc,
        tokens_dec,
        enc_mask,
        dec_mask,
        enc_dec_mask,
        tokentype_ids=None,
        lm_labels=lm_labels,
    )

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0("> building train, validation, and test datasets " "for T5 ...")
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=args.encoder_seq_length,
        max_seq_length_dec=args.decoder_seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        dataset_type="t5",
    )
    print_rank_0("> finished creating T5 datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_and_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "BertWordPieceLowerCase"},
    )
