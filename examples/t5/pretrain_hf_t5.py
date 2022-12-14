# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright (c) 2020, NVIDIA CORPORATION.
# See: https://github.com/NVIDIA/Megatron-LM/blob/master/pretrain_t5.py

"""Pretrain T5"""
from functools import partial
import os

import torch

from megatron import (
    get_args,
    get_timers,
    mpu,
    print_rank_0
)
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

def model_schedule(model, config):
    import slapo
    import inspect
    import torch.distributed as dist

    print("Using model schedule to optimize")
    print(f"Model schedule with world size {dist.get_world_size()}, rank {dist.get_rank()}")

    args = get_args()
    disable_flash_attn = bool(int(os.environ.get("DISABLE_FLASH_ATTN", "0")))
    input_names = list(model.dummy_inputs.keys())
    input_names += ["attention_mask"]
    sig = inspect.signature(model.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    if args.fp16:
        print("Change model dtype to fp16")
        model.half()

    sch = slapo.create_schedule(
        model,
        world_size=dist.get_world_size(),
        rank=dist.get_rank(),
        tracer="huggingface",
        concrete_args=concrete_args,
    )

    model, _ = slapo.build(sch)
    if args.fp16:
        model.half()
    model.cuda()
    return model


def model_provider(pre_process=True, post_process=True,
                   add_encoder=True, add_decoder=True):
    """Build the model."""
    from transformers import AutoConfig, T5Model
    args = get_args()
    model_name = os.environ.get("MODEL_NAME", None)
    if model_name is None:
        raise RuntimeError(f"'MODEL_NAME' not found in environment")
    print_rank_0(f'Building HF {model_name} ...')
    config = AutoConfig.from_pretrained(model_name)
    config.vocab_size = args.padded_vocab_size
    config.use_cache = False
    print_rank_0(config)

    class T5WithLMHead(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            orig_model = T5Model(config)
            self.model = model_schedule(orig_model, config)
            print_rank_0(self.model)

            if post_process and add_decoder:
                self.lm_head = T5LMHead(self.model.encoder.embed_tokens.weight.size(0), True)

        def post_model_processing(self, lm_output, lm_labels=None):
            # Output. [s, b, h]
            lm_logits = self.lm_head(lm_output, self.model.encoder.embed_tokens.weight)

            if lm_labels is None:
                # [s b h] => [b s h]
                return lm_logits.transpose(0,1).contiguous()
            else:
                # [b s] => [s b]
                lm_labels = lm_labels.transpose(0,1).contiguous()
                if args.fp16_lm_cross_entropy:
                    assert lm_logits.dtype == torch.half
                    lm_loss = mpu.vocab_parallel_cross_entropy(lm_logits, lm_labels)
                else:
                    lm_loss = mpu.vocab_parallel_cross_entropy(lm_logits.float(), lm_labels)
                # [s b] => [b s]
                lm_loss = lm_loss.transpose(0,1).contiguous()
            return lm_loss

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

            output = self.model(input_ids=encoder_input_ids,
                                attention_mask=encoder_attn_mask,
                                decoder_input_ids=decoder_input_ids,
                                decoder_attention_mask=decoder_attn_mask,)
            lm_output = output["last_hidden_state"].transpose(0, 1).contiguous()

            output_tensor = self.post_model_processing(lm_output, lm_labels)
            return output_tensor

    model = T5WithLMHead(config)
    return model


def get_batch(data_iterator):
    """Build the batch."""

    keys = ['text_enc', 'text_dec', 'labels', 'loss_mask',
            'enc_mask', 'dec_mask', 'enc_dec_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_enc = data_b['text_enc'].long()
    tokens_dec = data_b['text_dec'].long()
    labels = data_b['labels'].long()
    loss_mask = data_b['loss_mask'].float()

    enc_mask = (data_b['enc_mask'] < 0.5)
    dec_mask = (data_b['dec_mask'] < 0.5)
    enc_dec_mask = (data_b['enc_dec_mask'] < 0.5)

    def post_process_mask(seq_length, mask):
        # The shape of attention mask is (batch, seq_length, seq_length) while
        # HF model expects (batch, 1, 1, seq_length).
        # (B, 1, S, S) -> (B, 1, S, 1)
        mask = mask[..., -1:]
        # (B, 1, S, 1) -> (B, 1, S)
        mask = mask.squeeze(-1)
        return mask
    enc_mask = post_process_mask(tokens_enc.shape[1], enc_mask)
    dec_mask = post_process_mask(tokens_dec.shape[1], dec_mask)

    return tokens_enc, tokens_dec, loss_mask, labels, \
           enc_mask, dec_mask, enc_dec_mask


def loss_func(loss_mask, output_tensor):
    lm_loss_ = output_tensor.float()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    loss = lm_loss
    averaged_losses = average_losses_across_data_parallel_group([lm_loss])

    return loss, {'lm loss': averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    tokens_enc, tokens_dec, loss_mask, lm_labels, enc_mask, dec_mask, enc_dec_mask \
        = get_batch(data_iterator)
    timers('batch generator').stop()

    # Forward model lm_labels
    output_tensor = model(tokens_enc,
                          tokens_dec,
                          enc_mask,
                          dec_mask,
                          enc_dec_mask,
                          tokentype_ids=None,
                          lm_labels=lm_labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for T5 ...')
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
        dataset_type='t5')
    print_rank_0("> finished creating T5 datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider, ModelType.encoder_and_decoder,
             forward_step, args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
