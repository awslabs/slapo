# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain GPT"""
import os

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, ModelType
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.model.gpt_model import post_language_model_processing


def model_schedule(model, config):
    import ms
    import inspect
    import torch.distributed as dist
    from gpt_schedule import (
        replace_qkv,
        replace_softmax,
        shard_word_embedding,
        shard_qkv,
        replace_and_shard_mlp,
        remove_cast,
        replace_attention,
        checkpoint,
    )

    print("Using model schedule to optimize")
    print("World size: {}, rank: {}".format(dist.get_world_size(), dist.get_rank()))

    is_gpt2 = "GPT2" in config.architectures[0]
    assert not is_gpt2, "GPT-2 schedule is not working"
    args = get_args()
    disable_flash_attn = bool(int(os.environ.get("DISABLE_FLASH_ATTN", "0")))
    input_names = list(model.dummy_inputs.keys())
    input_names += ["attention_mask", "position_ids"]#, "token_type_ids"]
    sig = inspect.signature(model.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }

    if args.fp16:
        print("Change model dtype to fp16")
        model.half()

    sch = ms.create_schedule(
        model,
        world_size=dist.get_world_size(),
        rank=dist.get_rank(),
        tracer="huggingface",
        concrete_args=concrete_args,
    )

    # Deal with attention.
    attn_path, out_proj_name = "h.N.attn.attention", "out_proj"
    n_layer, n_head, hidden_size, vocab_size = (
        config.num_layers,
        config.num_heads,
        config.hidden_size,
        config.vocab_size,
    )
    if not disable_flash_attn:
        replace_attention(sch, config, attn_path)
    else:
        remove_cast(sch)
        # replace_softmax(sch)
        replace_qkv(sch, n_layer, n_head, hidden_size)
        if sch.world_size > 1:
            shard_qkv(sch, n_layer, attn_path, out_proj_name=out_proj_name)

    # Deal with MLP.
    replace_and_shard_mlp(sch, config)

    # Deal with embedding.
    shard_word_embedding(sch, vocab_size)

    # Gradient checkpointing
    if args.recompute_method is not None:
        checkpoint(sch, config)

    model, _ = ms.build(sch)
    if args.fp16:
        model.half()
    model.cuda()
    return model


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    from transformers import AutoConfig, GPTNeoModel

    args = get_args()
    model_name = os.environ.get("MODEL_NAME", None)
    if model_name is None:
        raise RuntimeError(f"'MODEL_NAME' not found in environment")
    if "gpt-neo" not in model_name:
        raise RuntimeError(f"Only gpt-neo is supported for now, got {model_name}")

    print_rank_0(f"Building HF {model_name} ...")
    config = AutoConfig.from_pretrained(model_name)
    config.vocab_size = args.padded_vocab_size
    config.use_cache = False
    print(config)

    class GPTWithLMHead(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            orig_model = GPTNeoModel(config)
            self.gpt = model_schedule(orig_model, config)
            print(self.gpt)

        def forward(
            self,
            input_ids=None,
            position_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
        ):
            assert token_type_ids is None, "Not traced"
            output = self.gpt(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # token_type_ids=token_type_ids,
                position_ids=position_ids,
            )
            lm_output = output["last_hidden_state"].transpose(0, 1).contiguous()

            output_tensor = post_language_model_processing(
                lm_output, labels, self.gpt.wte.weight, True, args.fp16_lm_cross_entropy
            )
            return output_tensor

    model = GPTWithLMHead(config)
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ["text"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b["text"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    # TODO: This may not be necessary for GPT-Neo.
    batch_size = tokens.shape[0]
    seq_length = tokens.shape[1]
    # The shape of attention_mask is (1, 1, seq_length, seq_length) while the 3rd dim
    # is broadcast. The required shape for HF GPT is (batch, 1, 1, seq_length).
    # (1, 1, S, S) -> (1, 1, S, 1)
    attention_mask = attention_mask[..., -1:]
    # (1, 1, S, 1) -> (1, 1, 1, S)
    attention_mask = attention_mask.reshape((1, 1, 1, seq_length))
    # (1, 1, 1, S) -> (B, 1, 1, S)
    attention_mask = attention_mask.repeat(batch_size, 1, 1, 1)

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {"lm loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator").start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers("batch-generator").stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0("> building train, validation, and test datasets " "for GPT ...")
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
    )
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
