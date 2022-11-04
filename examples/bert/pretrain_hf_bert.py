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

"""Pretrain BERT"""

from functools import partial

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import mpu
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.model import ModelType
from megatron.model.module import MegatronModule
from megatron.model.bert_model import post_language_model_processing, BertLMHead
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from megatron.model.utils import init_method_normal, get_linear_layer


def model_schedule(model, config):
    import ms
    import inspect
    import torch.distributed as dist
    from bert_schedule import replace_qkv, shard_params

    print("Using model schedule to optimize")
    print("World size: {}, rank: {}".format(dist.get_world_size(), dist.get_rank()))

    input_names = list(model.dummy_inputs.keys())
    input_names += ["attention_mask", "token_type_ids"]
    sig = inspect.signature(model.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    model.half()
    sch = ms.create_schedule(
        model,
        None,
        world_size,
        rank,
        config={"tracer": "huggingface", "concrete_args": concrete_args},
    )

    replace_qkv(sch, config.hidden_size, config.num_attention_heads, config.num_hidden_layers)
    if world_size > 1:
        shard_params(sch, config.num_hidden_layers, fused_qkv=True, prefix="")

    model, _ = ms.build(sch)
    model.half()
    model.cuda()
    return model


def model_provider(pre_process=True, post_process=True):
    from transformers import AutoConfig, AutoModelForMaskedLM, BertModel
    from transformers import PreTrainedModel

    print_rank_0("building HF BERT model ...")
    config = AutoConfig.from_pretrained("bert-large-uncased")
    args = get_args()
    config.vocab_size = args.padded_vocab_size
    config.type_vocab_size = 2 if args.bert_binary_head else 0
    print(config)

    class BertWithLMHead(torch.nn.Module):
        def __init__(self, config, add_pooling_layer):
            super().__init__()
            self.bert = model_schedule(
                BertModel(config, add_pooling_layer=add_pooling_layer), config
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

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            #            output = self.bert(input_ids, attention_mask, token_type_ids,
            #                               position_ids, head_mask, inputs_embeds, encoder_hidden_states,
            #                               encoder_attention_mask, past_key_values, use_cache,
            #                               output_attentions, output_hidden_states, return_dict)
            output = self.bert(input_ids, attention_mask, token_type_ids)
            lm_output = output["last_hidden_state"].transpose(0, 1).contiguous()
            pooled_output = output["pooler_output"]
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

    model = BertWithLMHead(config, add_pooling_layer=args.bert_binary_head)
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
