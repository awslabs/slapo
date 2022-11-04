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
import os

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
    import torch
    import transformers.utils.fx as fx
    import torch.distributed as dist
    import torch.nn as nn

    print("Using model schedule to optimize")
    print("World size: {}, rank: {}".format(dist.get_world_size(), dist.get_rank()))

    args = get_args()
    input_names = list(model.dummy_inputs.keys())
    input_names += ["attention_mask", "token_type_ids"]
    sig = inspect.signature(model.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    class NewTracer(fx.HFTracer):

        def __init__(self) -> None:
            super(NewTracer, self).__init__()

        def trace(self, *args, **kwargs):
            graph = super().trace(*args, **kwargs)
            return graph

    if args.fp16:
        print("Change model dtype to fp16")
        model.half()
    traced_graph = NewTracer().trace(model, concrete_args=concrete_args)
    gm = fx.GraphModule(model, traced_graph)

    # Placeholder. Not effective for now.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    sch = ms.create_schedule(gm, optimizer, dist.get_world_size(), dist.get_rank())

    def replace_attention():
        print("Replace HF BertSelfAttention with xformer Attention")
        from ms.op.xformers_attn import BertSelfAttentionXFormer
    
        class SelfAttention_Pattern(ms.Pattern):
    
            def __init__(self, layer_num):
                super(SelfAttention_Pattern, self).__init__()
                self.layer_num = layer_num
    
            @staticmethod
            def func(x: torch.Tensor) -> torch.Tensor:
                return x
    
            def starting_point(self, node):
                if node.op != "call_module":
                    return False
                name = node.target
                if "layer.{}.attention.self".format(self.layer_num) in name:
                    return True
                else:
                    return False
    
        sch.trace_module()
        for i in range(config.num_hidden_layers):
            op_lst = sch.find(SelfAttention_Pattern(i))
            sch[op_lst[0][0].name.replace("_", ".")].replace(BertSelfAttentionXFormer, config)


    def replace_qkv():
        print("Replace HF QKV Dense with FusedQKV")

        class FusedQKV(nn.Module):
            
            def __init__(self, hidden_size = 768, num_heads = 12) -> None:
                super(FusedQKV, self).__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_size = hidden_size // num_heads
                self.fused_linear = nn.Linear(hidden_size, num_heads * self.head_size * 3)

            def transpose_for_scores(self, x):
                new_x_shape = x.size()[:-1] + (self.num_heads // dist.get_world_size(), self.head_size, 3)
                x = x.view(*new_x_shape)
                return x.permute(0, 2, 1, 3, 4)

            def forward(self, hidden_states): # [8, 512, 768]
                qkv = self.fused_linear(hidden_states)
                transposed_qkv = self.transpose_for_scores(qkv)
                return [torch.squeeze(t) for t in torch.split(transposed_qkv, 1, dim=-1)]

        class QKV_Pattern(ms.Pattern):
            
            def __init__(self, layer_num):
                super(QKV_Pattern, self).__init__()
                self.layer_num = layer_num

            @staticmethod
            def func(x: torch.Tensor) -> torch.Tensor:
                new_x_shape = x.size()[:-1] + (16, 1024)#(12, 768)
                x = x.view(new_x_shape)
                return x.permute(0, 2, 1, 3)

            def starting_point(self, node):
                if node.op != "call_module":
                    return False
                name = node.target
                if "layer.{}.".format(self.layer_num) in name and "self" in name:
                    if "query" in name or "key" in name or "value" in name:
                        return True
                return False

        for i in range(config.num_hidden_layers):
            op_lst = sch.find(QKV_Pattern(i))
            sch[op_lst].replace(FusedQKV, kwargs={"hidden_size": config.hidden_size, "num_heads": config.num_attention_heads}, seq=False)

    # Disable for now. Need retracing with a different granularity.
    #sch.trace_module()
    #replace_attention()

    sch.trace_module()
    replace_qkv()
    sch.trace_module()

    if dist.get_world_size() > 1:
        sch.trace_module()
        for i in range(config.num_hidden_layers):
            # MLP
            sch["encoder.layer.{}.intermediate.dense".format(i)].shard(axis=1, param="weight")
            sch["encoder.layer.{}.output.dense".format(i)].shard(axis=0, param="weight")
            sch["encoder.layer.{}.output.dense".format(i)].gather()
            sch["encoder.layer.{}.intermediate.dense".format(i)].bw_gather()

            # Attention
            name = "FusedQKV" if i == 0 else "FusedQKV_{}".format(i)
            sch[name].shard(axis=1, param="fused_linear.weight")
            sch["encoder.layer.{}.attention.output.dense".format(i)].shard(axis=0, param="weight")
            sch["encoder.layer.{}.attention.output.dense".format(i)].gather()
            sch[name].bw_gather(axis=1)

        # fix number of heads
        import operator
        for node in sch.gm.graph.nodes:
            if node.op == "call_function" and node.target == operator.add:
                if isinstance(node.args[1], tuple):
                    lst = list(node.args[1])
                    lst[0] = lst[0] // sch.world_size # num of heads
                    node.args = (node.args[0], tuple(lst))

    model, optimizer = ms.build(sch)
    if args.fp16:
        model.half()
    model.cuda()
    return model


def model_provider(pre_process=True, post_process=True):
    from transformers import AutoConfig, BertModel
    args = get_args()
    model_name = os.environ.get("MODEL_NAME", None)
    if model_name is None:
        raise RuntimeError("'MODEL_NAME' not found in environment")

    print_rank_0(f'Building HF {model_name} ...')
    config = AutoConfig.from_pretrained(model_name)
    config.vocab_size = args.padded_vocab_size
    config.type_vocab_size = 2 if args.bert_binary_head else 0
    print(config)


    class BertWithLMHead(torch.nn.Module):
        def __init__(self, config, add_pooling_layer):
            super().__init__()
            self.bert = model_schedule(BertModel(config, add_pooling_layer=add_pooling_layer), config)
            init_method = init_method_normal(args.init_method_std)
            self.binary_head = torch.nn.Linear(args.hidden_size, 2)
            self.lm_head = BertLMHead(self.bert.embeddings.word_embeddings.weight.size(0),
                    args.hidden_size, init_method, args.layernorm_epsilon, parallel_output=True)

        def forward(
            self,
            input_ids = None,
            attention_mask = None,
            token_type_ids = None,
            position_ids = None,
            labels = None,
        ):
            # Note: other arguments (e.g., head_mask) are not supported yet.
            output = self.bert(input_ids, attention_mask, token_type_ids)
            lm_output = output["last_hidden_state"].transpose(0, 1).contiguous()
            pooled_output = output["pooler_output"]
            output_tensor = post_language_model_processing(
                    lm_output, pooled_output,
                    self.lm_head, self.binary_head,
                    labels,
                    self.bert.embeddings.word_embeddings.weight,
                    args.fp16_lm_cross_entropy)
            return output_tensor

    model = BertWithLMHead(config, add_pooling_layer=args.bert_binary_head)
    return model

def get_batch(data_iterator):
    """Build the batch."""

    # Items and their type.
    keys = ['text', 'types', 'labels', 'is_random', 'loss_mask', 'padding_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['text'].long()
    types = data_b['types'].long()
    sentence_order = data_b['is_random'].long()
    loss_mask = data_b['loss_mask'].float()
    lm_labels = data_b['labels'].long()
    padding_mask = data_b['padding_mask'].long()

    return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask


def loss_func(loss_mask, sentence_order, output_tensor):
    lm_loss_, sop_logits = output_tensor

    lm_loss_ = lm_loss_.float()
    loss_mask = loss_mask.float()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    if sop_logits is not None:
        sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(),
                                   sentence_order.view(-1),
                                   ignore_index=-1)
        sop_loss = sop_loss.float()
        loss = lm_loss + sop_loss
        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss, sop_loss])
        return loss, {'lm loss': averaged_losses[0],
                      'sop loss': averaged_losses[1]}

    else:
        loss = lm_loss
        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss])
        return loss, {'lm loss': averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    if not args.bert_binary_head:
        types = None

    # Forward pass through the model.
    output = model(tokens, padding_mask, labels=lm_labels, token_type_ids=types)

    return output, partial(loss_func, loss_mask, sentence_order)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for BERT ...')
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
        binary_head=args.bert_binary_head)
    print_rank_0("> finished creating BERT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step, args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
