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
    import torch
    import transformers
    import transformers.utils.fx as fx
    import torch.distributed as dist
    import torch.nn as nn

    print("Using model schedule to optimize")
    print("World size: {}, rank: {}".format(dist.get_world_size(), dist.get_rank()))

    args = get_args()
    input_names = list(model.dummy_inputs.keys())
    input_names += ["attention_mask", "position_ids"]
    sig = inspect.signature(model.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    class HFGPTTracer(fx.HFTracer):

        def __init__(self, **config) -> None:
            super().__init__()

        def trace(self, *args, **kwargs):
            graph = super().trace(*args, **kwargs)
            return graph

        def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
            # Note that we currently cannot shard Conv1D, so HF gpt2 series models are not
            # supported yet.
            if isinstance(m, transformers.pytorch_utils.Conv1D):
                return True
            return ((not self._stateless_mod_instanciation_depends_on_proxies(m)) and
                    super().is_leaf_module(m, module_qualified_name))

    if args.fp16:
        print("Change model dtype to fp16")
        model.half()

    sch = ms.create_schedule(
        model,
        world_size=dist.get_world_size(),
        rank=dist.get_rank(),
        tracer=HFGPTTracer,
        concrete_args=concrete_args)

    def replace_qkv():
        print("Replace HF QKV Linear with FusedQKV")
        num_heads = config.num_heads
        hidden_size = config.hidden_size

        class FusedQKV(nn.Module):
            
            def __init__(self, hidden_size, num_heads) -> None:
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

        class QKVPattern(ms.Pattern):
            
            def __init__(self, layer_num):
                super(QKVPattern, self).__init__()
                self.layer_num = layer_num

            @staticmethod
            def func(x: torch.Tensor) -> torch.Tensor:
                new_x_shape = x.size()[:-1] + (num_heads, hidden_size)
                x = x.view(new_x_shape)
                return x.permute(0, 2, 1, 3)

            def starting_point(self, node):
                if node.op != "call_module":
                    return False
                name = node.target
                if f"h.{self.layer_num}." in name and "attention" in name:
                    if "k_proj" in name or "q_proj" in name or "v_proj" in name:
                        return True
                return False

        for i in range(config.num_layers):
            op_lst = sch.find(QKVPattern(i))
            assert op_lst, "Cannot find QKV pattern"
            sch[op_lst].replace(FusedQKV, hidden_size=hidden_size, num_heads=num_heads)


    def fix_view_after_shard():
        # fix number of heads
        import operator
        ops = sch.find_method(lambda node:
            node.op == "call_method" and
            node.target == "view" and
            len(node.args) == 2 and
            node.args[0].target == "contiguous" and
            isinstance(node.args[1], torch.fx.Node) and
            node.args[1].target == operator.add)

        def new_view(tensor, old_shape):
            new_shape = old_shape[:-1] + (-1,)
            return tensor.view(new_shape)

        for op in ops:
            sch[op].replace(new_view)

    sch.trace_module()
    replace_qkv()
    sch.trace_module()

    # Sharding.
    if dist.get_world_size() > 1:
        # Embedding
        sch["wte"].shard("weight", axis=0)
        # Build the mask
        vocab_start_index = sch.rank * config.vocab_size // sch.world_size
        vocab_end_index = (sch.rank + 1) * config.vocab_size // sch.world_size

        def fw_pre_hook(_input):
            # Mask the input
            input_mask = (_input[0] < vocab_start_index) | (_input[0] >= vocab_end_index)
            masked_input = _input[0].clone() - vocab_start_index
            masked_input[input_mask] = 0
            return masked_input

        sch["wte"].hook("fw_pre", fw_pre_hook)

        def fw_post_hook(_input, output):
            # Mask the output embedding
            input_mask = (_input[0] < vocab_start_index) | (_input[0] >= vocab_end_index)
            output[input_mask, :] = 0.0
            # Reduce across all the model parallel GPUs
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
            return output

        sch["wte"].hook("fw_post", fw_post_hook)

        sch.trace_module()
        for i in range(config.num_layers):
            # Attention. Assuming QKV is fused.
            sch[f"h.{i}.attn.attention.FusedQKV_0"].shard("weight", axis=0)
            sch[f"h.{i}.attn.attention.FusedQKV_0"].shard("bias", axis=0)
            sch[f"h.{i}.attn.attention.FusedQKV_0"].sync(backward=True)
            sch[f"h.{i}.attn.attention.out_proj"].shard("weight", axis=1)
            sch[f"h.{i}.attn.attention.out_proj"].sync()

            # MLP
            sch[f"h.{i}.mlp.c_fc"].shard("weight", axis=0)
            sch[f"h.{i}.mlp.c_fc"].shard("bias", axis=0)
            sch[f"h.{i}.mlp.c_proj"].shard("weight", axis=1)
            sch[f"h.{i}.mlp.c_proj"].sync()
            sch[f"h.{i}.mlp.c_fc"].sync(backward=True)

        sch.trace_module()
        fix_view_after_shard()


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
    
    print_rank_0(f'Building HF {model_name} ...')
    config = AutoConfig.from_pretrained(model_name)
    config.vocab_size = args.padded_vocab_size
    hidden_size = config.hidden_size
    print(config)
    
    class GPTNeoWithLMHead(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.gpt = model_schedule(GPTNeoModel(config), config)

        def forward(
            self,
            input_ids = None,
            position_ids = None,
            attention_mask = None,
            labels = None,
            token_type_ids = None,            
        ):
            assert token_type_ids is None, "Not traced"
            batch_size = input_ids.shape[0]
            seq_length = input_ids.shape[1]
            # The shape of attention_mask is (1, 1, seq_length, seq_length) while the 3rd dim
            # is broadcast. The required shape for HF GPT is (batch, 1, 1, seq_length).
            # (1, 1, S, S) -> (1, 1, S, 1)
            attention_mask = attention_mask[..., -1:]
            # (1, 1, S, 1) -> (1, 1, 1, S)
            attention_mask = attention_mask.reshape((1, 1, 1, seq_length))
            # (1, 1, 1, S) -> (B, 1, 1, S)
            attention_mask = attention_mask.repeat(batch_size, 1, 1, 1)
            output = self.gpt(input_ids=input_ids, attention_mask=attention_mask,
                              position_ids=position_ids)
            lm_output = output["last_hidden_state"].transpose(0, 1).contiguous()

            output_tensor = post_language_model_processing(
                lm_output, labels, self.gpt.wte.weight, True,
                args.fp16_lm_cross_entropy)
            return output_tensor

    model = GPTNeoWithLMHead(config)
    return model

def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step, args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
