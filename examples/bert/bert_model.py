"""HuggingFace Bert with model schedule."""
import inspect
import torch.distributed as dist

import ms
from bert_schedule import (
    replace_qkv,
    shard_params,
    replace_xformer_attention,
    checkpoint,
)


def model_schedule(model, config, disable_flash_attn=False, fp16=True, ckpt_ratio=0.0):
    def print_rank_0(message):
        """If distributed is initialized, print only on rank 0."""
        if dist.is_initialized():
            if dist.get_rank() == 0:
                print(message, flush=True)
        else:
            print(message, flush=True)

    print(
        f"Model schedule with world size {dist.get_world_size()}, rank {dist.get_rank()}"
    )

    input_names = list(model.dummy_inputs.keys())  # only has "input_ids"
    input_names += ["attention_mask", "token_type_ids"]  # "position_ids"
    sig = inspect.signature(model.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }

    if fp16:
        print_rank_0("Change model dtype to fp16")
        model.half()

    sch = ms.create_schedule(
        model,
        tracer="huggingface",
        concrete_args=concrete_args,
    )
    if not disable_flash_attn:
        print_rank_0("Replace HF BertSelfAttention with xformer Attention")
        replace_xformer_attention(sch, config)
        if sch.world_size > 1:
            shard_params(sch, config, fused_qkv=None, prefix="")
    else:
        print_rank_0("Replace HF QKV Dense with FusedQKV")
        replace_qkv(sch, config)
        if sch.world_size > 1:
            shard_params(sch, config, fused_qkv=True)

    if ckpt_ratio > 0.0:
        n_ckpt = checkpoint(sch, config, ckpt_ratio=ckpt_ratio)
        print_rank_0(f"Checkpointing {n_ckpt} layers")

    model, _ = ms.build(sch)
    if fp16:
        model.half()
    model.cuda()
    return model


def get_scheduled_bert(
    model_name,
    padded_vocab_size=None,
    binary_head=False,
    add_pooling_layer=True,
    disable_flash_attn=False,
    fp16=True,
    ckpt_ratio=0.0,
):
    from transformers import AutoConfig, BertModel

    config = AutoConfig.from_pretrained(model_name)
    if padded_vocab_size is not None:
        config.vocab_size = padded_vocab_size
    config.type_vocab_size = 2 if binary_head else 0

    model = model_schedule(
        BertModel(config, add_pooling_layer=add_pooling_layer),
        config,
        disable_flash_attn=disable_flash_attn,
        fp16=fp16,
        ckpt_ratio=ckpt_ratio,
    )
    return model
