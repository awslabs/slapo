import os
import inspect
import argparse
from transformers import BertModel, AutoConfig
import torch
import torch.nn as nn
import ms
from bert_schedule import (
    replace_layernorm,
    replace_xformer_attention,
    replace_qkv,
    shard_params,
    checkpoint,
)
from ms.utils import report_memory
import deepspeed
from deepspeed.utils import RepeatingLoader


def train(args):
    print("Use deepspeed to initialize")
    deepspeed.init_distributed(dist_backend="nccl")
    # https://huggingface.co/bert-large-uncased/blob/main/config.json
    bert_config = AutoConfig.from_pretrained("bert-large-uncased")
    bert = BertModel(bert_config)
    optimizer = torch.optim.AdamW(bert.parameters(), lr=0.001)
    bert.half()

    input_names = list(bert.dummy_inputs.keys())
    input_names += ["attention_mask", "token_type_ids"]
    sig = inspect.signature(bert.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }

    rank = int(os.environ["LOCAL_RANK"])
    sch = ms.create_schedule(
        bert,
        optimizer,
        args.world_size,
        rank,
        tracer="huggingface",
        concrete_args=concrete_args,
    )
    if args.checkpoint:
        print("Use gradient checkpoint")
        checkpoint(sch, bert_config, prefix="")

    print("Use pipeline parallelism")
    sch["encoder.layer.12"].partition()

    report_memory(rank)
    device = "cuda:{}".format(rank)
    # https://github.com/microsoft/DeepSpeed/blob/ff427438651943ee473ab37547337f5f3d8c2279/tests/unit/model_parallelism/test_configurable_parallel_pp.py#L20
    ds_config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {"type": "AdamW", "params": {"lr": 0.0001}},
    }
    # criteria = nn.CrossEntropyLoss()
    def loss_fn(outputs, label):
        return outputs[0].mean()
        # output = outputs[0].transpose(1, 2).contiguous()
        # return criteria(output, label)

    model, optimizer = ms.build(
        sch, target="deepspeed", config=ds_config_dict, loss_fn=loss_fn
    )
    model.half()
    model.cuda()
    report_memory(rank)

    bs = 8
    seq_length = 512
    bert_input_dict = {
        "input_ids": torch.zeros(
            bs, seq_length, dtype=torch.long, device=device
        ).random_(bert.config.vocab_size),
        "attention_mask": torch.ones(bs, seq_length, dtype=torch.long, device=device),
        "token_type_ids": torch.ones(bs, seq_length, dtype=torch.long, device=device),
        "labels": torch.zeros(bs, seq_length, dtype=torch.long, device=device).random_(
            bert.config.vocab_size
        ),
    }

    loader = RepeatingLoader(
        [
            # First batch
            # (inputs, labels)
            (
                (
                    bert_input_dict["input_ids"],
                    bert_input_dict["attention_mask"],
                    bert_input_dict["token_type_ids"],
                ),
                bert_input_dict["labels"],
            ),
            # Rest of the batches
            # ...
        ]
    )
    data_iter = iter(loader)

    baseline = model.train_batch(data_iter=data_iter)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    # This is passed in via cmd
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=n_gpus)
    parser.add_argument("--iter_nums", type=int, default=10)
    parser.add_argument(
        "--checkpoint", action="store_true", help="Enable gradient checkpointing"
    )
    args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    train(args)
