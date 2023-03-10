# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial


class LossTestDataset(Dataset):
    def __init__(self, dataset, fn) -> None:
        super().__init__()
        self.dataset = dataset
        self.fn = fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        entry = self.dataset[index]
        return self.fn(entry)


def get_dataloader(
    model_name,
    dataset_name,
    micro_batch_size,
    enable_pipeline,
    collate_fn=None,
    getitem_fn=None,
    cache_dir=None,
    mpu=None,
    max_seq_length=1024,
):
    raw_dataset = load_dataset(
        dataset_name.split("-")[0], dataset_name, cache_dir=cache_dir
    )

    if "bert" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    if "gpt" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    train, val = preprocessing_datasets(
        raw_dataset, tokenizer, model_name, max_seq_length
    )
    train_dataset = LossTestDataset(train, getitem_fn)
    val_dataset = LossTestDataset(val, getitem_fn)

    num_replicas = None
    rank = None
    if mpu:
        num_replicas = mpu.get_data_parallel_world_size()
        rank = mpu.get_data_parallel_rank()

    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        sampler=DistributedSampler(train_dataset, num_replicas=num_replicas, rank=rank),
        collate_fn=partial(collate_fn, enable_pipeline=enable_pipeline),
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=micro_batch_size,
        sampler=DistributedSampler(val_dataset, num_replicas=num_replicas, rank=rank),
        collate_fn=partial(collate_fn, enable_pipeline=enable_pipeline),
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader


def preprocessing_datasets(datasets, tokenizer, model_name, max_seq_length=1024):
    column_names = datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if tokenizer.model_max_length < max_seq_length:
        raise ValueError(
            f"The tokenizer ({tokenizer.__class__.__name__}) has a maximum sequence "
            f"length of {tokenizer.model_max_length}, which cannot support "
            f"`max_seq_length={max_seq_length}`"
        )

    # we tokenize every text, then concatenate them together before splitting them in smaller parts.
    # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
    # efficient when it receives the `special_tokens_mask`.
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column_name],
            return_special_tokens_mask=True if "bert" in model_name else False,
        )

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        load_from_cache_file=True,
    )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + max_seq_length]
                for i in range(0, total_length, max_seq_length)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
    )

    return lm_datasets["train"], lm_datasets["validation"]


if __name__ == "__main__":
    # some tests
    train, val = get_dataloader("gpt-neo-2.7B", "wikitext-103-v1", 4, True)
    for b in train:
        print(b)
    # import pdb; pdb.set_trace()
