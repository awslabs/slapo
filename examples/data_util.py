# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch


class LossTestDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        entry = self.dataset[index]
        ret = [
            entry["input_ids"],
            entry["attention_mask"],
            # position_ids
            torch.arange(len(entry["input_ids"])),
            entry["labels"],
        ]
        return ret


def get_data_move_and_group_fn(enable_pipeline):
    def _collate_fn(batch):
        device = torch.cuda.current_device()
        input_ids = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
        attention_mask = torch.tensor(
            [x[1] for x in batch], dtype=torch.float16, device=device
        )
        position_ids = torch.stack([x[2] for x in batch]).to(device=device)
        labels = torch.tensor([x[3] for x in batch], dtype=torch.long, device=device)

        ret = [input_ids, attention_mask, position_ids, labels]
        if not enable_pipeline:
            # insert None in second and fourth position
            ret.insert(1, None)  # past_key_values
            ret.insert(3, None)  # token_type_ids

        # group first inputs
        return [ret[:-1], ret[-1]]

    return _collate_fn


def get_dataloader(model_name, micro_batch_size, enable_pipeline, cache_dir=None):
    # wiki_option_datafile = 'wikitext-2-v1'
    wiki_option_datafile = "wikitext-103-v1"
    raw_dataset = load_dataset("wikitext", wiki_option_datafile, cache_dir=cache_dir)

    if "bert" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    if "gpt" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    train, val = preprocessing_datasets(raw_dataset, tokenizer, model_name)
    train_dataset = LossTestDataset(train)
    val_dataset = LossTestDataset(val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        sampler=DistributedSampler(train_dataset),
        collate_fn=get_data_move_and_group_fn(enable_pipeline),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=micro_batch_size,
        sampler=DistributedSampler(val_dataset),
        collate_fn=get_data_move_and_group_fn(enable_pipeline),
    )
    return train_loader, val_loader


def preprocessing_datasets(datasets, tokenizer, model_name):
    column_names = datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    max_seq_length = tokenizer.model_max_length
    if max_seq_length > 1024:
        max_seq_length = 1024

    # we tokenize every text, then concatenate them together before splitting them in smaller parts.
    # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
    # efficient when it receives the `special_tokens_mask`.
    if "bert" in model_name:

        def tokenize_function(examples):
            return tokenizer(
                examples[text_column_name], return_special_tokens_mask=True
            )

    else:

        def tokenize_function(examples):
            output = tokenizer(examples[text_column_name])
            return output

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        load_from_cache_file=True,
    )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    if "bert" in model_name:

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
            return result

    elif "gpt" in model_name:

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
    train, val = get_dataloader("gpt-neo-2.7B", 4, True)
    for b in train:
        print(b)
    # import pdb; pdb.set_trace()
