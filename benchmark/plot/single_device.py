# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import seaborn as sns
# sns.set_theme(context="paper", style="whitegrid", palette=sns.color_palette("Set3", 10))

NORMALIZE = False


def normalize(data):
    tmp_matrix = np.array(list(data.values()))
    max_val = tmp_matrix.max(axis=0)
    max_val = (
        np.repeat(max_val, tmp_matrix.shape[0], axis=0)
        .reshape(tmp_matrix.shape[::-1])
        .transpose()
    )
    normalized_data = tmp_matrix / max_val
    for i, key in enumerate(data):
        data[key] = normalized_data[i]
    return data


def draw_bar(
    data,
    ax,
    x_ticklabels,
    x_label=None,
    y_label=None,
    legends=None,
    title=None,
    outfile="test",
):
    x = np.arange(0, len(x_ticklabels) * 3, 3)
    width = 0.23
    interval = np.arange(-len(data) + 1, len(data), 2)
    for i, key in enumerate(data):
        label = legends.get(key, key) if legends is not None else key
        ax.bar(x + interval[i] * width, data[key], width * 2, alpha=0.95, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticklabels)
    # ax.set_yticks(np.arange(0, 8.5, 1.0))
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    # for i, v in enumerate(baseline/best):
    #     ax.text(i, v, str(v), color='black', fontweight='bold')
    ax.set_axisbelow(True)
    if NORMALIZE:
        ax.legend(bbox_to_anchor=(0.5, 1.3), ncol=3, loc="upper center")
    else:
        ax.legend(loc=0)  # , prop={"size": 10})
    ax.grid(axis="y")
    if title is not None:
        ax.set_title(title)
    plt.savefig(f"{outfile}.pdf", format="pdf", dpi=200, bbox_inches="tight")
    plt.show()


def plot(file_name):
    with open(file_name, "r") as csv_file:
        for line in csv_file:
            if "Impl" in line:
                break
        headers = line.strip().split("\t")
        lines = []
        for line in csv_file.readlines():
            lines.append(line.strip().split("\t"))
        results = pd.DataFrame(lines, columns=headers)
    model_name_mapping = {
        "BERT": "bert-large-uncased",
        "RoBERTa": "roberta-large",
        "ALBERT": "albert-large-v2",
        "GPT": "EleutherAI/gpt-neo-125M",
        "OPT": "facebook/opt-350m",
        "T5": "t5-base",
    }
    legend_name_mapping = {
        "eager": "PyTorch Eager",
        "eager-ckpt": "PyTorch Eager+Ckpt",
        "torchscript": "TorchScript",
        "slapo": "Slapo",
        "slapo-ckpt": "Slapo+Ckpt",
    }
    data = {
        "eager": [],
        "eager-ckpt": [],
        "torchscript": [],
        "slapo": [],
        "slapo-ckpt": [],
    }
    for impl in data:
        if "ckpt" in impl:
            new_impl_name = impl.split("-")[0]
        else:
            new_impl_name = impl
        if new_impl_name == "slapo":
            new_impl_name = "slapo-megatron"
        res = results[results["Impl"] == new_impl_name]
        for long_name in model_name_mapping.values():
            selected_model_res = res[res["Model"] == long_name]
            cond = (
                (selected_model_res["Ckpt"] == "0")
                if "ckpt" not in impl
                else (selected_model_res["Ckpt"] != "0")
            )
            thrpt = selected_model_res[cond]["Thrpt"].values.tolist()
            if len(thrpt) == 0 or thrpt[0] in ["fail", "oom"]:
                data[impl].append(0)
            else:
                data[impl].append(float(thrpt[0]))
    if NORMALIZE:
        fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))
        data = normalize(data)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 3))
    print(data)
    draw_bar(
        data,
        ax,
        x_ticklabels=list(model_name_mapping.keys()),
        x_label="",
        y_label="Normalized Throughput" if NORMALIZE else "Throughput (samples/sec)",
        legends=legend_name_mapping,
        title=None,
        outfile="single_device_v100{}".format("_norm" if NORMALIZE else ""),
    )


if __name__ == "__main__":
    assert len(sys.argv) > 1
    file_name = sys.argv[1]
    plot(file_name)
