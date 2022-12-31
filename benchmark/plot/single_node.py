# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import seaborn as sns
# sns.set_theme(context="paper", style="whitegrid", palette=sns.color_palette("Set3", 10))


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
    norm=False,
):
    x = np.arange(0, len(x_ticklabels) * 4, 4)
    width = 0.2
    interval = np.arange(-len(data) + 1, len(data), 2)
    bars = []
    for i, key in enumerate(data):
        label = legends.get(key, key) if legends is not None else key
        bars.append(
            ax.bar(
                x + interval[i] * width, data[key], width * 2, alpha=0.95, label=label
            )
        )
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticklabels)
    # ax.set_yticks(np.arange(0, 8.5, 1.0))
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    for bar in bars:
        for patch in bar.patches:
            height = patch.get_height()
            if height == 0:
                ax.text(
                    patch.get_x() + patch.get_width() / 2.0,
                    height,
                    "X",
                    ha="center",
                    va="bottom",
                    color=patch.get_facecolor(),
                    fontweight="bold",
                )
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    if title is not None:
        ax.set_title(title)
    return bars


def plot(file_name, norm=False):
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
        "GPT": "EleutherAI/gpt-neo-1.3B",
        "OPT": "facebook/opt-350m",
        "T5": "t5-large",
        "WideResNet": "wideresnet-250M",
    }
    legend_name_mapping = {
        "megatron": "Megatron",
        "slapo-megatron": "Slapo w/ TP",
        "deepspeed": "DeepSpeed",
        "slapo-deepspeed": "Slapo w/ ZeRO3",
        # "megatron|ckpt": "Megatron+Ckpt",
        # "slapo-megatron|ckpt": "Slapo w/ TP+Ckpt",
        # "deepspeep|ckpt": "DeepSpeed+Ckpt",
        # "slapo-deepspeed|ckpt": "Slapo w/ ZeRO3+Ckpt",
    }
    fig, axs = plt.subplots(2, 4, figsize=(12, 4.5))
    for i, (model, long_name) in enumerate(model_name_mapping.items()):
        data = {key: [] for key in legend_name_mapping}
        for impl in data:
            if "ckpt" in impl:
                new_impl_name = impl.split("|")[0]
            else:
                new_impl_name = impl
            res = results[results["Impl"] == new_impl_name]
            res = res[res["Model"] == long_name]
            for n_gpu in [2, 4, 8]:
                selected_model_res = res[res["nGPU"] == str(n_gpu)]
                cond = (
                    (selected_model_res["Ckpt"] == "0")
                    if "ckpt" not in impl
                    else (selected_model_res["Ckpt"] != "0")
                )
                thrpt = selected_model_res[cond]["Thrpt"].values.tolist()
                assert len(thrpt) <= 1
                if len(thrpt) == 0 or thrpt[0] in ["fail", "oom"]:
                    data[impl].append(0)
                else:
                    data[impl].append(float(thrpt[0]))
        print(model, data)
        if norm:
            data = normalize(data)
        draw_bar(
            data,
            axs[i // 4][i % 4],
            x_ticklabels=list(["2", "4", "8"]),
            x_label="# of GPUs",
            y_label="Normalized Throughput" if norm else "Throughput (samples/sec)",
            legends=legend_name_mapping,
            title=model,
            norm=norm,
        )
    # legend as a separate figure
    label_params = axs[1][2].get_legend_handles_labels()
    axs[1][3].axis(False)
    axs[1][3].legend(*label_params, loc="center", frameon=False)
    plt.tight_layout()
    plt.savefig(
        "single_node_v100{}.pdf".format("_norm" if norm else ""),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    assert len(sys.argv) > 1
    file_name = sys.argv[1]
    plot(file_name, norm=True if len(sys.argv) > 2 else False)
