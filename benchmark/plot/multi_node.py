# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import seaborn as sns
# sns.set_theme(context="paper", style="whitegrid", palette=sns.color_palette("Set3", 10))


def draw_bar(
    data,
    ax,
    x_ticklabels,
    x_label=None,
    y_label=None,
    legends=None,
    title=None,
):
    x = np.arange(0, len(x_ticklabels) * 4, 4)
    width = 0.3
    interval = np.arange(-len(data) + 1, len(data), 2)
    bars = []
    plt.rcParams["hatch.linewidth"] = 0.6
    hatches = ["//", "\\\\", "..", "x", "|", "-", "o", "O", ".", "*"]
    color_map = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = [color_map[0], color_map[2], color_map[1]]
    for i, key in enumerate(data):
        label = legends.get(key, key) if legends is not None else key
        hatch = None
        kwargs = {}
        kwargs["alpha"] = 0.95
        kwargs["label"] = label
        kwargs["hatch"] = hatch
        kwargs["color"] = color_map[i]
        # if i >= 4:
        #     kwargs["color"] = bars[i % 4].patches[0].get_facecolor()
        bars.append(ax.bar(x + interval[i] * width, data[key], width * 2, **kwargs))
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticklabels, fontsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=12)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=12)
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
    ax.legend(loc=0, prop={"size": 12})
    if title is not None:
        ax.set_title(title)
    return bars


def plot(file_name=None):
    data = {
        "megatron": [9.21, 16.69, 28.16],
        "deepspeed": [9.75, 19.14, 23.9],
        "slapo": [11.29, 22.01, 30.53],
    }
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
        "megatron": "Megatron-LM",
        "slapo": "Slapo",
        "deepspeed": "DeepSpeed",
    }
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 2.4))
    draw_bar(
        data,
        ax,
        x_ticklabels=list(["16", "32", "64"]),
        x_label="# of GPUs",
        y_label="Throughput\n(samples/sec)",
        legends=legend_name_mapping,
    )
    plt.tight_layout()
    plt.savefig(
        "multi-node.pdf",
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    plot()
