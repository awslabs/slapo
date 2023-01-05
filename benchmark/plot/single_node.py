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
    idx,
    x_ticklabels,
    x_label=None,
    y_label=None,
    legends=None,
    title=None,
    ckpt=False,
):
    x = np.arange(0, len(x_ticklabels) * 4, 4)
    width = 0.25
    interval = np.arange(-len(data) + 1, len(data), 2)
    bars = []
    plt.rcParams["hatch.linewidth"] = 0.6
    hatches = ["//", "\\\\", "..", "x", "|", "-", "o", "O", ".", "*"]
    for i, key in enumerate(data):
        label = legends.get(key, key) if legends is not None else key
        hatch = None  # if "ckpt" not in key else hatches[i % 4]
        kwargs = {}
        kwargs["alpha"] = 0.95 if "ckpt" not in key else 0.95
        kwargs["label"] = label
        kwargs["hatch"] = hatch
        # if i >= 4:
        #     kwargs["color"] = bars[i % 4].patches[0].get_facecolor()
        bars.append(ax.bar(x + interval[i] * width, data[key], width * 2, **kwargs))
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticklabels)
    if ckpt:
        if idx > 3 and x_label is not None:
            ax.set_xlabel(x_label)
    else:
        if x_label is not None:
            ax.set_xlabel(x_label)
    if ckpt:
        if idx % 2 != 0:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
        else:
            if y_label is not None and idx == 2:
                ax.set_ylabel(y_label)
    else:
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
    if ckpt:
        if title is not None:
            ax.set_title(title, loc="left", x=0.05, y=1.0, pad=-12, fontsize=10)
    else:
        if title is not None:
            ax.set_title(title)
    return bars


def plot(file_name, ckpt=False):
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
    if not ckpt:
        legend_name_mapping = {
            "megatron": "Megatron",
            "slapo-megatron": "Slapo w/ TP",
            "deepspeed": "DeepSpeed",
            "slapo-deepspeed": "Slapo w/ ZeRO3",
        }
        fig, axs = plt.subplots(2, 4, figsize=(12, 4.5))
    else:
        model_name_mapping.pop("ALBERT")
        legend_name_mapping = {
            "megatron|ckpt": "Megatron+Ckpt",
            "slapo-megatron|ckpt": "Slapo w/ TP+Ckpt",
            "deepspeed|ckpt": "DeepSpeed+Ckpt",
            "slapo-deepspeed|ckpt": "Slapo w/ ZeRO3+Ckpt",
        }
        fig, axs = plt.subplots(3, 2, figsize=(6, 4.6))
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
        draw_bar(
            data,
            axs[i // 4][i % 4] if not ckpt else axs[i // 2][i % 2],
            idx=i,
            x_ticklabels=list(["2", "4", "8"]),
            x_label="# of GPUs",
            y_label="Throughput (samples/sec)",
            legends=legend_name_mapping,
            title=model,
            ckpt=ckpt,
        )
    if not ckpt:
        # legend as a separate figure
        label_params = axs[1][2].get_legend_handles_labels()
        axs[1][3].axis(False)
        axs[1][3].legend(*label_params, ncol=1, loc="center", frameon=False)
        axs[1][3].text(
            0.5,
            0,
            "X: Unsupported or OOM",
            ha="center",
            va="bottom",
            color="black",
            fontweight="bold",
        )
    else:
        label_params = axs[0][0].get_legend_handles_labels()
        fig.legend(
            *label_params,
            ncol=2,
            bbox_to_anchor=(0.5, 1.09),
            borderaxespad=0,
            loc="upper center"
        )
    plt.tight_layout()
    plt.savefig(
        "single_node_v100{}.pdf".format("-ckpt" if ckpt else ""),
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    assert len(sys.argv) > 1
    file_name = sys.argv[1]
    plot(file_name, ckpt=True if len(sys.argv) > 2 else False)
