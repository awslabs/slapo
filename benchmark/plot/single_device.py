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
    draw_labels=False,
    mark_na=False,
):
    x = np.arange(0, len(x_ticklabels) * 3, 3)
    width = 0.23
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
    # if y_label is not None:
    #     ax.set_ylabel(y_label)
    if mark_na:
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
    if draw_labels:
        ax.legend(loc=0, ncol=2)  # , prop={"size": 10})
    ax.grid(axis="y")


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
        "WideResNet": "wideresnet-250M",
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
    fig, axs = plt.subplots(2, 1, figsize=(6, 2.5), sharex=True)
    fig.subplots_adjust(hspace=0.1)
    print(data)
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
    for i, ax in enumerate(axs):
        draw_bar(
            data,
            ax,
            x_ticklabels=list(model_name_mapping.keys()),
            x_label="",
            y_label="Throughput (samples/sec)",
            legends=legend_name_mapping,
            draw_labels=(i == 0),
            mark_na=(i == len(axs) - 1),
        )
    ax1, ax2 = axs
    ax1.set_ylim(235, 275)  # outliers only
    ax2.set_ylim(0, 50)  # most of the data

    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax1.xaxis.set_ticks_position("none")  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # Now, let's turn towards the cut-out slanted lines.
    # We create line objects in axes coordinates, in which (0,0), (0,1),
    # (1,0), and (1,1) are the four corners of the axes.
    # The slanted lines themselves are markers at those locations, such that the
    # lines keep their angle and position, independent of the axes size or scale
    # Finally, we need to disable clipping.

    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    fig.supylabel("Throughput (samples/sec)")
    # plt.tight_layout()
    plt.savefig("single_device_v100.pdf", format="pdf", dpi=200, bbox_inches="tight")
    plt.show()
    speedup_eager = np.array(data["slapo"]) / np.array(data["eager"])
    speedup_ts = np.array(data["slapo"][:3] + data["slapo"][4:]) / np.array(
        data["torchscript"][:3] + data["torchscript"][4:]
    )
    print(
        "Speedup vs Eager: min {:.2f}x, max {:.2f}x, mean {:.2f}x".format(
            speedup_eager.min(), speedup_eager.max(), speedup_eager.mean()
        )
    )
    print(
        "Speedup vs TorchScript: min {:.2f}x, max {:.2f}x, mean {:.2f}x".format(
            speedup_ts.min(), speedup_ts.max(), speedup_ts.mean()
        )
    )


if __name__ == "__main__":
    assert len(sys.argv) > 1
    file_name = sys.argv[1]
    plot(file_name)
