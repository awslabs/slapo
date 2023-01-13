# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import matplotlib.pyplot as plt

thrpt = [
    [123.7, 124.3, 126.9, 0.0, 0.0, 0.0, 0.0],
    [121.1, 123.0, 125.9, 0.0, 0.0, 0.0, 0.0],
    [120.7, 122.8, 124.0, 0.0, 0.0, 0.0, 0.0],
    [118.2, 121.0, 122.6, 127.6, 0.0, 0.0, 0.0],
    [121.3, 122.6, 124.0, 129.3, 0.0, 0.0, 0.0],
    [118.4, 120.4, 122.7, 127.9, 0.0, 0.0, 0.0],
    [118.4, 120.4, 120.9, 126.0, 130.9, 0.0, 0.0],
    [115.8, 118.2, 120.4, 123.1, 128.8, 0.0, 0.0],
    [115.4, 116.7, 118.1, 121.4, 127.2, 0.0, 0.0],
    [113.7, 115.0, 116.0, 120.2, 125.4, 128.4, 0.0],
    [110.1, 111.6, 114.1, 117.1, 121.5, 123.8, 127.1],
    [108.2, 108.3, 109.5, 113.1, 116.3, 118.4, 122.4],
    [101.5, 106.5, 106.9, 107.8, 112.0, 114.9, 115.4],
]

X = [192, 184, 176, 168, 160, 152, 144, 136, 128, 120, 112, 104, 96]
Y = [1.0, 0.92, 0.84, 0.67, 0.5, 0.34, 0.25]
thrpt = np.array(thrpt).reshape((len(X), len(Y)))
print(X, Y, thrpt)

fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.5))
CS = ax.contourf(
    Y,
    X,
    thrpt,
    levels=[0, 100, 105, 110, 115, 120, 125, 130, 135],
    colors=[
        "#9e0142",
        "#d53e4f",
        "#f46d43",
        "#fdae61",
        "#fee08b",
        "#ffffbf",
        "#e6f598",
        "#abdda4",
        "#66c2a5",
        "#3288bd",
        "#5e4fa2",
    ][:-1][::-1],
)  # , cmap="viridis")
# ax.clabel(
# CS,
# inline=True,
# fontsize=5,
# colors="black"
# manual=[
#     (0.6, 175),
#     (0.3, 120),
#     (0.66, 159),
#     (0.46, 135),
#     (0.67, 110),
#     (0.84, 100),
#     (0.82, 184),
# ],
# )
# best result
explored_points = [
    [1.0, 176],
    [0.92, 176],
    [0.84, 176],
    [0.67, 176],
    [0.84, 168],
    [0.67, 168],
    [0.5, 168],
    [0.67, 160],
    [0.5, 160],
    [0.67, 152],
    [0.5, 152],
    [0.5, 144],
    [0.34, 144],
    [0.5, 136],
    [0.34, 136],
    [0.5, 128],
    [0.34, 128],
]
x = [point[0] for point in explored_points]
y = [point[1] for point in explored_points]
ax.plot(x, y, "*", color="purple", markersize=5, linewidth=1)
# ax.plot([0.5], [144], "x", color='red', markersize=10)
ax.text(0.5, 144, "$\\times$", color="red", va="center", ha="center", fontsize=20)
ax.set_xlim(0.25, 1.0)
ax.set_ylim(96, 192)
ax.set_xlabel("Activation Checkpointing Ratio")
ax.set_ylabel("Batch Size")
ax.text(
    0.4,
    168,
    "OOM",
    weight="bold",
    horizontalalignment="center",
    color="black",
    fontsize=10,
)
ax.text(
    0.55,
    135,
    "125",
    horizontalalignment="center",
    weight="bold",
    color="black",
    fontsize=10,
)
ax.text(
    0.67,
    128,
    "120",
    horizontalalignment="center",
    weight="bold",
    color="black",
    fontsize=10,
)
ax.text(
    0.75,
    115,
    "115",
    horizontalalignment="center",
    weight="bold",
    color="black",
    fontsize=10,
)
ax.text(
    0.84,
    107,
    "110",
    horizontalalignment="center",
    weight="bold",
    color="black",
    fontsize=10,
)
ax.text(
    0.92,
    100,
    "105",
    horizontalalignment="center",
    weight="bold",
    color="black",
    fontsize=10,
)
ax.text(
    1.16,
    100,
    "Throughput (samples / sec)",
    horizontalalignment="center",
    color="black",
    rotation=270,
    fontsize=10,
)
ax.set_xticks([0.34, 0.5, 0.67, 0.84, 1.0])
ax.set_yticks([184, 168, 152, 136, 120, 104])
# make a colorbar for the contour lines
CB = fig.colorbar(CS)
# draw search space
# from matplotlib.patches import Polygon
# y = np.array([[0.25, 104], [0.25, 176], [1.0, 176], [1.0, 120], [0.67, 120], [0.67, 104]])
# p_defined = Polygon(y, facecolor="#FFE7B2")
# y = np.array([[0.75, 200], [0.67, 176], [0.5, 152], [0.34, 128], [0, 100], [0, 200]])
# p_invalid = Polygon(y, facecolor=(0.1, 0.2, 0.5, 0.3))
# ax.add_patch(p_defined)
# ax.add_patch(p_invalid)
plt.show()
plt.savefig("autotune.pdf", format="pdf", dpi=200, bbox_inches="tight")
