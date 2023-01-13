# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

y = np.array(
    [[0.25, 104], [0.25, 176], [1.0, 176], [1.0, 120], [0.67, 120], [0.67, 104]]
)
p_defined = Polygon(y, facecolor="#FFE7B2")
y = np.array([[0.75, 200], [0.67, 176], [0.5, 152], [0.34, 128], [0, 100], [0, 200]])
p_invalid = Polygon(y, facecolor=(0.1, 0.2, 0.5, 0.3))

fig, ax = plt.subplots(1, 1, figsize=(3.4, 3))

ax.add_patch(p_defined)
ax.add_patch(p_invalid)
ax.set_xlim([0.1, 1.2])
ax.set_ylim([90, 200])
ax.set_xlabel("ckpt_ratio")
ax.set_ylabel("batch_size", rotation=270)
ax.text(0.45, 180, "Invalid space\n(OOM)", horizontalalignment="center", fontsize=10)
ax.text(
    0.63,
    140,
    "User-defined polygon\nsearch space",
    horizontalalignment="center",
    color="red",
    fontsize=10,
)
ax.text(
    0.94,
    100,
    "Inefficient\nspace",
    horizontalalignment="center",
    color="gray",
    fontsize=10,
)
ax.yaxis.set_label_position("right")
ax.yaxis.set_label_coords(1.25, 0.5)
ax.yaxis.tick_right()
plt.tight_layout()
plt.show()
plt.savefig("polygon.png", format="png", dpi=200, bbox_inches="tight")
