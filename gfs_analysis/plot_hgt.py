"""
Plots the results of gfs_hgz_factor.py
"""

import color_scheme
import gfs

import os
import numpy as np
from matplotlib import pyplot as plt
import json


RUN_ID = "_2"

SELF_FN = os.path.dirname(os.path.abspath(__file__))
IN_FN = os.path.join(SELF_FN, "data", "gfs", f"gfs_hgt_factor{RUN_ID}.csv")
IN_PARAM_FN = os.path.join(SELF_FN, "data", "gfs", "gfs_hgt_factor.json")

PLOT_HGT_FN_PNG = os.path.join(SELF_FN, "data", "gfs", f"gfs_hgt{RUN_ID}.pdf")
PLOT_HGT_FN_SVG = os.path.join(SELF_FN, "data", "gfs", f"gfs_hgt{RUN_ID}.svg")
PLOT_HGT_NON_NORMA_FN_PNG = os.path.join(SELF_FN, "data", "gfs", f"gfs_hgt_non_norma{RUN_ID}.pdf")
PLOT_HGT_NON_NORMA_FN_SVG = os.path.join(SELF_FN, "data", "gfs", f"gfs_hgt_non_norma{RUN_ID}.svg")

PLOT_FAC_FN_PNG = os.path.join(SELF_FN, "data", "gfs", f"gfs_hgt_factor{RUN_ID}.pdf")
PLOT_FAC_FN_SVG = os.path.join(SELF_FN, "data", "gfs", f"gfs_hgt_factor{RUN_ID}.svg")

PLOT_HGT_LEVEL_FN_PNG = os.path.join(SELF_FN, "data", "gfs", f"gfs_hgt_levels{RUN_ID}.pdf")
PLOT_HGT_LEVEL_FN_SVG = os.path.join(SELF_FN, "data", "gfs", f"gfs_hgt_levels{RUN_ID}.svg")
plt.rcParams["font.family"] = "Bahnschrift"
plt.rcParams["font.weight"] = "ultralight"
plt.rcParams["font.size"] = "16"


def ma(x, w=10):
    x = list(x)
    pad_left = [x[0] for _ in range(0, w - 1)]
    return np.convolve(pad_left + x, np.ones(w), "valid") / w


def normalise_gfs(gfs):
    total_genes = sum(gfs)
    return [g / total_genes for g in gfs]


with open(IN_FN, "r") as f:
    data = f.readlines()

data = [d.strip().split(",") for d in data][1:]
data = [
    (
        float(hgt),
        float(fixed_egfs_loss),
        [float(g) for g in gfs[1:-1].split(" ") if g],
        float(random_fixed_loss),
        [float(g) for g in gfs2[1:-1].split(" ") if g],
        float(random_egfs_loss),
    )
    for hgt, fixed_egfs_loss, gfs, random_fixed_loss, gfs2, random_egfs_loss in data
]

with open(IN_PARAM_FN, "r") as f:
    param = json.load(f)

theta = param["theta"]
rho = param["rho"]
num_samples = param["num_samples"]

data = sorted(data)
egfs = gfs.expected_gfs(n=num_samples, theta=theta, rho=rho)

## Normalised
plt.figure(figsize=(8, 4))
plt.plot(
    range(1, num_samples + 1),
    normalise_gfs(egfs),
    color=color_scheme.secondary,
    label="Expected GFS, random tree, neutral evolution.",
)

# Plot gfs with hgt = 0
print(data)
hgt, _, mean_gfs, _, _, _ = min(data)
plt.plot(
    range(1, num_samples + 1),
    normalise_gfs(mean_gfs),
    label="Fixed tree, γ = 0.",
    color=color_scheme.primary,
)


# Plot gfs with max hgt loss
# hgt, minimum_loss, mean_gfs, _, _ = min(data, key=lambda d: d[1])
hgt, minimum_loss, mean_gfs, _, _, _ = max(data)
# hgt, _, mean_gfs, _, _ = max(data)
plt.plot(
    range(1, num_samples + 1),
    normalise_gfs(mean_gfs),
    label=f"Fixed tree, γ = {hgt:.4f}.",
    color=color_scheme.blue,
    # linestyle=(0, (5, 10)),
)

hgt_at_minimum_loss = hgt

plt.legend()
plt.xlabel("GF Class")
plt.ylabel("Gene Frequency")
plt.tight_layout()
plt.savefig(PLOT_HGT_FN_PNG)
plt.savefig(PLOT_HGT_FN_SVG)
plt.show()

## Non Normalised
plt.figure(figsize=(8, 4))
plt.plot(
    range(1, num_samples + 1),
    egfs,
    color=color_scheme.secondary,
    label="Random tree, γ = 0.",
)

# Plot gfs with hgt = 0
hgt, _, mean_gfs, _, _, _ = min(data)
plt.plot(
    range(1, num_samples + 1),
    mean_gfs,
    label="Fixed tree, γ = 0.",
    color=color_scheme.primary,
)


# Plot gfs with max loss
# hgt, minimum_loss, mean_gfs, _, _ = min(data, key=lambda d: d[1])
hgt, minimum_loss, mean_gfs, _, _, _ = max(data)
# hgt, _, mean_gfs, _, _ = max(data)
plt.plot(
    range(1, num_samples + 1),
    mean_gfs,
    label=f"Fixed tree, γ = {hgt:.4f}.",
    color=color_scheme.blue,
    # linestyle=(0, (5, 10)),
)

plt.legend()
plt.xlabel("GF Class")
plt.ylabel("Gene Frequency")
plt.tight_layout()
plt.savefig(PLOT_HGT_NON_NORMA_FN_PNG)
plt.savefig(PLOT_HGT_NON_NORMA_FN_SVG)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

hgt, fixed_egfs_loss, _, random_fixed_loss, _, random_egfs = zip(*data)
ax.plot(
    hgt,
    ma(fixed_egfs_loss),
    color=color_scheme.secondary,
    label="Against random tree, γ = 0.",
)
ax.plot(
    hgt,
    ma(random_fixed_loss),
    color=color_scheme.primary,
    label="Against random, identical γ.",
)
# ax.plot(
#    hgt,
#    ma(random_egfs),
#    color=color_scheme.background,
#    label="Random tree same γ, against random tree, γ = 0.",
# )
plt.axvline(
    x=hgt_at_minimum_loss,
    color=color_scheme.blue,
    linestyle="--",
    label=f"Minimum: ({hgt_at_minimum_loss:.4f}, {minimum_loss:.0f})",
)


ax.set_ylabel("Mean χ²-like error")
ax.set_xlabel("HGT Rate")
plt.legend()
plt.tight_layout()

plt.savefig(PLOT_FAC_FN_PNG)
plt.savefig(PLOT_FAC_FN_SVG)
plt.show()

## Plot gfs of non fixated at different hgt levels
plt.figure(figsize=(8, 4))
plt.plot(
    range(1, num_samples + 1),
    normalise_gfs(egfs),
    color=color_scheme.primary,
    label=r"γ = 0",
)

## Plot gfs with hgt =
# hgt, _, _, _, mean_gfs_random = data[0]
# plt.plot(
#    range(1, num_samples + 1),
#    normalise_gfs(mean_gfs_random),
#    label=f"HGT = {hgt:.5f}",
#    color=color_scheme.primary,
# )

hgt, _, _, _, mean_gfs_random, _ = data[5]
plt.plot(
    range(1, num_samples + 1),
    normalise_gfs(mean_gfs_random),
    label=f"γ = {hgt:.5f}",
    color=color_scheme.secondary,
    # linestyle=(0, (5, 10)),
)

hgt, _, _, _, mean_gfs_random, _ = data[50]
plt.plot(
    range(1, num_samples + 1),
    normalise_gfs(mean_gfs_random),
    label=f"γ = {hgt:.5f}",
    color=color_scheme.blue,
    # linestyle=(0, (5, 10)),
)

plt.legend()
plt.xlabel("GF Class")
plt.ylabel("Gene Frequency")
plt.tight_layout()
plt.savefig(PLOT_HGT_LEVEL_FN_PNG)
plt.savefig(PLOT_HGT_LEVEL_FN_SVG)
plt.show()
