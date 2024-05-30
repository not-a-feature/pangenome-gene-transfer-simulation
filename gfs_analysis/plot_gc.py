"""
Plots the results of gfs_gc_factor.py
"""

import color_scheme
import gfs

import os
import numpy as np
from matplotlib import pyplot as plt
import json


SELF_FN = os.path.dirname(os.path.abspath(__file__))
IN_FN = os.path.join(SELF_FN, "data", "gfs", "gfs_gc_factor.csv")
IN_PARAM_FN = os.path.join(SELF_FN, "data", "gfs", "gfs_gc_factor.json")


PLOT_GC_FN_PDF = os.path.join(SELF_FN, "data", "gfs", "gfs_gc.pdf")
PLOT_GC_FN_SVG = os.path.join(SELF_FN, "data", "gfs", "gfs_gc.svg")
PLOT_GC_NON_NORMA_FN_PDF = os.path.join(SELF_FN, "data", "gfs", "gfs_gc_non_norma.pdf")
PLOT_GC_NON_NORMA_FN_SVG = os.path.join(SELF_FN, "data", "gfs", "gfs_gc_non_norma.svg")

PLOT_FAC_FN_PDF = os.path.join(SELF_FN, "data", "gfs", "gfs_gc_factor.pdf")
PLOT_FAC_FN_SVG = os.path.join(SELF_FN, "data", "gfs", "gfs_gc_factor.svg")
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
        float(gc),
        float(fixed_egfs_loss),
        [float(g) for g in gfs[1:-1].split(" ") if g],
        float(random_fixed_loss),
        [float(g) for g in gfs2[1:-1].split(" ") if g],
        float(random_egfs_loss),
    )
    for gc, fixed_egfs_loss, gfs, random_fixed_loss, gfs2, random_egfs_loss in data
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

# Plot gfs with gc = 0
print(data[0])
gene_conv, zero_loss, mean_gfs, _, _, _ = min(data)
plt.plot(
    range(1, num_samples + 1),
    normalise_gfs(mean_gfs),
    label="Fixed tree, κ = 0.",
    color=color_scheme.primary,
)


# Plot gfs with minimum loss
# marked_gc, marked_loss, mean_gfs, _, _ = min(data, key=lambda d: d[1])
# plt.plot(
#     range(1, num_samples + 1),
#     normalise_gfs(mean_gfs),
#     label=f"Fixed tree, gene conversion = {marked_gc:.4f}",
#     color=color_scheme.blue,
# )

# Plot gfs with gc =
marked_gc, marked_loss, mean_gfs, _, _, _ = data[50]
plt.plot(
    range(1, num_samples + 1),
    normalise_gfs(mean_gfs),
    label=f"Fixed tree, κ = {marked_gc:.4f}.",
    color=color_scheme.blue,
)

plt.legend()
plt.xlabel("GF Class")
plt.ylabel("Gene Frequency")
plt.tight_layout()
plt.savefig(PLOT_GC_FN_PDF)
plt.savefig(PLOT_GC_FN_SVG)
plt.show()

## Non Normalised
plt.figure(figsize=(8, 4))
plt.plot(
    range(1, num_samples + 1),
    egfs,
    color=color_scheme.secondary,
    label="Random tree, κ = 0.",
)

# Plot gfs with gc = 0
gene_conv, _, mean_gfs, _, _, _ = min(data)
plt.plot(
    range(1, num_samples + 1),
    mean_gfs,
    label="Fixed tree, κ = 0.",
    color=color_scheme.primary,
)


# # Plot gfs with minimum loss
# gene_conv, _, mean_gfs, _, _ = min(data, key=lambda d: d[1])
# plt.plot(
#     range(1, num_samples + 1),
#     mean_gfs,
#     label=f"Fixed tree, gc = {gene_conv:.4f}",
#     color=color_scheme.blue,
# )

# Plot gfs with gc =
gene_conv, _, mean_gfs, _, _, _ = data[50]
plt.plot(
    range(1, num_samples + 1),
    mean_gfs,
    label=f"Fixed tree, κ = {gene_conv:.4f}",
    color=color_scheme.blue,
)

plt.legend()
plt.ylabel("Gene Frequency")
plt.xlabel("GF Class")
plt.tight_layout()
plt.savefig(PLOT_GC_NON_NORMA_FN_PDF)
plt.savefig(PLOT_GC_NON_NORMA_FN_SVG)
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(8, 4))

gene_conv, fixed_egfs_loss, _, random_fixed_loss, _, _ = zip(*data)
ax.plot(
    gene_conv,
    ma(fixed_egfs_loss),
    color=color_scheme.secondary,
    label="Against random trees, neutral evolution.",
)

ax.plot(
    gene_conv,
    ma(random_fixed_loss),
    color=color_scheme.primary,
    label="Against random trees, indentical κ.",
)

plt.axvline(
    x=0,
    color=color_scheme.primary,
    linestyle="--",
    label=f"κ = {0:.4f}",  # , χ²-like error: {zero_loss:.0f}",
)
plt.axvline(
    x=marked_gc,
    color=color_scheme.blue,
    linestyle="--",
    label=f"κ = {marked_gc:.4f}",  # , χ²-like error: {marked_loss:.0f}",
)


ax.set_ylabel("Mean χ²-like error")
ax.set_xlabel("Gene Conversion")
plt.legend()
plt.tight_layout()

plt.savefig(PLOT_FAC_FN_PDF)
plt.savefig(PLOT_FAC_FN_SVG)
plt.show()
