import numpy as np

from sklearn.datasets import make_blobs
from scipy.spatial.distance import pdist, squareform

from mi_density import *

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.style.use("ggplot")
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "monospace"

# set a custom color cycle
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=["#348ABD", "#A60628", "#7A68A6", "#467821", "#D95204", "#D982BA"]
)

colors = [
    "#348abd",  # Start color
    "#4c7dc1",
    "#696ebd",
    "#865baf",
    "#9c4497",
    "#aa2a76",
    "#ae0d50",
    "#a60628",  # End color
]
n_bins = 256
cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=n_bins)


SAVE = True
FIGSIZE = (3.5, 3.5)
POS = [0.17, 0.17, 0.8, 0.8]

# Plot the Dummy Data ##################################################################

# Define the parameters for each blob
blob_params = [
    {"center": [0, 0], "std": 0.1, "random_state": 10},
    {"center": [1, 1], "std": 0.3, "random_state": 5},
    {"center": [2, 2], "std": 0.5, "random_state": 6},
    {"center": [2, 1], "std": 0.1, "random_state": 11},
]

# Generate all blobs and concatenate
X = np.vstack(
    [
        make_blobs(
            n_samples=20,
            centers=[params["center"]],
            cluster_std=params["std"],
            random_state=params["random_state"],
        )[0]
        for params in blob_params
    ]
)

X_dm = squareform(pdist(X, metric="euclidean"))

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=500, layout="constrained")

ax.set_position(POS, which="both")

plt.scatter(X[:, 0], X[:, 1], c="k", marker=".", s=85)

plt.scatter(X[50, 0], X[50, 1], c="C0", marker=".", s=95, ec="k")
plt.scatter(X[56, 0], X[56, 1], c="C1", marker=".", s=95, ec="k")
plt.scatter(X[47, 0], X[47, 1], c="C2", marker=".", s=95, ec="k")

# annotate those points of interest A, B, C
plt.annotate(
    "A", (X[50, 0] + 0.05, X[50, 1] + 0.05), color="k", fontsize=8, weight="bold"
)
plt.annotate(
    "B", (X[56, 0] + 0.05, X[56, 1] + 0.05), color="k", fontsize=8, weight="bold"
)
plt.annotate(
    "C", (X[47, 0] + 0.05, X[47, 1] + 0.05), color="k", fontsize=8, weight="bold"
)

plt.yticks([0.0, 1.0, 2.0, 3.0], ["0.0", "1.0", "2.0", "3.0"])
plt.xticks([0.0, 1.0, 2.0, 3.0], ["0.0", "1.0", "2.0", "3.0"])

plt.xlim(-0.5, 3.5)
plt.ylim(-0.5, 3.25)
plt.xlabel("X$_1$ [a.u.]")
plt.ylabel("X$_2$ [a.u.]")

if SAVE:
    plt.savefig("figures/dummydata.png", dpi=500)
else:
    plt.show(block=False)

# Plot the Cummulative #################################################################

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, dpi=500, layout="constrained")

ax.set_position(POS, which="both")

for i, row in enumerate(X_dm):
    count = np.ones(row.shape[0])
    cs = np.cumsum(count) / np.sum(count)

    # add annotations above the lines
    if i in [50, 56, 47]:
        label = "A" if i == 50 else "B" if i == 56 else "C"
        plt.text(
            x=np.sort(row)[-1],
            y=cs[-1] + 0.005,
            s=f"{label}",
            color="k",
            fontsize=8,
            ha="center",
            zorder=100,
            weight="bold",
        )

    # Correct line color assignment using proper elif structure
    line_color = "C0" if i == 50 else "C1" if i == 56 else "C2" if i == 47 else "k"

    # Correct line alpha using proper comparison
    line_alpha = 1.0 if i in [50, 56, 47] else 0.1

    # Correct line zorder
    line_zorder = 2 if i in [50, 56, 47] else 1

    # correct the lw
    lw = 2 if i in [50, 56, 47] else 1

    ax.step(
        np.sort(row),
        cs,
        color=line_color,
        alpha=line_alpha,
        zorder=line_zorder,
        lw=lw,
        where="post",
    )


plt.xlim(0, None)
plt.ylim(0, 1)
plt.xlabel("Distance [a.u.]")
plt.ylabel("Cumulative Probability")

if SAVE:
    plt.savefig("figures/cumulativeprob.png", dpi=500)
else:
    plt.show(block=False)

# Cutoff Distance and Decay ############################################################


def get_sigma_for_decay(cutoff, target_decay):
    """Calculate sigma needed to achieve target decay value at cutoff."""
    return cutoff / np.sqrt(-2 * np.log(target_decay))


cutoff, mi_profile = get_MI_profile(X_dm, 200)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, dpi=500, layout="constrained")

ax.set_position(POS, which="both")

for i, row in enumerate(X_dm):
    count = np.ones(row.shape[0])
    cs = np.cumsum(count) / np.sum(count)
    ax.step(np.sort(row), cs, color="k", alpha=0.15, zorder=1, lw=1, where="post")

x = np.linspace(0, 1.75, 500)

x = np.linspace(np.min(X_dm), np.max(X_dm), 200)
mi_decay = 1 - mi_profile / np.max(mi_profile)

mi_decay = mi_decay[x < cutoff]
x = x[x < cutoff]

ax.plot(x, mi_decay, color="C1", lw=2)
ax.axvline(cutoff, color="C1", linestyle="--", lw=2)

# put the word "cutoff" above the line
plt.text(
    x=cutoff - 0.05,
    y=1.015,
    s="CUTOFF",
    color="k",
    fontsize=8,
    ha="left",
    va="center",
    weight="bold",
)

ax.set_ylim(0, 1.0)
ax.set_xlim(0, np.max(X_dm))

ax.set_xlabel("Distance [a.u.]")
ax.set_ylabel("Decay Factor / Probability")

if SAVE:
    plt.savefig("figures/decayfunc.png", dpi=500)
else:
    plt.show(block=False)

# Compute Dummy Density ################################################################

cutoff = 1.3522078158872688  # computed elsewhere

# dens = compute_gaussian_density(X_dm, cutoff, end_decay=0.01)
cutoff, mi_profile = get_MI_profile(X_dm, 200)
dens = compute_mi_density(X_dm, cutoff, mi_profile)

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=500)

ax.set_position(POS, which="both")

ax.scatter(X[:, 0], X[:, 1], c=dens, marker=".", s=95, cmap="rainbow", ec="k")

# annotate the lowest 5 densities with the lowest starting at 1
for i, d in enumerate(np.argsort(dens)[:5]):
    plt.annotate(
        f"{i+1}", (X[d, 0] + 0.05, X[d, 1] + 0.05), color="k", fontsize=8, weight="bold"
    )

# draw a circle around the 50th point with a radius of the cutoff
ax.add_artist(plt.Circle((X[50, 0], X[50, 1]), cutoff, color="k", fill=False, ls=":"))
ax.add_artist(plt.Circle((X[50, 0], X[50, 1]), 0.115, color="k", fill=False, ls=":"))
ax.add_artist(plt.Circle((X[50, 0], X[50, 1]), 0.236, color="k", fill=False, ls=":"))
ax.add_artist(plt.Circle((X[50, 0], X[50, 1]), 0.645, color="k", fill=False, ls=":"))

# add cbar and make it short and put it within the plot
cax = ax.inset_axes([0.6, 0.2, 0.3, 0.05])  # [x, y, width, height] in axes coordinates
cbar = plt.colorbar(ax.collections[0], cax=cax, orientation="horizontal", pad=0.1)
cax.set_facecolor("white")  # Set background color
cbar.set_label("Density")

ax.add_artist(
    plt.Rectangle(
        (0.59, 0.03),  # Match inset_axes x,y
        0.32,  # Match inset_axes width
        0.23,  # Match inset_axes height
        fc="white",
        ec="k",
        transform=ax.transAxes,
        zorder=1,
    )
)

plt.yticks([0.0, 1.0, 2.0, 3.0], ["0.0", "1.0", "2.0", "3.0"])
plt.xticks([0.0, 1.0, 2.0, 3.0], ["0.0", "1.0", "2.0", "3.0"])

plt.xlim(-0.5, 3.5)
plt.ylim(-0.5, 3.25)
plt.xlabel("X$_1$ [a.u.]")
plt.ylabel("X$_2$ [a.u.]")


if SAVE:
    plt.savefig("figures/dummydensity.png", dpi=500)
else:
    plt.show(block=True)
