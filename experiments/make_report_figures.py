"""Summary figures for experiments/WRITEUP.md.

Usage: uv run python experiments/make_report_figures.py
Writes experiments/results/report_fig1_forward.png and report_fig2_tradeoff.png.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# Reference palette (dataviz skill), light mode; first three categorical slots.
SURFACE, INK, INK2, GRID, CONTEXT = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#a8a7a1"
SERIES = {"MINov": "#2a78d6", "1-NN distance": "#eb6834", "Gaussian KDE, median bandwidth": "#1baf7a"}
NARROW = "#1baf7a"  # slot 3 reused in figure 2 for the narrow MI kernel (separate figure)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def fig_forward():
    c = pd.read_csv(os.path.join(RES, "e7_fwd2000_candidates.csv.gz"))
    names = {"MINov (exact)": "MINov", "1-NN distance": "1-NN distance",
             "KDE (median bw)": "Gaussian KDE, median bandwidth"}
    groups = [("a_G", "elmd", "GNoME\nElMD"), ("a_G", "lostop", "GNoME\nLoStOP"),
              ("b_R23new", "elmd", "Later\ndiscoveries\nElMD"), ("b_R23new", "lostop", "Later\ndiscoveries\nLoStOP")]
    stats = {}
    for score, label in names.items():
        for pool, metric, _ in groups:
            d = c[(c.score == score) & (c.pool == pool) & (c.metric == metric)]
            flips = (d.novelY5 != d.novelF5).mean() * 100
            cover = ((d.rankF >= d.rank_lo) & (d.rankF <= d.rank_hi)).mean() * 100
            stats[(label, pool, metric)] = (flips, cover)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(groups))
    width = 0.26
    for k, (ax, title, ylabel) in enumerate([
        (axes[0], "Novelty calls that changed as the reference grew", "% of candidates (call flipped)"),
        (axes[1], "Coverage of nominal 90% rank intervals", "% of candidates (future rank inside interval)"),
    ]):
        for i, label in enumerate(SERIES):
            vals = [stats[(label, p, m)][k] for p, m, _ in groups]
            bars = ax.bar(x + (i - 1) * width, vals, width, color=SERIES[label], label=label,
                          edgecolor=SURFACE, linewidth=2)
            if k == 0:
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", va="bottom",
                            fontsize=8, color=INK2)
        ax.set_xticks(x, [g[2] for g in groups])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
    axes[1].axhline(90, color=INK2, linestyle="--", linewidth=1)
    axes[1].text(3.45, 91.5, "nominal 90%", ha="right", fontsize=8, color=INK2)
    axes[1].set_ylim(0, 100)
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("Forward test: reference = experimental Li compounds known by 2000; future = full record",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(RES, "report_fig1_forward.png"), dpi=200)


def fig_tradeoff():
    f = pd.read_csv(os.path.join(RES, "e9_frontier.csv"))
    a = pd.read_csv(os.path.join(RES, "e9_discovery_auc.csv"))
    a = a[(a.Y == 2000) & (a.ref == "loo") & (a.comparison == "post")]
    sweep = f[f.family == "MI_q"].copy()
    sweep["q"] = sweep.score.str.replace("MI q=", "").astype(float)
    sweep = sweep.sort_values("q")
    highlight = [("MI q=0.5", "MINov (median cutoff)", SERIES["MINov"]),
                 ("MI q=0.02", "narrow MI kernel (q = 0.02)", NARROW),
                 ("kNN p=0", "1-NN distance", SERIES["1-NN distance"])]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
    for j, metric in enumerate(["lostop", "elmd"]):
        fm = f[f.metric == metric].set_index("score")
        am = a[a.metric == metric].set_index("score")
        sm = sweep[sweep.metric == metric]
        for i, (ylabel, getter) in enumerate([
            ("Sensitivity to a local gap\n(novelty rise when 20 neighbours removed)", lambda s: fm.loc[s, "delta_knn_20"]),
            ("AUC: reported after 2000\nvs before (0.5 = chance)", lambda s: am.loc[s, "auc"]),
        ]):
            ax = axes[i, j]
            xs = sm.stab_0_1 if "stab_0_1" in sm else sm["stab_0.1"]
            ys = [getter(s) for s in sm.score]
            ax.plot(xs, ys, color=CONTEXT, linewidth=2, zorder=1, label="MI kernel, width swept (narrow left, median right)")
            offsets = {"MI q=0.5": (-10, 8, "right"), "MI q=0.02": (-10, -16, "right"), "kNN p=0": (10, -4, "left")}
            for s, label, color in highlight:
                xv, yv = fm.loc[s, "stab_0.1"], getter(s)
                ax.scatter([xv], [yv], s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=3, label=label)
                dx, dy, ha = offsets[s]
                ax.annotate(label, (xv, yv), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8.5, color=INK)
            if i == 1:
                ax.axhline(0.5, color=INK2, linestyle="--", linewidth=1)
                ax.set_xlabel("Stability: rank correlation with full reference\nwhen only 10% of it is kept")
            ax.set_ylabel(ylabel)
            ax.set_title({"lostop": "LoStOP (structure)", "elmd": "ElMD (composition)"}[metric] if i == 0 else "")
    axes[0, 0].legend(frameon=False, loc="lower left", fontsize=8.5)
    fig.suptitle("Kernel width trades stability for local sensitivity (reference: 2,095 experimental MP Li compounds)",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(RES, "report_fig2_tradeoff.png"), dpi=200)


if __name__ == "__main__":
    fig_forward()
    fig_tradeoff()
    print("wrote report_fig1_forward.png, report_fig2_tradeoff.png")
