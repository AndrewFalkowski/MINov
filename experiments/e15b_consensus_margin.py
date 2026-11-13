"""E15b: Consensus across reference databases, rank-within-pool calls, and whether E13's margin band covers
reference-choice uncertainty.

Inputs: E15 outputs (e15_scores.npz, e15_references.csv), E13 calibration (e13_calibration.csv), and the Li
distance matrices (only to recompute reference A's leave-one-out scores for the margin).
References: A (R23), B (+N25_theo), C (+G_new), D (all MP-2023 Li, ElMD only); see e15_reference_choice.py.
Pools: G = the 2,628 G candidates eligible under every reference (the 372 imported into MP-2025 are excluded);
  P44 = the 30 eligible P44 candidates.

1. Consensus. LOO call (E15 rule: score < 5th percentile of the reference's LOO scores): share novel under A,
   under all of the references, under any, and under A only (novel under A and under none of the others).
   Rank call: the top p% most novel (lowest score; p = 5 / 10 / 20) within the candidate pool under each
   reference, no reference threshold. For P44 the pool is G + P44 (eligible), and the P44 share is reported.
   Reference sets: ElMD {A,B,C,D} and {A,B,C}; LoStOP {A,B,C}. Per call type also the A->X call-change share.
   retention = share under all / share under A (1 = every A call survives every reference).
2. Margin band (E13). Margin = percentile of the candidate's A score among A's LOO scores (E13's
   percentile_of) minus 5, in pp; uncertain if |margin| <= c, c = E13 li2000 pooled percentile c_cap90 (and
   c_cap95). Reported: share of G calls uncertain under A, share of A->B / A->C / A->D call changes inside the
   band, and the c that would capture 90% of those changes.

Outputs (experiments/results/): e15b_consensus.csv, e15b_margin.csv, e15b_consensus.png.
Usage: uv run python experiments/e15b_consensus_margin.py   (single process, < 1 min)
"""

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
import e15_reference_choice as e15  # noqa: E402
from e13_margin_rule import percentile_of  # noqa: E402

OUT = e15.OUT
NAMES = e15.NAMES
Q = 5
TOPS = (5, 10, 20)
SETS = {("elmd", "ABCD"): ["A", "B", "C", "D"], ("elmd", "ABC"): ["A", "B", "C"], ("lostop", "ABC"): ["A", "B", "C"]}


def loo_A(metric):
    U, mp, _ = e15.load()
    ref = e15.entries(U, mp)["A"]
    Mx = e15.matrices()[metric]
    r, m, _ = e15.ref_rows(ref, metric)
    D = e15.ref_block(Mx, r)
    prof = e15.pair_profile(D, m)
    out = {k: [] for k in NAMES}
    for s in range(0, len(r), e15.BLOCK):
        rr = np.arange(s, min(s + e15.BLOCK, len(r)))
        b = e15.scores_block(D[rr], m, prof, self_rows=rr, mult_self=m[rr])
        for k in NAMES:
            out[k].append(b[k])
    return {k: np.repeat(np.concatenate(v), m) for k, v in out.items()}


def top_call(s, pool_idx, p):
    """Boolean over all candidates: in the top p% most novel (lowest s) within pool_idx."""
    k = max(1, int(round(p / 100 * len(pool_idx))))
    o = pool_idx[np.argsort(s[pool_idx], kind="stable")[:k]]
    c = np.zeros(len(s), bool)
    c[o] = True
    return c


def main():
    Z = np.load(os.path.join(OUT, "e15_scores.npz"), allow_pickle=False)
    cset = Z["cand_set"]
    refs = pd.read_csv(os.path.join(OUT, "e15_references.csv")).set_index(["ref", "metric"])
    isG = cset == "G"
    elig = ~(Z["B|elmd|excluded"] | Z["C|elmd|excluded"] | Z["D|elmd|excluded"] | Z["B|lostop|excluded"])
    pools = {"G": np.flatnonzero(isG & elig), "P44": np.flatnonzero(~isG & elig)}
    rank_pool = np.flatnonzero(elig)  # G + P44 eligible, used to rank P44

    rows = []
    for (metric, tag), R in SETS.items():
        for k in NAMES:
            S = {r: Z[f"{r}|{metric}|{k}"] for r in R}
            calls = {"LOO 5%": {r: S[r] < refs.loc[(r, metric), f"thr|{k}"] for r in R}}
            for p in TOPS:
                calls[f"rank top {p}%"] = {
                    "G": {r: top_call(S[r], pools["G"], p) for r in R},
                    "P44": {r: top_call(S[r], rank_pool, p) for r in R}}
            for ctype, cc in calls.items():
                for pool, idx in pools.items():
                    C = {r: (cc[pool][r] if ctype != "LOO 5%" else cc[r])[idx] for r in R}
                    M = np.array([C[r] for r in R])
                    a = C["A"].mean()
                    row = dict(metric=metric, refs=tag, score=k, call=ctype, pool=pool, n=len(idx),
                               novel_A=a, novel_all=M.all(0).mean(), novel_any=M.any(0).mean(),
                               novel_A_only=(C["A"] & ~M[1:].any(0)).mean(),
                               retention=M.all(0).mean() / a if a else np.nan)
                    for r in R[1:]:
                        row[f"change_A->{r}"] = (C["A"] != C[r]).mean()
                    rows.append(row)
    cons = pd.DataFrame(rows)
    cons.to_csv(os.path.join(OUT, "e15b_consensus.csv"), index=False)

    # 2. margin band
    cal = pd.read_csv(os.path.join(OUT, "e13_calibration.csv"))
    cal = cal[(cal.calibration == "li2000") & (cal.margin == "pct") & (cal.scope == "pooled")].set_index("score")
    mrows = []
    for metric in e15.METRICS:
        L = loo_A(metric)
        for k in NAMES:
            sA = Z[f"A|{metric}|{k}"]
            thr = refs.loc[("A", metric), f"thr|{k}"]
            margin = percentile_of(sA, L[k]) - Q
            assert np.array_equal(margin < 0, sA < thr), "margin sign must match the A call"
            for pool in ("G", "P44"):
                idx = pools[pool]
                am = np.abs(margin[idx])
                for rule in ("c_cap90", "c_cap95"):
                    c = cal.loc[k, rule]
                    unc = am <= c
                    row = dict(metric=metric, score=k, pool=pool, n=len(idx), rule=rule, c_pp=c,
                               uncertain_under_A=unc.mean(), novel_under_A=(margin[idx] < 0).mean())
                    for r in ["B", "C", "D"] if metric == "elmd" else ["B", "C"]:
                        ch = (sA[idx] < thr) != (Z[f"{r}|{metric}|{k}"][idx] < refs.loc[(r, metric), f"thr|{k}"])
                        row[f"n_change_A->{r}"] = int(ch.sum())
                        row[f"changes_in_band_A->{r}"] = unc[ch].mean() if ch.any() else np.nan
                        row[f"c90_needed_A->{r}"] = np.percentile(am[ch], 90) if ch.any() else np.nan
                        row[f"median_abs_margin_changes_A->{r}"] = np.median(am[ch]) if ch.any() else np.nan
                    mrows.append(row)
    marg = pd.DataFrame(mrows)
    marg.to_csv(os.path.join(OUT, "e15b_margin.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(cons[cons.pool == "G"].round(3).to_string(index=False))
    print(marg.round(3).to_string(index=False))
    fig(cons)


def fig(cons):
    import matplotlib.pyplot as plt
    import make_report_figures as mrf

    labels = {"MINov (exact)": "MINov", "1-NN distance": "1-NN distance",
              "KDE (median bw)": "Gaussian KDE, median bandwidth"}
    ctypes = ["LOO 5%", "rank top 5%", "rank top 10%", "rank top 20%"]
    xt = ["reference LOO\nthreshold (5%)", "top 5% of\ncandidates", "top 10% of\ncandidates", "top 20% of\ncandidates"]
    d = cons[cons.pool == "G"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    width = 0.26
    for ax, (metric, tag, title) in zip(axes, [("elmd", "ABCD", "ElMD (composition), references A, B, C, D"),
                                                ("lostop", "ABC", "LoStOP (structure), references A, B, C")]):
        x = np.arange(len(ctypes))
        for i, (k, lab) in enumerate(labels.items()):
            v = [d[(d.metric == metric) & (d.refs == tag) & (d.score == k) & (d.call == c)].retention.iloc[0] * 100
                 for c in ctypes]
            bars = ax.bar(x + (i - 1) * width, v, width, color=mrf.SERIES[lab], label=lab, edgecolor=mrf.SURFACE,
                          linewidth=2)
            for b, val in zip(bars, v):
                ax.text(b.get_x() + b.get_width() / 2, val + 1.5, f"{val:.0f}", ha="center", va="bottom", fontsize=7,
                        color=mrf.INK2)
        ax.set_xticks(x, xt, fontsize=8)
        ax.set_ylim(0, 110)
        ax.set_title(title)
        ax.set_ylabel("% of A's novel calls that are novel\nunder every reference")
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=3, fontsize=9)
    fig.suptitle("Agreement across reference databases: threshold calls vs ranking within the candidate pool "
                 "(2,628 GNoME candidates)", x=0.01, ha="left", fontsize=10, color=mrf.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(OUT, "e15b_consensus.png"), dpi=200)


if __name__ == "__main__":
    main()
