"""E17: An absolute novelty rule ("fewer than k effective known analogues") vs the relative 5% LOO rule.

E15 showed that most reference-choice call changes come from the relative rule (novel = score below the 5th
percentile of the reference's own leave-one-out scores), not from the candidates' scores. Here the threshold is a
fixed number instead:
  MINov (exact)    novel if density < k. Density = sum of MI weights in [0, 1] over reference entries, i.e. a
                   weighted count of known neighbours within the MI cutoff ("effective known analogues").
  KDE (median bw)  novel if Gaussian kernel sum < k (kernel = 1 at distance 0, so also a weighted count).
  1-NN distance    novel if nearest reference distance > d, d = f x MI cutoff of the baseline reference
                   (A for reference choice, reference_Y for growth), held fixed when the reference changes.
  k, f are choices. Grids: k in K_GRID (the requested {0.5, 1, 2, 5} plus larger values, because MINov and KDE
  densities with the median cutoff/bandwidth are tens to hundreds), f in F_GRID. Also "frozen": the baseline
  reference's own 5% LOO threshold kept as a fixed number (same calls as the relative rule on the baseline,
  absolute afterwards).

1. Reference choice: references A, B, C, D and size-matched Xsub of E15 (scores from e15_scores.npz; pools G and
   P44, eligibility as E15). Share novel, call-change share and direction for A->X. Profile freeze: MINov and KDE
   rescored against X's entries with A's MI profile / bandwidth (e15 engine), separating the cutoff moving from
   the entries added.
2. Reference growth: E7 (Li) and E12 (Na, nitride) forward tables, Y = 1990 / 2000 / 2005, pools a (GNoME) and b
   (later discoveries). Stored sY, sF (future = full record, pool-b masking) and stored relative calls; MI cutoffs
   of reference_Y recomputed for the 1-NN rule.
3. Informativeness under A: share novel among A's own members (leave-one-out) vs among GNoME candidates per rule.

Outputs (experiments/results/): e17_choice.csv, e17_growth.csv (per cell), e17_growth_summary.csv,
  e17_informative.csv, e17_cutoffs.csv, e17_call_changes.png.
Usage: uv run python experiments/e17_absolute_rule.py [fig]   (single process, < 1 min; 'fig' redraws only)
"""

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
import e12_new_chemistries as e12  # noqa: E402
import e15_reference_choice as e15  # noqa: E402
from e15b_consensus_margin import loo_A  # noqa: E402

OUT = e15.OUT
NAMES = e15.NAMES
K_GRID = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
F_GRID = [0.05, 0.1, 0.25, 0.5]
K_FIG, F_FIG = 1, 0.25
N_DRAWS = e15.N_DRAWS


def rules(name):
    """(rule, param) pairs for a score."""
    out = [("relative", 5), ("frozen", np.nan)]
    if name == "1-NN distance":
        out += [("absolute", f) for f in F_GRID]
    else:
        out += [("absolute", k) for k in K_GRID]
    return out


def call(name, rule, param, s, thr_own, thr_base, cutoff_base):
    if rule == "relative":
        return s < thr_own
    if rule == "frozen":
        return s < thr_base
    if name == "1-NN distance":
        return -s > param * cutoff_base
    return s < param


# 1. Reference choice ######################################################################


def frozen_profile_scores(metric):
    """MINov and KDE of all candidates against B, C, D entries with A's MI profile and KDE bandwidth."""
    U, mp, _ = e15.load()
    refs = e15.entries(U, mp)
    Mx = e15.matrices()[metric]
    cands = U[U.set.isin(["G", "P44"])].reset_index(drop=True)
    q = e15.cand_rows(cands, metric)
    rA, mA, _ = e15.ref_rows(refs["A"], metric)
    profA = e15.pair_profile(e15.ref_block(Mx, rA), mA)
    out = {}
    for X in ["B", "C", "D"] if metric == "elmd" else ["B", "C"]:
        r, m, _ = e15.ref_rows(refs[X], metric)
        acc = {k: [] for k in NAMES}
        for s in range(0, len(q), e15.BLOCK):
            b = e15.scores_block(e15.query_block(Mx, q[s:s + e15.BLOCK], r), m, profA)
            for k in NAMES:
                acc[k].append(b[k])
        out[X] = {k: np.concatenate(v) for k, v in acc.items()}
    return out


def stage_choice():
    Z = np.load(os.path.join(OUT, "e15_scores.npz"))
    R = pd.read_csv(os.path.join(OUT, "e15_references.csv")).set_index(["ref", "metric"])
    isG = Z["cand_set"] == "G"
    rows = []
    for metric in e15.METRICS:
        frz = frozen_profile_scores(metric)
        cut = R.loc[("A", metric), "mi_cutoff"]
        targets = ["B", "C", "D"] if metric == "elmd" else ["B", "C"]
        for k in NAMES:
            sA = Z[f"A|{metric}|{k}"]
            thrA = R.loc[("A", metric), f"thr|{k}"]
            comps = []
            for X in targets:
                comps.append((X, X, "own", Z[f"{X}|{metric}|{k}"], R.loc[(X, metric), f"thr|{k}"]))
                if k != "1-NN distance":
                    comps.append((X, X, "A profile", frz[X][k], np.nan))
                for d in range(N_DRAWS):
                    Xs = f"{X}sub{d}"
                    comps.append((f"{X}sub", Xs, "own", Z[f"{Xs}|{metric}|{k}"], R.loc[(Xs, metric), f"thr|{k}"]))
            for fam, ref, prof, sX, thrX in comps:
                elig = ~Z[f"A|{metric}|excluded"] & ~Z[f"{ref}|{metric}|excluded"]
                for rule, param in rules(k):
                    if rule == "relative" and prof != "own":
                        continue
                    cA = call(k, rule, param, sA, thrA, thrA, cut)
                    cX = call(k, rule, param, sX, thrX, thrA, cut)
                    for pool, sel in [("G", isG & elig), ("P44", ~isG & elig)]:
                        a, b = cA[sel], cX[sel]
                        rows.append(dict(metric=metric, score=k, rule=rule, param=param, pair=f"A->{fam}", ref=ref,
                                         profile=prof, pool=pool, n=int(sel.sum()), novel_A=a.mean(), novel_X=b.mean(),
                                         call_change=(a != b).mean(), n_novel_to_not=int((a & ~b).sum()),
                                         n_not_to_novel=int((~a & b).sum())))
        e15.log(f"choice {metric}")
    df = pd.DataFrame(rows)
    keys = ["metric", "score", "rule", "param", "pair", "profile", "pool"]
    agg = df.groupby(keys, dropna=False).agg(n=("n", "mean"), novel_A=("novel_A", "mean"), novel_X=("novel_X", "mean"),
                                             call_change=("call_change", "mean"), cc_min=("call_change", "min"),
                                             cc_max=("call_change", "max"), n_novel_to_not=("n_novel_to_not", "mean"),
                                             n_not_to_novel=("n_not_to_novel", "mean"), n_draws=("ref", "size"))
    agg.reset_index().to_csv(os.path.join(OUT, "e17_choice.csv"), index=False)
    return agg.reset_index()


# 3. Informativeness under A ##############################################################


def stage_informative():
    Z = np.load(os.path.join(OUT, "e15_scores.npz"))
    R = pd.read_csv(os.path.join(OUT, "e15_references.csv")).set_index(["ref", "metric"])
    isG = Z["cand_set"] == "G"
    rows = []
    for metric in e15.METRICS:
        L = loo_A(metric)
        cut = R.loc[("A", metric), "mi_cutoff"]
        for k in NAMES:
            thr = R.loc[("A", metric), f"thr|{k}"]
            for rule, param in rules(k):
                if rule == "frozen":
                    continue
                g = call(k, rule, param, Z[f"A|{metric}|{k}"][isG], thr, thr, cut)
                ref = call(k, rule, param, L[k], thr, thr, cut)
                rows.append(dict(metric=metric, score=k, rule=rule, param=param,
                                 threshold_value=param * cut if (rule == "absolute" and k == "1-NN distance") else
                                 (param if rule == "absolute" else thr),
                                 novel_G=g.mean(), novel_A_members_loo=ref.mean(),
                                 ratio=g.mean() / ref.mean() if ref.mean() else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e17_informative.csv"), index=False)
    return df


# 2. Reference growth #####################################################################


def cutoffs():
    """MI cutoff of reference_Y (experimental entries first reported <= Y) per chem, Y, metric."""
    rows = []
    Uli = pd.read_csv(os.path.join(e15.PROC, "U_index.csv"))
    fy = pd.read_csv(os.path.join(e15.PROC, "r23_first_year.csv"))[["row", "first_year"]]
    Uli = Uli.merge(fy, on="row", how="left")
    specs = {"li": (Uli[Uli.set == "R23"], e15.PROC)}
    for chem in ["na", "nitride"]:
        U = e12.load_index(chem)
        specs[chem] = (U[U.set == "R"], e12.proc(chem))
    for chem, (Ur, proc) in specs.items():
        for metric in e15.METRICS:
            Mx = np.load(os.path.join(proc, f"U_{metric}.npy"), mmap_mode="r")
            for Y in [1990, 2000, 2005]:
                r = np.sort(Ur.row.values[Ur.first_year.values <= Y])
                prof = e15.pair_profile(e15.ref_block(Mx, r), np.ones(len(r), int))
                rows.append(dict(chem=chem, Y=Y, metric=metric, n_ref=len(r), mi_cutoff=prof["cutoff"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e17_cutoffs.csv"), index=False)
    return df


def stage_growth():
    cut = cutoffs().set_index(["chem", "Y", "metric"])
    tabs = [pd.read_csv(os.path.join(OUT, f"e7_fwd{Y}_candidates.csv.gz")).assign(chem="li") for Y in [1990, 2000, 2005]]
    tabs += [e12.read_csv(os.path.join(OUT, f"e12_{c}_fwd_candidates.csv.gz"), dtype={"material_id": str})
             for c in ["na", "nitride"]]
    rows = []
    for t in tabs:
        t = t.copy()
        t["pool"] = np.where(t.pool.str.startswith("a"), "a_G", "b_later")
        for (chem, Y, metric, k, pool), d in t.groupby(["chem", "Y", "metric", "score", "pool"]):
            c = cut.loc[(chem, Y, metric), "mi_cutoff"]
            sY, sF, thrY = d.sY.values, d.sF.values, d.thrY5.values
            for rule, param in rules(k):
                if rule == "relative":
                    a, b = d.novelY5.values.astype(bool), d.novelF5.values.astype(bool)
                else:
                    a = call(k, rule, param, sY, None, thrY, c)
                    b = call(k, rule, param, sF, None, thrY, c)
                rows.append(dict(chem=chem, Y=Y, metric=metric, score=k, pool=pool, rule=rule, param=param, n=len(d),
                                 novel_Y=a.mean(), novel_F=b.mean(), flip=(a != b).mean(),
                                 n_novel_to_not=int((a & ~b).sum()), n_not_to_novel=int((~a & b).sum())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e17_growth.csv"), index=False)
    g = df.groupby(["metric", "score", "pool", "rule", "param"], dropna=False)
    s = g.agg(n_cells=("flip", "size"), flip_median=("flip", "median"), flip_min=("flip", "min"),
              flip_max=("flip", "max"), novel_Y_median=("novel_Y", "median"), novel_Y_min=("novel_Y", "min"),
              novel_Y_max=("novel_Y", "max")).reset_index()
    s.to_csv(os.path.join(OUT, "e17_growth_summary.csv"), index=False)
    e15.log("growth")
    return df


# Figure ##################################################################################


def stage_fig():
    import matplotlib.pyplot as plt
    import make_report_figures as mrf

    ch = pd.read_csv(os.path.join(OUT, "e17_choice.csv"))
    gr = pd.read_csv(os.path.join(OUT, "e17_growth.csv"))
    ch = ch[(ch.pool == "G") & (ch.profile == "own")]
    labels = {"MINov (exact)": "MINov", "1-NN distance": "1-NN distance",
              "KDE (median bw)": "Gaussian KDE, median bandwidth"}
    rl = [("relative", "relative: below 5% of the reference's LOO scores"),
          ("frozen", "absolute: A's 5% threshold kept fixed"),
          ("absolute", f"absolute: density < {K_FIG} (1-NN: distance > {F_FIG} x MI cutoff)")]
    hatch = {"relative": "", "frozen": "///", "absolute": ""}
    alpha = {"relative": 0.35, "frozen": 1.0, "absolute": 1.0}

    def pick(d, k, rule):
        if rule == "absolute":
            return d[(d.rule == rule) & (d.param == (F_FIG if k == "1-NN distance" else K_FIG))]
        return d[d.rule == rule]

    groups = [("A->B", "+ MP-2025\nadditions"), ("A->C", "+ other GNoME\npredictions"),
              ("A->D", "all MP-2023\n(incl. theoretical)"), ("li2000a_G", "growth Li 2000\nGNoME"),
              ("li2000b_later", "growth Li 2000\nlater discoveries")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), sharey="row")
    w = 0.27
    for i, metric in enumerate(["elmd", "lostop"]):
        for j, (k, lab) in enumerate(labels.items()):
            ax = axes[i, j]
            gs = [g for g in groups if not (metric == "lostop" and g[0] == "A->D")]
            x = np.arange(len(gs))
            for r, (rule, rlab) in enumerate(rl):
                v = []
                for fam, _ in gs:
                    if fam.startswith("li2000"):
                        d = gr[(gr.chem == "li") & (gr.Y == 2000) & (gr.metric == metric) & (gr.score == k)
                               & (gr.pool == fam[6:])]
                        v.append(pick(d, k, rule).flip.iloc[0] * 100)
                    else:
                        d = ch[(ch.pair == fam) & (ch.metric == metric) & (ch.score == k)]
                        v.append(pick(d, k, rule).call_change.iloc[0] * 100)
                bars = ax.bar(x + (r - 1) * w, v, w, color=mrf.SERIES[lab], alpha=alpha[rule], hatch=hatch[rule],
                              edgecolor=mrf.SURFACE, linewidth=1.5, label=rlab)
                for b, val in zip(bars, v):
                    ax.text(b.get_x() + b.get_width() / 2, val + 0.8, f"{val:.1f}", ha="center", va="bottom",
                            fontsize=6.5, color=mrf.INK2)
            dA = ch[(ch.pair == "A->C") & (ch.metric == metric) & (ch.score == k)]
            ax.text(1.0, 1.01, f"novel under A: relative {pick(dA, k, 'relative').novel_A.iloc[0] * 100:.1f}%, "
                    f"absolute {pick(dA, k, 'absolute').novel_A.iloc[0] * 100:.1f}%", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.5, color=mrf.INK2)
            ax.axvline(len(gs) - 2.5, color=mrf.CONTEXT, linewidth=1, linestyle="--")
            ax.set_xticks(x, [g[1] for g in gs], fontsize=7)
            ax.set_title(f"{lab.replace('Gaussian KDE, median bandwidth', 'KDE (median bw)')}, {'ElMD' if metric == 'elmd' else 'LoStOP'}", fontsize=10)
            if j == 0:
                ax.set_ylabel("% of candidates whose call changed")
            ax.grid(axis="x", visible=False)
            ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    hs = [Patch(facecolor=mrf.CONTEXT, alpha=0.6, label=rl[0][1]),
          Patch(facecolor=mrf.CONTEXT, hatch="///", edgecolor=mrf.SURFACE, label=rl[1][1]),
          Patch(facecolor=mrf.INK2, label=rl[2][1])]
    fig.legend(handles=hs, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.95), ncol=3, fontsize=8.5)
    fig.suptitle("Call changes under a relative vs an absolute novelty rule: reference choice (A -> X, 2,628-3,000 "
                 "GNoME candidates) and reference growth (E7)", x=0.01, ha="left", fontsize=10, color=mrf.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(os.path.join(OUT, "e17_call_changes.png"), dpi=200)
    e15.log("figure")


def main():
    if sys.argv[1:] == ["fig"]:
        return stage_fig()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 500)
    ch = stage_choice()
    inf = stage_informative()
    gr = stage_growth()
    stage_fig()
    print(ch[(ch.pool == "G") & ch.pair.isin(["A->B", "A->C", "A->D"])].round(3).to_string(index=False))
    print(inf.round(4).to_string(index=False))
    s = pd.read_csv(os.path.join(OUT, "e17_growth_summary.csv"))
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
