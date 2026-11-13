"""E19: Is reference-choice sensitivity specific to GNoME candidates, or does it hold for real discoveries too?

E15 swapped the reference from experimental MP-2023 (A) to other databases and found 28-41% of GNoME calls change
under MINov/ElMD. GNoME candidates sit on the 5% LOO threshold (median at the 5.7th percentile), so this tests
whether the effect is a near-threshold artefact by repeating the swap for real later discoveries.

Data: data/raw/processed (Li). Year split as in E7: A_Y = R23 entries first reported <= Y (Y = 2000; 1990 and 2005
as checks). Candidate pools, scored against the same references:
  later  R23 entries first reported after Y (E7 pool b). At Y no reference contains them or their later siblings.
  G      the 3,000 GNoME Dec-2023 sample (E7 pool a). E15 exclusion: self-copies in the reference are dropped.
Alternative references (E15 engine; ElMD formula-level with multiplicities, LoStOP entry-level):
  B_Y = A_Y + N25_theo;  C_Y = A_Y + G_new;  D_Y = A_Y + all MP-2023 theoretical Li entries (ElMD only).
  Sibling exclusion (default): added entries whose fkey equals any later-discovery candidate's fkey are dropped
  (a theoretical entry of a later-reported formula would leak the future). Variant "noexcl" keeps them (only run
  where something was dropped). Size-matched Xsub: N_DRAWS random subsets of X_Y with |A_Y| entries.
Rule: novel = score < 5th percentile of the reference's LOO scores (E7/E15); everything refit per reference.
Outputs per Y x metric x score x pool x swap: call change, direction, Spearman, top-10% overlap, rank vs
  threshold decomposition (E15), score vs threshold-population decomposition (population = A_Y members).
Near-threshold test: percentile margin under A_Y (E13 percentile_of on A_Y LOO scores, minus 5 pp). G reweighted to
  the later pool's margin distribution (fine bins, importance weights; bins without G are reported as uncovered),
  and call changes per coarse margin bin for both pools. Growth comparison: E7 forward flips (A_Y -> full R23,
  stored) for both pools and for margin-matched G.
Consensus (E15b-style): share novel under A_Y, all, any, A-only, retention; rank-within-pool top 5/10/20%.

Check: A_Y scores and thresholds must reproduce E7's stored sY / thrY5 (e19_check.csv).
Outputs (experiments/results/): e19_references.csv, e19_pairs.csv, e19_pairs_summary.csv, e19_margin_bins.csv,
  e19_matched.csv, e19_consensus.csv, e19_check.csv, e19_call_changes.png.
Usage: uv run python experiments/e19_choice_discoveries.py [fig]   (single process, ~5 min; 'fig' redraws only)
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
YEARS = [2000, 1990, 2005]
N_DRAWS = 5
SEED = 19
FINE = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7.5, 10, 15, 20, 30, 50, 101])
COARSE = np.array([-5, -2.5, 0, 2.5, 5, 10, 20, 40, 101])
TOPS = (5, 10, 20)


# Data ##################################################################################


def setup():
    U, mp, Mt = e15.load()
    fy = pd.read_csv(os.path.join(e15.PROC, "r23_first_year.csv"))[["row", "first_year"]]
    U = U.merge(fy, on="row", how="left")
    return U, mp, Mt


def table(d, uid, member):
    return pd.DataFrame(dict(uid=uid, material_id=d.material_id.values,
                             gnome_id=d.gnome_id.values if "gnome_id" in d else np.nan,
                             fkey=d.fkey.values, frow=d.frow.values, is_r23=member))


def references(U, mp, Y, rng):
    R = U[U.set == "R23"]
    AY = R[R.first_year <= Y]
    L = set(R[R.first_year > Y].fkey)
    A = table(AY, AY.row.values, True)
    add = {"B": U[U.set == "N25_theo"], "C": U[U.set == "G_new"]}
    theo = mp[mp.theoretical.astype(bool)]
    refs, dropped = {"A": A}, {}
    for X in ["B", "C", "D"]:
        d = add[X] if X != "D" else theo
        uid = d.row.values if X != "D" else np.full(len(d), -1)
        sib = d.fkey.isin(L).values
        dropped[X] = int(sib.sum())
        refs[X] = pd.concat([A, table(d[~sib], uid[~sib], False)], ignore_index=True)
        if sib.any():
            refs[f"{X}_noexcl"] = pd.concat([A, table(d, uid, False)], ignore_index=True)
        for k in range(N_DRAWS):
            refs[f"{X}sub{k}"] = refs[X].iloc[np.sort(rng.choice(len(refs[X]), len(A), replace=False))]
    return refs, dropped


def metrics_for(name):
    return ["elmd"] if name.startswith("D") else e15.METRICS


def evaluate(Mx, rows, m, mA, qrows):
    """e15.evaluate plus the reference LOO scores (entry level) and those of the A_Y members."""
    D = e15.ref_block(Mx, rows)
    prof = e15.pair_profile(D, m)
    loo = {k: [] for k in NAMES}
    for s in range(0, len(rows), e15.BLOCK):
        r = np.arange(s, min(s + e15.BLOCK, len(rows)))
        b = e15.scores_block(D[r], m, prof, self_rows=r, mult_self=m[r])
        for k in NAMES:
            loo[k].append(b[k])
    del D
    sq = {k: [] for k in NAMES}
    for s in range(0, len(qrows), e15.BLOCK):
        b = e15.scores_block(e15.query_block(Mx, qrows[s:s + e15.BLOCK], rows), m, prof)
        for k in NAMES:
            sq[k].append(b[k])
    res = dict(cutoff=prof["cutoff"], scores={}, thr={}, thr_pop={}, loo={})
    for k in NAMES:
        L = np.concatenate(loo[k])
        res["scores"][k] = np.concatenate(sq[k])
        res["loo"][k] = np.repeat(L, m)
        res["thr"][k] = float(np.percentile(res["loo"][k], Q))
        res["thr_pop"][k] = float(np.percentile(np.repeat(L, mA), Q))
    return res


def g_excluded(cands, ref, Mt):
    """E15 rule for GNoME candidates; later discoveries are never in a reference (only siblings, see docstring)."""
    ex = e15.excluded(cands, ref, Mt)
    return ex & (cands.pool == "G").values


# Run ###################################################################################


def run_year(Y, U, mp, Mt, Mx, rng):
    refs, dropped = references(U, mp, Y, rng)
    R = U[U.set == "R23"]
    cands = pd.concat([R[R.first_year > Y].assign(pool="later"), U[U.set == "G"].assign(pool="G")],
                      ignore_index=True)
    res, rinfo = {}, []
    for name, ref in refs.items():
        ex = g_excluded(cands, ref, Mt)
        for metric in metrics_for(name):
            r, m, mA = e15.ref_rows(ref, metric)
            out = evaluate(Mx[metric], r, m, mA, e15.cand_rows(cands, metric))
            out["excluded"] = ex
            res[(name, metric)] = out
            if "sub" not in name:
                row = dict(Y=Y, ref=name, metric=metric, n_entries=len(ref), n_unique_rows=len(r),
                           n_AY_members=int(mA.sum()), mi_cutoff=out["cutoff"],
                           n_added_dropped_as_siblings=dropped.get(name, 0),
                           n_G_excluded=int(ex[(cands.pool == "G").values].sum()))
                for k in NAMES:
                    row[f"thr|{k}"] = out["thr"][k]
                    row[f"thr_pop|{k}"] = out["thr_pop"][k]
                rinfo.append(row)
    e15.log(f"Y={Y}: {len(refs)} references")
    return cands, res, pd.DataFrame(rinfo), dropped


def margins(res, metric, cands):
    A = res[("A", metric)]
    return {k: percentile_of(A["scores"][k], A["loo"][k]) - Q for k in NAMES}


def weighted_change(change, mg, pool_mask, target_mask):
    """Call-change share of pool reweighted to the target pool's fine-bin margin distribution."""
    bp = np.clip(np.digitize(mg[pool_mask], FINE) - 1, 0, len(FINE) - 2)
    bt = np.clip(np.digitize(mg[target_mask], FINE) - 1, 0, len(FINE) - 2)
    nb = len(FINE) - 1
    cp = np.bincount(bp, minlength=nb)
    ct = np.bincount(bt, minlength=nb) / len(bt)
    covered = cp > 0
    w = np.where(covered[bp], ct[bp] / np.maximum(cp[bp], 1), 0.0)
    return float((w * change).sum() / w.sum()), float(ct[covered].sum())


def stage_run():
    U, mp, Mt = setup()
    Mx = e15.matrices()
    rng = np.random.default_rng(SEED)
    rows, bins, matched, cons, checks, rinfos = [], [], [], [], [], []
    for Y in YEARS:
        cands, res, rinfo, dropped = run_year(Y, U, mp, Mt, Mx, rng)
        rinfos.append(rinfo)
        pool = cands.pool.values
        # check against E7
        e7 = pd.read_csv(os.path.join(OUT, f"e7_fwd{Y}_candidates.csv.gz"))
        e7["pool2"] = np.where(e7.pool == "a_G", "G", "later")
        for metric in e15.METRICS:
            for k in NAMES:
                d = e7[(e7.metric == metric) & (e7.score == k)].set_index(["pool2", "material_id"])
                key = list(zip(pool, cands.material_id))
                sY = d.loc[key, "sY"].values
                mine = res[("A", metric)]["scores"][k]
                checks.append(dict(Y=Y, metric=metric, score=k, max_abs_diff_sY=float(np.max(np.abs(sY - mine))),
                                   thr_e7=float(d.thrY5.iloc[0]), thr_mine=res[("A", metric)]["thr"][k]))
        for metric in e15.METRICS:
            mg = margins(res, metric, cands)
            e7m = e7[e7.metric == metric]
            swaps = [("A", X) for X in ["B", "C", "D", "B_noexcl", "C_noexcl", "D_noexcl"]]
            swaps += [("A", f"{X}sub{d}") for X in ["B", "C", "D"] for d in range(N_DRAWS)]
            for r1, r2 in swaps:
                if (r2, metric) not in res:
                    continue
                a, b = res[(r1, metric)], res[(r2, metric)]
                elig = ~a["excluded"] & ~b["excluded"]
                for p in ["later", "G"]:
                    sel = elig & (pool == p)
                    for k in NAMES:
                        d = e15.compare(a["scores"][k][sel], a["thr"][k], b["scores"][k][sel], b["thr"][k],
                                        b["thr_pop"][k])
                        pc = mg[k][sel] + Q
                        d.update(Y=Y, metric=metric, score=k, pool=p, ref2=r2,
                                 median_pct_A=float(np.median(pc)),
                                 near_thr_2p5_10=float(((pc >= 2.5) & (pc < 10)).mean()))
                        rows.append(d)
                # margin bins and margin-matched G (main swaps only)
                if "sub" in r2:
                    continue
                for k in NAMES:
                    ch = (a["scores"][k] < a["thr"][k]) != (b["scores"][k] < b["thr"][k])
                    lat, g = elig & (pool == "later"), elig & (pool == "G")
                    cb = np.clip(np.digitize(mg[k], COARSE) - 1, 0, len(COARSE) - 2)
                    for p, msk in [("later", lat), ("G", g)]:
                        for j in range(len(COARSE) - 1):
                            s = msk & (cb == j)
                            bins.append(dict(Y=Y, metric=metric, score=k, ref2=r2, pool=p,
                                             margin_lo=COARSE[j], margin_hi=COARSE[j + 1], n=int(s.sum()),
                                             call_change=ch[s].mean() if s.any() else np.nan))
                    wc, cov = weighted_change(ch[g], mg[k], g, lat)
                    matched.append(dict(Y=Y, metric=metric, score=k, ref2=r2, later=ch[lat].mean(), G=ch[g].mean(),
                                        G_margin_matched=wc, later_mass_covered=cov))
            # growth (E7 stored), incl. margin-matched G
            for k in NAMES:
                d = e7m[e7m.score == k].set_index(["pool2", "material_id"])
                fl = (d.loc[list(zip(pool, cands.material_id)), "novelY5"].values
                      != d.loc[list(zip(pool, cands.material_id)), "novelF5"].values)
                lat, g = pool == "later", pool == "G"
                wc, cov = weighted_change(fl[g], mg[k], g, lat)
                matched.append(dict(Y=Y, metric=metric, score=k, ref2="growth (E7)", later=fl[lat].mean(),
                                    G=fl[g].mean(), G_margin_matched=wc, later_mass_covered=cov))
                cb = np.clip(np.digitize(mg[k], COARSE) - 1, 0, len(COARSE) - 2)
                for p, msk in [("later", lat), ("G", g)]:
                    for j in range(len(COARSE) - 1):
                        s = msk & (cb == j)
                        bins.append(dict(Y=Y, metric=metric, score=k, ref2="growth (E7)", pool=p,
                                         margin_lo=COARSE[j], margin_hi=COARSE[j + 1], n=int(s.sum()),
                                         call_change=fl[s].mean() if s.any() else np.nan))
            # consensus
            R_ = ["A", "B", "C", "D"] if metric == "elmd" else ["A", "B", "C"]
            elig = ~np.any([res[(r, metric)]["excluded"] for r in R_], axis=0)
            for p in ["later", "G"]:
                idx = np.flatnonzero(elig & (pool == p))
                for k in NAMES:
                    S = {r: res[(r, metric)]["scores"][k] for r in R_}
                    calls = {"LOO 5%": {r: (S[r] < res[(r, metric)]["thr"][k])[idx] for r in R_}}
                    for t in TOPS:
                        cc = {}
                        for r in R_:
                            kk = max(1, int(round(t / 100 * len(idx))))
                            c = np.zeros(len(idx), bool)
                            c[np.argsort(S[r][idx], kind="stable")[:kk]] = True
                            cc[r] = c
                        calls[f"rank top {t}%"] = cc
                    for ct, C in calls.items():
                        M = np.array([C[r] for r in R_])
                        a_ = C["A"].mean()
                        cons.append(dict(Y=Y, metric=metric, refs="".join(R_), score=k, call=ct, pool=p, n=len(idx),
                                         novel_A=a_, novel_all=M.all(0).mean(), novel_any=M.any(0).mean(),
                                         novel_A_only=(C["A"] & ~M[1:].any(0)).mean(),
                                         retention=M.all(0).mean() / a_ if a_ else np.nan))
        e15.log(f"Y={Y} analysed")
    pd.DataFrame(checks).to_csv(os.path.join(OUT, "e19_check.csv"), index=False)
    pd.concat(rinfos).to_csv(os.path.join(OUT, "e19_references.csv"), index=False)
    P = pd.DataFrame(rows)
    lead = ["Y", "metric", "score", "pool", "ref2", "n"]
    P = P[lead + [c for c in P.columns if c not in lead]]
    P.to_csv(os.path.join(OUT, "e19_pairs.csv"), index=False)
    P["swap"] = "A->" + P.ref2.str.replace(r"\d+$", "", regex=True)
    num = [c for c in P.columns if c not in ["Y", "metric", "score", "pool", "ref2", "swap"]]
    g = P.groupby(["Y", "metric", "score", "pool", "swap"])
    S = g[num].mean()
    S["n_draws"] = g.size()
    S["cc_min"], S["cc_max"] = g.call_change.min(), g.call_change.max()
    S.reset_index().to_csv(os.path.join(OUT, "e19_pairs_summary.csv"), index=False)
    pd.DataFrame(bins).to_csv(os.path.join(OUT, "e19_margin_bins.csv"), index=False)
    pd.DataFrame(matched).to_csv(os.path.join(OUT, "e19_matched.csv"), index=False)
    pd.DataFrame(cons).to_csv(os.path.join(OUT, "e19_consensus.csv"), index=False)
    print(pd.DataFrame(checks).to_string(), flush=True)


# Figure ################################################################################


def stage_fig():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    import make_report_figures as mrf

    M = pd.read_csv(os.path.join(OUT, "e19_matched.csv"))
    M = M[M.Y == 2000]
    labels = {"MINov (exact)": "MINov", "1-NN distance": "1-NN distance",
              "KDE (median bw)": "Gaussian KDE, median bandwidth"}
    groups = [("B", "+ MP-2025\nadditions"), ("C", "+ other GNoME\npredictions"),
              ("D", "+ MP-2023\ntheoretical"), ("growth (E7)", "growth\n2000 -> full")]
    pools = [("later", "later discoveries", 1.0, ""), ("G", "GNoME", 0.35, ""),
             ("G_margin_matched", "GNoME, reweighted to the later pool's margins", 1.0, "///")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), sharey="row")
    w = 0.27
    for i, metric in enumerate(["elmd", "lostop"]):
        for j, (k, lab) in enumerate(labels.items()):
            ax = axes[i, j]
            gs = [g for g in groups if not (metric == "lostop" and g[0] == "D")]
            x = np.arange(len(gs))
            for r, (col, _, al, ht) in enumerate(pools):
                v = [M[(M.metric == metric) & (M.score == k) & (M.ref2 == g)][col].iloc[0] * 100 for g, _ in gs]
                bars = ax.bar(x + (r - 1) * w, v, w, color=mrf.SERIES[lab], alpha=al, hatch=ht,
                              edgecolor=mrf.SURFACE, linewidth=1.5)
                for b, val in zip(bars, v):
                    ax.text(b.get_x() + b.get_width() / 2, val + 0.6, f"{val:.1f}", ha="center", va="bottom",
                            fontsize=6.5, color=mrf.INK2)
            ax.axvline(len(gs) - 1.5, color=mrf.CONTEXT, linewidth=1, linestyle="--")
            ax.set_xticks(x, [g[1] for g in gs], fontsize=7.5)
            short = lab.replace("Gaussian KDE, median bandwidth", "KDE (median bw)")
            ax.set_title(f"{short}, {'ElMD' if metric == 'elmd' else 'LoStOP'}", fontsize=10)
            if j == 0:
                ax.set_ylabel("% of candidates whose call changed")
            ax.grid(axis="x", visible=False)
            ax.set_axisbelow(True)
    hs = [Patch(facecolor=mrf.INK2, label=pools[0][1]), Patch(facecolor=mrf.INK2, alpha=0.35, label=pools[1][1]),
          Patch(facecolor=mrf.INK2, hatch="///", edgecolor=mrf.SURFACE, label=pools[2][1])]
    fig.legend(handles=hs, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.95), ncol=3, fontsize=8.5)
    fig.suptitle("Reference swap from experimental Li compounds known by 2000 (A_2000) to other databases: "
                 "call changes for real later discoveries vs GNoME candidates", x=0.01, ha="left", fontsize=10,
                 color=mrf.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(os.path.join(OUT, "e19_call_changes.png"), dpi=200)
    e15.log("figure")


def main():
    if sys.argv[1:] != ["fig"]:
        stage_run()
    stage_fig()


if __name__ == "__main__":
    main()
