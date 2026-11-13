"""E9: Does the stability-sensitivity frontier (E1b/E5) and the two-scale MINov transfer to the
larger MP-2023 reference, and which kernel width is informative about later discoveries?

Data: data/raw/processed U_{lostop,elmd}.npy restricted to R23 (2095 experimental MP-2023 Li
entries, polymorphs included) and G (3000 GNoME Li); r23_first_year.csv (earliest literature year).
Scores (HIGHER = DENSER = LESS NOVEL, every parameter refit on the reference in use):
  MI_q   w = 1 - H(min(F/(2q), 0.5)) for q in e5.Q_FINE (q = 0.5 is MINov)
  KDE    Gaussian, h = q-quantile of reference pairs (E1b sweep)
  kNN    -mean distance to the k = max(1, round(p n)) nearest (p = 0 is 1-NN)
  RA     rank averages over the query set (e5.COMBOS), incl. the two-scale
         RA MINov & MI q=0.01 w=0.75

Parts
  1. Transfer (E1b/E5 protocol on R23): 150 seeds + 200 controls held out, base = rest.
     Sensitivity: paired rise of the seed's novelty percentile among controls after removing a
     5/20/100-NN hole (same metric) or a 20/100-NN hole in the other metric (and a random-200
     control). Stability: Spearman vs base-reference scores at 10% / 30% subsamples (20 draws),
     queries = seeds + controls + 300 random G. Gap above the KDE+kNN Pareto frontier with a joint
     bootstrap (seeds x draws), frontier re-estimated per replicate (as e5.bootstrap_gaps).
     --variant formula keeps one R23 entry per formula (lowest row) to mimic the original 1834 set.
  2. Discovery detection (year split Y in 1990, 2000, 2005): Reference_Y = R23 first reported
     <= Y. Post-Y compounds are scored against Reference_Y; pre-Y compounds leave-one-out (LOO)
     and, as a check, leave-out of every entry sharing the formula and first year ("group").
     AUC = P(post-Y scored more novel than pre-Y); also G vs pre-Y. Stratified bootstrap CIs
     and paired differences vs MINov.
  3. Stability on the small published sets (perovskite n=85, diverse n=60): every item a query,
     LOO against the full set vs against 30% / 50% subsamples (self masked), 50 draws.

Usage: uv run python experiments/e9_two_scale_transfer.py [--parts 123] [--variant all|formula]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
from e1b_stability_sensitivity import (  # noqa: E402
    KDE_Q, KNN_P, hole_for_seed, kde_name, knn_name, novelty_percentile)
from e5_kernel_shape import (  # noqa: E402
    COMBOS, MINOV, Q_FINE, combo_name, mi_name, mi_weight, pareto_gap, score_meta, sorted_quantile)

OUT = os.path.join(REPO, "experiments", "results")
PROC = os.path.join(REPO, "data", "raw", "processed")
N_SEEDS, N_CONTROLS, N_G_QUERIES = 150, 200, 300
FRACTIONS = [0.1, 0.3]
HOLES = [("none", 0), ("knn", 5), ("knn", 20), ("knn", 100), ("cross", 20), ("cross", 100), ("random", 200)]
GAP_HOLES = ["knn_5", "knn_20", "knn_100", "cross_20", "cross_100"]
TWO_SCALE = combo_name(MINOV, mi_name(0.01), 0.75)
YEARS = [1990, 2000, 2005]


# Scores ###############################################################################


def base_scores(ref_dm, query_dm):
    """MI_q, KDE and kNN sweeps. Entries of query_dm set to inf are masked (weight 0)."""
    n = len(ref_dm)
    sp = np.sort(ref_dm[np.triu_indices(n, k=1)])
    F = np.searchsorted(sp, query_dm, side="right") / len(sp)
    out = {mi_name(q): mi_weight(F, q).sum(axis=1) for q in Q_FINE}
    for q, h in zip(KDE_Q, sorted_quantile(sp, KDE_Q)):
        out[kde_name(q)] = np.exp(-0.5 * (query_dm / max(h, 1e-12)) ** 2).sum(axis=1)
    srt = np.sort(query_dm, axis=1)
    for p in KNN_P:
        k = max(1, int(round(p * n)))
        out[knn_name(p)] = -srt[:, :k].mean(axis=1)
    return out


def add_combos(s):
    s = dict(s)
    n = len(next(iter(s.values())))
    for a, b, alpha in COMBOS:
        s[combo_name(a, b, alpha)] = (alpha * rankdata(s[a]) + (1 - alpha) * rankdata(s[b])) / n
    return s


def all_scores(ref_dm, query_dm):
    return add_combos(base_scores(ref_dm, query_dm))


# Data #################################################################################


def load(variant):
    u = pd.read_csv(os.path.join(PROC, "U_index.csv"))
    fy = pd.read_csv(os.path.join(PROC, "r23_first_year.csv"))
    r = u[u.set == "R23"].merge(fy[["row", "first_year"]], on="row").sort_values("row")
    if variant == "formula":
        r = r.drop_duplicates("fkey", keep="first")
    g = u[u.set == "G"].sort_values("row")
    rows = np.concatenate([r.row.values, g.row.values])
    dms = {}
    for m in ["lostop", "elmd"]:
        big = np.load(os.path.join(PROC, f"U_{m}.npy"), mmap_mode="r")
        dms[m] = np.asarray(big[np.ix_(rows, rows)], dtype=np.float64)
    meta = pd.concat([r.assign(kind="R23"), g.assign(kind="G", first_year=np.nan)]).reset_index(drop=True)
    return meta, dms, len(r)


# Part 1: transfer #####################################################################


def run_seed(seed, metric, dm, dm_other, base, controls):
    rng = np.random.default_rng(seed)
    queries = np.concatenate([[seed], controls])
    recs = []
    for kind, param in HOLES:
        hole = hole_for_seed(seed, kind, param, dm, dm_other, base, None, rng)
        ref = np.setdiff1d(base, hole)
        for name, s in all_scores(dm[np.ix_(ref, ref)], dm[np.ix_(queries, ref)]).items():
            recs.append(dict(metric=metric, hole=kind, param=param, seed=seed, score=name,
                             percentile=novelty_percentile(s[0], s[1:])))
    return recs


def stability(dm, pool, queries, draws, rng, metric):
    full = all_scores(dm[np.ix_(pool, pool)], dm[np.ix_(queries, pool)])
    recs = []
    for f in FRACTIONS:
        for d in range(draws):
            ref = pool[rng.random(len(pool)) < f]
            s = all_scores(dm[np.ix_(ref, ref)], dm[np.ix_(queries, ref)])
            recs += [dict(metric=metric, fraction=f, draw=d, score=k, spearman=spearmanr(s[k], full[k])[0])
                     for k in full]
    return recs


def bootstrap_gaps(holes, stab, meta, n_boot=1000, seed=0):
    """Joint bootstrap (hole seeds x subsample draws) of the stability gap above the KDE + kNN
    Pareto frontier, frontier re-estimated per replicate (same as e5.bootstrap_gaps)."""
    rng = np.random.default_rng(seed)
    keep = [mi_name(q) for q in Q_FINE] + [combo_name(*c) for c in COMBOS]
    rows = []
    for metric in ["lostop", "elmd"]:
        st = stab[(stab.metric == metric) & (stab.fraction == 0.1)].pivot(index="score", columns="draw",
                                                                         values="spearman")
        for col in GAP_HOLES:
            dp = holes[(holes.metric == metric) & (holes.col == col)].pivot(index="score", columns="seed",
                                                                           values="dp")
            scores = st.index.intersection(dp.index)
            S, P = st.loc[scores].values, dp.loc[scores].values
            fam = meta.set_index("score").loc[scores, "family"].values
            reps = []
            for _ in range(n_boot):
                di = rng.integers(0, S.shape[1], S.shape[1])
                si = rng.integers(0, P.shape[1], P.shape[1])
                df = pd.DataFrame({"score": scores, "family": fam, "x": S[:, di].mean(1), "y": P[:, si].mean(1)})
                reps.append(pareto_gap(df, "x", "y").values)
            reps = np.array(reps)
            for i, sc in enumerate(scores):
                if sc not in keep:
                    continue
                g = reps[:, i][np.isfinite(reps[:, i])]
                ok = len(g) > 0
                rows.append(dict(metric=metric, hole=col, score=sc,
                                 gap_mean=np.mean(g) if ok else np.nan,
                                 gap_lo=np.quantile(g, 0.025) if ok else np.nan,
                                 gap_hi=np.quantile(g, 0.975) if ok else np.nan,
                                 p_gap_pos=np.mean(g > 0) if ok else np.nan, frac_in_range=len(g) / n_boot))
    return pd.DataFrame(rows)


def part1(meta, dms, n_r, args, tag):
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_r)
    seeds, controls = perm[:N_SEEDS], perm[N_SEEDS:N_SEEDS + N_CONTROLS]
    base = np.sort(perm[N_SEEDS + N_CONTROLS:])
    gq = n_r + rng.choice(len(meta) - n_r, N_G_QUERIES, replace=False)
    queries = np.concatenate([seeds, controls, gq])
    recs = []
    for metric, other in [("lostop", "elmd"), ("elmd", "lostop")]:
        t = time.perf_counter()
        dm, dmo = dms[metric][:n_r, :n_r], dms[other][:n_r, :n_r]
        out = Parallel(n_jobs=args.jobs)(delayed(run_seed)(s, metric, dm, dmo, base, controls) for s in seeds)
        recs += [r for rs in out for r in rs]
        print(f"[holes {tag}] {metric}: {time.perf_counter() - t:.0f}s", flush=True)
    holes = pd.DataFrame(recs)
    p0 = holes[holes.hole == "none"].set_index(["metric", "seed", "score"]).percentile
    holes = holes.join(p0.rename("p0"), on=["metric", "seed", "score"])
    holes["dp"] = holes.percentile - holes.p0
    holes.to_csv(os.path.join(OUT, f"e9_holes{tag}.csv.gz"), index=False, float_format="%.4g")

    srec = []
    for metric in ["lostop", "elmd"]:
        t = time.perf_counter()
        srec += stability(dms[metric], base, queries, args.draws, np.random.default_rng(1), metric)
        print(f"[stability {tag}] {metric}: {time.perf_counter() - t:.0f}s", flush=True)
    stab = pd.DataFrame(srec)
    stab.to_csv(os.path.join(OUT, f"e9_stability{tag}.csv"), index=False)
    return summarize1(holes, stab, tag)


def summarize1(holes, stab, tag):
    holes = holes[holes.hole != "none"].assign(col=lambda d: d.hole + "_" + d.param.map("{:g}".format))
    dl = holes.pivot_table(index=["metric", "score"], columns="col", values="dp", aggfunc="mean").add_prefix("delta_")
    st = stab.groupby(["metric", "score", "fraction"]).spearman.mean().unstack("fraction")
    st.columns = [f"stab_{c:g}" for c in st.columns]
    fr = st.join(dl).reset_index()
    meta = score_meta(fr.score.unique())
    fr = fr.merge(meta, on="score")
    for metric in fr.metric.unique():
        m = fr.metric == metric
        for col in GAP_HOLES:
            fr.loc[m, "gap_" + col] = pareto_gap(fr[m], "stab_0.1", "delta_" + col)
    # stab95 rule: narrowest MI_q with 10% stability >= 0.95
    for metric in fr.metric.unique():
        mi = fr[(fr.metric == metric) & (fr.family == "MI_q")].sort_values("width")
        ok = mi[mi["stab_0.1"] >= 0.95]
        fr.loc[fr.metric == metric, "stab95_q"] = ok.width.iloc[0] if len(ok) else np.nan
    fr.to_csv(os.path.join(OUT, f"e9_frontier{tag}.csv"), index=False)
    boot = bootstrap_gaps(holes, stab, meta)
    boot.to_csv(os.path.join(OUT, f"e9_gap_bootstrap{tag}.csv"), index=False)
    return fr, boot


# Part 2: discovery detection ##########################################################


def auc(neg_score, pos_score):
    """P(pos more novel, i.e. LOWER density score, than neg); ties count half."""
    r = rankdata(-np.concatenate([neg_score, pos_score]))
    n0, n1 = len(neg_score), len(pos_score)
    return (r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def auc_matrix(S, n0, idx0, idx1):
    """AUC per row of S (scores x items, first n0 items negatives) for resampled indices."""
    sub = np.concatenate([S[:, idx0], S[:, n0 + idx1]], axis=1)
    r = rankdata(-sub, axis=1)
    n1 = len(idx1)
    return (r[:, len(idx0):].sum(1) - n1 * (n1 + 1) / 2) / (len(idx0) * n1)


def score_split(dm, ref, pre, others, group_mask=None):
    """Scores of `pre` (LOO within ref; ref == pre) and of each array in `others` vs ref.
    group_mask (len(pre) x len(ref) bool) masks extra reference entries for pre queries."""
    ref_dm = dm[np.ix_(ref, ref)]
    q = dm[np.ix_(pre, ref)].copy()
    q[np.arange(len(pre)), np.arange(len(pre))] = np.inf  # pre == ref (same order)
    if group_mask is not None:
        q[group_mask] = np.inf
    s_pre = base_scores(ref_dm, q)
    s_oth = [base_scores(ref_dm, dm[np.ix_(o, ref)]) for o in others]
    return s_pre, s_oth


def part2(meta, dms, n_r, n_boot=500):
    rng = np.random.default_rng(2)
    fy = meta.first_year.values
    is_r = np.arange(len(meta)) < n_r
    g_idx = np.flatnonzero(~is_r)
    rows, boots = [], []
    for Y in YEARS:
        pre = np.flatnonzero(is_r & (fy <= Y))
        post = np.flatnonzero(is_r & (fy > Y))
        pre_keys = set(meta.fkey.values[pre])
        post_newf = np.array([k not in pre_keys for k in meta.fkey.values[post]])
        # group mask: same formula and same first year as the pre query (co-reported polymorphs)
        fk, yr = meta.fkey.values[pre], fy[pre]
        gmask = (fk[:, None] == fk[None, :]) & (yr[:, None] == yr[None, :])
        for metric, dm in dms.items():
            t = time.perf_counter()
            for loo, gm in [("loo", None), ("group", gmask)]:
                s_pre, (s_post, s_g) = score_split(dm, pre, pre, [post, g_idx], gm)
                comps = [("post", s_post, np.ones(len(post), bool)), ("post_newformula", s_post, post_newf),
                         ("G", s_g, np.ones(len(g_idx), bool))]
                for comp, s_pos, msk in comps:
                    if loo == "group" and comp == "G":
                        continue
                    # rank averages over the union of the compared query sets
                    joint = add_combos({k: np.concatenate([s_pre[k], s_pos[k][msk]]) for k in s_pre})
                    n0, n1 = len(pre), int(msk.sum())
                    names = list(joint)
                    S = np.array([joint[k] for k in names])
                    for k in ["nelements", "nsites"]:  # trivial baselines (negated: more = "more novel")
                        names.append(k)
                        v = meta[k].values.astype(float)
                        S = np.vstack([S, -np.concatenate([v[pre], v[(post if comp != "G" else g_idx)][msk]])])
                    a = auc_matrix(S, n0, np.arange(n0), np.arange(n1))
                    B = np.array([auc_matrix(S, n0, rng.integers(0, n0, n0), rng.integers(0, n1, n1))
                                  for _ in range(n_boot)])
                    i0 = names.index(MINOV)
                    i2 = names.index(TWO_SCALE)
                    for i, nm in enumerate(names):
                        d = B[:, i] - B[:, i0]
                        d2 = B[:, i] - B[:, i2]
                        rows.append(dict(Y=Y, metric=metric, ref=loo, comparison=comp, n_neg=n0, n_pos=n1,
                                         score=nm, auc=a[i], auc_lo=np.quantile(B[:, i], 0.025),
                                         auc_hi=np.quantile(B[:, i], 0.975), d_minov=a[i] - a[i0],
                                         d_minov_lo=np.quantile(d, 0.025), d_minov_hi=np.quantile(d, 0.975),
                                         d_two_scale=a[i] - a[i2], d_two_scale_lo=np.quantile(d2, 0.025),
                                         d_two_scale_hi=np.quantile(d2, 0.975)))
            print(f"[year {Y}] {metric}: n_pre={len(pre)} n_post={len(post)} (new formula {post_newf.sum()}) "
                  f"{time.perf_counter() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e9_discovery_auc.csv"), index=False)
    return df


# Part 3: small sets ###################################################################


def part3(draws=50):
    rng = np.random.default_rng(3)
    recs = []
    for ds in ["perovskite_dataset", "diverse_dataset"]:
        for metric in ["lostop", "elmd"]:
            dm = np.load(os.path.join(REPO, "precomputed", f"{ds}_{metric}_dm.npy")).astype(float)
            n = len(dm)
            q = dm.copy()
            np.fill_diagonal(q, np.inf)
            full = all_scores(dm, q)
            for f in [0.3, 0.5]:
                for d in range(draws):
                    ref = np.flatnonzero(rng.random(n) < f)
                    qq = dm[:, ref].copy()
                    qq[ref, np.arange(len(ref))] = np.inf
                    s = all_scores(dm[np.ix_(ref, ref)], qq)
                    recs += [dict(dataset=ds, metric=metric, n=n, fraction=f, draw=d, score=k,
                                  spearman=spearmanr(s[k], full[k])[0]) for k in full]
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(OUT, "e9_smallsets_stability.csv"), index=False)
    return df


# Plot #################################################################################


def plot(fr, disc, path):
    import matplotlib.pyplot as plt

    fams = [("KDE (h=q-quantile)", "o-", "C0"), ("kNN sweep", "s-", "C1"), ("MI_q", "*-", "k")]
    comps = [("post", "post-2000 discoveries vs pre-2000 (LOO)"), ("G", "GNoME vs pre-2000 (LOO)")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for i, metric in enumerate(["lostop", "elmd"]):
        f = fr[fr.metric == metric].set_index("score")
        for j, (comp, title) in enumerate(comps):
            ax = axes[i, j]
            d = disc[(disc.Y == 2000) & (disc.metric == metric) & (disc.ref == "loo") & (disc.comparison == comp)]
            d = d.set_index("score").join(f[["stab_0.1", "family", "width"]], how="inner")
            for fam, style, c in fams:
                m = d[d.family == fam].sort_values("width")
                ax.errorbar(m["stab_0.1"], m.auc, yerr=[m.auc - m.auc_lo, m.auc_hi - m.auc], fmt=style, color=c,
                            label=fam, ms=6, lw=1, capsize=2, elinewidth=0.7)
                for s, r in m.iterrows():
                    ax.annotate(f"{r.width:g}", (r["stab_0.1"], r.auc), fontsize=6.5, xytext=(3, 3),
                                textcoords="offset points", color=c)
            ra = d[d.family == "rank average"]
            ax.plot(ra["stab_0.1"], ra.auc, "P", color="C5", ms=7, alpha=0.6, label="rank averages")
            ts = d.loc[[TWO_SCALE]]
            ax.plot(ts["stab_0.1"], ts.auc, "P", color="C3", mec="k", ms=12, label="two-scale (MINov & MI 0.01, w=.75)")
            mn = d.loc[[MINOV]]
            ax.plot(mn["stab_0.1"], mn.auc, "*", color="gold", mec="k", ms=18, label="MINov (q=0.5)")
            ax.axhline(0.5, color="grey", lw=0.6)
            ax.set_xlim(max(0.3, d["stab_0.1"].min() - 0.03), 1.005)
            ax.set_xlabel("stability: Spearman vs full reference at 10% subsample (R23)")
            ax.set_ylabel("discovery-detection AUC (95% CI)")
            ax.set_title(f"{metric}: {title}", fontsize=10)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("E9: stability vs real-data informativeness (labels: KDE q / kNN p / MI_q q)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


# Main #################################################################################


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="123")
    ap.add_argument("--variant", default="all", choices=["all", "formula"])
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = "" if args.variant == "all" else "_formula"
    meta, dms, n_r = load(args.variant)
    print(f"[load] variant={args.variant} n_R23={n_r} n_G={len(meta) - n_r}", flush=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.precision", 3)

    if "1" in args.parts:
        fr, boot = part1(meta, dms, n_r, args, tag)
        cols = ["stab_0.1", "stab_0.3"] + ["delta_" + c for c in GAP_HOLES] + ["gap_knn_20", "gap_cross_100"]
        print(fr.set_index(["metric", "family", "score"])[cols].sort_index())
        print(boot[boot.score.isin([TWO_SCALE, MINOV, mi_name(0.15)])])
    if "2" in args.parts and args.variant == "all":
        disc = part2(meta, dms, n_r)
        print(disc[(disc.Y == 2000) & (disc.ref == "loo")].pivot_table(
            index="score", columns=["metric", "comparison"], values="auc"))
    if "3" in args.parts and args.variant == "all":
        sm = part3()
        print(sm.groupby(["dataset", "metric", "fraction", "score"]).spearman.mean().unstack(["dataset", "metric",
                                                                                              "fraction"]))
    fp = os.path.join(OUT, "e9_frontier.csv")
    dp = os.path.join(OUT, "e9_discovery_auc.csv")
    if args.variant == "all" and os.path.exists(fp) and os.path.exists(dp):
        plot(pd.read_csv(fp), pd.read_csv(dp), os.path.join(OUT, "e9_stability_vs_discovery.png"))


if __name__ == "__main__":
    main()
