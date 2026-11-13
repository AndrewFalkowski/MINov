"""E11: Abramson square-root adaptive MINov. Does local bandwidth adaptation with no new free
parameter move MINov off the stable/global end of the stability-sensitivity trade-off?

Adaptive MINov (AMINov). On a reference R with sorted pair distances and empirical CDF F:
  pilot   p_j = sum_{k != j} w(d_jk) + 1   (MINov leave-one-out density, +1 = own weight w(0) ~ 1)
  lambda  lambda_j = (p_j / g)^(-alpha),   g = geometric mean of p_j   (alpha = 1/2: Abramson)
  score   s(x) = sum_j w(d(x, j) / lambda_j),  w = MINov weight 1 - H(min(F, 1/2)) (same F, cutoff)
Each reference's kernel widens where the reference is sparse and narrows where it is dense. Pilot
and lambdas are refit on every reference (subset). The control is the same rule for a Gaussian
KDE with median bandwidth: pilot = its own LOO kernel sum + 1, h_j = h0 * lambda_j (unnormalised
kernel sums, as kde_median in novelty_scores.py). alpha = 0.5 is the headline; 0.25 and 0.75 are
sensitivity checks only. HIGHER = DENSER = LESS NOVEL throughout.

Parts (protocols reused unchanged, so E9 / E1b / E7 numbers can be joined per seed / draw):
  1. R23 stability + holes (E9 part 1: same seeds, controls, holes, subsample draws), gap above
     the E9 KDE + kNN Pareto frontier with the joint (seeds x draws) bootstrap.
  2. Synthetic Gaussian mixtures (E1b / E5 generator, d = 2, 5, 10, 3 reps): accuracy = Spearman
     with the true log density; stability at 10% / 30% subsamples.
  3. Year split Y = 2000 (E9 part 2, LOO for pre-Y): AUC post-Y vs pre-Y and G vs pre-Y.
  4. Forward call flips (E7 task 2, Y = 2000): pools G and post-Y R23, 5% LOO threshold,
     Reference_Y vs full R23 (pool b masks same-formula rows first reported after Y).

Usage: uv run python experiments/e11_abramson.py [--parts 12345] [--jobs 4] [--draws 20]
Part 5 = summary table + figure from the CSVs.
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
from e1b_stability_sensitivity import (  # noqa: E402
    KDE_Q, hole_for_seed, kde_name, log_density, make_mixture, novelty_percentile, sample_mixture)
from e5_kernel_shape import mi_weight, pareto_gap, sorted_quantile  # noqa: E402
import e9_two_scale_transfer as e9  # noqa: E402

OUT = os.path.join(REPO, "experiments", "results")
ALPHAS = [0.5, 0.25, 0.75]
FRACTIONS = [0.1, 0.3]
Y = 2000
MINOV, MI15, KDEM, NN1 = "MINov", "MI q=0.15", "KDE median", "1-NN"
SYN_Q = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]  # extra MI_q widths (synthetic)
E9_EQUIV = {MINOV: "MI q=0.5", MI15: "MI q=0.15", KDEM: "KDE q=0.5", NN1: "kNN p=0"}


def amin(a):
    return f"AMINov a={a:g}"


def akde(a):
    return f"AKDE a={a:g}"


HEAD = [MINOV, amin(0.5), KDEM, akde(0.5), MI15, NN1]
MAIN = HEAD + [amin(0.25), amin(0.75), akde(0.25), akde(0.75)]


# Scores ###############################################################################


def lambdas(pilot, alpha):
    lp = np.log(pilot)
    return np.exp(-alpha * (lp - lp.mean()))


def scores(ref_dm, query_dm, sweep=False, return_lambda=False):
    """MINov, MI q=0.15, median KDE, their Abramson versions and 1-NN (inf entries are masked)."""
    n = len(ref_dm)
    iu = np.triu_indices(n, k=1)
    upper = ref_dm[iu]  # reference matrices are symmetric, so pairs suffice for the pilots
    order = np.argsort(upper, kind="stable")
    sp = upper[order]
    m = len(sp)
    Fq = np.searchsorted(sp, query_dm, side="right") / m
    out = {MINOV: mi_weight(Fq, 0.5).sum(1), MI15: mi_weight(Fq, 0.15).sum(1)}
    # pilots on the reference itself (leave-one-out, + own weight 1)
    Fu = np.empty(m)
    Fu[order] = np.searchsorted(sp, sp, side="right") / m
    Wu = mi_weight(Fu, 0.5)
    p_mi = np.bincount(iu[0], Wu, n) + np.bincount(iu[1], Wu, n) + 1.0
    h0 = max(sorted_quantile(sp, 0.5), 1e-12)
    Ku = np.exp(-0.5 * (upper / h0) ** 2)
    p_kde = np.bincount(iu[0], Ku, n) + np.bincount(iu[1], Ku, n) + 1.0
    del Fu, Wu, Ku, upper, order
    out[KDEM] = np.exp(-0.5 * (query_dm / h0) ** 2).sum(1)
    lam = {}
    for a in ALPHAS:
        lm = lambdas(p_mi, a)
        lam[a] = lm
        out[amin(a)] = mi_weight(np.searchsorted(sp, query_dm / lm, side="right") / m, 0.5).sum(1)
        hk = h0 * lambdas(p_kde, a)
        out[akde(a)] = np.exp(-0.5 * (query_dm / hk) ** 2).sum(1)
    out[NN1] = -query_dm.min(1)
    if sweep:
        for q in SYN_Q:
            out[f"MI q={q:g}"] = mi_weight(Fq, q).sum(1)
        for q, h in zip(KDE_Q, sorted_quantile(sp, KDE_Q)):
            out[kde_name(q)] = np.exp(-0.5 * (query_dm / max(h, 1e-12)) ** 2).sum(1)
    return (out, lam) if return_lambda else out


# Part 1: R23 holes + stability (E9 protocol) ##########################################


WCTX = {}


def init_worker(r23_rows):
    """Each worker reads its own R23 x R23 blocks from the memory-mapped U matrices."""
    for m in ["lostop", "elmd"]:
        big = np.load(os.path.join(e9.PROC, f"U_{m}.npy"), mmap_mode="r")
        WCTX[m] = np.asarray(big[np.ix_(r23_rows, r23_rows)], dtype=np.float64)


def run_seed(args):
    seed, metric, other, base, controls = args
    dm, dm_other = WCTX[metric], WCTX[other]
    rng = np.random.default_rng(seed)  # as e9.run_seed (random-hole stream)
    queries = np.concatenate([[seed], controls])
    recs = []
    for kind, param in e9.HOLES:
        hole = hole_for_seed(seed, kind, param, dm, dm_other, base, None, rng)
        ref = np.setdiff1d(base, hole)
        for name, s in scores(dm[np.ix_(ref, ref)], dm[np.ix_(queries, ref)]).items():
            recs.append(dict(metric=metric, hole=kind, param=param, seed=seed, score=name,
                             percentile=novelty_percentile(s[0], s[1:])))
    return recs


def stab_draw(dm, ref, queries, full, metric, f, d):
    s = scores(dm[np.ix_(ref, ref)], dm[np.ix_(queries, ref)])
    return [dict(metric=metric, fraction=f, draw=d, score=k, spearman=spearmanr(s[k], full[k])[0]) for k in full]


def part1(meta, dms, n_r, jobs, draws):
    rng = np.random.default_rng(0)  # identical to e9.part1
    perm = rng.permutation(n_r)
    seeds, controls = perm[:e9.N_SEEDS], perm[e9.N_SEEDS:e9.N_SEEDS + e9.N_CONTROLS]
    base = np.sort(perm[e9.N_SEEDS + e9.N_CONTROLS:])
    gq = n_r + rng.choice(len(meta) - n_r, e9.N_G_QUERIES, replace=False)
    queries = np.concatenate([seeds, controls, gq])
    recs = []
    with Pool(jobs, initializer=init_worker, initargs=(meta.row.values[:n_r],)) as pool:
        for metric, other in [("lostop", "elmd"), ("elmd", "lostop")]:
            t = time.perf_counter()
            out = pool.map(run_seed, [(s, metric, other, base, controls) for s in seeds], chunksize=2)
            recs += [r for rs in out for r in rs]
            print(f"[holes] {metric}: {time.perf_counter() - t:.0f}s", flush=True)
    holes = pd.DataFrame(recs)
    p0 = holes[holes.hole == "none"].set_index(["metric", "seed", "score"]).percentile
    holes = holes.join(p0.rename("p0"), on=["metric", "seed", "score"])
    holes["dp"] = holes.percentile - holes.p0
    holes.to_csv(os.path.join(OUT, "e11_holes.csv.gz"), index=False, float_format="%.4g")

    srec, lam_rows = [], []
    for metric in ["lostop", "elmd"]:
        t = time.perf_counter()
        dm = dms[metric]
        full, lam = scores(dm[np.ix_(base, base)], dm[np.ix_(queries, base)], return_lambda=True)
        for a, lm in lam.items():
            lam_rows.append(dict(metric=metric, alpha=a, **{f"lambda_p{p}": np.percentile(lm, p)
                                                           for p in [1, 5, 25, 50, 75, 95, 99]}))
        r = np.random.default_rng(1)  # identical draw stream to e9.stability
        refs = [(f, d, base[r.random(len(base)) < f]) for f in FRACTIONS for d in range(draws)]
        for f, d, ref in refs:
            srec += stab_draw(dm, ref, queries, full, metric, f, d)
        print(f"[stability] {metric}: {time.perf_counter() - t:.0f}s", flush=True)
    stab = pd.DataFrame(srec)
    stab.to_csv(os.path.join(OUT, "e11_stability.csv"), index=False)
    pd.DataFrame(lam_rows).to_csv(os.path.join(OUT, "e11_lambda.csv"), index=False)


def summarize1(n_boot=1000):
    holes = pd.read_csv(os.path.join(OUT, "e11_holes.csv.gz"))
    stab = pd.read_csv(os.path.join(OUT, "e11_stability.csv"))
    holes = holes[holes.hole != "none"].assign(col=lambda d: d.hole + "_" + d.param.map("{:g}".format))
    # check: our MINov / MI 0.15 / KDE median / 1-NN reproduce E9 per seed and per draw
    h9 = pd.read_csv(os.path.join(OUT, "e9_holes.csv.gz"))
    h9 = h9[h9.hole != "none"].assign(col=lambda d: d.hole + "_" + d.param.map("{:g}".format))
    s9 = pd.read_csv(os.path.join(OUT, "e9_stability.csv"))
    for ours, theirs in E9_EQUIV.items():
        a = holes[holes.score == ours].set_index(["metric", "col", "seed"]).dp
        b = h9[h9.score == theirs].set_index(["metric", "col", "seed"]).dp
        c = stab[stab.score == ours].set_index(["metric", "fraction", "draw"]).spearman
        e = s9[s9.score == theirs].set_index(["metric", "fraction", "draw"]).spearman
        print(f"[check vs E9] {ours:10s} max|d dp|={np.nanmax(np.abs(a - b.reindex(a.index))):.2g} "
              f"max|d stab|={np.nanmax(np.abs(c - e.reindex(c.index))):.2g}", flush=True)

    fr9 = pd.read_csv(os.path.join(OUT, "e9_frontier.csv"))
    dl = holes.pivot_table(index=["metric", "score"], columns="col", values="dp", aggfunc="mean").add_prefix("delta_")
    sem = holes.pivot_table(index=["metric", "score"], columns="col", values="dp", aggfunc="sem").add_prefix("sem_")
    st = stab.groupby(["metric", "score", "fraction"]).spearman.mean().unstack("fraction")
    st.columns = [f"stab_{c:g}" for c in st.columns]
    fr = st.join(dl).join(sem).reset_index().assign(family="E11", width=np.nan)
    ctx = fr9[fr9.family.isin(["KDE (h=q-quantile)", "kNN sweep"])]
    for metric in fr.metric.unique():
        both = pd.concat([ctx[ctx.metric == metric], fr[fr.metric == metric]], ignore_index=True)
        for col in e9.GAP_HOLES:
            g = pareto_gap(both, "stab_0.1", "delta_" + col).values[-(fr.metric == metric).sum():]
            fr.loc[fr.metric == metric, "gap_" + col] = g
    fr.to_csv(os.path.join(OUT, "e11_frontier.csv"), index=False)

    # joint bootstrap (seeds x draws) of the gap, frontier re-estimated per replicate (as e9)
    rng = np.random.default_rng(0)
    fam9 = ctx.drop_duplicates("score").set_index("score").family
    rows = []
    for metric in ["lostop", "elmd"]:
        S9 = s9[(s9.metric == metric) & (s9.fraction == 0.1) & s9.score.isin(fam9.index)]
        S11 = stab[(stab.metric == metric) & (stab.fraction == 0.1)]
        St = pd.concat([S9, S11]).pivot(index="score", columns="draw", values="spearman")
        for col in e9.GAP_HOLES:
            P9 = h9[(h9.metric == metric) & (h9.col == col) & h9.score.isin(fam9.index)]
            P11 = holes[(holes.metric == metric) & (holes.col == col)]
            Pt = pd.concat([P9, P11]).pivot(index="score", columns="seed", values="dp")
            names = list(St.index.intersection(Pt.index))
            S, P = St.loc[names].values, Pt.loc[names].values
            fam = np.array([fam9.get(s, "E11") for s in names])
            reps = []
            for _ in range(n_boot):
                di = rng.integers(0, S.shape[1], S.shape[1])
                si = rng.integers(0, P.shape[1], P.shape[1])
                df = pd.DataFrame({"score": names, "family": fam, "x": S[:, di].mean(1), "y": P[:, si].mean(1)})
                reps.append(pareto_gap(df, "x", "y").values)
            reps = np.array(reps)
            for i, sc in enumerate(names):
                if fam[i] != "E11":
                    continue
                g = reps[:, i][np.isfinite(reps[:, i])]
                ok = len(g) > 0
                rows.append(dict(metric=metric, hole=col, score=sc, gap_mean=np.mean(g) if ok else np.nan,
                                 gap_lo=np.quantile(g, 0.025) if ok else np.nan,
                                 gap_hi=np.quantile(g, 0.975) if ok else np.nan,
                                 p_gap_pos=np.mean(g > 0) if ok else np.nan, frac_in_range=len(g) / n_boot))
    boot = pd.DataFrame(rows)
    boot.to_csv(os.path.join(OUT, "e11_gap_bootstrap.csv"), index=False)
    return fr, boot


# Part 2: synthetic ####################################################################


def run_synthetic(dim, rep, draws, n_ref=1000, n_mix_query=400, n_unif_query=100):
    rng = np.random.default_rng(1000 * dim + rep)  # identical stream to E1b / E5
    mix = make_mixture(dim, rng)
    ref_x = sample_mixture(n_ref, mix, rng)
    lo, hi = ref_x.min(0), ref_x.max(0)
    q_x = np.vstack([sample_mixture(n_mix_query, mix, rng), rng.uniform(lo, hi, (n_unif_query, dim))])
    truth = log_density(q_x, mix)
    in_mix = np.arange(len(q_x)) < n_mix_query
    ref_dm, q_dm = cdist(ref_x, ref_x), cdist(q_x, ref_x)
    full = scores(ref_dm, q_dm, sweep=True)
    recs = [dict(dim=dim, rep=rep, fraction=1.0, draw=0, score=k, spearman=1.0, acc_all=spearmanr(s, truth)[0],
                 acc_mix=spearmanr(s[in_mix], truth[in_mix])[0]) for k, s in full.items()]
    for f in FRACTIONS:
        for d in range(draws):
            ref = np.flatnonzero(rng.random(n_ref) < f)
            s = scores(ref_dm[np.ix_(ref, ref)], q_dm[:, ref], sweep=True)
            recs += [dict(dim=dim, rep=rep, fraction=f, draw=d, score=k, spearman=spearmanr(s[k], full[k])[0],
                          acc_all=spearmanr(s[k], truth)[0], acc_mix=np.nan) for k in full]
    return recs


def part2(jobs, draws, reps=3):
    t = time.perf_counter()
    out = Parallel(n_jobs=jobs)(delayed(run_synthetic)(dim, rep, draws) for dim in [2, 5, 10] for rep in range(reps))
    syn = pd.DataFrame([r for rs in out for r in rs])
    syn.to_csv(os.path.join(OUT, "e11_synthetic.csv"), index=False)
    g = syn.groupby(["dim", "score", "fraction"])
    ss = pd.concat([g.spearman.mean().unstack("fraction").drop(columns=1.0).add_prefix("stab_"),
                    g.acc_all.mean().unstack("fraction").add_prefix("acc_")], axis=1)
    ss.columns = [c.replace("_1.0", "_full") for c in map(str, ss.columns)]
    ss = ss.reset_index()
    ss.to_csv(os.path.join(OUT, "e11_synthetic_summary.csv"), index=False)
    print(f"[synthetic] {time.perf_counter() - t:.0f}s", flush=True)


# Part 3: year split ###################################################################


def part3(meta, dms, n_r, n_boot=500):
    rng = np.random.default_rng(2)
    fy = meta.first_year.values
    is_r = np.arange(len(meta)) < n_r
    pre = np.flatnonzero(is_r & (fy <= Y))
    post = np.flatnonzero(is_r & (fy > Y))
    g_idx = np.flatnonzero(~is_r)
    rows = []
    for metric, dm in dms.items():
        ref_dm = dm[np.ix_(pre, pre)]
        q = ref_dm.copy()
        np.fill_diagonal(q, np.inf)  # leave-one-out for pre-Y compounds
        s_pre = scores(ref_dm, q)
        for comp, idx in [("post", post), ("G", g_idx)]:
            s_pos = scores(ref_dm, dm[np.ix_(idx, pre)])
            names = list(s_pre)
            S = np.array([np.concatenate([s_pre[k], s_pos[k]]) for k in names])
            n0, n1 = len(pre), len(idx)
            a = e9.auc_matrix(S, n0, np.arange(n0), np.arange(n1))
            B = np.array([e9.auc_matrix(S, n0, rng.integers(0, n0, n0), rng.integers(0, n1, n1))
                          for _ in range(n_boot)])
            i0 = names.index(MINOV)
            for i, nm in enumerate(names):
                d = B[:, i] - B[:, i0]
                rows.append(dict(Y=Y, metric=metric, comparison=comp, n_neg=n0, n_pos=n1, score=nm, auc=a[i],
                                 auc_lo=np.quantile(B[:, i], 0.025), auc_hi=np.quantile(B[:, i], 0.975),
                                 d_minov=a[i] - a[i0], d_minov_lo=np.quantile(d, 0.025),
                                 d_minov_hi=np.quantile(d, 0.975)))
        print(f"[year {Y}] {metric}: n_pre={len(pre)} n_post={len(post)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e11_year_auc.csv"), index=False)
    e9a = pd.read_csv(os.path.join(OUT, "e9_discovery_auc.csv"))
    e9a = e9a[(e9a.Y == Y) & (e9a.ref == "loo")]
    for ours, theirs in E9_EQUIV.items():
        for comp in ["post", "G"]:
            for metric in dms:
                a = df[(df.score == ours) & (df.comparison == comp) & (df.metric == metric)].auc.iloc[0]
                b = e9a[(e9a.score == theirs) & (e9a.comparison == comp) & (e9a.metric == metric)].auc.iloc[0]
                if abs(a - b) > 1e-9:
                    print(f"[check vs E9 AUC] {ours} {comp} {metric}: {a:.4f} vs {b:.4f}", flush=True)
    return df


# Part 4: forward call flips (E7 task 2) ###############################################


def loo_threshold(ref_dm, q=5):
    loo = ref_dm.copy()
    np.fill_diagonal(loo, np.inf)
    return {k: np.percentile(v, q) for k, v in scores(ref_dm, loo).items()}


def part4(meta, dms, n_r, n_boot=2000):
    rng = np.random.default_rng(4)
    fy = meta.first_year.values[:n_r]
    fk = meta.fkey.values
    r23 = np.arange(n_r)
    refY = r23[fy <= Y]
    poolb = r23[fy > Y]
    g = np.arange(n_r, len(meta))
    mask_b = (fk[poolb][:, None] == fk[r23][None, :]) & (fy[None, :] > Y)
    recs = []
    for metric, dm in dms.items():
        t = time.perf_counter()
        DY = dm[np.ix_(refY, refY)]
        DF = dm[np.ix_(r23, r23)]
        thrY, thrF = loo_threshold(DY), loo_threshold(DF)
        sY = {"a_G": scores(DY, dm[np.ix_(g, refY)]), "b_R23new": scores(DY, dm[np.ix_(poolb, refY)])}
        sF = {"a_G": scores(DF, dm[np.ix_(g, r23)]),
              "b_R23new": scores(DF, np.where(mask_b, np.inf, dm[np.ix_(poolb, r23)]))}
        for pool in sY:
            flips = {k: (sY[pool][k] < thrY[k]) != (sF[pool][k] < thrF[k]) for k in sY[pool]}
            n = len(flips[MINOV])
            for k, fl in flips.items():
                d = fl.astype(float) - flips[MINOV]
                bi = rng.integers(0, n, (n_boot, n))
                bd = d[bi].mean(1)
                recs.append(dict(Y=Y, metric=metric, pool=pool, score=k, n_pool=n,
                                 n_novelY5=int((sY[pool][k] < thrY[k]).sum()),
                                 n_novelF5=int((sF[pool][k] < thrF[k]).sum()), n_flip5=int(fl.sum()),
                                 flip_rate=fl.mean(), d_minov=d.mean(), d_minov_lo=np.quantile(bd, 0.025),
                                 d_minov_hi=np.quantile(bd, 0.975)))
        print(f"[flips] {metric}: {time.perf_counter() - t:.0f}s", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(OUT, "e11_call_flips.csv"), index=False)
    e7 = pd.read_csv(os.path.join(OUT, "e7_c1_tests.csv"))
    e7 = e7[(e7.Y == Y) & (e7.test == "summary")]
    for ours, theirs in [(MINOV, "MINov (exact)"), (NN1, "1-NN distance"), (KDEM, "KDE (median bw)")]:
        for _, r in e7[e7.score == theirs].iterrows():
            o = df[(df.score == ours) & (df.metric == r.metric) & (df.pool == r.pool)].iloc[0]
            print(f"[check vs E7] {ours:10s} {r.metric:6s} {r.pool:8s} flips {o.n_flip5} vs {r.n_flip5}", flush=True)
    return df


# Part 5: summary table + figure #######################################################


def summary():
    fr = pd.read_csv(os.path.join(OUT, "e11_frontier.csv"))
    syn = pd.read_csv(os.path.join(OUT, "e11_synthetic_summary.csv"))
    yr = pd.read_csv(os.path.join(OUT, "e11_year_auc.csv"))
    fl = pd.read_csv(os.path.join(OUT, "e11_call_flips.csv"))
    best_kde = {}
    for dim in syn.dim.unique():
        k = syn[(syn.dim == dim) & syn.score.str.startswith("KDE q=")]
        best_kde[dim] = k.loc[k.acc_full.idxmax()]
    rows = []
    for sc in MAIN + ["best fixed KDE (oracle)"]:
        r = dict(score=sc)
        for metric in ["lostop", "elmd"]:
            f = fr[(fr.metric == metric) & (fr.score == sc)]
            if len(f):
                f = f.iloc[0]
                r.update({f"{metric}_stab10": f["stab_0.1"], f"{metric}_stab30": f["stab_0.3"],
                          f"{metric}_d20": f["delta_knn_20"], f"{metric}_dx100": f["delta_cross_100"],
                          f"{metric}_gap20": f["gap_knn_20"]})
            y = yr[(yr.metric == metric) & (yr.score == sc) & (yr.comparison == "post")]
            if len(y):
                r[f"{metric}_aucY"] = y.auc.iloc[0]
            for pool in ["a_G", "b_R23new"]:
                x = fl[(fl.metric == metric) & (fl.score == sc) & (fl.pool == pool)]
                if len(x):
                    r[f"{metric}_flip_{pool}"] = 100 * x.flip_rate.iloc[0]
        for dim in sorted(syn.dim.unique()):
            if sc.startswith("best"):
                b = best_kde[dim]
                r[f"acc_d{dim}"], r[f"synstab10_d{dim}"] = b.acc_full, b["stab_0.1"]
                r[f"bestkde_d{dim}"] = b.score
            else:
                s = syn[(syn.dim == dim) & (syn.score == sc)].iloc[0]
                r[f"acc_d{dim}"], r[f"synstab10_d{dim}"] = s.acc_full, s["stab_0.1"]
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "e11_summary.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60, "display.precision", 3):
        print(tab.set_index("score").T)
    figure(fr)


def figure(fr):
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2, GRID, CONTEXT = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#a8a7a1"
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })
    BLUE, BLUE_D, ORANGE, AQUA, AQUA_D = "#2a78d6", "#184f95", "#eb6834", "#1baf7a", "#0e7a54"
    f9 = pd.read_csv(os.path.join(OUT, "e9_frontier.csv"))
    pts = [(MINOV, "MINov", BLUE, "o"), (amin(0.5), "adaptive MINov (Abramson, α = 1/2)", BLUE_D, "D"),
           (KDEM, "KDE, median bandwidth", AQUA, "o"), (akde(0.5), "adaptive median KDE (α = 1/2)", AQUA_D, "D")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, metric in zip(axes, ["lostop", "elmd"]):
        sm = f9[(f9.metric == metric) & (f9.family == "MI_q")].sort_values("width")
        ax.plot(sm["stab_0.1"], sm.delta_knn_20, color=CONTEXT, lw=2, zorder=1,
                label="MI kernel, width q swept (narrow left, MINov = q 0.5)")
        for _, r in sm[sm.width.isin([0.05, 0.1, 0.15, 0.2, 0.3])].iterrows():
            ax.scatter(r["stab_0.1"], r.delta_knn_20, s=14, color=CONTEXT, zorder=1)
            ax.annotate(f"q {r.width:g}", (r["stab_0.1"], r.delta_knn_20), textcoords="offset points",
                        xytext=(5, 2), fontsize=7.5, color=INK2)
        fm = fr[fr.metric == metric].set_index("score")
        off25 = (7, -12, "left") if metric == "lostop" else (-8, -4, "right")
        for a, dx, dy, ha in [(0.25, *off25), (0.75, -7, 6, "right")]:
            x, y = fm.loc[amin(a), "stab_0.1"], fm.loc[amin(a), "delta_knn_20"]
            ax.scatter(x, y, s=40, marker="D", facecolor="none", edgecolor=BLUE_D, linewidth=1.2, zorder=2)
            ax.annotate(f"α {a:g}", (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha,
                        fontsize=7.5, color=INK2)
        for (s0, s1) in [(MINOV, amin(0.5)), (KDEM, akde(0.5))]:
            ax.annotate("", xy=(fm.loc[s1, "stab_0.1"], fm.loc[s1, "delta_knn_20"]),
                        xytext=(fm.loc[s0, "stab_0.1"], fm.loc[s0, "delta_knn_20"]),
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=1, shrinkA=5, shrinkB=5), zorder=2)
        for s, label, color, mk in pts:
            ax.scatter([fm.loc[s, "stab_0.1"]], [fm.loc[s, "delta_knn_20"]], s=150 if s == KDEM else 75, marker=mk,
                       color=color,
                       edgecolor=SURFACE, linewidth=2, zorder=3, label=label)
        nn = fm.loc[NN1]
        ax.text(0.02, 0.97, f"off scale: 1-NN distance at stability {nn['stab_0.1']:.2f}, "
                f"sensitivity {nn.delta_knn_20:.2f}", transform=ax.transAxes, fontsize=8, color=INK2, va="top")
        ax.set_xlim(0.93, 1.0)
        ax.set_ylim(0, 0.12)
        ax.set_xlabel("Stability: rank correlation with full reference\nwhen only 10% of it is kept")
        ax.set_ylabel("Sensitivity to a local gap\n(novelty rise when 20 neighbours removed)")
        ax.set_title({"lostop": "LoStOP (structure)", "elmd": "ElMD (composition)"}[metric])
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="lower center", ncol=3, fontsize=8.5)
    fig.suptitle("Abramson adaptation moves MINov along the width trade-off, not above it (stable end shown;"
                 " hollow: α = 0.25, 0.75; reference: 2,095 experimental MP Li compounds)", x=0.01, ha="left",
                 fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(os.path.join(OUT, "e11_tradeoff.png"), dpi=200)


# Main #################################################################################


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="12345")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--draws", type=int, default=20)
    args = ap.parse_args()
    jobs = min(args.jobs, 4)  # shared machine
    t0 = time.perf_counter()
    if any(p in args.parts for p in "134"):
        meta, dms, n_r = e9.load("all")
        print(f"[load] n_R23={n_r} n_G={len(meta) - n_r}", flush=True)
    if "1" in args.parts:
        part1(meta, dms, n_r, jobs, args.draws)
        fr, boot = summarize1()
        with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 3):
            print(fr[fr.score.isin(MAIN)].set_index(["metric", "score"]).filter(regex="stab|delta|gap"))
            print(boot)
    if "2" in args.parts:
        part2(jobs, args.draws)
    if "3" in args.parts:
        part3(meta, dms, n_r)
    if "4" in args.parts:
        part4(meta, dms, n_r)
    if "5" in args.parts:
        summary()
    print(f"[done] {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
