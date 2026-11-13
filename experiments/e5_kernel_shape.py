"""E5: Kernel shape and width of the MI density; can the width be chosen from the data?

MINov sums w(d) = 1 - H(F(d)) over the reference (F = empirical CDF of reference pair
distances, H = binary entropy in bits), so its cutoff always sits at the median pair distance.
E5 compresses that kernel to a cutoff at the q-quantile,

    MI_q:   w_q(d) = 1 - H(min(F(d) / (2q), 0.5))        (q = 0.5 recovers MINov),

and compares it with other kernel shapes matched to the same half-weight point. MI_q has
w = 0.5 at F = u50 = 2 * H^-1(0.5) * q = 0.220 q, so for each q the comparison kernels are
    Gauss  : Gaussian in distance, half weight at d50 = (u50-quantile of pairs), h = d50 / 1.1774
    GaussF : Gaussian in quantile space, w = exp(-ln2 (F / u50)^2)
    TopHat : w = [d <= d50]  (ball count)
plus the E1b sweeps (KDE with h = q-quantile of pairs; kNN with k = p * n) for reference.

Parts (protocols, seeds, holes, subsamples and synthetic data are identical to E1b, so the
E1b sweeps recomputed here reproduce e1b_frontier.csv / e1b_synthetic_summary.csv):
  1. Neighbourhood holes on the Li dataset (sensitivity delta).
  2. Stability under 10% / 30% subsampling (Li), plus split-half agreement per q.
  3. Synthetic Gaussian mixtures (accuracy vs true log density, stability).
  4. Data-driven rules for q (H5b), on the Li base reference and on each synthetic reference:
       loo      : leave-one-out log likelihood of the reference under the normalised MI_q
                  kernel density, volume r^D with D = TwoNN intrinsic dimension (also D/2, 2D);
                  defensive mixture with 1% of the q=0.5 density (strict LOO is -inf for
                  narrow q on real data); "loo median" uses the median log density instead
       rankmi   : peak of the MI between "j in kNN(i)" and "i in kNN(j)" over k (q = k/(n-1))
       ranknmi  : peak of that MI divided by H(k/(n-1)) (mutuality coefficient)
       stab95   : narrowest q whose 10%-subsample stability >= 0.95
       knee     : knee (max distance above the chord) of 10% stability vs log q
       shpen    : argmax split-half agreement - lam * (normalised log q), lam = 0.05
  5. Duplicates (H5c): collapse reference materials within t of a kept one (greedy, index
     order), t = 1e-8 (exact), the 1e-4 and 1e-3 quantiles of pair distances; re-run
     stability; size-matched random removal as a control.
Combined scores (H5d) are rank averages over the query set of MINov and a narrower score.

Usage: uv run python experiments/e5_kernel_shape.py [--parts 12345] [--jobs 8]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import brentq
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import rankdata, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
from MINOV.density import MIProfile  # noqa: E402
from e1b_stability_sensitivity import (  # noqa: E402
    FRACTIONS, HOLES, KDE_Q, KNN_P, N_CONTROLS, N_MP, hole_for_seed, kde_name, knn_name,
    log_density, make_mixture, novelty_percentile, sample_mixture)

OUT = os.path.join(REPO, "experiments", "results")
Q_FINE = [0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
Q_COARSE = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
SCORED_HOLES = [("none", 0), ("knn", 5), ("knn", 20), ("knn", 100), ("radius", 0.01), ("cross", 100),
                ("random", 200)]
LN2 = np.log(2)
SH_SPLITS = 5
SH_LAMBDA = 0.05


def binary_entropy(p):
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    m = (p > 0) & (p < 1)
    pm = p[m]
    out[m] = -(pm * np.log2(pm) + (1 - pm) * np.log2(1 - pm))
    return out


P_HALF = brentq(lambda p: binary_entropy(np.array([p]))[0] - 0.5, 1e-9, 0.5)  # 0.1100
C50 = 2 * P_HALF  # MI_q falls to half weight at F = C50 * q


def mi_weight(F, q):
    return 1.0 - binary_entropy(np.minimum(F / (2 * q), 0.5))


def mi_name(q):
    return f"MI q={q:g}"


MINOV = mi_name(0.5)
COMBOS = [(MINOV, mi_name(0.02), 0.5), (MINOV, mi_name(0.01), 0.25), (MINOV, mi_name(0.01), 0.5),
          (MINOV, mi_name(0.01), 0.75), (MINOV, mi_name(0.005), 0.5), (MINOV, knn_name(0.2), 0.5),
          (MINOV, knn_name(0.03), 0.5), (MINOV, knn_name(0.0), 0.5), (kde_name(0.1), kde_name(0.01), 0.5)]


def combo_name(a, b, alpha):
    return f"RA {a} & {b} w={alpha:g}"


def sorted_quantile(sp, q):
    """np.quantile(..., method='linear') on an already sorted array."""
    pos = np.asarray(q, dtype=float) * (len(sp) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(sp) - 1)
    return sp[lo] + (pos - lo) * (sp[hi] - sp[lo])


# Scores ###############################################################################


def kernel_scores(ref_dm, query_dm):
    """All E5 scores (HIGHER = DENSER = LESS NOVEL), every parameter refit on ref_dm."""
    n = len(ref_dm)
    sp = np.sort(ref_dm[np.triu_indices(n, k=1)])
    F = np.searchsorted(sp, query_dm, side="right") / len(sp)
    out = {}
    for q in Q_FINE:
        out[mi_name(q)] = mi_weight(F, q).sum(axis=1)
    for q, d50 in zip(Q_COARSE, sorted_quantile(sp, C50 * np.array(Q_COARSE))):
        u50 = C50 * q
        h = max(d50, 1e-12) / np.sqrt(2 * LN2)
        out[f"Gauss q={q:g}"] = np.exp(-0.5 * (query_dm / h) ** 2).sum(axis=1)
        out[f"GaussF q={q:g}"] = np.exp(-LN2 * (F / u50) ** 2).sum(axis=1)
        out[f"TopHat q={q:g}"] = (query_dm <= d50).sum(axis=1).astype(float)
    for q, h in zip(KDE_Q, sorted_quantile(sp, KDE_Q)):  # E1b KDE sweep
        out[kde_name(q)] = np.exp(-0.5 * (query_dm / max(h, 1e-12)) ** 2).sum(axis=1)
    srt = np.sort(query_dm, axis=1)
    for p in KNN_P:  # E1b kNN sweep
        k = max(1, int(round(p * n)))
        out[knn_name(p)] = -srt[:, :k].mean(axis=1)
    for a, b, alpha in COMBOS:
        out[combo_name(a, b, alpha)] = (alpha * rankdata(out[a]) + (1 - alpha) * rankdata(out[b])) / len(F)
    return out


def score_meta(names):
    rows = []
    for s in names:
        if s.startswith("RA "):
            rows.append(dict(score=s, family="rank average", width=np.nan))
            continue
        fam, w = s.split(" ", 1)
        fam = {"MI": "MI_q", "KDE": "KDE (h=q-quantile)", "kNN": "kNN sweep"}.get(fam, fam)
        rows.append(dict(score=s, family=fam, width=float(w.split("=")[1])))
    return pd.DataFrame(rows)


# Part 1: holes ########################################################################


def run_seed(seed, metric, dm, dm_other, base, controls, pair_q):
    rng = np.random.default_rng(seed)  # same stream as E1b (seed 0 -> rng_seed = s)
    queries = np.concatenate([[seed], controls])
    records = []
    for kind, param in HOLES:  # draw every E1b hole so the random holes match E1b
        hole = hole_for_seed(seed, kind, param, dm, dm_other, base, pair_q, rng)
        if (kind, param) not in SCORED_HOLES:
            continue
        ref = np.setdiff1d(base, hole)
        scores = kernel_scores(dm[np.ix_(ref, ref)], dm[np.ix_(queries, ref)])
        for name, s in scores.items():
            records.append(dict(metric=metric, hole=kind, param=param, seed=seed, score=name,
                                percentile=novelty_percentile(s[0], s[1:])))
    return records


# Part 2: stability and split-half ######################################################


def stability(dm, pool, queries, draws, rng, label, masks=None):
    full = kernel_scores(dm[np.ix_(pool, pool)], dm[np.ix_(queries, pool)])
    masks = masks or {"all": np.ones(len(queries), bool)}
    records = []
    for f in FRACTIONS:
        for d in range(draws):
            ref = pool[rng.random(len(pool)) < f]
            s = kernel_scores(dm[np.ix_(ref, ref)], dm[np.ix_(queries, ref)])
            for name in full:
                rec = dict(**label, fraction=f, draw=d, score=name)
                for mk, m in masks.items():
                    rec[f"spearman_{mk}"] = spearmanr(s[name][m], full[name][m])[0]
                records.append(rec)
    return records


def split_half(dm, pool, queries, rng, n_splits=SH_SPLITS):
    """Mean Spearman between MI_q scores of the queries against two disjoint random halves."""
    out = {q: [] for q in Q_FINE}
    for _ in range(n_splits):
        perm = rng.permutation(pool)
        a, b = perm[: len(pool) // 2], perm[len(pool) // 2:]
        sa = kernel_scores(dm[np.ix_(a, a)], dm[np.ix_(queries, a)])
        sb = kernel_scores(dm[np.ix_(b, b)], dm[np.ix_(queries, b)])
        for q in Q_FINE:
            out[q].append(spearmanr(sa[mi_name(q)], sb[mi_name(q)])[0])
    return {q: float(np.mean(v)) for q, v in out.items()}


# Part 4: rules for q ##################################################################


def twonn_dimension(A):
    B = A.copy()
    np.fill_diagonal(B, np.inf)
    r = np.sort(B, axis=1)[:, :2]
    ok = r[:, 0] > 1e-8
    mu = r[ok, 1] / r[ok, 0]
    return ok.sum() / np.sum(np.log(mu)), float(np.mean(mu == 1.0))


def loo_loglik(A, D, eps=0.01):
    """LOO log density of each reference material under the MI_q kernel normalised with volume r^D.

    A compact kernel gives zero density to any material with no neighbour inside the cutoff, so
    the strict LOO likelihood is -inf for every narrow q on real data. Two robust versions:
      mix    : mean log of the defensive mixture (1 - eps) p_q + eps p_0.5   (eps = 0.01)
      median : median of log p_q over the reference
    Returns {q: (mix, median, n_zero)}."""
    n = len(A)
    sp = np.sort(A[np.triu_indices(n, k=1)])
    m = len(sp)
    F = np.searchsorted(sp, A, side="right") / m
    # On [sp[k-1], sp[k]) (sp[-1] := 0) the CDF is k / m; shell volume ~ sp[k]^D - sp[k-1]^D
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hi = np.log(sp)
        log_lo = np.concatenate([[-np.inf], log_hi[:-1]])
        log_shell = D * log_hi + np.log1p(-np.exp(D * (log_lo - log_hi)))
    log_shell = np.where(np.isnan(log_shell), -np.inf, log_shell)
    f_int = np.arange(m) / m
    logp = {}
    for q in Q_FINE:
        w_int = mi_weight(f_int, q)
        v = w_int > 0
        log_z = logsumexp(np.log(w_int[v]) + log_shell[v])
        W = mi_weight(F, q)
        np.fill_diagonal(W, 0.0)
        with np.errstate(divide="ignore"):
            logp[q] = np.log(W.sum(axis=1) / (n - 1)) - log_z
    wide = logp[0.5]
    return {q: (float(np.mean(np.logaddexp(np.log1p(-eps) + lp, np.log(eps) + wide))),
                float(np.median(lp)), int(np.sum(~np.isfinite(lp)))) for q, lp in logp.items()}


def rank_mi_profile(A):
    """MI between [j in kNN_k(i)] and [i in kNN_k(j)] over pairs, for every k."""
    n = len(A)
    B = A.copy()
    np.fill_diagonal(B, np.inf)
    R = np.argsort(np.argsort(B, axis=1, kind="stable"), axis=1) + 1
    iu = np.triu_indices(n, k=1)
    x, y = R[iu], R.T[iu]
    m = len(x)
    ks = np.arange(1, n - 1)
    nx = np.searchsorted(np.sort(x), ks, side="right")
    ny = np.searchsorted(np.sort(y), ks, side="right")
    n11 = np.searchsorted(np.sort(np.maximum(x, y)), ks, side="right")
    p11, p10, p01 = n11 / m, (nx - n11) / m, (ny - n11) / m
    p00, px, py = 1 - p11 - p10 - p01, nx / m, ny / m

    def term(pj, a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(pj > 0, pj * np.log2(pj / (a * b)), 0.0)

    mi = term(p11, px, py) + term(p10, px, 1 - py) + term(p01, 1 - px, py) + term(p00, 1 - px, 1 - py)
    qk = ks / (n - 1)
    return qk, mi, mi / binary_entropy(qk)


def knee(qs, y):
    x = np.log(qs)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    return qs[int(np.argmax(yn - xn))]


def choose_q(A, stab10, sh):
    """Every ground-truth-free rule -> chosen q (snapped to Q_FINE), plus the curves used."""
    qs = np.array(Q_FINE)
    snap = lambda q: qs[int(np.argmin(np.abs(np.log(qs) - np.log(q))))]  # noqa: E731
    D, tie_frac = twonn_dimension(A)
    rules, curves = {}, []
    for tag, d in [("loo", D), ("loo D/2", D / 2), ("loo 2D", 2 * D)]:
        ll = loo_loglik(A, d)
        rules[tag] = qs[int(np.argmax([ll[q][0] for q in qs]))]
        curves += [dict(curve=tag, q=q, value=ll[q][0]) for q in qs]
        if tag == "loo":
            rules["loo median"] = qs[int(np.argmax([ll[q][1] for q in qs]))]
            curves += [dict(curve="loo median", q=q, value=ll[q][1]) for q in qs]
            curves += [dict(curve="loo n_zero", q=q, value=ll[q][2]) for q in qs]
    qk, mi, nmi = rank_mi_profile(A)
    rules["rankmi"] = snap(qk[np.argmax(mi)])
    rules["ranknmi"] = snap(qk[np.argmax(nmi)])
    rules["rankmi raw"] = qk[np.argmax(mi)]
    rules["ranknmi raw"] = qk[np.argmax(nmi)]
    s10 = np.array([stab10[q] for q in qs])
    ok = s10 >= 0.95
    rules["stab95"] = qs[np.argmax(ok)] if ok.any() else np.nan
    rules["knee"] = knee(qs, s10)
    shv = np.array([sh[q] for q in qs])
    pen = (np.log(qs) - np.log(qs[0])) / (np.log(qs[-1]) - np.log(qs[0]))
    rules["shpen"] = qs[int(np.argmax(shv - SH_LAMBDA * pen))]
    curves += [dict(curve="stab10", q=q, value=v) for q, v in zip(qs, s10)]
    curves += [dict(curve="splithalf", q=q, value=v) for q, v in zip(qs, shv)]
    sel = np.unique(np.round(np.geomspace(1, len(qk), 60)).astype(int)) - 1
    curves += [dict(curve="rankmi", q=qk[i], value=mi[i]) for i in sel]
    curves += [dict(curve="ranknmi", q=qk[i], value=nmi[i]) for i in sel]
    return rules, curves, dict(twonn_D=D, twonn_tie_frac=tie_frac)


# Part 3: synthetic #####################################################################


def run_synthetic(dim, rep, draws, n_ref=1000, n_mix_query=400, n_unif_query=100):
    rng = np.random.default_rng(1000 * dim + rep)  # identical stream to E1b
    mix = make_mixture(dim, rng)
    ref_x = sample_mixture(n_ref, mix, rng)
    lo, hi = ref_x.min(0), ref_x.max(0)
    q_x = np.vstack([sample_mixture(n_mix_query, mix, rng), rng.uniform(lo, hi, (n_unif_query, dim))])
    truth = log_density(q_x, mix)
    in_mix = np.arange(len(q_x)) < n_mix_query
    ref_dm, q_dm = cdist(ref_x, ref_x), cdist(q_x, ref_x)

    full = kernel_scores(ref_dm, q_dm)
    records = [dict(dim=dim, rep=rep, fraction=1.0, draw=0, score=n, spearman=1.0,
                    acc_all=spearmanr(s, truth)[0], acc_mix=spearmanr(s[in_mix], truth[in_mix])[0])
               for n, s in full.items()]
    for f in FRACTIONS:
        for d in range(draws):
            ref = np.flatnonzero(rng.random(n_ref) < f)
            s = kernel_scores(ref_dm[np.ix_(ref, ref)], q_dm[:, ref])
            for name in full:
                records.append(dict(dim=dim, rep=rep, fraction=f, draw=d, score=name,
                                    spearman=spearmanr(s[name], full[name])[0], acc_all=np.nan,
                                    acc_mix=np.nan))
    df = pd.DataFrame(records)
    stab10 = df[df.fraction == 0.1].groupby("score").spearman.mean()
    sh = split_half(np.vstack([np.hstack([ref_dm, q_dm.T]), np.hstack([q_dm, np.zeros((len(q_x),) * 2)])]),
                    np.arange(n_ref), n_ref + np.arange(len(q_x)), np.random.default_rng(7 + 1000 * dim + rep))
    rules, curves, info = choose_q(ref_dm, {q: stab10[mi_name(q)] for q in Q_FINE}, sh)
    ll_true = loo_loglik(ref_dm, dim)
    rules["loo trueD"] = Q_FINE[int(np.argmax([ll_true[q][0] for q in Q_FINE]))]
    lab = dict(dataset=f"synthetic d={dim}", dim=dim, rep=rep)
    return records, [dict(**lab, rule=k, q=v) for k, v in rules.items()], \
        [dict(**lab, **c) for c in curves], dict(**lab, **info)


# Part 5: duplicates ###################################################################


def collapse(dm, pool, t):
    """Greedy t-net in index order: keep a material, drop every later one within t of it."""
    sub = dm[np.ix_(pool, pool)]
    alive = np.ones(len(pool), bool)
    for i in range(len(pool)):
        if alive[i]:
            close = sub[i] <= t
            close[: i + 1] = False
            alive[close] = False
    return pool[alive]


# Summaries and plots ##################################################################


def pareto_gap(df, x, y):
    """Stability gap of each score above the Pareto frontier of the E1b KDE + kNN sweeps.

    Frontier points: (stability x, sensitivity y), both higher-is-better; gap = x - frontier x
    at the score's y (linear interpolation between non-dominated points; NaN outside range)."""
    ref = df[df.family.isin(["KDE (h=q-quantile)", "kNN sweep"])][[x, y]].dropna().values
    nd = np.array([not np.any((ref[:, 0] >= a) & (ref[:, 1] >= b) & ((ref[:, 0] > a) | (ref[:, 1] > b)))
                   for a, b in ref])
    p = ref[nd][np.argsort(ref[nd][:, 1])]
    fx = np.interp(df[y], p[:, 1], p[:, 0], left=np.nan, right=np.nan)
    return df[x] - fx


BOOT_SCORES = [mi_name(q) for q in [0.015, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]] + [
    "Gauss q=0.05", "Gauss q=0.1", "Gauss q=0.2", "GaussF q=0.1", "TopHat q=0.2"] + [
    combo_name(*c) for c in COMBOS]


def bootstrap_gaps(fr, stab, n_boot=1000, seed=0):
    """Joint bootstrap over hole seeds (sensitivity) and subsample draws (stability) of the gap
    above the KDE + kNN Pareto frontier; the frontier itself is re-estimated in every replicate.
    Also the paired gap difference of each rank average minus the MI_q score of nearest Delta."""
    holes = pd.read_csv(os.path.join(OUT, "e5_holes.csv.gz"))
    holes = holes[holes.hole != "none"].assign(col=lambda d: d.hole + "_" + d.param.map("{:g}".format))
    rng = np.random.default_rng(seed)
    rows = []
    for metric in ["lostop", "elmd"]:
        meta = fr[fr.metric == metric][["score", "family"]].set_index("score")
        st = stab[(stab.metric == metric) & (stab.fraction == 0.1)].pivot(index="score", columns="draw",
                                                                         values="spearman")
        for col in ["knn_20", "knn_100", "cross_100"]:
            dp = holes[(holes.metric == metric) & (holes.col == col)].pivot(index="score", columns="seed",
                                                                           values="dp")
            scores = st.index.intersection(dp.index)
            S, P = st.loc[scores].values, dp.loc[scores].values
            fam = meta.loc[scores, "family"].values
            reps = []
            for _ in range(n_boot):
                di = rng.integers(0, S.shape[1], S.shape[1])
                si = rng.integers(0, P.shape[1], P.shape[1])
                df = pd.DataFrame({"score": scores, "family": fam, "x": S[:, di].mean(1), "y": P[:, si].mean(1)})
                reps.append(pareto_gap(df, "x", "y").values)
            reps = np.array(reps)
            for i, sc in enumerate(scores):
                if sc in BOOT_SCORES:
                    g = reps[:, i]
                    g = g[np.isfinite(g)]
                    rows.append(dict(metric=metric, hole=col, score=sc, gap_mean=np.mean(g) if len(g) else np.nan,
                                     gap_lo=np.quantile(g, 0.025) if len(g) else np.nan,
                                     gap_hi=np.quantile(g, 0.975) if len(g) else np.nan,
                                     p_gap_pos=np.mean(g > 0) if len(g) else np.nan,
                                     frac_in_range=len(g) / n_boot))
    return pd.DataFrame(rows)


def plot_frontier(fr, rules, path):
    import matplotlib.pyplot as plt

    fams = [("KDE (h=q-quantile)", "o-", "C0"), ("kNN sweep", "s-", "C1"), ("MI_q", "*-", "k"),
            ("Gauss", "^--", "C2"), ("GaussF", "v--", "C4"), ("TopHat", "D:", "C3")]
    holes = [("delta_knn_20", "20-NN hole"), ("delta_knn_100", "100-NN hole"),
             ("delta_cross_100", "100-NN hole in OTHER metric")]
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    for i, metric in enumerate(["lostop", "elmd"]):
        df = fr[fr.metric == metric]
        for j, (col, title) in enumerate(holes):
            ax = axes[i, j]
            for fam, style, c in fams:
                m = df[df.family == fam].sort_values("width")
                ax.plot(m["stab_0.1"], m[col], style, color=c, label=fam, ms=6 if fam != "MI_q" else 8, lw=1)
                if fam == "MI_q":
                    for _, r in m.iterrows():
                        ax.annotate(f"{r.width:g}", (r["stab_0.1"], r[col]), fontsize=6.5, xytext=(3, 3),
                                    textcoords="offset points")
            ra = df[df.family == "rank average"]
            ax.plot(ra["stab_0.1"], ra[col], "P", color="C5", ms=8, label="rank averages (H5d)")
            mn = df[df.score == MINOV]
            ax.plot(mn["stab_0.1"], mn[col], "*", color="gold", mec="k", ms=18, label="MINov (q=0.5)")
            rr = rules[rules.dataset == metric]
            for k, (rule, mk) in enumerate([("loo", "o"), ("ranknmi", "s"), ("knee", "D"), ("stab95", "^"),
                                            ("shpen", "v")]):
                q = rr[rr.rule == rule].q.values
                if len(q) and np.isfinite(q[0]):
                    r = df[df.score == mi_name(q[0])]
                    ax.plot(r["stab_0.1"], r[col], mk, mfc="none", mec="r", ms=10 + 2 * k, mew=1.2,
                            label=f"rule {rule} (q={q[0]:g})" if j == 0 else f"rule {rule}")
            ax.set_xlim(max(0.45, df["stab_0.1"].min() - 0.02), 1.0)
            ax.set_xlabel("stability: Spearman vs full reference at 10% subsample")
            ax.set_ylabel("sensitivity: rise in seed novelty percentile")
            ax.set_title(f"{metric}: {title}", fontsize=10)
            ax.grid(alpha=0.3)
    for i in range(2):
        h, lab = axes[i, 0].get_legend_handles_labels()
        axes[i, 2].legend(h, lab, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("E5: E1b frontier with the compressed MI kernel MI_q (labels: q) and width-matched shapes")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def plot_synthetic(ss, rules, path):
    import matplotlib.pyplot as plt

    fams = [("KDE (h=q-quantile)", "o-", "C0"), ("kNN sweep", "s-", "C1"), ("MI_q", "*-", "k"),
            ("Gauss", "^--", "C2"), ("GaussF", "v--", "C4"), ("TopHat", "D:", "C3")]
    dims = sorted(ss.dim.unique())
    fig, axes = plt.subplots(2, len(dims), figsize=(5.5 * len(dims), 9))
    for k, dim in enumerate(dims):
        df = ss[ss.dim == dim]
        ax = axes[0, k]
        for fam, style, c in fams:
            m = df[df.family == fam].sort_values("width")
            ax.plot(m["stab_0.1"], m["acc_full"], style, color=c, label=fam, lw=1)
        mn = df[df.score == MINOV]
        ax.plot(mn["stab_0.1"], mn["acc_full"], "*", color="gold", mec="k", ms=16, label="MINov")
        ax.set_xlabel("stability at 10%")
        ax.set_ylabel("accuracy: Spearman vs true log density")
        ax.set_title(f"Gaussian mixture d={dim}")
        ax.grid(alpha=0.3)
        ax = axes[1, k]
        m = df[df.family == "MI_q"].sort_values("width")
        ax.plot(m.width, m.acc_full, "k*-", label="MI_q accuracy")
        ax.plot(m.width, m["stab_0.1"], "C7.--", label="MI_q stability 10%")
        rr = rules[rules.dim == dim]
        for j, rule in enumerate(["loo", "loo trueD", "rankmi", "ranknmi", "stab95", "knee", "shpen"]):
            qs = rr[rr.rule == rule].q.values
            ax.plot(qs, np.full(len(qs), 0.5 + 0.04 * j) + np.linspace(-0.01, 0.01, len(qs)), "|", ms=12,
                    mew=2, label=f"rule {rule} (y offset)", color=f"C{j}")
        ax.set_xscale("log")
        ax.set_xlabel("q (MI_q cutoff quantile)")
        ax.set_ylim(0.45, 1.0)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="lower left")
    axes[1, 0].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


# Main #################################################################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=150)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--parts", default="12345")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)  # E1b's stream: permutation, then stability draws
    perm = rng.permutation(N_MP)
    seeds, controls = perm[: args.seeds], perm[args.seeds: args.seeds + N_CONTROLS]
    base = np.sort(perm[args.seeds + N_CONTROLS:])
    dms = {m: np.load(os.path.join(REPO, "precomputed", f"Li_{m}_dm.npy")) for m in ["lostop", "elmd"]}
    queries = np.concatenate([seeds, controls, np.arange(N_MP, len(dms["lostop"]))])

    # sanity: MI q=0.5 is MINov
    for metric, dm in dms.items():
        a = kernel_scores(dm[np.ix_(base, base)], dm[np.ix_(queries, base)])[MINOV]
        b = MIProfile(dm[np.ix_(base, base)]).density(dm[np.ix_(queries, base)])
        print(f"[check] {metric}: MI q=0.5 vs MINov exact: max |diff| = {np.max(np.abs(a - b)):.2e}", flush=True)

    if "1" in args.parts:
        records = []
        for metric, other in [("lostop", "elmd"), ("elmd", "lostop")]:
            t = time.perf_counter()
            dm = dms[metric]
            bp = np.sort(dm[np.ix_(base, base)][np.triu_indices(len(base), k=1)])
            pair_q = {q: sorted_quantile(bp, q) for kind, q in HOLES if kind == "radius"}
            out = Parallel(n_jobs=args.jobs)(delayed(run_seed)(s, metric, dm, dms[other], base, controls, pair_q)
                                             for s in seeds)
            records += [r for rs in out for r in rs]
            print(f"[holes] {metric}: {time.perf_counter() - t:.0f}s", flush=True)
        holes = pd.DataFrame(records)
        p0 = holes[holes.hole == "none"].set_index(["metric", "seed", "score"]).percentile
        holes = holes.join(p0.rename("p0"), on=["metric", "seed", "score"])
        holes["dp"] = holes.percentile - holes.p0
        holes.to_csv(os.path.join(OUT, "e5_holes.csv.gz"), index=False, float_format="%.4g")
        (holes.groupby(["metric", "hole", "param", "score"])
         .agg(auc=("percentile", "mean"), delta=("dp", "mean"), delta_sem=("dp", "sem")).reset_index()
         .to_csv(os.path.join(OUT, "e5_holes_summary.csv"), index=False))

    if "2" in args.parts:
        records, sh_rows = [], []
        for metric, dm in dms.items():
            t = time.perf_counter()
            records += stability(dm, base, queries, args.draws, rng, dict(metric=metric))
            sh = split_half(dm, base, queries, np.random.default_rng(11))
            sh_rows += [dict(metric=metric, q=q, splithalf=v) for q, v in sh.items()]
            print(f"[stability] {metric}: {time.perf_counter() - t:.0f}s", flush=True)
        pd.DataFrame(records).rename(columns={"spearman_all": "spearman"}).to_csv(
            os.path.join(OUT, "e5_stability.csv"), index=False)
        pd.DataFrame(sh_rows).to_csv(os.path.join(OUT, "e5_splithalf.csv"), index=False)

    if "3" in args.parts:
        t = time.perf_counter()
        out = Parallel(n_jobs=args.jobs)(delayed(run_synthetic)(dim, rep, args.draws)
                                         for dim in [2, 5, 10] for rep in range(args.reps))
        pd.DataFrame([r for o in out for r in o[0]]).to_csv(os.path.join(OUT, "e5_synthetic.csv"), index=False)
        pd.DataFrame([r for o in out for r in o[1]]).to_csv(os.path.join(OUT, "e5_rules_synthetic.csv"),
                                                            index=False)
        pd.DataFrame([r for o in out for r in o[2]]).to_csv(os.path.join(OUT, "e5_rule_curves_synthetic.csv"),
                                                            index=False)
        pd.DataFrame([o[3] for o in out]).to_csv(os.path.join(OUT, "e5_dimension_synthetic.csv"), index=False)
        print(f"[synthetic] {time.perf_counter() - t:.0f}s", flush=True)

    if "4" in args.parts:
        stab = pd.read_csv(os.path.join(OUT, "e5_stability.csv"))
        shd = pd.read_csv(os.path.join(OUT, "e5_splithalf.csv"))
        rule_rows, curve_rows, info_rows = [], [], []
        for metric, dm in dms.items():
            s10 = stab[(stab.metric == metric) & (stab.fraction == 0.1)].groupby("score").spearman.mean()
            sh = shd[shd.metric == metric].set_index("q").splithalf
            rules, curves, info = choose_q(dm[np.ix_(base, base)], {q: s10[mi_name(q)] for q in Q_FINE},
                                           {q: sh[q] for q in Q_FINE})
            rule_rows += [dict(dataset=metric, rule=k, q=v) for k, v in rules.items()]
            curve_rows += [dict(dataset=metric, **c) for c in curves]
            info_rows.append(dict(dataset=metric, **info))
            print(f"[rules] {metric}: {info} {rules}", flush=True)
        pd.DataFrame(rule_rows).to_csv(os.path.join(OUT, "e5_rules_real.csv"), index=False)
        pd.DataFrame(curve_rows).to_csv(os.path.join(OUT, "e5_rule_curves_real.csv"), index=False)
        pd.DataFrame(info_rows).to_csv(os.path.join(OUT, "e5_dimension_real.csv"), index=False)

    if "5" in args.parts:
        rows, summ = [], []
        keep_scores = [knn_name(0.0), "kNN p=0.002", knn_name(0.01), MINOV, mi_name(0.02), mi_name(0.005),
                       kde_name(0.02)]
        for metric, dm in dms.items():
            bp = np.sort(dm[np.ix_(base, base)][np.triu_indices(len(base), k=1)])
            for tag, t in [("none", -1.0), ("exact (1e-8)", 1e-8), ("quantile 1e-4", sorted_quantile(bp, 1e-4)),
                           ("quantile 1e-3", sorted_quantile(bp, 1e-3))]:
                pool = base if t < 0 else collapse(dm, base, t)
                nn_q = dm[np.ix_(queries, pool)].min(axis=1)
                masks = {"all": np.ones(len(queries), bool), "clean": nn_q > max(t, 1e-8)}
                conds = [("dedup", pool)]
                if t >= 0:
                    rr = np.random.default_rng(3)
                    conds.append(("random removal", np.sort(rr.choice(base, len(pool), replace=False))))
                for cond, pl in conds:
                    lab = dict(metric=metric, threshold=tag, t=t, condition=cond, n_pool=len(pl),
                               n_removed=len(base) - len(pl), pairs_below_t=int(np.sum(bp <= t)),
                               n_clean_queries=int(masks["clean"].sum()))
                    recs = stability(dm, pl, queries, args.draws, np.random.default_rng(5), lab, masks)
                    rows += [r for r in recs if r["score"] in keep_scores]
                    print(f"[dups] {metric} {tag} {cond}: n_pool={len(pl)}", flush=True)
        dd = pd.DataFrame(rows)
        dd.to_csv(os.path.join(OUT, "e5_duplicates_raw.csv"), index=False)
        g = dd.groupby(["metric", "threshold", "t", "condition", "n_pool", "n_removed", "pairs_below_t",
                        "n_clean_queries", "score", "fraction"])
        g.agg(stab_all=("spearman_all", "mean"), stab_clean=("spearman_clean", "mean"),
              sem_all=("spearman_all", "sem")).reset_index().to_csv(
            os.path.join(OUT, "e5_duplicates.csv"), index=False)

    # Summaries ##########################################################################
    hs = pd.read_csv(os.path.join(OUT, "e5_holes_summary.csv"))
    hs = hs[hs.hole != "none"].assign(col=lambda d: d.hole + "_" + d.param.map("{:g}".format))
    dl = hs.pivot_table(index=["metric", "score"], columns="col", values="delta").add_prefix("delta_")
    se = hs.pivot_table(index=["metric", "score"], columns="col", values="delta_sem").add_prefix("sem_")
    stab = pd.read_csv(os.path.join(OUT, "e5_stability.csv"))
    st = stab.groupby(["metric", "score", "fraction"]).spearman.mean().unstack("fraction")
    st.columns = [f"stab_{c:g}" for c in st.columns]
    sts = stab[stab.fraction == 0.1].groupby(["metric", "score"]).spearman.sem().rename("stab_0.1_sem")
    fr = st.join(sts).join(dl).join(se).reset_index()
    fr = fr.merge(score_meta(fr.score.unique()), on="score")
    for metric in fr.metric.unique():
        m = fr.metric == metric
        for col in ["delta_knn_20", "delta_knn_100", "delta_cross_100"]:
            fr.loc[m, "gap_" + col[6:]] = pareto_gap(fr[m], "stab_0.1", col)
    fr.to_csv(os.path.join(OUT, "e5_frontier.csv"), index=False)
    if os.path.exists(os.path.join(OUT, "e5_holes.csv.gz")):
        bootstrap_gaps(fr, stab).to_csv(os.path.join(OUT, "e5_gap_bootstrap.csv"), index=False)

    syn = pd.read_csv(os.path.join(OUT, "e5_synthetic.csv"))
    g = syn.groupby(["dim", "score", "fraction"])
    ss = pd.concat([g.spearman.mean().unstack("fraction").drop(columns=1.0).add_prefix("stab_"),
                    syn[syn.fraction == 1.0].groupby(["dim", "score"]).acc_all.mean().rename("acc_full"),
                    syn[syn.fraction == 1.0].groupby(["dim", "score"]).acc_mix.mean().rename("accmix_full")],
                   axis=1).reset_index()
    ss = ss.merge(score_meta(ss.score.unique()), on="score")
    ss.to_csv(os.path.join(OUT, "e5_synthetic_summary.csv"), index=False)

    # rule evaluation: synthetic accuracy at the chosen q vs oracle; real-data frontier position
    rs = pd.read_csv(os.path.join(OUT, "e5_rules_synthetic.csv"))
    acc = syn[(syn.fraction == 1.0) & syn.score.str.startswith("MI q=")].copy()
    acc["q"] = acc.score.str[5:].astype(float)
    s10 = syn[(syn.fraction == 0.1)].groupby(["dim", "rep", "score"]).spearman.mean().rename("stab10")
    acc = acc.join(s10, on=["dim", "rep", "score"])
    oracle = acc.loc[acc.groupby(["dim", "rep"]).acc_all.idxmax(), ["dim", "rep", "q", "acc_all"]]
    oracle = oracle.rename(columns={"q": "oracle_q", "acc_all": "oracle_acc"})
    ev = rs[~rs.rule.str.endswith("raw")].merge(acc[["dim", "rep", "q", "acc_all", "stab10"]],
                                                on=["dim", "rep", "q"], how="left")
    ev = ev.merge(oracle, on=["dim", "rep"])
    ev["regret"] = ev.oracle_acc - ev.acc_all
    ev.to_csv(os.path.join(OUT, "e5_rules_eval_synthetic.csv"), index=False)

    rr = pd.read_csv(os.path.join(OUT, "e5_rules_real.csv"))
    rr = rr[~rr.rule.str.endswith("raw")].assign(score=lambda d: d.q.map(lambda q: mi_name(q) if np.isfinite(q) else ""))
    rr = rr.merge(fr.rename(columns={"metric": "dataset"})[["dataset", "score", "stab_0.1", "stab_0.3",
                                                             "delta_knn_20", "delta_knn_100", "delta_cross_100"]],
                  on=["dataset", "score"], how="left")
    rr.to_csv(os.path.join(OUT, "e5_rules_eval_real.csv"), index=False)

    plot_frontier(fr, rr, os.path.join(OUT, "e5_frontier.png"))
    plot_synthetic(ss, rs, os.path.join(OUT, "e5_synthetic.png"))

    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 3,
                           "display.max_rows", 500):
        cols = ["stab_0.1", "stab_0.3", "delta_knn_5", "delta_knn_20", "delta_knn_100", "delta_radius_0.01",
                "delta_cross_100", "delta_random_200", "gap_knn_20", "gap_knn_100", "gap_cross_100"]
        print(fr.set_index(["metric", "family", "score"])[cols].sort_index())
        print(ss.set_index(["dim", "family", "score"])[["stab_0.1", "stab_0.3", "acc_full", "accmix_full"]]
              .sort_index())
        print(ev.groupby(["dim", "rule"])[["q", "acc_all", "oracle_q", "oracle_acc", "regret", "stab10"]].mean())
        print(rr)


if __name__ == "__main__":
    main()
