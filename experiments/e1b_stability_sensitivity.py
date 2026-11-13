"""E1b: Do wide kernels buy stability at the cost of sensitivity to local gaps?

Three parts, all with the score convention of novelty_scores.py (HIGHER = DENSER = LESS NOVEL)
and every parameter (bandwidth, k, cutoff) refit on whatever reference is in use.

1. Neighbourhood holdouts (Li dataset, LoStOP and ElMD).
   150 seed and 200 control MP materials are drawn at random and removed from the reference;
   the base reference R0 is the rest of MP. For each seed and each hole, the reference is
   R0 minus the hole, and the seed plus all controls are scored against it. The seed's
   sensitivity is its novelty percentile among the controls (0.5 = typical, 1 = most novel);
   averaged over seeds this is the AUC of holed seeds vs controls. The headline number is the
   paired rise (delta) of that percentile relative to the same seed with no hole. Hole types:
     knn       : the r nearest R0 neighbours of the seed in the scoring metric
     radius    : all R0 materials within the q-quantile of R0 pair distances (scoring metric)
     cross     : the r nearest R0 neighbours in the OTHER metric (breaks the circularity of
                 defining the hole with the distance the score uses)
     random    : r random R0 materials (size-only control; should give AUC ~ 0.5)
2. Stability of the same scores: queries = seeds + controls + GNoME scored against R0 and
   against uniform 10% / 30% subsamples of R0 (20 draws); Spearman vs the R0 scores.
   Together with part 1 this places every score on a stability-sensitivity plane, including
   a Gaussian-KDE bandwidth sweep (h = q-quantile of reference pair distances) and a kNN
   sweep (k = fraction p of the reference, score = -mean distance to the k nearest).
3. Synthetic ground truth: Gaussian mixtures with known density in d = 2, 5, 10.
   Accuracy = Spearman(score, true log density) on queries; stability as in part 2.

Usage: uv run python experiments/e1b_stability_sensitivity.py [--seeds 150] [--draws 20] [--jobs 8]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import rankdata, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
from novelty_scores import SCORES  # noqa: E402

OUT = os.path.join(REPO, "experiments", "results")
N_MP = 1834
N_CONTROLS = 200

KDE_Q = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]  # bandwidth = q-quantile of pairs
KNN_P = [0.0, 0.002, 0.005, 0.01, 0.03, 0.1, 0.2, 0.5]  # k = max(1, round(p * n_ref)); p=0 -> 1-NN
HOLES = ([("none", 0)]
         + [("knn", r) for r in [5, 20, 50, 100, 200]]
         + [("radius", q) for q in [0.002, 0.01, 0.05]]
         + [("cross", r) for r in [20, 100]]
         + [("random", r) for r in [20, 200]])
FRACTIONS = [0.1, 0.3]


# Scores ###############################################################################


def kde_name(q):
    return f"KDE q={q:g}"


def knn_name(p):
    return f"kNN p={p:g}"


def all_scores(ref_dm, query_dm):
    """Every score in novelty_scores.SCORES plus the KDE and kNN sweeps."""
    out = {name: f(ref_dm, query_dm) for name, f in SCORES.items()}
    pairs = ref_dm[np.triu_indices(len(ref_dm), k=1)]
    for q, h in zip(KDE_Q, np.quantile(pairs, KDE_Q)):
        h = max(h, 1e-12)
        out[kde_name(q)] = np.exp(-0.5 * (query_dm / h) ** 2).sum(axis=1)
    srt = np.sort(query_dm, axis=1)
    for p in KNN_P:
        k = max(1, int(round(p * len(ref_dm))))
        out[knn_name(p)] = -srt[:, :k].mean(axis=1)
    return out


def score_meta():
    """Family and width parameter of each score, for plotting."""
    rows = [dict(score=n, family="baseline", width=np.nan) for n in SCORES]
    rows += [dict(score=kde_name(q), family="KDE sweep", width=q) for q in KDE_Q]
    rows += [dict(score=knn_name(p), family="kNN sweep", width=p) for p in KNN_P]
    return pd.DataFrame(rows)


def novelty_percentile(seed_score, control_scores):
    """Fraction of controls less novel (denser) than the seed; ties count half."""
    return (np.sum(control_scores > seed_score) + 0.5 * np.sum(control_scores == seed_score)) / len(
        control_scores)


# Part 1: neighbourhood holes ##########################################################


def hole_for_seed(seed, kind, param, dm, dm_other, base, pair_q, rng):
    d = dm[seed, base]
    if kind == "none":
        return np.array([], dtype=int)
    if kind == "knn":
        return base[np.argsort(d, kind="stable")[:param]]
    if kind == "radius":
        return base[d <= pair_q[param]]
    if kind == "cross":
        return base[np.argsort(dm_other[seed, base], kind="stable")[:param]]
    if kind == "random":
        return rng.choice(base, param, replace=False)
    raise ValueError(kind)


def run_seed(seed, metric, dm, dm_other, base, controls, pair_q, rng_seed):
    rng = np.random.default_rng(rng_seed)
    queries = np.concatenate([[seed], controls])
    records = []
    for kind, param in HOLES:
        hole = hole_for_seed(seed, kind, param, dm, dm_other, base, pair_q, rng)
        ref = np.setdiff1d(base, hole)
        scores = all_scores(dm[np.ix_(ref, ref)], dm[np.ix_(queries, ref)])
        for name, s in scores.items():
            records.append(dict(metric=metric, hole=kind, param=param, seed=seed, n_removed=len(hole),
                                score=name, percentile=novelty_percentile(s[0], s[1:])))
    return records


# Part 2: stability under subsampling ##################################################


def stability(dm, pool, queries, draws, rng, label):
    full = all_scores(dm[np.ix_(pool, pool)], dm[np.ix_(queries, pool)])
    records = []
    for f in FRACTIONS:
        for d in range(draws):
            ref = pool[rng.random(len(pool)) < f]
            s = all_scores(dm[np.ix_(ref, ref)], dm[np.ix_(queries, ref)])
            for name in full:
                records.append(dict(**label, fraction=f, draw=d, score=name,
                                    spearman=spearmanr(s[name], full[name])[0]))
    return records


# Part 3: synthetic Gaussian mixtures ##################################################


def make_mixture(dim, rng):
    """Blob mixture in the spirit of methods_analysis.ipynb, plus a broad background."""
    k = 5
    centers = rng.uniform(0, 2.5, size=(k, dim))
    stds = np.array([0.1, 0.3, 0.5, 0.1, 0.2])
    weights = np.array([0.3, 0.25, 0.2, 0.1, 0.1])
    centers = np.vstack([centers, np.full(dim, 1.25)])  # broad background component
    stds = np.append(stds, 1.0)
    weights = np.append(weights, 0.05)
    return centers, stds, weights


def sample_mixture(n, mix, rng):
    centers, stds, weights = mix
    comp = rng.choice(len(weights), n, p=weights)
    return centers[comp] + rng.standard_normal((n, centers.shape[1])) * stds[comp, None]


def log_density(x, mix):
    centers, stds, weights = mix
    dim = x.shape[1]
    sq = cdist(x, centers, "sqeuclidean")
    logp = np.log(weights) - 0.5 * sq / stds**2 - dim * np.log(stds) - 0.5 * dim * np.log(2 * np.pi)
    return logsumexp(logp, axis=1)


def run_synthetic(dim, rep, draws, n_ref=1000, n_mix_query=400, n_unif_query=100):
    rng = np.random.default_rng(1000 * dim + rep)
    mix = make_mixture(dim, rng)
    ref_x = sample_mixture(n_ref, mix, rng)
    lo, hi = ref_x.min(0), ref_x.max(0)
    q_x = np.vstack([sample_mixture(n_mix_query, mix, rng), rng.uniform(lo, hi, (n_unif_query, dim))])
    truth = log_density(q_x, mix)
    in_mix = np.arange(len(q_x)) < n_mix_query
    ref_dm, q_dm = cdist(ref_x, ref_x), cdist(q_x, ref_x)

    def acc(s):
        return spearmanr(s, truth)[0], spearmanr(s[in_mix], truth[in_mix])[0]

    full = all_scores(ref_dm, q_dm)
    records = [dict(dim=dim, rep=rep, fraction=1.0, draw=0, score=n, spearman=1.0,
                    acc_all=acc(s)[0], acc_mix=acc(s)[1]) for n, s in full.items()]
    for f in FRACTIONS:
        for d in range(draws):
            ref = np.flatnonzero(rng.random(n_ref) < f)
            s = all_scores(ref_dm[np.ix_(ref, ref)], q_dm[:, ref])
            for name in full:
                a_all, a_mix = acc(s[name])
                records.append(dict(dim=dim, rep=rep, fraction=f, draw=d, score=name,
                                    spearman=spearmanr(s[name], full[name])[0],
                                    acc_all=a_all, acc_mix=a_mix))
    return records


# Plots ################################################################################


def plot_frontier(frontier, path):
    import matplotlib.pyplot as plt

    holes = [("knn", 20, "20-NN hole (same metric)"), ("radius", 0.01, "radius hole, 1% pair quantile"),
             ("cross", 100, "100-NN hole in the OTHER metric")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for i, metric in enumerate(["lostop", "elmd"]):
        for j, (kind, param, title) in enumerate(holes):
            ax = axes[i, j]
            df = frontier[(frontier.metric == metric)]
            y = df[f"delta_{kind}_{param:g}"]
            x = df["stab_0.1"]
            for fam, style in [("KDE sweep", "o-"), ("kNN sweep", "s-")]:
                m = (df.family == fam).values
                order = np.argsort(df.width.values[m])
                ax.plot(x.values[m][order], y.values[m][order], style, label=fam, ms=5)
                for xx, yy, w in zip(x.values[m], y.values[m], df.width.values[m]):
                    ax.annotate(f"{w:g}", (xx, yy), fontsize=7, xytext=(3, 3), textcoords="offset points")
            for name, marker in [("MINov (exact)", "*"), ("KDE (paper)", "D"), ("kNN (paper)", "^"),
                                 ("KDE (global bw)", "v"), ("KDE (median bw)", "P"), ("LOF", "X")]:
                m = (df.score == name).values
                ax.plot(x.values[m], y.values[m], marker, ms=14 if marker == "*" else 9, label=name,
                        color="k" if name.startswith("MINov") else None)
            ax.set_xlabel("stability: Spearman vs full reference at 10% subsample")
            ax.set_ylabel("sensitivity: rise in seed novelty percentile")
            ax.set_title(f"{metric}: {title}", fontsize=10)
            ax.axhline(0, color="grey", lw=0.5)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Stability-sensitivity frontier (labels: KDE q-quantile bandwidth / kNN k as fraction of n)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_synthetic(summary, path):
    import matplotlib.pyplot as plt

    dims = sorted(summary.dim.unique())
    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4.5))
    for ax, dim in zip(axes, dims):
        df = summary[summary.dim == dim]
        for fam, style in [("KDE sweep", "o-"), ("kNN sweep", "s-")]:
            m = df[df.family == fam].sort_values("width")
            ax.plot(m["stab_0.1"], m["acc_full"], style, label=fam, ms=5)
            for _, r in m.iterrows():
                ax.annotate(f"{r.width:g}", (r["stab_0.1"], r["acc_full"]), fontsize=7, xytext=(3, 3),
                            textcoords="offset points")
        for name, marker in [("MINov (exact)", "*"), ("KDE (paper)", "D"), ("kNN (paper)", "^"),
                             ("KDE (global bw)", "v"), ("LOF", "X")]:
            r = df[df.score == name]
            ax.plot(r["stab_0.1"], r["acc_full"], marker, ms=14 if marker == "*" else 9, label=name,
                    color="k" if name.startswith("MINov") else None)
        ax.set_title(f"Gaussian mixture, d={dim}")
        ax.set_xlabel("stability: Spearman vs full reference at 10%")
        ax.set_ylabel("accuracy: Spearman vs true log density")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# Main #################################################################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=150)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parts", default="123")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    meta = score_meta()
    rng = np.random.default_rng(args.seed)

    perm = rng.permutation(N_MP)
    seeds, controls = perm[: args.seeds], perm[args.seeds: args.seeds + N_CONTROLS]
    base = np.sort(perm[args.seeds + N_CONTROLS:])
    dms = {m: np.load(os.path.join(REPO, "precomputed", f"Li_{m}_dm.npy")) for m in ["lostop", "elmd"]}

    if "1" in args.parts:
        records = []
        for metric, other in [("lostop", "elmd"), ("elmd", "lostop")]:
            t = time.perf_counter()
            dm = dms[metric]
            base_pairs = dm[np.ix_(base, base)][np.triu_indices(len(base), k=1)]
            pair_q = {q: np.quantile(base_pairs, q) for kind, q in HOLES if kind == "radius"}
            out = Parallel(n_jobs=args.jobs)(
                delayed(run_seed)(s, metric, dm, dms[other], base, controls, pair_q, args.seed * 10**6 + s)
                for s in seeds)
            records += [r for rs in out for r in rs]
            print(f"[holes] {metric}: {time.perf_counter() - t:.0f}s", flush=True)
        pd.DataFrame(records).to_csv(os.path.join(OUT, "e1b_holes.csv"), index=False)

    if "2" in args.parts:
        records = []
        queries = np.concatenate([seeds, controls, np.arange(N_MP, len(dms["lostop"]))])
        for metric, dm in dms.items():
            t = time.perf_counter()
            records += stability(dm, base, queries, args.draws, rng, dict(metric=metric))
            print(f"[stability] {metric}: {time.perf_counter() - t:.0f}s", flush=True)
        pd.DataFrame(records).to_csv(os.path.join(OUT, "e1b_stability.csv"), index=False)

    if "3" in args.parts:
        t = time.perf_counter()
        out = Parallel(n_jobs=args.jobs)(delayed(run_synthetic)(dim, rep, args.draws)
                                         for dim in [2, 5, 10] for rep in range(args.reps))
        syn = pd.DataFrame([r for rs in out for r in rs])
        syn.to_csv(os.path.join(OUT, "e1b_synthetic.csv"), index=False)
        print(f"[synthetic] {time.perf_counter() - t:.0f}s", flush=True)

    # summaries and plots (from whatever CSVs exist)
    holes = pd.read_csv(os.path.join(OUT, "e1b_holes.csv"))
    no_hole = holes[holes.hole == "none"].set_index(["metric", "seed", "score"]).percentile
    holes = holes.join(no_hole.rename("p0"), on=["metric", "seed", "score"])
    holes["dp"] = holes.percentile - holes.p0  # paired rise in the seed's novelty percentile
    holes_s = (holes.groupby(["metric", "hole", "param", "score"])
               .agg(auc=("percentile", "mean"), flag10=("percentile", lambda p: np.mean(p > 0.9)),
                    delta=("dp", "mean"), delta_sem=("dp", "sem"), n_removed=("n_removed", "median"))
               .reset_index())
    holes_s.to_csv(os.path.join(OUT, "e1b_holes_summary.csv"), index=False)
    stab = pd.read_csv(os.path.join(OUT, "e1b_stability.csv"))
    st = stab.groupby(["metric", "score", "fraction"]).spearman.mean().unstack("fraction")
    st.columns = [f"stab_{c:g}" for c in st.columns]
    hs = holes_s[holes_s.hole != "none"].assign(col=lambda d: d.hole + "_" + d.param.map("{:g}".format))
    hs = hs.pivot_table(index=["metric", "score"], columns="col", values="delta").add_prefix("delta_")
    frontier = st.join(hs).reset_index().merge(meta, on="score")
    frontier.to_csv(os.path.join(OUT, "e1b_frontier.csv"), index=False)
    plot_frontier(frontier, os.path.join(OUT, "e1b_frontier.png"))

    syn = pd.read_csv(os.path.join(OUT, "e1b_synthetic.csv"))
    g = syn.groupby(["dim", "score", "fraction"])
    ss = pd.concat([g.spearman.mean().unstack("fraction").drop(columns=1.0).add_prefix("stab_"),
                    g.acc_all.mean().unstack("fraction").add_prefix("acc_"),
                    g.acc_mix.mean().unstack("fraction").add_prefix("accmix_")], axis=1)
    ss.columns = [c.replace("_1.0", "_full") for c in map(str, ss.columns)]
    ss = ss.reset_index().merge(meta, on="score")
    ss.to_csv(os.path.join(OUT, "e1b_synthetic_summary.csv"), index=False)
    plot_synthetic(ss, os.path.join(OUT, "e1b_synthetic.png"))

    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.precision", 3,
                           "display.max_rows", 200):
        print(frontier.drop(columns=["family"]).set_index(["metric", "score"]))
        print(ss.drop(columns=["family"]).set_index(["dim", "score"]))


if __name__ == "__main__":
    main()
