"""E14: Does reference-relative novelty flag where a property model is wrong, beyond the model's own uncertainty?

Question: is novelty (vs the training set) a domain-of-applicability signal that adds information to an ensemble's
spread sigma, and does MINov (wide, global kernel) or 1-NN distance (local) carry more of it?

Data (matminer load_dataset, cached in data/raw/e14/):
  elastic_tensor_2015  target log10(K_VRH), rows with K_VRH <= 0 dropped; novelty on ElMD and LoStOP.
  matbench_expt_gap    target experimental band gap (eV); novelty on ElMD only.
  matbench_phonons     target last phonon-DOS peak (cm^-1); novelty on ElMD and LoStOP.
  Rows dropped: elements off the modified Pettifor scale (ElMD), NaN Magpie features, LoStOP failures (no site
  featurized); counts are logged.

Model: RandomForestRegressor(300 trees, n_jobs=3) on Magpie (ElementProperty "magpie") features, plus LoStOP
  features for structure datasets. mu = forest mean, sigma = std of the 300 tree predictions.

Splits: 5-fold KFold(shuffle) x 5 repeats (seeds 0-4) = "random"; plus one GroupKFold(5) on chemical system
  (sorted element set) = "chemsys" (extrapolation: no test chemical system is in the training fold).

Novelty per test point vs its training fold, for "MINov (exact)", "1-NN distance", "KDE (median bw)" from
  experiments/novelty_scores.py (profile / bandwidth fitted on the training-fold distance matrix). Scores are
  higher = denser; novelty = -score, converted to a within-fold percentile before folds are pooled.

Analyses (per repeat, folds pooled so each material appears once; random-CV numbers are means over the 5 repeats;
  95% CIs = percentile bootstrap over materials, N_BOOT resamples, same resample applied to every repeat):
  1. Spearman(|err|, novelty) and partial Spearman given sigma (ranks of |err| and novelty each OLS-residualized
     on rank(sigma), Pearson of residuals). Also Spearman(|err|, sigma) and Spearman(sigma, novelty).
  2. Worst 10% of |err|: AUC and recall@10% (top-10% flagged) for sigma alone, novelty alone, and
     rank average (rank(sigma) + rank(novelty)) / 2 (equal weights, no tuning); delta vs sigma alone.
  3. Coverage of mu +/- 1.645 sigma overall and by novelty tercile; delta = top (most novel) - bottom tercile.
  2b. Supplement: worst-10% AUC of a 5-fold cross-fitted logistic regression on [rank sigma, rank novelty] minus
     the same on rank sigma alone (point estimate, range over repeats).
  Added because the data forced it: expt_gap is ~half exact zeros (metals), so it is also analysed restricted to
  non-metals (y > 0, same predictions) as "expt_gap_nonmetal".
  4. Figure e14_summary.png: partial Spearman (CI) and coverage by novelty tercile.

Machine limits: LoStOP (n_jobs=3), Magpie (n_jobs=3) and RF (n_jobs=3) run one at a time; float32 distance matrices.
LoStOP distance = Euclidean distance between LoStOP fingerprints (what MINOV.distance.get_lostop_dm computes); the
fingerprints come from MINOV.distance.process_structure, called per structure so failures can be logged and dropped
instead of raising.

Outputs: experiments/results/e14_<dataset>_cache.npz (features + distance matrices, git-ignored),
  e14_predictions.csv.gz, e14_model_metrics.csv, e14_partial.csv, e14_worst10.csv, e14_coverage.csv,
  e14_summary.png, e14_log.txt.

Usage: uv run python experiments/e14_model_error.py [--stages features models analyze fig] [--datasets ...]
"""

import argparse
import os
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import pdist, squareform  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
from novelty_scores import SCORES  # noqa: E402
from make_report_figures import SERIES, SURFACE, INK, INK2, GRID, CONTEXT  # noqa: E402,F401

OUT = os.path.join(REPO, "experiments", "results")
RAW = os.path.join(REPO, "data", "raw", "e14")
NAMES = ["MINov (exact)", "1-NN distance", "KDE (median bw)"]
SHORT = {"MINov (exact)": "MINov", "1-NN distance": "1-NN", "KDE (median bw)": "median KDE"}
COLOR = {"MINov (exact)": SERIES["MINov"], "1-NN distance": SERIES["1-NN distance"],
         "KDE (median bw)": SERIES["Gaussian KDE, median bandwidth"]}
DATASETS = {
    "elastic": dict(name="elastic_tensor_2015", structure=True),
    "expt_gap": dict(name="matbench_expt_gap", structure=False),
    "phonons": dict(name="matbench_phonons", structure=True),
}
REPEATS = [0, 1, 2, 3, 4]
N_FOLDS = 5
N_TREES = 300
WORKERS = 3
N_BOOT = 1000
Z90 = 1.645
SEED = 0

warnings.filterwarnings("ignore")
LOG = open(os.path.join(OUT, "e14_log.txt"), "a")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


# Data and features ###################################################################


def load(key):
    from matminer.datasets import load_dataset

    os.makedirs(RAW, exist_ok=True)
    df = load_dataset(DATASETS[key]["name"], data_home=RAW)
    if key == "elastic":
        n0 = len(df)
        df = df[df.K_VRH > 0].reset_index(drop=True)
        log(f"[{key}] dropped K_VRH <= 0: {n0 - len(df)}")
        df["y"] = np.log10(df.K_VRH.astype(float))
        df["comp"] = [s.composition for s in df.structure]
    elif key == "expt_gap":
        from pymatgen.core import Composition

        df["y"] = df["gap expt"].astype(float)
        df["comp"] = [Composition(c) for c in df.composition]
    else:
        df["y"] = df["last phdos peak"].astype(float)
        df["comp"] = [s.composition for s in df.structure]
    df["formula"] = [c.reduced_formula for c in df.comp]
    df["chemsys"] = ["-".join(sorted(e.symbol for e in c.elements)) for c in df.comp]
    return df


def build_features(key):
    """Magpie (+ LoStOP) features and float32 distance matrices, cached to e14_<key>_cache.npz."""
    path = os.path.join(OUT, f"e14_{key}_cache.npz")
    if os.path.exists(path):
        log(f"[{key}] cache exists: {path}")
        return
    from matminer.featurizers.composition import ElementProperty
    from MINOV.distance import _PETTIFOR_INDEX, get_elmd_dm

    df = load(key)
    n_raw = len(df)
    keep = np.array([all(e.symbol in _PETTIFOR_INDEX for e in c.elements) for c in df.comp])
    log(f"[{key}] raw {n_raw}; off modified-Pettifor scale: {(~keep).sum()}")
    df = df[keep].reset_index(drop=True)

    t = time.time()
    ep = ElementProperty.from_preset("magpie")
    ep.set_n_jobs(1)  # matminer's pool re-spawns workers on macOS and is slower than serial here
    magpie = np.array(ep.featurize_many(df.comp.tolist(), pbar=False), dtype=float)
    bad = np.isnan(magpie).any(axis=1)
    log(f"[{key}] Magpie {magpie.shape} in {time.time() - t:.0f}s; rows with NaN: {bad.sum()}")
    df, magpie = df[~bad].reset_index(drop=True), magpie[~bad]

    arrays = {}
    n_lostop_fail = 0
    if DATASETS[key]["structure"]:
        from joblib import Parallel, delayed
        from MINOV.distance import process_structure

        t = time.time()
        res = Parallel(n_jobs=WORKERS, backend="loky")(
            delayed(process_structure)(i, s, verbose=False) for i, s in enumerate(df.structure))
        ok = np.array([r[0] is not None for r in res])
        n_lostop_fail = int((~ok).sum())
        log(f"[{key}] LoStOP in {(time.time() - t) / 60:.1f} min; failures (no site featurized): {n_lostop_fail}"
            f" {df.formula[~ok].tolist()}")
        lostop = np.array([r[0] for r, o in zip(res, ok) if o], dtype=float)
        df, magpie = df[ok].reset_index(drop=True), magpie[ok]
        arrays["lostop"] = lostop
        arrays["D_lostop"] = squareform(pdist(lostop, metric="euclidean")).astype(np.float32)

    t = time.time()
    arrays["D_elmd"] = get_elmd_dm(df.formula.tolist()).astype(np.float32)
    log(f"[{key}] ElMD {arrays['D_elmd'].shape} in {time.time() - t:.0f}s")
    np.savez_compressed(path, magpie=magpie, y=df.y.values, formula=df.formula.values.astype(str),
                        chemsys=df.chemsys.values.astype(str), n_raw=n_raw, n_lostop_fail=n_lostop_fail, **arrays)
    log(f"[{key}] final n = {len(df)}; unique formulas {df.formula.nunique()}; chemsys {df.chemsys.nunique()}")


def read_cache(key):
    z = np.load(os.path.join(OUT, f"e14_{key}_cache.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}


# Models and novelty ##################################################################


def splits(c):
    from sklearn.model_selection import GroupKFold, KFold

    n = len(c["y"])
    for r in REPEATS:
        for f, (tr, te) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=r).split(np.arange(n))):
            yield "random", r, f, tr, te
    for f, (tr, te) in enumerate(GroupKFold(N_FOLDS).split(np.arange(n), groups=c["chemsys"])):
        yield "chemsys", 0, f, tr, te


def run_models(keys):
    from sklearn.ensemble import RandomForestRegressor

    rows = []
    for key in keys:
        c = read_cache(key)
        X = np.hstack([c["magpie"], c["lostop"]]) if "lostop" in c else c["magpie"]
        metrics = [m for m in ("elmd", "lostop") if f"D_{m}" in c]
        log(f"[{key}] models: X {X.shape}, metrics {metrics}")
        t = time.time()
        for split, r, f, tr, te in splits(c):
            rf = RandomForestRegressor(n_estimators=N_TREES, n_jobs=WORKERS, random_state=1000 * r + f)
            rf.fit(X[tr], c["y"][tr])
            trees = np.stack([tree.predict(X[te]) for tree in rf.estimators_])
            d = pd.DataFrame(dict(dataset=key, split=split, repeat=r, fold=f, idx=te, y=c["y"][te],
                                  mu=trees.mean(0), sigma=trees.std(0)))
            for m in metrics:
                D = c[f"D_{m}"]
                ref, q = D[np.ix_(tr, tr)], D[np.ix_(te, tr)]
                for s in NAMES:
                    nov = -np.asarray(SCORES[s](ref, q), dtype=float)
                    d[f"nov|{s}|{m}"] = nov
                    d[f"pct|{s}|{m}"] = rankdata(nov) / len(nov)  # within-fold percentile, pooled across folds
            rows.append(d)
            log(f"[{key}] {split} r{r} f{f} done ({time.time() - t:.0f}s)")
    out = pd.concat(rows, ignore_index=True)
    path = os.path.join(OUT, "e14_predictions.csv.gz")
    if os.path.exists(path):
        old = pd.read_csv(path)
        out = pd.concat([old[~old.dataset.isin(keys)], out], ignore_index=True)
    out.to_csv(path, index=False)
    log(f"wrote {path} ({len(out)} rows)")


# Statistics ##########################################################################


def _rk(x):
    return rankdata(x)


def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def _resid(y, x):
    x = x - x.mean()
    return (y - y.mean()) - x * ((x * (y - y.mean())).sum() / (x * x).sum())


def _auc(score_rank, pos):
    n1, n0 = pos.sum(), (~pos).sum()
    return float((score_rank[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _recall(score, pos):
    k = int(np.ceil(0.1 * len(score)))
    flagged = np.argsort(-score, kind="stable")[:k]
    return float(pos[flagged].sum() / pos.sum())


def stats_one(err, sig, cov, novs):
    """All statistics for one set of materials. novs: {score: novelty percentile}. Returns flat dict."""
    out = {}
    re, rs = _rk(err), _rk(sig)
    pos = re > np.quantile(re, 0.9)  # worst 10% by |err|
    out["rho_err_sigma"] = _pearson(re, rs)
    out["auc|sigma"] = _auc(rs, pos)
    out["rec|sigma"] = _recall(rs, pos)
    out["cov|all"] = float(cov.mean())
    er = _resid(re, rs)
    for s, nv in novs.items():
        rn = _rk(nv)
        out[f"rho|{s}"] = _pearson(re, rn)
        out[f"partial|{s}"] = _pearson(er, _resid(rn, rs))
        out[f"rho_sigma_nov|{s}"] = _pearson(rs, rn)
        out[f"auc|nov|{s}"] = _auc(rn, pos)
        out[f"rec|nov|{s}"] = _recall(rn, pos)
        comb = _rk((rs + rn) / 2)
        out[f"auc|comb|{s}"] = _auc(comb, pos)
        out[f"rec|comb|{s}"] = _recall(comb, pos)
        out[f"dauc|{s}"] = out[f"auc|comb|{s}"] - out["auc|sigma"]
        out[f"drec|{s}"] = out[f"rec|comb|{s}"] - out["rec|sigma"]
        t1, t2 = np.quantile(nv, [1 / 3, 2 / 3])
        bins = [nv <= t1, (nv > t1) & (nv <= t2), nv > t2]
        for k, b in enumerate(bins):
            out[f"cov|T{k + 1}|{s}"] = float(cov[b].mean())
            out[f"sig|T{k + 1}|{s}"] = float(np.median(sig[b]))
            out[f"err|T{k + 1}|{s}"] = float(np.median(err[b]))
        out[f"dcov|{s}"] = out[f"cov|T3|{s}"] - out[f"cov|T1|{s}"]
    return out


def lr_auc(err, sig, cov, novs):
    """Supplement to the untuned rank average: worst-10% AUC of a logistic regression on [rank sigma, rank novelty],
    5-fold cross-fitted over materials, vs the same on rank sigma alone. Point estimate only (no bootstrap)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    re, rs = _rk(err), _rk(sig)
    pos = re > np.quantile(re, 0.9)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    base = cross_val_predict(LogisticRegression(), (rs / len(rs))[:, None], pos, cv=cv, method="predict_proba")[:, 1]
    out = {"auc_lr|sigma": _auc(_rk(base), pos)}
    for s, nv in novs.items():
        X = np.c_[rs, _rk(nv)] / len(rs)
        pr = cross_val_predict(LogisticRegression(), X, pos, cv=cv, method="predict_proba")[:, 1]
        out[f"auc_lr|comb|{s}"] = _auc(_rk(pr), pos)
        out[f"dauc_lr|{s}"] = out[f"auc_lr|comb|{s}"] - out["auc_lr|sigma"]
    return out


def analyze():
    p = pd.read_csv(os.path.join(OUT, "e14_predictions.csv.gz"))
    p["err"] = (p.y - p.mu).abs()
    p["cov"] = p.err <= Z90 * p.sigma
    # Data-forced addition: ~half of expt_gap targets are exactly 0 (metals), so |err| and sigma are dominated by
    # the metal / non-metal call. Re-analyse the non-metals (y > 0) from the same predictions.
    nm = p[(p.dataset == "expt_gap") & (p.y > 0)].copy()
    nm["dataset"] = "expt_gap_nonmetal"
    p = pd.concat([p, nm], ignore_index=True)
    rng = np.random.default_rng(SEED)
    model_rows, stat_rows = [], []
    for (key, split), g in p.groupby(["dataset", "split"], sort=False):
        reps = [d.sort_values("idx") for _, d in g.groupby("repeat")]
        for r, d in enumerate(reps):
            from sklearn.metrics import mean_absolute_error, r2_score

            model_rows.append(dict(dataset=key, split=split, repeat=r, n=len(d), r2=r2_score(d.y, d.mu),
                                   mae=mean_absolute_error(d.y, d.mu), median_sigma=d.sigma.median(),
                                   coverage90=d["cov"].mean()))
        metrics = sorted({c.split("|")[2] for c in g.columns if c.startswith("pct|")})
        metrics = [m for m in metrics if g[f"pct|{NAMES[0]}|{m}"].notna().all()]
        n = len(reps[0])
        boot_idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]
        for m in metrics:
            arrs = [(d.err.values, d.sigma.values, d["cov"].values,
                     {s: d[f"pct|{s}|{m}"].values for s in NAMES}) for d in reps]
            point = pd.DataFrame([{**stats_one(*a), **lr_auc(*a)} for a in arrs])
            boots = []
            for bi in boot_idx:
                boots.append(pd.DataFrame([stats_one(e[bi], s[bi], c[bi], {k: v[bi] for k, v in nv.items()})
                                           for e, s, c, nv in arrs]).mean())
            boots = pd.DataFrame(boots)
            for col in point.columns:
                lo, hi = (boots[col].quantile(0.025), boots[col].quantile(0.975)) if col in boots else (np.nan, np.nan)
                stat_rows.append(dict(dataset=key, split=split, metric=m, stat=col, n=n,
                                      est=point[col].mean(), rep_min=point[col].min(), rep_max=point[col].max(),
                                      lo=lo, hi=hi))
            log(f"[{key}] {split} {m}: bootstrap done")
    mm = pd.DataFrame(model_rows)
    mm.to_csv(os.path.join(OUT, "e14_model_metrics.csv"), index=False)
    st = pd.DataFrame(stat_rows)
    st.to_csv(os.path.join(OUT, "e14_stats_long.csv"), index=False)
    tables(mm, st)


def _fmt(r):
    return f"{r.est:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]"


def tables(mm, st):
    log("\nModel metrics (random: mean over repeats; chemsys: one run)")
    log(mm.groupby(["dataset", "split"], sort=False)[["n", "r2", "mae", "median_sigma", "coverage90"]]
        .agg(["mean", "std"]).round(3).to_string())
    st = st.set_index(["dataset", "split", "metric", "stat"])
    keys = st.index.droplevel(3).unique()

    def row(k, stat):
        return st.loc[(*k, stat)]

    part, worst, cov = [], [], []
    for k in keys:
        for s in NAMES:
            part.append(dict(dataset=k[0], split=k[1], metric=k[2], score=SHORT[s],
                             rho_err_sigma=row(k, "rho_err_sigma").est,
                             rho_sigma_nov=row(k, f"rho_sigma_nov|{s}").est,
                             rho_err_nov=_fmt(row(k, f"rho|{s}")), partial=_fmt(row(k, f"partial|{s}")),
                             partial_est=row(k, f"partial|{s}").est, partial_lo=row(k, f"partial|{s}").lo,
                             partial_hi=row(k, f"partial|{s}").hi,
                             partial_rep_range=f"{row(k, f'partial|{s}').rep_min:+.3f}..{row(k, f'partial|{s}').rep_max:+.3f}"))
            worst.append(dict(dataset=k[0], split=k[1], metric=k[2], score=SHORT[s],
                              auc_sigma=row(k, "auc|sigma").est, auc_nov=row(k, f"auc|nov|{s}").est,
                              auc_comb=row(k, f"auc|comb|{s}").est, dauc=_fmt(row(k, f"dauc|{s}")),
                              rec_sigma=row(k, "rec|sigma").est, rec_nov=row(k, f"rec|nov|{s}").est,
                              rec_comb=row(k, f"rec|comb|{s}").est, drec=_fmt(row(k, f"drec|{s}")),
                              dauc_lr_crossfit=row(k, f"dauc_lr|{s}").est,
                              dauc_lr_rep_range=f"{row(k, f'dauc_lr|{s}').rep_min:+.3f}..{row(k, f'dauc_lr|{s}').rep_max:+.3f}"))
            c = dict(dataset=k[0], split=k[1], metric=k[2], score=SHORT[s], cov_all=row(k, "cov|all").est)
            for t in (1, 2, 3):
                r_ = row(k, f"cov|T{t}|{s}")
                c[f"cov_T{t}"] = r_.est
                c[f"cov_T{t}_lo"], c[f"cov_T{t}_hi"] = r_.lo, r_.hi
                c[f"medsig_T{t}"] = row(k, f"sig|T{t}|{s}").est
                c[f"mederr_T{t}"] = row(k, f"err|T{t}|{s}").est
            c["dcov_T3_T1"] = _fmt(row(k, f"dcov|{s}"))
            cov.append(c)
    for name, rows in [("partial", part), ("worst10", worst), ("coverage", cov)]:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT, f"e14_{name}.csv"), index=False)
        show = df.drop(columns=[c for c in df.columns if c.endswith(("_lo", "_hi", "_est"))])
        log(f"\n{name}\n" + show.round(3).to_string(index=False))


# Figure ##############################################################################


def fig():
    part = pd.read_csv(os.path.join(OUT, "e14_partial.csv"))
    cov = pd.read_csv(os.path.join(OUT, "e14_coverage.csv"))
    label = {"elastic": "Elastic log K", "expt_gap": "Expt. band gap", "expt_gap_nonmetal": "Band gap,\nnon-metals",
             "phonons": "Phonon peak"}
    mlabel = {"elmd": "ElMD", "lostop": "LoStOP"}
    combos = part[["dataset", "metric"]].drop_duplicates().values.tolist()
    order = [c for c in [("elastic", "elmd"), ("elastic", "lostop"), ("expt_gap", "elmd"),
                                                            ("expt_gap_nonmetal", "elmd"), ("phonons", "elmd"),
                         ("phonons", "lostop")] if list(c) in combos]
    ylabels = [f"{label[d]}\n{mlabel[m]}" for d, m in order]
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.9 * len(order) + 2.2), gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    y0 = np.arange(len(order))[::-1]
    width = 0.13
    ax = axes[0]
    for i, s in enumerate(NAMES):
        for j, split in enumerate(["random", "chemsys"]):
            d = part[(part.score == SHORT[s]) & (part.split == split)].set_index(["dataset", "metric"])
            d = d.loc[order]
            yy = y0 + (i - 1) * 2 * width + (j - 0.5) * width
            ax.errorbar(d.partial_est, yy, xerr=[d.partial_est - d.partial_lo, d.partial_hi - d.partial_est],
                        fmt="o" if split == "random" else "s", color=COLOR[s], mfc=COLOR[s] if split == "random" else SURFACE,
                        ms=5, capsize=0, lw=1.4, label=f"{SHORT[s]} ({'random CV' if split == 'random' else 'chem. system held out'})")
    ax.axvline(0, color=INK2, lw=1, ls="--")
    ax.set_yticks(y0, ylabels)
    ax.set_xlabel("Partial Spearman(|error|, novelty | sigma)")
    ax.set_title("Novelty vs error beyond sigma")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    for ax, split in zip(axes[1:], ["random", "chemsys"]):
        for i, s in enumerate(NAMES):
            d = cov[(cov.score == SHORT[s]) & (cov.split == split)].set_index(["dataset", "metric"]).loc[order]
            yy = y0 + (i - 1) * 0.25
            ax.hlines(yy, d.cov_T1 * 100, d.cov_T3 * 100, color=COLOR[s], lw=1.2, zorder=2)
            ax.scatter(d.cov_T1 * 100, yy, marker="o", facecolor=SURFACE, edgecolor=COLOR[s], s=30, zorder=3)
            ax.scatter(d.cov_T3 * 100, yy, marker="o", color=COLOR[s], s=30, zorder=3, label=SHORT[s])
        allc = cov[(cov.split == split) & (cov.score == "MINov")].set_index(["dataset", "metric"]).loc[order]
        ax.scatter(allc.cov_all * 100, y0 + 0.42, marker="|", color=INK, s=60, label="all test points")
        ax.scatter([], [], marker="o", facecolor=SURFACE, edgecolor=INK2, s=30, label="least-novel tercile")
        ax.scatter([], [], marker="o", color=INK2, s=30, label="most-novel tercile")
        ax.axvline(90, color=INK2, lw=1, ls="--")
        ax.set_xlim(88, 100.5)
        ax.set_yticks(y0, ylabels if split == "random" else [""] * len(order))
        ax.set_xlabel("Coverage of mu +/- 1.645 sigma (%)")
        ax.set_title("Coverage by novelty tercile\n" + ("random CV" if split == "random" else "chem. system held out"))
        ax.grid(axis="y", visible=False)
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(1.0, -0.16), ncol=3)
    fig.suptitle("E14: does novelty vs the training set flag random-forest errors beyond the ensemble spread?",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "e14_summary.png"), dpi=200)
    log("wrote e14_summary.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["features", "models", "analyze", "fig"])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    a = ap.parse_args()
    log(f"\n=== E14 run {time.strftime('%Y-%m-%d %H:%M')} stages={a.stages} datasets={a.datasets}")
    if "features" in a.stages:
        for k in a.datasets:
            build_features(k)
    if "models" in a.stages:
        run_models(a.datasets)
    if "analyze" in a.stages:
        analyze()
    if "fig" in a.stages:
        fig()
