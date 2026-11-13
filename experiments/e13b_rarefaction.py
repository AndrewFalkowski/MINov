"""E13b: Can we tell at time Y, without the future, that Reference_Y is too unconverged for margin calls?

Rarefaction of the novelty threshold. For each chemistry x Y x metric (E13 / E7 / E12 reference definitions),
Reference_Y is subsampled at fractions f in FRACS (DRAWS draws each; f = 1 is the full reference), every score is
refit on the subsample and its 5% leave-one-out (LOO) threshold recomputed. The threshold is expressed
  - on the full-Y percentile scale (percentile among the full Reference_Y LOO scores, E13's percentile_of), and
  - as the raw ratio thr(f) / thr(1) (1-NN: ratio of the distance thresholds, -thr).
Per group (chem x Y x metric x score, 54 groups), with n(f) = round(f * n_Y), OLS over f in SLOPE_FRACS of the
draw-mean against log n:
  slope_pct  = d percentile / d log n  (pp per e-fold of reference size),
  elasticity = d log thr / d log n     (log of the raw ratio; ~1 for a sum-type density under random thinning).
Extrapolation to the future size n_F (the full record; uses the future size only, not future data):
  pct_hat = 5 + slope_pct * log(n_F / n_Y),  logratio_hat = elasticity * log(n_F / n_Y).
Realized: thrF5 percentile on the Y scale and log(thrF5 / thrY5) (from e13_sanity.csv); the group's confident-call
flip rate under E13's band (Li-2000 calibration, percentile margin, pooled c, rule c_cap90), max over the two pools
and pooled; a group is "bad" if either pool's confident flip rate exceeds 1%.

Tests
 1. Spearman (bootstrap 95% CI over groups, 2000 resamples) of each predictor (slope_pct, elasticity, pct_hat,
    excess elasticity = elasticity - 1 for MINov / KDE, n_Y) against realized drift and flip rates; all 54 groups and
    per score (18 each).
 2. Flag rules, calibrated on Li (18 groups) per score, tested on Na + nitrides (36 groups):
      flag if predictor >= s, s = smallest Li value among bad Li groups (catch every bad Li group);
      predictors: slope_pct, elasticity, n_Y (flag if n_Y <= largest bad Li n_Y).
    Flagged group -> (a) whole cell unreliable, or (b) band widened to c + |pct_hat - 5| (percentile margin).
    Reported: bad groups caught / flagged good groups, and the confident flip rate / % uncertain after (b).
 3. Reference size alone (n_Y) vs slope: same tests.

Outputs (experiments/results/): e13b_rarefaction.csv (per draw), e13b_groups.csv (per group predictors, realized,
flags), e13b_spearman.csv, e13b_flag_rules.csv, e13b_widen.csv, e13b_rarefaction.png.

Machine limits: 3 worker processes; matrices read with mmap_mode="r"; one reference block per worker at a time.

Usage: uv run python experiments/e13b_rarefaction.py [--stages draws analyze fig] [--workers 3]
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
import e12_new_chemistries as e12  # noqa: E402
import e13_margin_rule as e13  # noqa: E402
from novelty_scores import SCORES  # noqa: E402

OUT = e13.OUT
NAMES = e13.NAMES
FRACS = (0.2, 0.3, 0.5, 0.7, 0.85, 1.0)
SLOPE_FRACS = (0.5, 0.7, 0.85, 1.0)
DRAWS = 20
SEED = 0
N_BOOT = 2000
RULE = "c_cap90"


def log(msg, t0=[time.perf_counter()]):  # noqa: B006
    print(f"[{time.perf_counter() - t0[0]:6.0f}s] {msg}", flush=True)


# Reference blocks (same definitions as E13 / E7 / E12) ##################################


def ref_info():
    """(chem, Y) -> (loader(metric) -> Dref, n_Y, n_F)."""
    info = {}
    U = e13.li_index()
    r23 = U.index[U.set == "R23"].values
    fy = U.first_year.values
    for Y in e13.YEARS:
        refY = r23[fy[r23] <= Y]

        def load(m, refY=refY):
            D = e13.li_block(m, refY, refY)
            D = (D + D.T) / 2
            np.fill_diagonal(D, 0.0)
            return D
        info[("li", Y)] = (load, len(refY), len(r23))
    for chem in ("na", "nitride"):
        U = e12.load_index(chem)
        rows = U.row.values
        r = np.flatnonzero((U.set == "R").values)
        fy = U.first_year.values
        for Y in e13.YEARS:
            refY = r[fy[r] <= Y]
            info[(chem, Y)] = ((lambda m, chem=chem, refY=refY, rows=rows: e12.ref_block(chem, m, rows[refY])),
                               len(refY), len(r))
    return info


_CACHE = {}


def job(task):
    """task = (chem, Y, metric, f): DRAWS subsamples (1 for f = 1); LOO 5% threshold of every score."""
    chem, Y, m, f = task
    key = (chem, Y, m)
    if key not in _CACHE:
        _CACHE.clear()
        D = ref_info()[(chem, Y)][0](m)
        q = D.copy()
        np.fill_diagonal(q, np.inf)
        full = {name: np.sort(SCORES[name](D, q)) for name in NAMES}
        _CACHE[key] = (D, full)
    D, full = _CACHE[key]
    n = len(D)
    k = int(round(f * n))
    out = []
    for d in range(1 if f == 1.0 else DRAWS):
        rng = np.random.default_rng((SEED, Y, e13.CHEMS.index(chem), e13.METRICS.index(m), int(f * 1000), d))
        idx = np.arange(n) if f == 1.0 else np.sort(rng.choice(n, k, replace=False))
        ref = D[np.ix_(idx, idx)]
        q = ref.copy()
        np.fill_diagonal(q, np.inf)
        for name in NAMES:
            loo = SCORES[name](ref, q)
            thr = np.percentile(loo, e13.Q)
            thr1 = np.percentile(full[name], e13.Q)
            ratio = (-thr) / (-thr1) if name == "1-NN distance" else thr / thr1
            out.append(dict(chem=chem, Y=Y, metric=m, score=name, f=f, draw=d, n_sub=k, n_Y=n, thr=thr,
                            pct_on_Y=float(e13.percentile_of([thr], full[name])[0]), ratio=ratio))
    return out


def stage_draws(workers):
    info = ref_info()
    tasks = [(c, Y, m, f) for (c, Y) in info for m in e13.METRICS for f in FRACS]
    rows = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap(job, tasks)):  # ordered: consecutive tasks share a reference block
            rows += r
            if (i + 1) % 18 == 0:
                log(f"{i + 1}/{len(tasks)} tasks")
    df = pd.DataFrame(rows)
    nF = {k: v[2] for k, v in info.items()}
    df["n_F"] = [nF[(c, y)] for c, y in zip(df.chem, df.Y)]
    df.to_csv(os.path.join(OUT, "e13b_rarefaction.csv"), index=False)


# Analysis ################################################################################


def boot_spearman(x, y, rng):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho = spearmanr(x, y)[0]
    b = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(x), len(x))
        if np.ptp(x[k]) > 0 and np.ptp(y[k]) > 0:
            b.append(spearmanr(x[k], y[k])[0])
    return dict(n=len(x), rho=rho, lo=np.nanpercentile(b, 2.5), hi=np.nanpercentile(b, 97.5))


def groups_table():
    r = pd.read_csv(os.path.join(OUT, "e13b_rarefaction.csv"))
    m = r.groupby(["chem", "Y", "metric", "score", "f"]).agg(
        n_sub=("n_sub", "first"), n_Y=("n_Y", "first"), n_F=("n_F", "first"), pct=("pct_on_Y", "mean"),
        pct_sd=("pct_on_Y", "std"), logratio=("ratio", lambda v: np.mean(np.log(v)))).reset_index()
    rows = []
    for key, d in m.groupby(["chem", "Y", "metric", "score"]):
        s = d[d.f.isin(SLOPE_FRACS)]
        x = np.log(s.n_sub.values)
        slope_pct = np.polyfit(x, s.pct.values, 1)[0]
        elast = np.polyfit(x, s.logratio.values, 1)[0]
        nY, nF = d.n_Y.iloc[0], d.n_F.iloc[0]
        g = np.log(nF / nY)
        rec = dict(zip(["chem", "Y", "metric", "score"], key), n_Y=nY, n_F=nF, slope_pct=slope_pct, elasticity=elast,
                   excess_elasticity=elast - (0 if key[3] == "1-NN distance" else 1),
                   pct_hat=5 + slope_pct * g, logratio_hat=elast * g,
                   pct_f05=d[d.f == 0.5].pct.iloc[0], pct_f02=d[d.f == 0.2].pct.iloc[0])
        rows.append(rec)
    G = pd.DataFrame(rows)
    sa = pd.read_csv(os.path.join(OUT, "e13_sanity.csv"))
    sa["real_pct"] = sa.thrF5_pct_on_Y_scale
    sa["real_logratio"] = np.log(sa.thrF5 / sa.thrY5_recomputed)  # 1-NN: both negative -> distance ratio
    G = G.merge(sa[["chem", "Y", "metric", "score", "real_pct", "real_logratio"]], on=["chem", "Y", "metric", "score"])
    G["real_excess_logratio"] = G.real_logratio - np.where(G.score == "1-NN distance", 0, np.log(G.n_F / G.n_Y))
    t = pd.read_csv(os.path.join(OUT, "e13_tests.csv"))
    t = t[(t.calibration == "li2000") & (t.margin == "pct") & (t.scope == "pooled") & (t.rule == RULE)]
    fl = t.groupby(["chem", "Y", "metric", "score"]).apply(lambda x: pd.Series(dict(
        c=x.c.iloc[0], conf_flip_max=x.conf_flip_rate.max(), conf_flip_pooled=x.n_conf_flip.sum() / x.n_conf.sum(),
        flip_rate_all=x.n_flip.sum() / x.n.sum(), n_calls=x.n.sum(), n_conf=x.n_conf.sum(),
        n_conf_flip=x.n_conf_flip.sum(), unc_pooled=1 - x.n_conf.sum() / x.n.sum())), include_groups=False)
    G = G.merge(fl.reset_index(), on=["chem", "Y", "metric", "score"])
    G["bad"] = G.conf_flip_max > 0.01
    return G


def widen_eval(G):
    """Band (b): flagged groups use c + |pct_hat - 5|; returns per group conf flip / uncertain after widening."""
    mg = e12.read_csv(os.path.join(OUT, "e13_margins.csv.gz"), dtype={"material_id": str},
                      usecols=["chem", "Y", "metric", "score", "pool", "margin", "flip"])
    mg["a"] = mg.margin.abs()
    rows = []
    for g in G.itertuples():
        d = mg[(mg.chem == g.chem) & (mg.Y == g.Y) & (mg.metric == g.metric) & (mg.score == g.score)]
        c = g.c + abs(g.pct_hat - 5)
        for p, x in d.groupby("pool"):
            conf = x.a.values > c
            rows.append(dict(chem=g.chem, Y=g.Y, metric=g.metric, score=g.score, pool=p, c_wide=c, n=len(x),
                             n_conf=int(conf.sum()), n_conf_flip=int((conf & x.flip.values).sum())))
    return pd.DataFrame(rows)


def stage_analyze():
    rng = np.random.default_rng(SEED)
    G = groups_table()
    preds = ["slope_pct", "elasticity", "excess_elasticity", "pct_hat", "logratio_hat", "pct_f05", "n_Y"]
    targets = ["real_pct", "real_logratio", "real_excess_logratio", "conf_flip_max", "conf_flip_pooled",
               "flip_rate_all"]
    sp = []
    for scope, d in [("all", G)] + list(G.groupby("score")):
        for p in preds:
            for tg in targets:
                sp.append(dict(scope=scope, predictor=p, target=tg, **boot_spearman(d[p], d[tg], rng)))
    sp = pd.DataFrame(sp)
    sp.to_csv(os.path.join(OUT, "e13b_spearman.csv"), index=False)
    # flag rules: calibrate on Li per score
    W = widen_eval(G)
    rules, flags = [], []
    for name in NAMES:
        for p, sign in (("slope_pct", 1), ("elasticity", 1), ("excess_elasticity", 1), ("pct_hat", 1),
                        ("n_Y", -1)):
            d = G[G.score == name]
            li = d[d.chem == "li"]
            v = sign * li[p]
            s = v[li.bad].min() if li.bad.any() else np.inf
            flag = sign * d[p] >= s
            for role, sel in (("li_calib", d.chem == "li"), ("test_na_nitride", d.chem != "li")):
                x, fx = d[sel], flag[sel]
                rules.append(dict(score=name, predictor=p, threshold=sign * s, role=role, n_groups=len(x),
                                  n_bad=int(x.bad.sum()), caught=int((fx & x.bad).sum()),
                                  flagged_good=int((fx & ~x.bad).sum()), n_good=int((~x.bad).sum()),
                                  flagged=", ".join(f"{a.chem}{a.Y} {a.metric}" for a in x[fx].itertuples()),
                                  missed=", ".join(f"{a.chem}{a.Y} {a.metric}" for a in x[~fx & x.bad].itertuples()),
                                  # (a) drop flagged groups: remaining confident calls
                                  dropped_conf_flip=x[~fx].n_conf_flip.sum() / max(x[~fx].n_conf.sum(), 1),
                                  dropped_calls_pct=100 * x[fx].n_calls.sum() / x.n_calls.sum(),
                                  base_conf_flip=x.n_conf_flip.sum() / x.n_conf.sum()))
            flags.append(pd.DataFrame(dict(chem=d.chem, Y=d.Y, metric=d.metric, score=name, predictor=p,
                                           flag=flag.values)))
    R = pd.DataFrame(rules)
    R.to_csv(os.path.join(OUT, "e13b_flag_rules.csv"), index=False)
    F = pd.concat(flags, ignore_index=True)
    F.to_csv(os.path.join(OUT, "e13b_flags.csv"), index=False)
    # (b) widening, slope_pct flag
    fs = F[F.predictor == "slope_pct"]
    W = W.merge(fs[["chem", "Y", "metric", "score", "flag"]], on=["chem", "Y", "metric", "score"])
    W.to_csv(os.path.join(OUT, "e13b_widen.csv"), index=False)
    G.to_csv(os.path.join(OUT, "e13b_groups.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 500,
                           "display.precision", 3):
        print(G[["chem", "Y", "metric", "score", "n_Y", "n_F", "slope_pct", "elasticity", "pct_hat", "real_pct",
                 "real_logratio", "logratio_hat", "conf_flip_max", "bad"]].to_string())
        print(sp[sp.target.isin(["real_pct", "real_logratio", "conf_flip_max"])].to_string())
        print(R.drop(columns=["flagged"]).to_string())


def stage_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import make_report_figures as mrf
    COL = {"MINov (exact)": mrf.SERIES["MINov"], "1-NN distance": mrf.SERIES["1-NN distance"],
           "KDE (median bw)": mrf.SERIES["Gaussian KDE, median bandwidth"]}
    LABEL = {"MINov (exact)": "MINov", "1-NN distance": "1-NN distance",
             "KDE (median bw)": "Gaussian KDE, median bandwidth"}
    MARK = {"li": "o", "na": "s", "nitride": "^"}
    G = pd.read_csv(os.path.join(OUT, "e13b_groups.csv"))
    G["ratio_hat"], G["real_ratio"] = np.exp(G.logratio_hat), np.exp(G.real_logratio)
    r = pd.read_csv(os.path.join(OUT, "e13b_rarefaction.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # (a) rarefaction curves, MINov ElMD, raw ratio vs n / n_Y
    ax = axes[0]
    mm = r[(r.score == "MINov (exact)") & (r.metric == "elmd")].groupby(["chem", "Y", "f"]).ratio.mean().reset_index()
    for (c, Y), d in mm.groupby(["chem", "Y"]):
        bad = bool(G[(G.chem == c) & (G.Y == Y) & (G.metric == "elmd") & (G.score == "MINov (exact)")].bad.iloc[0])
        ax.plot(d.f, d.ratio, marker=MARK[c], ms=4, lw=1.2, color=mrf.INK if bad else mrf.CONTEXT,
                label=f"{e13.CHEM_LABEL[c]} {Y}" if bad else None)
    ax.plot([0.2, 1], [0.2, 1], ls=":", color=mrf.INK2, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.2, 0.3, 0.5, 0.7, 1.0], ["0.2", "0.3", "0.5", "0.7", "1"])
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)
    ax.set_yticks([0.1, 0.2, 0.5, 1.0], ["0.1", "0.2", "0.5", "1"])
    ax.set_xlabel("fraction of Reference_Y kept")
    ax.set_ylabel("5% LOO threshold / full-Y threshold")
    ax.set_title("MINov, ElMD: threshold rarefaction", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, title="dark = flagged bad by E13", title_fontsize=7.5)
    for ax, (xc, xl), (yc, yl), ylog in (
            (axes[1], ("ratio_hat", "extrapolated thr(future) / thr(Y), from rarefaction at Y"),
             ("real_ratio", "realized thr(future) / thr(Y)"), False),
            (axes[2], ("slope_pct", "slope: d percentile / d log n (f 0.5 to 1)"),
             ("conf_flip_max", "% confident calls flipped (worse pool)"), False)):
        for name in NAMES:
            for c in MARK:
                d = G[(G.score == name) & (G.chem == c)]
                y = d[yc] * (100 if yc == "conf_flip_max" else 1)
                ax.scatter(d[xc], y, marker=MARK[c], s=30, color=COL[name], edgecolor=mrf.SURFACE, lw=0.5,
                           label=f"{LABEL[name]}" if c == "li" else None)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    axes[1].plot([0.55, 4.2], [0.55, 4.2], ls=":", color=mrf.INK2, lw=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    tk = [0.6, 0.8, 1, 1.5, 2, 3, 4]
    axes[1].set_xticks(tk, [f"{v:g}" for v in tk])
    axes[1].set_yticks(tk, [f"{v:g}" for v in tk])
    axes[1].set_xticks([], minor=True)
    axes[1].set_yticks([], minor=True)
    axes[1].set_title("Extrapolated vs realized threshold growth (1-NN: distance)", fontsize=10)
    axes[2].axhline(1, ls="--", color=mrf.INK2, lw=1)
    axes[2].set_yscale("symlog", linthresh=1)
    axes[2].set_yticks([0, 0.5, 1, 2, 5, 10, 20], ["0", "0.5", "1", "2", "5", "10", "20"])
    axes[2].set_ylim(-0.05, 25)
    axes[2].set_title("Slope vs confident-call flips (E13 band)", fontsize=10)
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper left", title="circle Li, square Na, triangle nitrides",
                   title_fontsize=7.5)
    fig.suptitle("Rarefaction of the 5% novelty threshold at Y (no future data) vs what happened as the reference "
                 "grew; 54 groups = chemistry x Y x metric x score", x=0.01, ha="left", fontsize=9.5, color=mrf.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "e13b_rarefaction.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["draws", "analyze", "fig"])
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    if "draws" in args.stages:
        stage_draws(min(args.workers, 3))
    if "analyze" in args.stages:
        stage_analyze()
    if "fig" in args.stages:
        stage_fig()


if __name__ == "__main__":
    main()
