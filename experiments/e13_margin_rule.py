"""E13: A margin band around the novelty threshold: which calls survive reference growth, and does it transfer?

Inputs: the forward-test candidate tables of E7 (Li, Y = 1990 / 2000 / 2005) and E12 (Na, ternary nitrides), plus
the distance matrices they were computed from (data/raw/processed*/U_{elmd,lostop}.npy, read with mmap_mode="r").
Scores: MINov (exact), 1-NN distance, KDE (median bw); metrics ElMD and LoStOP. A call = (candidate, metric, score).

Margin. Reference_Y = experimental entries first reported <= Y (E7 / E12 definitions, same code paths). The
reference leave-one-out (LOO) scores at Y are recomputed from the U matrices with novelty_scores.SCORES. The
candidate's percentile within them (inverse of np.percentile's linear rule; ties take the upper end of the tie
block, i.e. the empirical CDF) minus 5 is the margin in percentage points: negative = called novel. Sanity check:
recomputed sY, thrY5 and novelY5 must equal the stored ones. Secondary raw margin: log(sY / thrY5) for MINov and
KDE, log(dthr / d) for 1-NN (d = -score), positive = not novel. The percentile margin is bounded below by -5 (a
score below every LOO score has percentile 0); the raw margin is not.

Rule. A call is "uncertain" if |margin| <= c, otherwise "confident". A flip = novelY5 != novelF5 (future = full
record, E7 / E12 pool-b masking). c is calibrated per score as
  c_a05 / c_a1 / c_a2 : smallest c such that confident calls flip at most alpha = 0.5% / 1% / 2% of the time
                (first crossing over c in {0} U {|margin|}; the confident-call flip rate is not monotone in c);
  c_cap90 / c_cap95 : smallest c with >= 90% / 95% of flips inside the band (quantile of |margin| among flips).
Variants: c pooled over metrics (primary) or per metric; percentile (primary) or raw margin. Bootstrap (500
resamples of calls) 95% intervals for the Li-2000 pooled percentile c values.
Note: Li-2000's base flip rate is ~1% for MINov and KDE, so the alpha = 1% rule calibrates to c ~ 0 there; the
capture rules (c_cap90 / c_cap95) are the ones that give a non-trivial band (the figure shows c_cap90).

Calibration: Li, Y = 2000, both pools and both metrics pooled. Tests: Li 1990 / 2005, Na and nitrides at 1990 / 2000
/ 2005, each pool and metric separately (cell = chem, Y, pool, metric). Per cell: confident-call flip rate with a
Clopper-Pearson 95% CI, share of flips inside the band, share of calls labelled uncertain, the in-cell oracle c for
alpha = 1% and its % uncertain, AUC of -|margin| for flip vs hold. Leave-one-chemistry-out: calibrate on the other
two chemistries at Y = 2000 (both pools, both metrics), test the third at every Y.

Matched reliability (e13_matched.csv, percentile margin, test cells of the Li-2000 design): % of calls labelled
uncertain needed for confident-call flips <= 1% with (i) an oracle c per cell, (ii) one c for all test calls pooled,
(iii) one c that keeps every test cell <= 1% (also excluding nitrides 1990).

Outputs (experiments/results/): e13_margins.csv.gz (per call), e13_sanity.csv, e13_calibration.csv,
e13_tests.csv (all variants, cells), e13_summary.csv (score x calibration x rule aggregate), e13_matched.csv,
e13_margin_rule.png.

Single process; peak memory ~1 GB (largest block: Li candidates x reference, float64).

With e13b outputs present, the fig stage also writes e13_margin_rule_flag.png (band rule + E13b flag/widening).

Usage: uv run python experiments/e13_margin_rule.py [--stages margins analyze fig] [--flag]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
import e12_new_chemistries as e12  # noqa: E402
from novelty_scores import SCORES  # noqa: E402

OUT = os.path.join(REPO, "experiments", "results")
LI_PROC = os.path.join(REPO, "data", "raw", "processed")
METRICS = ["elmd", "lostop"]
NAMES = ["MINov (exact)", "1-NN distance", "KDE (median bw)"]
YEARS = [1990, 2000, 2005]
CHEMS = ["li", "na", "nitride"]
Q = 5
ALPHAS = (0.005, 0.01, 0.02)
CAPS = (0.9, 0.95)
TARGET = {"c_a05": 0.005, "c_a1": 0.01, "c_a2": 0.02}  # cap rules are judged against the 1% target
CAL = ("li", 2000)
N_BOOT = 500
CHEM_LABEL = {"li": "Li", "na": "Na", "nitride": "Nitr."}
POOL_LABEL = {"a_G": "GNoME", "b_Rnew": "later disc."}
METRIC_LABEL = {"elmd": "ElMD", "lostop": "LoStOP"}


def log(msg, t0=[time.perf_counter()]):  # noqa: B006
    print(f"[{time.perf_counter() - t0[0]:6.0f}s] {msg}", flush=True)


# Percentile margin ######################################################################


def percentile_of(s, loo):
    """Percentile of each s within loo: inverse of np.percentile(loo, p) (linear); a value equal to a tied block of
    loo values takes the block's upper end, so pct(s) < Q  <=>  s < np.percentile(loo, Q)."""
    xs = np.sort(np.asarray(loo, float))
    n = len(xs)
    u, first, cnt = np.unique(xs, return_index=True, return_counts=True)
    lo, hi = first.astype(float), (first + cnt - 1).astype(float)
    s = np.asarray(s, float)
    j = np.searchsorted(u, s, side="right") - 1
    jj = np.clip(j, 0, len(u) - 1)
    jn = np.clip(j + 1, 0, len(u) - 1)
    pos = np.where(s == u[jj], hi[jj],
                   hi[jj] + (s - u[jj]) / np.where(jn > jj, u[jn] - u[jj], 1.0) * (lo[jn] - hi[jj]))
    pos = np.where(j < 0, 0.0, np.where(j >= len(u) - 1, np.where(s == u[-1], hi[-1], n - 1.0), pos))
    return 100 * pos / (n - 1)


def raw_margin(name, s, thr):
    with np.errstate(divide="ignore", invalid="ignore"):
        if name == "1-NN distance":
            return np.log(-thr) - np.log(-np.asarray(s))  # d = -s; +inf for d = 0
        return np.log(np.asarray(s)) - np.log(thr)


# Stage 1: recompute reference LOO scores at Y ############################################


def li_index():
    """E7's index(): U_index merged with first years; matrix rows = positional index."""
    U = pd.read_csv(os.path.join(LI_PROC, "U_index.csv"))
    fy = pd.read_csv(os.path.join(LI_PROC, "r23_first_year.csv"))[["row", "first_year"]]
    return U.merge(fy, on="row", how="left")


def li_block(metric, a, b):
    D = np.load(os.path.join(LI_PROC, f"U_{metric}.npy"), mmap_mode="r")
    return np.asarray(D[np.ix_(a, b)], dtype=np.float64)


def setups():
    """Yield (chem, Y, stored table, metric -> (Dref loader, Dc loader), material ids of cands)."""
    U = li_index()
    r23 = U.index[U.set == "R23"].values
    g = U.index[U.set == "G"].values
    fy = U.first_year.values
    for Y in YEARS:
        refY, poolb = r23[fy[r23] <= Y], r23[fy[r23] > Y]
        cands = np.r_[g, poolb]
        st = pd.read_csv(os.path.join(OUT, f"e7_fwd{Y}_candidates.csv.gz"), float_precision="round_trip")
        st["pool"] = st.pool.replace({"b_R23new": "b_Rnew"})

        def loaders(m, refY=refY, cands=cands):
            def dref():
                D = li_block(m, refY, refY)
                D = (D + D.T) / 2
                np.fill_diagonal(D, 0.0)
                return D
            return dref, lambda: li_block(m, cands, refY)
        yield "li", Y, st, loaders, U.material_id.values[cands], len(g), len(refY)
    for chem in ("na", "nitride"):
        U = e12.load_index(chem)
        rows = U.row.values
        is_r = (U.set == "R").values
        fy, r, g = U.first_year.values, np.flatnonzero(is_r), np.flatnonzero(~is_r)
        st_all = e12.read_csv(os.path.join(OUT, f"e12_{chem}_fwd_candidates.csv.gz"), dtype={"material_id": str},
                              float_precision="round_trip")
        for Y in YEARS:
            refY, poolb = r[fy[r] <= Y], r[fy[r] > Y]
            cands = np.r_[g, poolb]
            st = st_all[st_all.Y == Y].copy()
            # E12 rows are written in cand order within (metric, score); recover the index
            st["cand"] = st.groupby(["metric", "score"]).cumcount().values

            def loaders(m, chem=chem, refY=refY, cands=cands, rows=rows):
                return (lambda: e12.ref_block(chem, m, rows[refY]),
                        lambda: e12.block(chem, m, rows[cands], rows[refY]))
            yield chem, Y, st, loaders, U.material_id.values[cands].astype(str), len(g), len(refY)


def stage_margins():
    out, sanity = [], []
    for chem, Y, st, loaders, mids, ng, nref in setups():
        for m in METRICS:
            dref, dc = loaders(m)
            Dref, Dc = dref(), dc()
            loo_q = Dref.copy()
            np.fill_diagonal(loo_q, np.inf)
            for name in NAMES:
                s = SCORES[name](Dref, Dc)
                loo = SCORES[name](Dref, loo_q)
                thr = np.percentile(loo, Q)
                d = st[(st.metric == m) & (st.score == name)].sort_values("cand")
                c = d.cand.values.astype(int)
                assert len(d) == len(s) and np.array_equal(c, np.arange(len(s)))
                assert np.array_equal(d.material_id.astype(str).values, mids[c])
                pct = percentile_of(s, loo)
                novel = s < thr
                thrF = d.thrF5.iloc[0]
                sanity.append(dict(chem=chem, Y=Y, metric=m, score=name, n_ref=nref, n_cand=len(s),
                                   thrY5_stored=d.thrY5.iloc[0], thrY5_recomputed=thr,
                                   thr_abs_diff=abs(thr - d.thrY5.iloc[0]),
                                   thr_exact=bool(thr == d.thrY5.iloc[0]),
                                   max_abs_sY_diff=float(np.max(np.abs(s - d.sY.values))),
                                   n_sY_not_exact=int(np.sum(s != d.sY.values)),
                                   n_novel_mismatch=int(np.sum(novel != d.novelY5.values.astype(bool))),
                                   n_sign_mismatch=int(np.sum((pct < Q) != novel)),
                                   n_loo_ties_at_thr=int(np.sum(loo == thr)),
                                   thrF5=thrF, thrF5_pct_on_Y_scale=float(percentile_of([thrF], loo)[0]),
                                   frac_cand_below_min_loo=float(np.mean(s < loo.min()))))
                out.append(pd.DataFrame(dict(
                    chem=chem, Y=Y, metric=m, score=name, pool=np.where(c < ng, "a_G", "b_Rnew"), cand=c,
                    material_id=mids[c], sY=s, sF=d.sF.values, thrY5=thr, thrF5=thrF, novelY5=novel,
                    novelF5=d.novelF5.values.astype(bool), pctY=pct, margin=pct - Q,
                    raw_margin=raw_margin(name, s, thr))))
            del Dref, Dc, loo_q
            log(f"{chem} Y={Y} {m}: n_ref={nref}, n_cand={len(mids)}")
    df = pd.concat(out, ignore_index=True)
    df["flip"] = df.novelY5 != df.novelF5
    df.to_csv(os.path.join(OUT, "e13_margins.csv.gz"), index=False)
    sa = pd.DataFrame(sanity)
    sa.to_csv(os.path.join(OUT, "e13_sanity.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 4):
        print(sa.to_string())


# Stage 2: calibrate and test #############################################################


def cp(k, n):
    if n == 0:
        return np.nan, np.nan
    return (beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0, beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)


def calibrate(a, f):
    """a = |margin|, f = flip (bool). Returns dict of c values."""
    a, f = np.asarray(a, float), np.asarray(f, bool)
    o = np.argsort(a, kind="stable")
    a_s, f_s = a[o], f[o]
    cs = np.unique(np.r_[0.0, a_s[np.isfinite(a_s)]])
    k = np.searchsorted(a_s, cs, side="right")  # calls with |m| <= c (uncertain)
    n_conf = len(a) - k
    f_conf = f.sum() - np.r_[0, np.cumsum(f_s)][k]
    rate = np.where(n_conf > 0, f_conf / np.maximum(n_conf, 1), 0.0)
    res = {}
    for al in ALPHAS:
        ok = np.flatnonzero(rate <= al)
        res[f"c_a{al * 100:g}".replace("0.", "0")] = float(cs[ok[0]]) if len(ok) else np.inf
    fa = np.sort(a[f])
    for cap in CAPS:
        res[f"c_cap{int(cap * 100)}"] = float(fa[int(np.ceil(cap * len(fa))) - 1]) if len(fa) else 0.0
    return res


GRID = np.round(np.arange(0, 95.0001, 0.01), 2)


def rate_curve(a, f, grid=GRID):
    """Confident-call flips and counts at each c in grid (confident = |m| > c)."""
    a, f = np.asarray(a, float), np.asarray(f, bool)
    o = np.argsort(a, kind="stable")
    k = np.searchsorted(a[o], grid, side="right")
    return len(a) - k, f.sum() - np.r_[0, np.cumsum(f[o])][k]


def matched(df):
    """Per score: % uncertain needed for confident-call flips <= 1%, (i) per-cell oracle, (ii) one c for the
    pooled test cells, (iii) one c that keeps every test cell <= 1% (also without nitride 1990)."""
    test = df[~((df.chem == CAL[0]) & (df.Y == CAL[1]))]
    rows = []
    for name in NAMES:
        d = test[test.score == name]
        n_all = len(d)
        curves = {k: rate_curve(x.abs_pct, x.flip) + (len(x),) for k, x in d.groupby(["chem", "Y", "pool", "metric"])}
        nc = sum(v[0] for v in curves.values())
        fc = sum(v[1] for v in curves.values())
        rec = dict(score=name, n_calls=n_all, base_flip_pct=100 * d.flip.mean())
        unc_cell = 0
        for k, (n_c, f_c, n) in curves.items():
            ok = np.flatnonzero(f_c <= 0.01 * n_c)
            unc_cell += n - n_c[ok[0]]
        rec["oracle_per_cell_uncertain_pct"] = 100 * unc_cell / n_all
        i = np.flatnonzero(fc <= 0.01 * nc)[0]
        rec.update(pooled_c=GRID[i], pooled_uncertain_pct=100 * (1 - nc[i] / n_all))
        for tag, keys in (("all", list(curves)), ("no_nitride1990", [k for k in curves if k[:2] != ("nitride", 1990)])):
            ok = np.all([curves[k][1] <= 0.01 * curves[k][0] for k in keys], axis=0)
            i = np.flatnonzero(ok)
            nn = sum(curves[k][2] for k in keys)
            if len(i):
                rec[f"allcells_{tag}_c"] = GRID[i[0]]
                rec[f"allcells_{tag}_uncertain_pct"] = 100 * (1 - sum(curves[k][0][i[0]] for k in keys) / nn)
            else:
                rec[f"allcells_{tag}_c"] = np.nan
                rec[f"allcells_{tag}_uncertain_pct"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def evaluate_cell(a, f, c):
    a, f = np.asarray(a, float), np.asarray(f, bool)
    conf = a > c
    n, nc, kf = len(a), int(conf.sum()), int((f & conf).sum())
    lo, hi = cp(kf, nc)
    return dict(n=n, n_flip=int(f.sum()), n_conf=nc, n_conf_flip=kf, conf_flip_rate=kf / nc if nc else np.nan,
                conf_flip_lo=lo, conf_flip_hi=hi, capture=(f & ~conf).sum() / f.sum() if f.any() else np.nan,
                frac_uncertain=1 - nc / n)


def auc_flip(a, f):
    f = np.asarray(f, bool)
    if not 0 < f.sum() < len(f):
        return np.nan
    a = np.nan_to_num(np.asarray(a, float), posinf=1e300)
    return roc_auc_score(f, -a)


def stage_analyze():
    df = e12.read_csv(os.path.join(OUT, "e13_margins.csv.gz"), dtype={"material_id": str})
    df["abs_pct"] = df.margin.abs()
    df["abs_raw"] = df.raw_margin.abs()
    cells = ["chem", "Y", "pool", "metric"]
    # calibration sets
    calsets = {"li2000": (df.chem == CAL[0]) & (df.Y == CAL[1])}
    for ch in CHEMS:
        calsets[f"loco_no_{ch}"] = (df.chem != ch) & (df.Y == 2000)
    cal_rows, test_rows = [], []
    for cal, sel in calsets.items():
        held = cal.replace("loco_no_", "") if cal.startswith("loco") else None
        for name in NAMES:
            for mcol in ("abs_pct", "abs_raw"):
                for scope in ("pooled", "by_metric"):
                    groups = [(None, sel & (df.score == name))] if scope == "pooled" else \
                        [(m, sel & (df.score == name) & (df.metric == m)) for m in METRICS]
                    cvals = {}
                    for m, s in groups:
                        d = df[s]
                        cv = calibrate(d[mcol], d.flip)
                        cvals[m] = cv
                        cal_rows.append(dict(calibration=cal, score=name, margin=mcol[4:], scope=scope,
                                             metric=m or "both", n=len(d), n_flip=int(d.flip.sum()), **cv))
                    # test cells
                    if held is None:
                        tsel = (df.score == name) & ~((df.chem == CAL[0]) & (df.Y == CAL[1]))
                        tsel_in = (df.score == name) & (df.chem == CAL[0]) & (df.Y == CAL[1])
                    else:
                        tsel = (df.score == name) & (df.chem == held)
                        tsel_in = None
                    for role, ts in (("test", tsel), ("in_sample", tsel_in)):
                        if ts is None:
                            continue
                        for key, d in df[ts].groupby(cells):
                            cv = cvals[None] if scope == "pooled" else cvals[key[3]]
                            orc = calibrate(d[mcol], d.flip)
                            for rule, c in cv.items():
                                ev = evaluate_cell(d[mcol], d.flip, c)
                                test_rows.append(dict(calibration=cal, role=role, score=name, margin=mcol[4:],
                                                      scope=scope, rule=rule, c=c, **dict(zip(cells, key)), **ev,
                                                      c_oracle_a1=orc["c_a1"],
                                                      frac_uncertain_oracle_a1=evaluate_cell(
                                                          d[mcol], d.flip, orc["c_a1"])["frac_uncertain"],
                                                      auc=auc_flip(d[mcol], d.flip),
                                                      flip_rate_all=d.flip.mean(),
                                                      n_novelY=int(d.novelY5.sum()),
                                                      n_flip_novel_to_not=int((d.flip & d.novelY5).sum())))
    cal = pd.DataFrame(cal_rows)
    # bootstrap (over calls) of the Li-2000 pooled percentile-margin c values
    rng = np.random.default_rng(0)
    for name in NAMES:
        d = df[calsets["li2000"] & (df.score == name)]
        a, f = d.abs_pct.values, d.flip.values
        B = pd.DataFrame([calibrate(a[k], f[k]) for k in (rng.integers(0, len(a), len(a)) for _ in range(N_BOOT))])
        k = (cal.calibration == "li2000") & (cal.score == name) & (cal.margin == "pct") & (cal.scope == "pooled")
        for col in B:
            cal.loc[k, f"{col}_boot_lo"] = np.percentile(B[col], 2.5)
            cal.loc[k, f"{col}_boot_hi"] = np.percentile(B[col], 97.5)
    t = pd.DataFrame(test_rows)
    t["target"] = t.rule.map(TARGET).fillna(0.01)
    t["meets_target"] = t.conf_flip_rate <= t.target
    t["ci_excludes_target"] = t.conf_flip_lo > t.target
    cal.to_csv(os.path.join(OUT, "e13_calibration.csv"), index=False)
    t.to_csv(os.path.join(OUT, "e13_tests.csv"), index=False)
    # aggregate over test cells
    g = t[t.role == "test"].groupby(["calibration", "score", "margin", "scope", "rule"])
    summ = g.apply(lambda x: pd.Series(dict(
        n_cells=len(x), cells_meet=int(x.meets_target.sum()), cells_ci_excl=int(x.ci_excludes_target.sum()),
        median_conf_flip_pct=100 * x.conf_flip_rate.median(), max_conf_flip_pct=100 * x.conf_flip_rate.max(),
        pooled_conf_flip_pct=100 * x.n_conf_flip.sum() / x.n_conf.sum(),
        pooled_capture_pct=100 * (x.n_flip.sum() - x.n_conf_flip.sum()) / x.n_flip.sum(),
        pooled_uncertain_pct=100 * (1 - x.n_conf.sum() / x.n.sum()),
        median_uncertain_pct=100 * x.frac_uncertain.median(),
        pooled_uncertain_oracle_a1_pct=100 * (x.frac_uncertain_oracle_a1 * x.n).sum() / x.n.sum(),
        median_auc=x.auc.median(), min_auc=x.auc.min())), include_groups=False).reset_index()
    summ.to_csv(os.path.join(OUT, "e13_summary.csv"), index=False)
    mt = matched(df)
    mt.to_csv(os.path.join(OUT, "e13_matched.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 500,
                           "display.precision", 3):
        print(cal[(cal.calibration == "li2000")].to_string())
        print(summ.to_string())
        print(mt.to_string())


# Stage 3: figure ########################################################################


def stage_fig(rule="c_cap90", flag=False):
    """flag=True: cells flagged by E13b (rarefaction slope rule) use the widened band c + |pct_hat - 5|
    (e13b_widen.csv); flagged cells are drawn with a dark outline / hatched bars. Writes e13_margin_rule_flag.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import make_report_figures as mrf  # sets the report rcParams; palette
    SERIES = {"MINov (exact)": mrf.SERIES["MINov"], "1-NN distance": mrf.SERIES["1-NN distance"],
              "KDE (median bw)": mrf.SERIES["Gaussian KDE, median bandwidth"]}
    LABEL = {"MINov (exact)": "MINov", "1-NN distance": "1-NN distance",
             "KDE (median bw)": "Gaussian KDE, median bandwidth"}
    t = pd.read_csv(os.path.join(OUT, "e13_tests.csv"))
    t = t[(t.calibration == "li2000") & (t.role == "test") & (t.margin == "pct") & (t.scope == "pooled")
          & (t.rule == rule)].copy()
    t["flagged"] = False
    if flag:
        w = pd.read_csv(os.path.join(OUT, "e13b_widen.csv"))
        t = t.merge(w[["chem", "Y", "metric", "score", "pool", "flag", "n_conf", "n_conf_flip"]].rename(
            columns={"n_conf": "w_conf", "n_conf_flip": "w_flip"}), on=["chem", "Y", "metric", "score", "pool"])
        t["flagged"] = t.flag.astype(bool)
        f = t.flagged
        t.loc[f, "n_conf"], t.loc[f, "n_conf_flip"] = t.loc[f, "w_conf"], t.loc[f, "w_flip"]
        t.loc[f, "conf_flip_rate"] = t.loc[f, "n_conf_flip"] / t.loc[f, "n_conf"]
        t.loc[f, "frac_uncertain"] = 1 - t.loc[f, "n_conf"] / t.loc[f, "n"]
        ci = [cp(int(a), int(b)) for a, b in zip(t.loc[f, "n_conf_flip"], t.loc[f, "n_conf"])]
        t.loc[f, "conf_flip_lo"], t.loc[f, "conf_flip_hi"] = [c[0] for c in ci], [c[1] for c in ci]
    cal = pd.read_csv(os.path.join(OUT, "e13_calibration.csv"))
    cal = cal[(cal.calibration == "li2000") & (cal.margin == "pct") & (cal.scope == "pooled")].set_index("score")
    order = [("li", 1990), ("li", 2005), ("na", 1990), ("na", 2000), ("na", 2005), ("nitride", 1990),
             ("nitride", 2000), ("nitride", 2005)]
    facets = [(p, m) for p in ("a_G", "b_Rnew") for m in METRICS]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), sharey="row")
    x = np.arange(len(order))
    w = 0.24

    def get(d, col):
        return np.array([d.loc[o, col] if o in d.index else np.nan for o in order]) * 100

    for j, (p, m) in enumerate(facets):
        for i in range(2):
            ax = axes[i, j]
            for k, name in enumerate(NAMES):
                d = t[(t.score == name) & (t.pool == p) & (t.metric == m)].set_index(["chem", "Y"])
                xx = x + (k - 1) * w
                if i == 0:
                    ax.vlines(xx, get(d, "conf_flip_lo"), get(d, "conf_flip_hi"), color=SERIES[name], lw=1.2,
                              alpha=0.5)
                    ax.scatter(xx, get(d, "flip_rate_all"), s=22, facecolor="none", edgecolor=SERIES[name],
                               lw=1, zorder=3)
                    fl = get(d, "flagged").astype(bool) if flag else np.zeros(len(order), bool)
                    ax.scatter(xx, get(d, "conf_flip_rate"), s=28, color=SERIES[name], zorder=4,
                               edgecolor=np.where(fl, mrf.INK, mrf.SURFACE), linewidths=np.where(fl, 1.4, 0.5),
                               label=f"{LABEL[name]}: band |margin| <= {cal.loc[name, rule]:.2f} pp")
                else:
                    fl = get(d, "flagged").astype(bool) if flag else np.zeros(len(order), bool)
                    ax.bar(xx, get(d, "frac_uncertain"), w, color=SERIES[name], edgecolor=mrf.SURFACE, linewidth=1,
                           hatch=None)
                    if fl.any():
                        ax.bar(xx[fl], get(d, "frac_uncertain")[fl], w, color="none", edgecolor=mrf.INK,
                               linewidth=0.8, hatch="///")
            if i == 0:
                ax.axhline(1, color=mrf.INK2, ls="--", lw=1)
                ax.set_yscale("symlog", linthresh=1, linscale=1.5)
                ax.set_yticks([0, 0.5, 1, 2, 5, 10, 20, 40], ["0", "0.5", "1", "2", "5", "10", "20", "40"])
                ax.set_ylim(0, 50)
                ax.set_title(f"{'GNoME candidates' if p == 'a_G' else 'Later discoveries'}, {METRIC_LABEL[m]}",
                             fontsize=10)
            else:
                ax.set_ylim(0, 100 if flag else 60)
            ax.set_xticks(x, [f"{CHEM_LABEL[c]}\n{y}" for c, y in order], fontsize=7.5)
            ax.grid(axis="x", visible=False)
            ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("% of confident calls that flipped\n(open: % of all calls, no band)")
    axes[1, 0].set_ylabel("% of calls labelled uncertain")
    axes[0, 0].text(-0.45, 1.08, "1% target", ha="left", va="bottom", fontsize=8, color=mrf.INK2)
    h, lab = axes[0, 0].get_legend_handles_labels()
    h = [plt.Line2D([], [], marker="o", ls="", color=SERIES[n], markersize=6) for n in NAMES]
    fig.legend(h, lab, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.935), ncol=3, fontsize=8.5)
    capname = {"c_cap90": "90%", "c_cap95": "95%"}.get(rule, rule)
    fig.suptitle(f"Margin band calibrated on Li, Y = 2000 (band holds {capname} of flips there), applied to other "
                 "split years and chemistries.\nMargin = percentile of the score among reference leave-one-out "
                 "scores at Y, minus 5; whiskers = 95% Clopper-Pearson CI",
                 x=0.01, ha="left", fontsize=9.5, color=mrf.INK2)
    if flag:
        fig.text(0.99, 0.935, "dark outline / hatched: cell flagged at Y by threshold rarefaction (E13b);\n"
                 "band widened to c + |extrapolated threshold shift|", ha="right", va="top", fontsize=8,
                 color=mrf.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, "e13_margin_rule_flag.png" if flag else "e13_margin_rule.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["margins", "analyze", "fig"])
    ap.add_argument("--flag", action="store_true", help="figure with the E13b rarefaction flag (needs e13b outputs)")
    args = ap.parse_args()
    if "margins" in args.stages:
        stage_margins()
    if "analyze" in args.stages:
        stage_analyze()
    if "fig" in args.stages:
        stage_fig()
        if args.flag or os.path.exists(os.path.join(OUT, "e13b_widen.csv")):
            stage_fig(flag=True)


if __name__ == "__main__":
    main()
