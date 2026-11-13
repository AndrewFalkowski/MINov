"""E16: History replay. Did each year's new compounds come from the crowded or the sparse part of what was
eventually reported, relative to what was known at the time?

Framing: later does not mean more novel. This measures how the field ORDERED what it eventually found, not whether
discoveries are globally novel. The pool is only what was eventually reported (experimental MP-2023 entries).

Data (first-report years; float32 distance matrices read with mmap_mode="r", only the R x R block is loaded):
  Li       data/raw/processed/        U_index.csv (set R23) + r23_first_year.csv, U_elmd.npy, U_lostop.npy
  Na       data/raw/processed_na/     U_index.csv (set R, first_year column), U_elmd.npy, U_lostop.npy
  nitride  data/raw/processed_nitride/ (same layout as Na)

Part 1, novelty percentile of real discoveries. Annual steps t = 1991 ... 2013 (2013 = last first-report year).
  Reference_t = entries first reported <= t-1; Pool_t = entries first reported >= t; picks_t = entries reported in t.
  Every pool member is scored against Reference_t (score refit on Reference_t): MINov (exact), 1-NN distance,
  Gaussian KDE (median bw); metrics ElMD and LoStOP. Novelty percentile within the pool = mid-rank of novelty,
  100 = most novel; the pool mean is exactly 50, so a random order gives an expected 50 for the picks.
  Reference_t and Pool_t do not depend on the start year Y, so the steps are computed once and aggregated over
  t = Y+1 ... 2012 for Y in {1990, 2000, 2005} (weighted by picks; equivalently the mean over all picks). The last
  step (2013) is degenerate (pool = picks, mean 50 by construction) and is left out of every aggregate and figure;
  2013 entries still sit in the pools of earlier steps (2013 holds 3-4x a normal year).
  CIs: (a) bootstrap over entries, stratified by step; (b) bootstrap over steps (years), which keeps the
  within-year clustering of families. Variants: "all" and "no_known_formula" (pool members whose fkey is already
  in Reference_t, i.e. polymorphs of known compounds, dropped from pool and picks; percentiles recomputed).
  Contrasts: paired (same entries) differences MINov - 1-NN and ElMD - LoStOP. Robustness: 5-year steps from
  Y+1 (reference fixed at the start of each bin; degenerate last bin dropped).

Part 2, counterfactual order. Pool = entries first reported after Y, Reference_0 = entries <= Y, same number of
  picks per year as history. Policies: real order; MINov-greedy (lowest density first; profile refit at each
  year, frozen within the year while each pick's column is added; ties broken by larger mean distance to the
  reference); 1-NN-greedy (farthest point, same tie-break); random (20 draws). The chooser uses ElMD or LoStOP;
  coverage is counted with other descriptors: new chemical systems (element sets) and new stoichiometry
  prototypes (anonymized reduced formula), relative to Reference_0. Space groups are not in the index tables and
  are skipped. Every policy ends with the whole pool, so curves are summarised by their area (AUC over the
  fraction of picks made, 1 = everything new found at once) and by coverage at half the picks.

Outputs (experiments/results/): e16_steps.csv, e16_picks.csv.gz, e16_summary.csv, e16_contrasts.csv,
  e16_bins5.csv, e16_cf_curves.csv, e16_cf_runs.csv, e16_cf_summary.csv, e16_percentile.png, e16_counterfactual.png, e16_log.txt.

Machine limits: single process, BLAS threads capped at 2, peak RSS about 1.4 GB (Na).

Usage: uv run python experiments/e16_history_replay.py [--stages replay cf fig]
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pymatgen.core import Composition  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
from MINOV.density import MIProfile  # noqa: E402
from novelty_scores import SCORES  # noqa: E402

warnings.filterwarnings("ignore")

OUT = os.path.join(REPO, "experiments", "results")
CHEMS = ["li", "na", "nitride"]
CHEM_LABEL = {"li": "Li", "na": "Na", "nitride": "Ternary nitrides"}
METRICS = ["elmd", "lostop"]
METRIC_LABEL = {"elmd": "ElMD (composition)", "lostop": "LoStOP (structure)"}
NAMES = ["MINov (exact)", "1-NN distance", "KDE (median bw)"]
YEARS = [1990, 2000, 2005]
FIRST_STEP, LAST_STEP = min(YEARS) + 1, 2013
N_BOOT = 2000
N_RANDOM = 20
SEED = 0

# Style from experiments/make_report_figures.py
SURFACE, INK, INK2, GRID, CONTEXT = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#a8a7a1"
SERIES = {"MINov": "#2a78d6", "1-NN distance": "#eb6834", "Gaussian KDE, median bandwidth": "#1baf7a"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

LOG = []


def log(msg, t0=[time.perf_counter()]):  # noqa: B006
    line = f"[{time.perf_counter() - t0[0]:7.0f}s] {msg}"
    print(line, flush=True)
    LOG.append(line)


def peak_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)


# Data #################################################################################


def proc(chem):
    return os.path.join(REPO, "data", "raw", "processed" if chem == "li" else f"processed_{chem}")


def read_csv(path, **kw):
    """Keep the formula "NaN" (sodium nitride) as a string; only empty cells are missing."""
    kw.setdefault("keep_default_na", False)
    kw.setdefault("na_values", [""])
    return pd.read_csv(path, **kw)


def load_entries(chem):
    """Experimental reference entries with first-report year, chemical system and prototype label."""
    U = read_csv(os.path.join(proc(chem), "U_index.csv"), dtype={"material_id": str})
    if chem == "li":
        fy = read_csv(os.path.join(proc(chem), "r23_first_year.csv"))[["row", "first_year"]]
        R = U[U.set == "R23"].drop(columns=[c for c in ["first_year"] if c in U]).merge(fy, on="row", how="left")
    else:
        R = U[(U.set == "R") & U.lostop_ok.astype(bool)].copy()
    assert R.first_year.notna().all()
    R["first_year"] = R.first_year.astype(int)
    R = R.sort_values("row").reset_index(drop=True)
    R["chemsys"] = ["-".join(sorted(tok.rstrip("0123456789.") for tok in f.split())) for f in R.fkey]
    R["proto"] = [Composition(f).reduced_composition.anonymized_formula for f in R.fkey]
    return R


def load_block(chem, metric, rows):
    D = np.load(os.path.join(proc(chem), f"U_{metric}.npy"), mmap_mode="r")
    B = np.asarray(D[np.ix_(rows, rows)], dtype=np.float64)
    B = (B + B.T) / 2  # exact symmetry for MIProfile(exact)
    np.fill_diagonal(B, 0.0)
    return B


def novelty_pct(s):
    """Mid-rank novelty percentile within the pool: 100 = most novel (lowest density score); pool mean = 50."""
    return 100 * (rankdata(-s, method="average") - 0.5) / len(s)


# Part 1: replay ########################################################################


def steps_for(kind, Y=None):
    """List of (label, start, end): picks are first_year in [start, end]; reference <= start-1; pool >= start."""
    if kind == "annual":
        return [(t, t, t) for t in range(FIRST_STEP, LAST_STEP + 1)]
    out, s = [], Y + 1
    while s <= LAST_STEP:
        e = min(s + 4, LAST_STEP)
        out.append((f"{s}-{e}", s, e))
        s = e + 1
    return out


def replay(chem, R, D, metric, steps):
    """Per pick: novelty percentile in the pool, for every score and both sibling variants."""
    fy, fk = R.first_year.values, R.fkey.values
    rows, step_rows = [], []
    for label, s, e in steps:
        ref = np.flatnonzero(fy <= s - 1)
        pool = np.flatnonzero(fy >= s)
        is_pick = fy[pool] <= e
        known = np.isin(fk[pool], fk[ref])
        Dref, Dq = D[np.ix_(ref, ref)], D[np.ix_(pool, ref)]
        for name in NAMES:
            sc = SCORES[name](Dref, Dq)
            frac_zero = float(np.mean(sc == 0)) if name == "MINov (exact)" else np.nan
            for variant, keep in (("all", np.ones(len(pool), bool)), ("no_known_formula", ~known)):
                p = np.full(len(pool), np.nan)
                p[keep] = novelty_pct(sc[keep])
                pk = p[is_pick & keep]
                step_rows.append(dict(chem=chem, metric=metric, score=name, variant=variant, step=label, start=s,
                                      end=e, n_ref=len(ref), n_pool=int(keep.sum()), n_picks=len(pk),
                                      mean_pct=pk.mean() if len(pk) else np.nan,
                                      frac_known_pool=known.mean(), frac_known_picks=known[is_pick].mean(),
                                      frac_zero_density=frac_zero))
                for j in np.flatnonzero(is_pick & keep):
                    rows.append((chem, metric, name, variant, label, R.material_id.values[pool[j]], fk[pool[j]],
                                 fy[pool[j]], bool(known[j]), sc[j], p[j]))
    picks = pd.DataFrame(rows, columns=["chem", "metric", "score", "variant", "step", "material_id", "fkey",
                                        "first_year", "known_formula", "density", "pct"])
    return picks, pd.DataFrame(step_rows)


def aggregate(picks, rng):
    """Mean percentile of picks over the steps in the group; CIs over entries (stratified) and over steps."""
    groups = [g.values for _, g in picks.groupby("step")]
    n = sum(len(g) for g in groups)
    mean = sum(g.sum() for g in groups) / n
    boot = np.zeros(N_BOOT)
    for g in groups:
        boot += g[rng.integers(0, len(g), (N_BOOT, len(g)))].sum(1)
    boot /= n
    sums = np.array([g.sum() for g in groups])
    cnts = np.array([len(g) for g in groups])
    idx = rng.integers(0, len(groups), (N_BOOT, len(groups)))
    sboot = sums[idx].sum(1) / cnts[idx].sum(1)
    step_means = sums / cnts
    return dict(n_picks=n, n_steps=len(groups), mean_pct=mean,
                ci_lo=np.quantile(boot, 0.025), ci_hi=np.quantile(boot, 0.975),
                ci_steps_lo=np.quantile(sboot, 0.025), ci_steps_hi=np.quantile(sboot, 0.975),
                n_steps_below_50=int((step_means < 50).sum()))


def informative(picks, steps):
    """Drop degenerate steps where the pool is exactly the picks (the last step): their mean is 50 by construction."""
    deg = steps[steps.n_pool == steps.n_picks][["score", "variant", "step"]].assign(_deg=True)
    m = picks.merge(deg, on=["score", "variant", "step"], how="left")
    return m[m._deg.isna()].drop(columns="_deg")


def contrasts(picks, rng):
    """Paired (same entries) differences in mean percentile: score vs score and ElMD vs LoStOP."""
    out = []
    p = picks[picks.variant == "all"]
    for chem in p.chem.unique():
        for Y in YEARS:
            g = p[(p.chem == chem) & (p.first_year > Y)]
            w = g.pivot_table(index=["step", "material_id"], columns=["metric", "score"], values="pct")
            pairs = [(("elmd", "MINov (exact)"), ("elmd", "1-NN distance")),
                     (("lostop", "MINov (exact)"), ("lostop", "1-NN distance")),
                     (("elmd", "MINov (exact)"), ("lostop", "MINov (exact)")),
                     (("elmd", "1-NN distance"), ("lostop", "1-NN distance"))]
            for a, b in pairs:
                d = (w[a] - w[b]).dropna().reset_index(level=1, drop=True)
                out.append(dict(chem=chem, Y=Y, a=" ".join(a), b=" ".join(b), **aggregate(d, rng)))
    return pd.DataFrame(out).rename(columns={"mean_pct": "mean_diff", "n_steps_below_50": "n_steps_a_below_b"})


def stage_replay(chems):
    rng = np.random.default_rng(SEED)
    all_picks, all_steps, summ, bins5 = [], [], [], []
    for chem in chems:
        R = load_entries(chem)
        log(f"{chem}: {len(R)} entries; first_year {R.first_year.min()}-{R.first_year.max()}")
        for metric in METRICS:
            t0 = time.perf_counter()
            D = load_block(chem, metric, R.row.values)
            picks, steps = replay(chem, R, D, metric, steps_for("annual"))
            all_picks.append(picks)
            all_steps.append(steps)
            for Y in YEARS:
                sel = informative(picks[picks.first_year > Y], steps)
                for (name, variant), g in sel.groupby(["score", "variant"]):
                    summ.append(dict(chem=chem, Y=Y, metric=metric, score=name, variant=variant,
                                     **aggregate(g.set_index("step").pct, rng)))
                p5, s5 = replay(chem, R, D, metric, steps_for("bins5", Y))
                for (name, variant), g in informative(p5, s5).groupby(["score", "variant"]):
                    bins5.append(dict(chem=chem, Y=Y, metric=metric, score=name, variant=variant,
                                      **aggregate(g.set_index("step").pct, rng)))
            del D
            log(f"{chem} {metric}: replay done in {time.perf_counter() - t0:.0f}s, peak RSS {peak_mb()} MB")
    pd.concat(all_picks).to_csv(os.path.join(OUT, "e16_picks.csv.gz"), index=False)
    steps = pd.concat(all_steps)
    # per-step entry-bootstrap CI for the figure
    picks = pd.concat(all_picks)
    ci = []
    for key, g in picks.groupby(["chem", "metric", "score", "variant", "step"]):
        v = g.pct.values
        b = v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(1)
        ci.append((*key, np.quantile(b, 0.025), np.quantile(b, 0.975)))
    ci = pd.DataFrame(ci, columns=["chem", "metric", "score", "variant", "step", "ci_lo", "ci_hi"])
    steps["step"] = steps.step.astype(str)
    ci["step"] = ci.step.astype(str)
    steps.merge(ci, on=["chem", "metric", "score", "variant", "step"], how="left").to_csv(
        os.path.join(OUT, "e16_steps.csv"), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "e16_summary.csv"), index=False)
    deg = steps[steps.n_pool == steps.n_picks].step.unique().tolist()
    contrasts(picks[~picks.step.astype(str).isin(deg)], rng).to_csv(os.path.join(OUT, "e16_contrasts.csv"),
                                                                     index=False)
    pd.DataFrame(bins5).to_csv(os.path.join(OUT, "e16_bins5.csv"), index=False)


# Part 2: counterfactual order ############################################################


def greedy_order(D, ref0, pool, ks, kind):
    """Pick sum(ks) pool members, ks[i] per year: refit at each year, sequential within the year."""
    ref = list(ref0)
    rem = np.array(pool)
    order = []
    for k in ks:
        Dq = D[np.ix_(rem, ref)]
        sumd = Dq.sum(1)
        if kind == "MINov":
            prof = MIProfile(D[np.ix_(ref, ref)])
            key = prof.density(Dq)
        else:
            key = -Dq.min(1)  # higher = less novel, as for density
        for _ in range(k):
            j = np.lexsort((-sumd / len(ref), key))[0]  # lowest key, then largest mean distance
            pick = rem[j]
            order.append(pick)
            ref.append(pick)
            keep = np.ones(len(rem), bool)
            keep[j] = False
            rem, key, sumd = rem[keep], key[keep], sumd[keep]
            d = D[rem, pick]
            sumd = sumd + d
            key = key + prof.weight(d) if kind == "MINov" else np.maximum(key, -d)
    return np.array(order)


def coverage_curve(order, ks, labels, known0):
    """Cumulative number of new descriptor values (not in Reference_0) after each year."""
    seen, out, i = set(), [], 0
    for k in ks:
        for p in order[i:i + k]:
            if labels[p] not in known0:
                seen.add(labels[p])
        i += k
        out.append(len(seen))
    return np.array(out)


def curve_stats(cum_picks, cum_new):
    x = np.r_[0, cum_picks / cum_picks[-1]]
    y = np.r_[0, cum_new / cum_new[-1]]
    return np.trapezoid(y, x), float(np.interp(0.5, x, y))


def stage_cf(chems):
    curves, summ = [], []
    for chem in chems:
        R = load_entries(chem)
        fy = R.first_year.values
        for metric in METRICS:
            D = load_block(chem, metric, R.row.values)
            for Y in YEARS:
                t0 = time.perf_counter()
                ref0 = np.flatnonzero(fy <= Y)
                years = np.arange(Y + 1, LAST_STEP + 1)
                ks = [int((fy == t).sum()) for t in years]
                real = np.concatenate([np.flatnonzero(fy == t) for t in years])
                orders = {("real", 0): real,
                          ("MINov-greedy", 0): greedy_order(D, ref0, real, ks, "MINov"),
                          ("1-NN-greedy", 0): greedy_order(D, ref0, real, ks, "1-NN")}
                rng = np.random.default_rng((SEED, Y, CHEMS.index(chem)))
                for d in range(N_RANDOM):
                    orders[("random", d)] = rng.permutation(real)
                cum_picks = np.cumsum(ks)
                for desc in ("chemsys", "proto"):
                    labels = R[desc].values
                    known0 = set(labels[ref0])
                    for (policy, d), o in orders.items():
                        c = coverage_curve(o, ks, labels, known0)
                        auc, half = curve_stats(cum_picks, c)
                        summ.append(dict(chem=chem, Y=Y, chooser_metric=metric, descriptor=desc, policy=policy,
                                         draw=d, n_pool=len(real), n_new_total=int(c[-1]), auc=auc,
                                         cov_at_half=half))
                        for t, cp, cn in zip(years, cum_picks, c):
                            curves.append((chem, Y, metric, desc, policy, d, t, cp, cn))
                log(f"{chem} {metric} Y={Y}: counterfactual done in {time.perf_counter() - t0:.0f}s "
                    f"(pool {len(real)}), peak RSS {peak_mb()} MB")
            del D
    pd.DataFrame(curves, columns=["chem", "Y", "chooser_metric", "descriptor", "policy", "draw", "year",
                                  "cum_picks", "cum_new"]).to_csv(os.path.join(OUT, "e16_cf_curves.csv"),
                                                                   index=False)
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(OUT, "e16_cf_runs.csv"), index=False)
    # one row per chem x Y x metric x descriptor: real and greedy vs the random distribution
    out = []
    for key, g in s.groupby(["chem", "Y", "chooser_metric", "descriptor"]):
        rnd = g[g.policy == "random"]
        row = dict(zip(["chem", "Y", "chooser_metric", "descriptor"], key), n_pool=g.n_pool.iloc[0],
                   n_new_total=g.n_new_total.iloc[0],
                   random_auc_mean=rnd.auc.mean(), random_auc_min=rnd.auc.min(), random_auc_max=rnd.auc.max(),
                   random_half_mean=rnd.cov_at_half.mean(), random_half_min=rnd.cov_at_half.min(),
                   random_half_max=rnd.cov_at_half.max())
        for policy in ("real", "MINov-greedy", "1-NN-greedy"):
            p = g[g.policy == policy].iloc[0]
            tag = policy.replace("-greedy", "").replace("-", "")
            row[f"{tag}_auc"] = p.auc
            row[f"{tag}_half"] = p.cov_at_half
            row[f"{tag}_auc_frac_random_below"] = (rnd.auc < p.auc).mean()
        out.append(row)
    pd.DataFrame(out).to_csv(os.path.join(OUT, "e16_cf_summary.csv"), index=False)


# Figures ###############################################################################


def fig_percentile():
    st = pd.read_csv(os.path.join(OUT, "e16_steps.csv"))
    st = st[(st.variant == "all") & (st.n_pool > st.n_picks)]  # last step: pool = picks, 50 by construction
    st["year"] = st.step.astype(int)
    names = {"MINov (exact)": "MINov", "1-NN distance": "1-NN distance"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for j, chem in enumerate(CHEMS):
        for i, metric in enumerate(METRICS):
            ax = axes[i, j]
            ax.axhline(50, color=INK2, linestyle="--", linewidth=1)
            for score, label in names.items():
                d = st[(st.chem == chem) & (st.metric == metric) & (st.score == score)].sort_values("year")
                ax.fill_between(d.year, d.ci_lo, d.ci_hi, color=SERIES[label], alpha=0.15, linewidth=0)
                ax.plot(d.year, d.mean_pct, color=SERIES[label], linewidth=1.8, marker="o", markersize=3,
                        label=label)
            if i == 0:
                ax.set_title(CHEM_LABEL[chem])
            if j == 0:
                ax.set_ylabel(f"{METRIC_LABEL[metric]}\nnovelty percentile in pool")
            if i == 1:
                ax.set_xlabel("year of first report (step)")
            ax.set_ylim(0, 100)
            ax.set_xticks([1995, 2000, 2005, 2010], ["1995", "2000", "2005", "2010"])
    axes[0, 0].text(1991, 52, "random order = 50", fontsize=8, color=INK2)
    axes[0, 0].legend(frameon=False, loc="lower left", fontsize=9)
    fig.suptitle("Where each year's new compounds sat in the remaining eventually-reported pool, relative to what was "
                 "known the year before (100 = most novel; bands: 95% bootstrap over entries)",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "e16_percentile.png"), dpi=200)
    plt.close(fig)


def fig_cf(chem="li", Y=2000):
    c = pd.read_csv(os.path.join(OUT, "e16_cf_curves.csv"))
    c = c[(c.chem == chem) & (c.Y == Y)]
    desc_label = {"chemsys": "new chemical systems found", "proto": "new stoichiometry prototypes found"}
    colours = {"real": INK, "MINov-greedy": SERIES["MINov"], "1-NN-greedy": SERIES["1-NN distance"]}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for i, desc in enumerate(["chemsys", "proto"]):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            d = c[(c.descriptor == desc) & (c.chooser_metric == metric)]
            rnd = d[d.policy == "random"].pivot(index="year", columns="draw", values="cum_new")
            ax.fill_between(rnd.index, rnd.min(1), rnd.max(1), color=CONTEXT, alpha=0.4, linewidth=0,
                            label="random order (20 draws, range)")
            for policy, col in colours.items():
                p = d[d.policy == policy].sort_values("year")
                ax.plot(p.year, p.cum_new, color=col, linewidth=2 if policy == "real" else 1.8,
                        label={"real": "real order"}.get(policy, policy + f" (chooses on {METRIC_LABEL[metric].split()[0]})"))
            ax.set_title(f"Chooser: {METRIC_LABEL[metric]}" if i == 0 else "")
            if j == 0:
                ax.set_ylabel(desc_label[desc])
            if i == 1:
                ax.set_xlabel("year (same number of picks per year as history)")
            ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle(f"Counterfactual order, {CHEM_LABEL[chem]}: pool = compounds first reported after {Y}; coverage "
                 "counted relative to compounds known by " + str(Y), x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "e16_counterfactual.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["replay", "cf", "fig"])
    ap.add_argument("--chems", nargs="+", default=CHEMS)
    a = ap.parse_args()
    if "replay" in a.stages:
        stage_replay(a.chems)
    if "cf" in a.stages:
        stage_cf(a.chems)
    if "fig" in a.stages:
        fig_percentile()
        fig_cf()
    log(f"done; peak RSS {peak_mb()} MB")
    with open(os.path.join(OUT, "e16_log.txt"), "a") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
