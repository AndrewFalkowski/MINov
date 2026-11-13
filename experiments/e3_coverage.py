"""E3: How complete is the reference set, globally and locally, and can we predict it?

Species-sampling estimators (Good-Turing coverage, Chao1, Good-Toulmin, Chao's
extrapolation) applied to the Li reference set (Materials Project, MP) under several
definitions of a "species" (unit):
  chemsys      chemical system (e.g. Li-O-P)
  anon         anonymized reduced formula (stoichiometry type, e.g. ABC2)
  sg_anon      space group number + anonymized formula (structure type proxy)
  clu_<m>_q<q> complete-linkage clusters of distance matrix m cut at the q-quantile of
               MP pair distances (clusters of diameter <= threshold)

Parts
  1. Global: singletons/doubletons, Good-Turing coverage, Chao1, accumulation curves,
     and a test on GNoME: does GT predict the fraction of GNoME candidates in units
     unseen in MP?
  2. Split validation (H3c, global): MP split into A/B (random 50/50, random 1/3 vs 2/3,
     and density-biased 50/50 where A over-samples dense regions); from A predict the
     number of units B adds.
  3. Local missing mass (H3a/H3b/H3c, local): MI-weighted Good-Turing missing mass in
     each query's MINov neighbourhood,
         M(q) = sum_j w(d_qj) [unit of j is a singleton in the reference] / sum_j w(d_qj)
     (local coverage = 1 - M). Queries: GNoME (reference = MP) and MP leave-one-out.
     Validated on A/B splits: M estimated on A vs the realized weighted fraction of B
     neighbours whose unit is absent from A.

All estimators assume the reference is a random sample from a fixed population of
units; MP is not (curation and research-interest bias), which part 1's GNoME test and
part 2's biased split try to quantify.

Usage: uv run python experiments/e3_coverage.py [--splits 200] [--local-splits 20]
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
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from scipy.special import gammaln  # noqa: E402
from scipy.stats import binom, binomtest, mannwhitneyu, spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO]
from MINOV.density import MIProfile  # noqa: E402

OUT = os.path.join(REPO, "experiments", "results")
N_MP = 1834
METRICS = {"lostop": "Li_lostop_dm.npy", "elmd": "Li_elmd_dm.npy"}
CLUSTER_QUANTILES = [0.005, 0.02]


# Data and units #######################################################################


def file_names(folder):
    return [f for f in os.listdir(os.path.join(REPO, "data", folder)) if f.endswith(".cif")]


def load_dms():
    dms = {}
    for m, f in METRICS.items():
        d = np.load(os.path.join(REPO, "precomputed", f)).astype(float)
        np.fill_diagonal(d, 0.0)
        dms[m] = d
    return dms


def compute_units(dms):
    """Unit labels for every material (MP rows first, then GNoME), cached to CSV."""
    path = os.path.join(OUT, "e3_units.csv")
    if os.path.exists(path):
        return pd.read_csv(path, keep_default_na=False)
    from pymatgen.core import Composition, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    rows = []
    for folder, source in [("MP_Li_dataset", "MP"), ("GNOME_Li_dataset", "GNoME")]:
        for f in file_names(folder):
            comp = Composition(f[:-4])
            sg, symprec = -1, None
            try:
                s = Structure.from_file(os.path.join(REPO, "data", folder, f))
                for sp in (0.1, 0.01, 0.001):
                    try:
                        sg = SpacegroupAnalyzer(s, symprec=sp).get_space_group_number()
                        symprec = sp
                        break
                    except Exception:
                        continue
            except Exception as e:  # unreadable structure
                warnings.warn(f"{f}: {e}")
            rows.append(
                dict(
                    source=source,
                    formula=f[:-4],
                    chemsys="-".join(sorted(e.symbol for e in comp.elements)),
                    anon=comp.reduced_composition.anonymized_formula,
                    spacegroup=sg,
                    symprec=symprec,
                )
            )
    units = pd.DataFrame(rows)
    units["sg_anon"] = [
        f"{sg}|{a}" if sg > 0 else f"fail{i}|{a}"
        for i, (sg, a) in enumerate(zip(units.spacegroup, units.anon))
    ]
    for m, d in dms.items():
        Z = linkage(squareform(d, checks=False), "complete")
        mp_pairs = d[:N_MP, :N_MP][np.triu_indices(N_MP, 1)]
        for q in CLUSTER_QUANTILES:
            units[f"clu_{m}_q{q * 100:g}"] = fcluster(Z, np.quantile(mp_pairs, q), "distance")
    units.to_csv(path, index=False)
    return units


def unit_columns(units):
    return ["chemsys", "anon", "sg_anon"] + [c for c in units.columns if c.startswith("clu_")]


def encode(labels):
    return np.unique(np.asarray(labels).astype(str), return_inverse=True)[1]


# Species-sampling estimators ##########################################################


def freq_counts(counts):
    """f[i] = number of units seen exactly i times (f[0] unused)."""
    return np.bincount(counts[counts > 0])


def f0_hat(n, f1, f2):
    """Chao1 estimate of the number of unseen units (bias-corrected when f2 = 0)."""
    if f2 > 0:
        return (n - 1) / n * f1**2 / (2 * f2)
    return (n - 1) / n * f1 * (f1 - 1) / 2


def global_stats(counts):
    counts = counts[counts > 0]
    n, S = counts.sum(), len(counts)
    f = np.append(freq_counts(counts), [0, 0, 0])
    f1, f2 = f[1], f[2]
    gt = 1 - f1 / n
    # Chao & Jost (2012) coverage estimator
    cj = 1 - f1 / n * ((n - 1) * f1 / ((n - 1) * f1 + 2 * f2)) if f1 > 0 else 1.0
    chao1 = S + f0_hat(n, f1, f2)
    return dict(n=n, S_obs=S, f1=f1, f2=f2, f3=f[3], max_count=counts.max(),
                coverage_GT=gt, coverage_ChaoJost=cj, chao1=chao1, completeness_S_over_chao1=S / chao1)


def predict_new(counts, m):
    """Predicted number of new units when m more individuals are sampled."""
    counts = counts[counts > 0]
    n = counts.sum()
    f = freq_counts(counts).astype(float)
    i = np.arange(len(f))
    t = m / n
    gt = -np.sum(((-t) ** i * f)[1:])
    if t > 1:  # Orlitsky, Suresh & Wu (2016), binomial smoothing
        k = int(np.ceil(0.5 * np.log(n * t**2 / (t - 1)) / np.log(3)))
        q = 2 / (t + 2)
        tail = binom.sf(i - 1, k, q)  # P(L >= i)
        sgt = -np.sum(((-t) ** i * tail * f)[1:])
    else:
        sgt = gt
    f1, f2 = f[1], (f[2] if len(f) > 2 else 0)
    f0 = f0_hat(n, f1, f2)
    chao = f0 * (1 - (1 - f1 / (n * f0 + f1)) ** m) if f0 > 0 else 0.0  # Shen, Chao & Lin 2003
    return dict(GoodToulmin=gt, SmoothedGT=sgt, Chao=chao)


def rarefaction(counts, ms):
    """Expected number of units in a random subsample of size m (hypergeometric)."""
    counts = counts[counts > 0]
    n = counts.sum()

    def lchoose(a, b):
        return gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)

    out = []
    for m in ms:
        p_absent = np.where(n - counts >= m, np.exp(lchoose(n - counts, m) - lchoose(n, m)), 0.0)
        out.append(np.sum(1 - p_absent))
    return np.array(out)


# Part 1: global #######################################################################


def part1_global(units, ucols):
    mp, gn = units.source == "MP", units.source == "GNoME"
    rows, curves = [], {}
    for u in ucols:
        lab = units[u].astype(str).values
        _, counts = np.unique(lab[mp], return_counts=True)
        st = global_stats(counts)
        mp_set = set(lab[mp])
        gnome_new = int(np.sum([x not in mp_set for x in lab[gn]]))
        p_gt = st["f1"] / st["n"]  # GT: P(next random draw is an unseen unit)
        st.update(
            unit=u,
            GT_pred_frac_new_next=p_gt,
            GNoME_frac_new=gnome_new / gn.sum(),
            GNoME_binom_p=binomtest(gnome_new, gn.sum(), p_gt).pvalue,
            MP_rows_spacegroup_failed=int(np.sum(units.spacegroup[mp] < 0)) if u == "sg_anon" else 0,
        )
        rows.append(st)
        n = st["n"]
        ms = np.linspace(1, n, 60).astype(int)
        ext = np.linspace(n, 2 * n, 30).astype(int)
        f0 = f0_hat(n, st["f1"], st["f2"])
        S_ext = st["S_obs"] + f0 * (1 - (1 - st["f1"] / (n * f0 + st["f1"])) ** (ext - n))
        curves[u] = (ms, rarefaction(counts, ms), ext, S_ext, st["chao1"])
    df = pd.DataFrame(rows).set_index("unit")
    df.to_csv(os.path.join(OUT, "e3_global.csv"))
    return df, curves


# Part 2: split validation #############################################################


def density_bias_keys(dms):
    """Mean rank of MINov density (both metrics) among MP; used to bias splits."""
    r = np.zeros(N_MP)
    for d in dms.values():
        dm = d[:N_MP, :N_MP]
        r += pd.Series(MIProfile(dm).density(dm)).rank().values
    return (r - r.mean()) / r.std()


def part2_splits(units, ucols, dms, n_splits, rng):
    codes = {u: encode(units[u].values[:N_MP]) for u in ucols}
    zdens = density_bias_keys(dms)
    designs = {"random 1/2": 0.5, "random 1/3": 1 / 3, "dense-biased 1/2": 0.5}
    rows = []
    for design, frac in designs.items():
        nA = int(round(frac * N_MP))
        for s in range(n_splits):
            if design.startswith("dense"):
                keys = 1.5 * zdens + rng.gumbel(size=N_MP)  # Gumbel top-k = weighted sampling w/o replacement
                A = np.argsort(-keys)[:nA]
            else:
                A = rng.permutation(N_MP)[:nA]
            inA = np.zeros(N_MP, bool)
            inA[A] = True
            for u, c in codes.items():
                cA = np.bincount(c[inA], minlength=c.max() + 1)
                actual = len(np.setdiff1d(np.unique(c[~inA]), np.flatnonzero(cA)))
                pred = predict_new(cA, N_MP - nA)
                for est, p in pred.items():
                    rows.append(dict(design=design, split=s, unit=u, estimator=est, actual=actual, predicted=p))
    df = pd.DataFrame(rows)
    df["rel_err"] = (df.predicted - df.actual) / df.actual
    summ = (
        df.groupby(["design", "unit", "estimator"])
        .agg(actual_mean=("actual", "mean"), actual_sd=("actual", "std"),
             pred_mean=("predicted", "mean"), pred_sd=("predicted", "std"),
             bias_pct=("rel_err", lambda x: 100 * x.mean()),
             rmse_pct=("rel_err", lambda x: 100 * np.sqrt(np.mean(x**2))))
        .reset_index()
    )
    summ.to_csv(os.path.join(OUT, "e3_split_validation.csv"), index=False)
    return summ


# Part 3: local missing mass ###########################################################


def local_mass(W, codes_q, codes_ref, counts_ref, loo):
    """MI-weighted Good-Turing missing mass per query.

    W         (n_q, n_ref) weights w(d_qj); self-weights already zeroed when loo
    counts_ref unit counts in the reference (including the query itself when loo)
    loo       if True, the query belongs to the reference and is removed from it, so
              its own unit loses one count
    """
    c = counts_ref[codes_ref][None, :].astype(int)
    if loo:
        c = c - (codes_q[:, None] == codes_ref[None, :])
    single = (c == 1).astype(float)
    tot = W.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W * single).sum(axis=1) / tot


def partial_spearman(x, y, z):
    """Spearman partial correlation of x and y given z."""
    r = pd.DataFrame(dict(x=x, y=y, z=z)).rank().corr().values
    rxy, rxz, ryz = r[0, 1], r[0, 2], r[1, 2]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))


def part3_local(units, ucols, dms):
    per_material, rows = [], []
    for m, d in dms.items():
        Dmp, Dgn = d[:N_MP, :N_MP], d[N_MP:, :N_MP]
        prof = MIProfile(Dmp)
        Wmp = prof.weight(Dmp)
        np.fill_diagonal(Wmp, 0.0)  # leave-one-out (profile fit on full MP; negligible change)
        Wgn = prof.weight(Dgn)
        dens_mp, dens_gn = Wmp.sum(1), Wgn.sum(1)
        for u in ucols:
            codes = encode(units[u].values)
            cmp_, cgn = codes[:N_MP], codes[N_MP:]
            counts = np.bincount(cmp_, minlength=codes.max() + 1)
            M_mp = local_mass(Wmp, cmp_, cmp_, counts, loo=True)
            M_gn = local_mass(Wgn, cgn, cmp_, counts, loo=False)
            new_mp = counts[cmp_] == 1  # unit unseen in the rest of MP
            new_gn = counts[cgn] == 0
            per_material.append(pd.DataFrame(dict(
                metric=m, unit=u, source=["MP"] * N_MP + ["GNoME"] * len(cgn),
                formula=units.formula.values, density=np.r_[dens_mp, dens_gn],
                missing_mass=np.r_[M_mp, M_gn], unit_unseen=np.r_[new_mp, new_gn])))
            ok = ~np.isnan(M_mp)
            rows.append(dict(
                metric=m, unit=u,
                M_MP_median=np.nanmedian(M_mp), M_GNoME_median=np.nanmedian(M_gn),
                M_MP_mean=np.nanmean(M_mp), M_GNoME_mean=np.nanmean(M_gn),
                MP_frac_unseen=new_mp.mean(), GNoME_frac_unseen=new_gn.mean(),
                H3a_MWU_p_GNoME_greater=mannwhitneyu(M_gn[~np.isnan(M_gn)], M_mp[ok], alternative="greater").pvalue,
                H3a_AUC_GNoME_vs_MP=roc_auc_score(np.r_[np.ones((~np.isnan(M_gn)).sum()), np.zeros(ok.sum())],
                                                  np.r_[M_gn[~np.isnan(M_gn)], M_mp[ok]]),
                H3b_spearman_M_density_MP=spearmanr(M_mp[ok], dens_mp[ok])[0],
                H3b_spearman_M_density_GNoME=spearmanr(M_gn, dens_gn, nan_policy="omit")[0],
                LOO_AUC_unseen_by_M=roc_auc_score(new_mp[ok], M_mp[ok]),
                LOO_AUC_unseen_by_negdensity=roc_auc_score(new_mp[ok], -dens_mp[ok]),
                GNoME_AUC_unseen_by_M=(roc_auc_score(new_gn, M_gn) if 0 < new_gn.sum() < len(new_gn) else np.nan),
            ))
    pm = pd.concat(per_material, ignore_index=True)
    pm.to_csv(os.path.join(OUT, "e3_local_per_material.csv"), index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "e3_local_summary.csv"), index=False)
    return summ, pm


def part3_local_validation(units, ucols, dms, n_splits, rng):
    """M estimated on half A vs realized weighted fraction of B neighbours in units new to A."""
    rows = []
    allcodes = {u: encode(units[u].values) for u in ucols}
    n_all = len(units)
    for s in range(n_splits):
        perm = rng.permutation(N_MP)
        A, B = np.sort(perm[: N_MP // 2]), np.sort(perm[N_MP // 2:])
        for m, d in dms.items():
            prof = MIProfile(d[np.ix_(A, A)])
            WA, WB = prof.weight(d[:, A]), prof.weight(d[:, B])
            WA[A, np.arange(len(A))] = 0.0  # a query never counts itself
            WB[B, np.arange(len(B))] = 0.0
            densA = WA.sum(1)
            for u, codes in allcodes.items():
                cA = np.bincount(codes[A], minlength=codes.max() + 1)
                M = local_mass(WA, codes, codes[A], cA, loo=False)
                newB = (cA[codes[B]] == 0).astype(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    R = (WB * newB[None, :]).sum(1) / WB.sum(1)
                ok = ~np.isnan(M) & ~np.isnan(R)
                glob = (cA == 1).sum() / len(A)
                for grp, mask in [("all", ok), ("GNoME", ok & (np.arange(n_all) >= N_MP))]:
                    rows.append(dict(
                        split=s, metric=m, unit=u, queries=grp,
                        realized_mean=R[mask].mean(), M_mean=M[mask].mean(), global_GT=glob,
                        rho_M_R=spearmanr(M[mask], R[mask])[0],
                        rho_negdens_R=spearmanr(-densA[mask], R[mask])[0],
                        partial_rho_M_R_given_dens=partial_spearman(M[mask], R[mask], densA[mask]),
                        MAE_local=np.mean(np.abs(M[mask] - R[mask])),
                        MAE_global=np.mean(np.abs(glob - R[mask])),
                    ))
    df = pd.DataFrame(rows)
    summ = df.groupby(["metric", "unit", "queries"]).mean(numeric_only=True).drop(columns="split").reset_index()
    summ.to_csv(os.path.join(OUT, "e3_local_validation.csv"), index=False)
    return summ


# Figures ##############################################################################


def plot(curves, local_pm, gl):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    colors = plt.cm.tab10(np.arange(len(curves)))
    for (u, (ms, S, ext, S_ext, chao1)), col in zip(curves.items(), colors):
        ax.plot(ms, S, color=col, label=f"{u} (C={gl.loc[u, 'coverage_GT']:.2f})")
        ax.plot(ext, S_ext, color=col, ls="--")
    ax.set_xlabel("number of MP materials sampled")
    ax.set_ylabel("distinct units")
    ax.set_title("Accumulation (solid: rarefied; dashed: Chao extrapolation)")
    ax.legend(fontsize=7)
    ax = axes[1]
    sel = local_pm[(local_pm.metric == "lostop")]
    us = list(curves)
    data, pos, cols = [], [], []
    for i, u in enumerate(us):
        for j, src in enumerate(["MP", "GNoME"]):
            x = sel[(sel.unit == u) & (sel.source == src)].missing_mass.dropna().values
            data.append(x)
            pos.append(i * 3 + j)
            cols.append("tab:blue" if src == "MP" else "tab:orange")
    bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_xticks([i * 3 + 0.5 for i in range(len(us))])
    ax.set_xticklabels(us, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("local missing mass M (LoStOP neighbourhood)")
    ax.set_title("Local missing mass: MP LOO (blue), GNoME (orange)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e3_coverage.png"), dpi=150)

    # Calibration under shift: predicted vs actual fraction of GNoME in units unseen in MP
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(us))
    gn = local_pm[(local_pm.metric == "lostop") & (local_pm.source == "GNoME")]
    local = [gn[gn.unit == u].missing_mass.mean() for u in us]
    ax.bar(x - 0.27, gl.loc[us, "GT_pred_frac_new_next"], 0.27, label="global Good-Turing f1/n")
    ax.bar(x, local, 0.27, label="local missing mass (LoStOP nbhd), GNoME mean")
    ax.bar(x + 0.27, gl.loc[us, "GNoME_frac_new"], 0.27, label="actual GNoME fraction in unseen units")
    ax.set_xticks(x)
    ax.set_xticklabels(us, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("fraction in units unseen in MP")
    ax.set_title("Missing-mass estimates vs GNoME candidates (non-random draws)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e3_gnome_calibration.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--local-splits", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    t0 = time.time()

    dms = load_dms()
    units = compute_units(dms)
    ucols = unit_columns(units)
    print(f"units ready ({time.time() - t0:.0f}s); spacegroup failures: {(units.spacegroup < 0).sum()}")

    gl, curves = part1_global(units, ucols)
    print("\n== Part 1: global ==\n", gl.round(3).T)

    sv = part2_splits(units, ucols, dms, args.splits, rng)
    print(f"\n== Part 2: split validation ({time.time() - t0:.0f}s) ==\n", sv.round(2).to_string())

    ls, pm = part3_local(units, ucols, dms)
    print(f"\n== Part 3: local ({time.time() - t0:.0f}s) ==\n", ls.round(3).T)

    lv = part3_local_validation(units, ucols, dms, args.local_splits, rng)
    print(f"\n== Part 3: local validation ({time.time() - t0:.0f}s) ==\n", lv.round(3).to_string())

    plot(curves, pm, gl)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
