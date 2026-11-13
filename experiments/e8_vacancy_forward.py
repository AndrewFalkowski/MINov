"""E8: Analogue vacancy at scale, under a different reference (H7), and forward in time (H8).

Vacancy as in E6 (generator and pi0 from e2_denovelling): per formula, n Li-keeping
element-substitution analogues (Hautier lambda table, isovalent, r >= 0.03, own formula
excluded), E = sum pi0(r_j), O = analogues whose reduced formula is in the reference,
V = log((E + 0.5) / (O + 0.5)). pi0(r) = sigmoid(a + b log r) fit on the reference
formulas' own analogues (leave-one-out is automatic: a formula is never its own analogue),
cross-fitted over random halves of the reference: reference formulas get out-of-fold pi0,
candidates the mean of the two fold fits; V averaged over N_CROSS repeats.
Everything is formula level (canonical reduced composition via fkey).

Parts
  1. Scale. Reference = MP-2023 experimental Li formulas (1857). Candidates = all GNoME
     Dec-2023 Li formulas not in the reference; split into retained (still in current
     GNoME), removed-only, and GNoME added after Dec-2023. AUC candidate vs reference (LOO).
     P44 sanity check against E6. Density correlation on the G sample (U rows), densities
     from MIProfile fit on R23 rows of U_lostop / U_elmd.
  2. H7. Reference = all MP-2023 Li formulas (9551, theoretical included). PASS if AUC
     GNoME vs MP >= 0.8.
  3. H8 forward. Y in (1990, 2000, 2005): Reference_Y = experimental formulas with
     first_year <= Y (formula first year = min over its R23 entries). pi0 refit on
     Reference_Y only. (a) AUC of V_Y, post-Y formulas vs pre-Y formulas (LOO).
     (b) pre-Y formulas: does V_Y predict how many of their analogues are first reported
     after Y (Spearman; AUC for any filled), with baselines n, E and partial correlations.
  4. Removed vs retained GNoME Dec-2023 entries (entry level, formula-level V) and formulas.

Usage: uv run python experiments/e8_vacancy_forward.py
"""

import os
import sys
import time
import warnings
from multiprocessing import get_context

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
import e2_denovelling as e2  # noqa: E402

DATA = os.path.join(REPO, "data", "raw", "processed")
OUT = os.path.join(REPO, "experiments", "results")
CACHE = os.path.join(OUT, "e8_analogues_cache.npz")
N_CROSS = 5
N_BOOT = 1000
YEARS = (1990, 2000, 2005)
SEED = 8
F_BLOCK = set("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split())

warnings.filterwarnings("ignore")


# Analogue table #######################################################################

_ODDS, _INDEX = None, None


def _init(index):
    global _ODDS, _INDEX
    _ODDS, _INDEX = e2.element_odds(), index


def _one(f):
    an = [a for a in e2.element_analogues(f, _ODDS) if a["formula"] != f]
    return (np.array([np.log(a["r"]) for a in an], np.float32),
            np.array([a["n_sub"] for a in an], np.int8),
            np.array([_INDEX.get(a["formula"], -1) for a in an], np.int32))


def analogue_table(F):
    """Flat table over all formulas F: owner, aid (index into F or -1), log r, n_sub."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if list(z["F"]) == list(F):
            return {k: z[k] for k in ("owner", "aid", "logr", "nsub")}
    index = {f: i for i, f in enumerate(F)}
    t = time.time()
    with get_context("fork").Pool(8, initializer=_init, initargs=(index,)) as pool:
        res = pool.map(_one, F, chunksize=50)
    n = np.array([len(r[0]) for r in res])
    T = dict(owner=np.repeat(np.arange(len(F), dtype=np.int32), n),
             logr=np.concatenate([r[0] for r in res]), nsub=np.concatenate([r[1] for r in res]),
             aid=np.concatenate([r[2] for r in res]))
    print(f"analogues: {len(T['owner']):,} rows for {len(F)} formulas in {time.time() - t:.0f}s")
    np.savez(CACHE, F=np.array(F, dtype=object), **T)
    return T


# Vacancy ##############################################################################


def vacancy(T, nF, ref, cand, rng, n_cross=N_CROSS):
    """V, E, O, n over all formulas (NaN outside ref | cand). ref, cand: bool masks over F."""
    use = ref | cand
    sel = use[T["owner"]]
    own, aid, logr = T["owner"][sel], T["aid"][sel], T["logr"][sel].astype(float)
    hit = np.zeros(len(own), bool)
    ok = aid >= 0
    hit[ok] = ref[aid[ok]]
    own_ref = ref[own]
    ref_idx = np.where(ref)[0]
    n = np.bincount(own, minlength=nF)
    O = np.bincount(own, weights=hit, minlength=nF)
    Vs, Es, coefs = [], [], []
    for _ in range(n_cross):
        fold = np.full(nF, -1)
        fold[ref_idx] = rng.permutation(np.arange(len(ref_idx)) % 2)
        f_own = fold[own]
        pis = []
        for k in (0, 1):
            m = own_ref & (f_own == k)
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(logr[m, None], hit[m])
            a, b = lr.intercept_[0], lr.coef_[0, 0]
            coefs.append((a, b))
            pis.append(1 / (1 + np.exp(-(a + b * logr))))
        pi = np.where(f_own == 0, pis[1], np.where(f_own == 1, pis[0], 0.5 * (pis[0] + pis[1])))
        E = np.bincount(own, weights=pi, minlength=nF)
        Es.append(E)
        Vs.append(np.log((E + 0.5) / (O + 0.5)))
    nan = np.where(use, 1.0, np.nan)
    return dict(V=np.mean(Vs, 0) * nan, E=np.mean(Es, 0) * nan, O=O * nan, n=n * nan,
                logit_a=np.mean([c[0] for c in coefs]), logit_b=np.mean([c[1] for c in coefs]),
                obs_over_exp_ref=O[ref].sum() / np.mean(Es, 0)[ref].sum())


# Statistics ###########################################################################


def auc_fast(pos, neg):
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    r = rankdata(np.r_[pos, neg])
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def auc_ci(pos, neg, rng, nb=N_BOOT):
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    b = [auc_fast(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(nb)]
    return auc_fast(pos, neg), *np.percentile(b, [2.5, 97.5])


def strat_auc(pos, neg, gpos, gneg):
    num = den = 0.0
    for g in np.unique(np.r_[gpos, gneg]):
        p, q = pos[gpos == g], neg[gneg == g]
        if len(p) and len(q):
            num += auc_fast(p, q) * len(p) * len(q)
            den += len(p) * len(q)
    return num / den


def partial_spearman(x, y, covs):
    rx, ry = rankdata(x), rankdata(y)
    C = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in covs])
    res = [v - C @ np.linalg.lstsq(C, v, rcond=None)[0] for v in (rx, ry)]
    return np.corrcoef(*res)[0, 1]


def boot_stat(fn, n, rng, nb=N_BOOT):
    vals = [fn(rng.integers(0, n, n)) for _ in range(nb)]
    return np.nanpercentile(vals, [2.5, 97.5])


# Main #################################################################################


def canon(fkeys):
    from pymatgen.core import Composition
    return {k: Composition(k).reduced_formula for k in set(fkeys)}


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    mp = pd.read_csv(os.path.join(DATA, "mp2023_li.csv"))
    gn = pd.read_csv(os.path.join(DATA, "gnome_li.csv"))
    yr = pd.read_csv(os.path.join(DATA, "r23_first_year.csv"))
    U = pd.read_csv(os.path.join(DATA, "U_index.csv"))
    cm = canon(pd.concat([mp["fkey"], gn["fkey"], U["fkey"]]))
    for d in (mp, gn, yr, U):
        d["f"] = d["fkey"].map(cm)
    F = sorted(set(mp["f"]) | set(gn["f"]) | set(U["f"]))
    fi = {f: i for i, f in enumerate(F)}
    nF = len(F)
    T = analogue_table(F)
    print(f"{nF} formulas, {len(T['owner']):,} analogue rows ({time.time() - t0:.0f}s)")

    from pymatgen.core import Composition
    comp = [Composition(f) for f in F]
    n_el = np.array([len(c.elements) for c in comp])
    fblock = np.array([bool(F_BLOCK & {e.symbol for e in c.elements}) for c in comp])

    def mask(fs):
        m = np.zeros(nF, bool)
        m[[fi[f] for f in fs]] = True
        return m

    ref_exp = mask(mp.loc[~mp["theoretical"], "f"])
    ref_all = mask(mp["f"])
    g1223 = mask(gn.loc[gn["in_gnome_1223"], "f"])
    g_ret = mask(gn.loc[gn["in_gnome_1223"] & gn["in_gnome_current"], "f"])
    g_rem_any = mask(gn.loc[gn["in_gnome_1223"] & ~gn["in_gnome_current"], "f"])
    g_cur = mask(gn.loc[gn["in_gnome_current"], "f"])
    g_rem_only = g_rem_any & ~g_cur
    g_new = mask(gn.loc[gn["in_gnome_current"] & ~gn["in_gnome_1223"], "f"]) & ~g1223
    p44 = mask(U.loc[U["set"] == "P44", "f"])
    gn_all = g1223 | g_cur
    rows_auc, summ = [], {}

    def add_auc(part, ref_name, pos_name, pos, neg_name, neg, Vd, extra=None):
        a, lo, hi = auc_ci(Vd["V"][pos], Vd["V"][neg], rng)
        r = dict(part=part, reference=ref_name, positive=pos_name, negative=neg_name,
                 n_pos=int(pos.sum()), n_neg=int(neg.sum()), AUC_V=a, CI_lo=lo, CI_hi=hi,
                 AUC_V_strat_n_elements=strat_auc(Vd["V"][pos], Vd["V"][neg], n_el[pos], n_el[neg]),
                 AUC_V_no_fblock=auc_fast(Vd["V"][pos & ~fblock], Vd["V"][neg & ~fblock]),
                 AUC_n=auc_fast(Vd["n"][pos], Vd["n"][neg]), AUC_E=auc_fast(Vd["E"][pos], Vd["E"][neg]),
                 AUC_negO=auc_fast(-Vd["O"][pos], -Vd["O"][neg]),
                 AUC_n_elements=auc_fast(n_el[pos].astype(float), n_el[neg].astype(float)),
                 mean_V_pos=np.nanmean(Vd["V"][pos]), mean_V_neg=np.nanmean(Vd["V"][neg]),
                 OE_pos=np.nansum(Vd["O"][pos]) / np.nansum(Vd["E"][pos]),
                 OE_neg=np.nansum(Vd["O"][neg]) / np.nansum(Vd["E"][neg]),
                 frac_O0_pos=np.mean(Vd["O"][pos] == 0), frac_O0_neg=np.mean(Vd["O"][neg] == 0))
        r.update(extra or {})
        rows_auc.append(r)
        print(f"{part:8s} {ref_name:14s} {pos_name:28s} vs {neg_name:22s} AUC {a:.3f} [{lo:.3f}, {hi:.3f}]"
              f"  strat_nel {r['AUC_V_strat_n_elements']:.3f}  O/E {r['OE_pos']:.3f} vs {r['OE_neg']:.3f}")

    # 1. scale ---------------------------------------------------------------------------
    cand_exp = gn_all & ~ref_exp | p44 & ~ref_exp
    V1 = vacancy(T, nF, ref_exp, cand_exp, rng)
    summ.update(exp_logit_a=V1["logit_a"], exp_logit_b=V1["logit_b"], exp_ref_OE=V1["obs_over_exp_ref"],
                n_ref_exp=int(ref_exp.sum()), n_ref_all=int(ref_all.sum()))
    for name, m in (("GNoME Dec-2023 (all)", g1223), ("GNoME Dec-2023 retained", g_ret),
                    ("GNoME Dec-2023 removed-only", g_rem_only), ("GNoME added after Dec-2023", g_new),
                    ("P44 (E6 candidates)", p44)):
        add_auc("1_scale", "MP23 exp", name, m & ~ref_exp, "MP23 exp (LOO)", ref_exp, V1)

    # density on U: MIProfile on R23 rows; G sample + R23 LOO
    from MINOV.density import MIProfile
    r23 = np.where(U["set"] == "R23")[0]
    gs = np.where(U["set"] == "G")[0]
    corr_rows = []
    vU = V1["V"][U["f"].map(fi).values]
    nU = V1["n"][U["f"].map(fi).values]
    nelU = n_el[U["f"].map(fi).values]
    for met in ("lostop", "elmd"):
        D = np.load(os.path.join(DATA, f"U_{met}.npy"), mmap_mode="r")
        R = np.asarray(D[np.ix_(r23, r23)], dtype=float)
        prof = MIProfile(R)
        loo = prof.density(R) - prof.weight(0.0)
        dg = prof.density(np.asarray(D[np.ix_(gs, r23)], dtype=float))
        ok = np.isfinite(vU[gs]) & ~U["f"].iloc[gs].map(lambda f: ref_exp[fi[f]]).values
        x, d = vU[gs][ok], dg[ok]
        rho = spearmanr(x, d)[0]
        ci = boot_stat(lambda ix: spearmanr(x[ix], d[ix])[0], len(x), rng, 500)
        a, lo, hi = auc_ci(-dg[ok], -loo, rng)
        av, lov, hiv = auc_ci(vU[gs][ok], vU[r23], rng)
        corr_rows.append(dict(metric=met, group="G sample (3000, formula not in MP exp)", n=int(ok.sum()),
                              spearman_V_density=rho, CI_lo=ci[0], CI_hi=ci[1],
                              partial_ctrl_n_nel=partial_spearman(x, d, [nU[gs][ok], nelU[gs][ok]]),
                              spearman_V_density_R23=spearmanr(vU[r23], loo)[0],
                              AUC_density_G_vs_R23=a, AUC_density_CI_lo=lo, AUC_density_CI_hi=hi,
                              AUC_V_G_vs_R23_entries=av, AUC_V_CI_lo=lov, AUC_V_CI_hi=hiv))
        print(corr_rows[-1])
    pd.DataFrame(corr_rows).to_csv(os.path.join(OUT, "e8_density_correlation.csv"), index=False)

    # 2. H7: reference = all MP-2023 Li formulas ----------------------------------------
    cand_all = gn_all & ~ref_all | p44 & ~ref_all
    V2 = vacancy(T, nF, ref_all, cand_all, rng)
    summ.update(all_logit_a=V2["logit_a"], all_logit_b=V2["logit_b"], all_ref_OE=V2["obs_over_exp_ref"])
    for name, m in (("GNoME Dec-2023 (all)", g1223), ("GNoME Dec-2023 retained", g_ret),
                    ("GNoME added after Dec-2023", g_new), ("P44 (E6 candidates)", p44)):
        add_auc("2_H7", "MP23 all", name, m & ~ref_all, "MP23 all (LOO)", ref_all, V2)
        add_auc("2_H7", "MP23 all", name, m & ~ref_all, "MP23 exp (LOO)", ref_exp, V2)
    add_auc("2_H7", "MP23 all", "MP23 theoretical-only", ref_all & ~ref_exp, "MP23 exp (LOO)", ref_exp, V2)
    # same GNoME formulas, V under each reference
    both = g1223 & ~ref_all
    summ.update(H7_spearman_V_exp_vs_V_all_GNoME=spearmanr(V1["V"][both], V2["V"][both])[0],
                H7_GNoME_meanV_exp=np.mean(V1["V"][both]), H7_GNoME_meanV_all=np.mean(V2["V"][both]),
                H7_MPexp_meanV_exp=np.mean(V1["V"][ref_exp]), H7_MPexp_meanV_all=np.mean(V2["V"][ref_exp]))
    # who fills GNoME analogues in MP-all: share of GNoME analogue hits that are theoretical formulas
    sel = both[T["owner"]] & (T["aid"] >= 0)
    hit_aid = T["aid"][sel][ref_all[T["aid"][sel]]]
    summ.update(H7_GNoME_hits_theoretical_frac=float(np.mean(~ref_exp[hit_aid])),
                H7_GNoME_hits_total=int(len(hit_aid)))
    sel = ref_exp[T["owner"]] & (T["aid"] >= 0)
    hit_aid = T["aid"][sel][ref_all[T["aid"][sel]]]
    summ.update(H7_MPexp_hits_theoretical_frac=float(np.mean(~ref_exp[hit_aid])))

    # 3. H8 forward -----------------------------------------------------------------------
    fy = yr.groupby("f")["first_year"].min()
    year = np.full(nF, np.nan)
    year[[fi[f] for f in fy.index]] = fy.values
    h8, VYs = [], {}
    for Y in YEARS:
        refY = ref_exp & (year <= Y)
        post = ref_exp & (year > Y)
        VY = vacancy(T, nF, refY, post | (g1223 & ~ref_exp), rng)
        add_auc(f"3_H8a_{Y}", f"exp <= {Y}", f"first reported > {Y}", post, f"exp <= {Y} (LOO)", refY, VY)
        add_auc(f"3_H8a_{Y}", f"exp <= {Y}", "GNoME Dec-2023 (all)", g1223 & ~ref_exp, f"exp <= {Y} (LOO)", refY, VY)
        add_auc(f"3_H8a_{Y}", f"exp <= {Y}", "GNoME Dec-2023 (all)", g1223 & ~ref_exp, f"first reported > {Y}",
                post, VY)
        v95 = np.percentile(VY["V"][refY], 95)
        rows_auc[-1].update(V95_preY=v95, frac_post_above_V95=np.mean(VY["V"][post] > v95),
                            frac_GNoME_above_V95=np.mean(VY["V"][g1223 & ~ref_exp] > v95))
        VYs[Y] = VY["V"]
        # (b) filling: analogues of pre-Y formulas first reported after Y
        sel = refY[T["owner"]] & (T["aid"] >= 0)
        filled = np.bincount(T["owner"][sel], weights=post[T["aid"][sel]], minlength=nF)
        idx = np.where(refY)[0]
        v, fill, n, E, O = VY["V"][idx], filled[idx], VY["n"][idx], VY["E"][idx], VY["O"][idx]
        anyf = fill > 0
        rho = spearmanr(v, fill)[0]
        ci_rho = boot_stat(lambda ix: spearmanr(v[ix], fill[ix])[0], len(v), rng, 500)
        a, lo, hi = auc_ci(v[anyf], v[~anyf], rng)
        haz = fill / np.maximum(n - O, 1)  # fraction of absent analogues that got filled
        hn = n - O > 0
        pr = partial_spearman(v, fill, [np.log1p(n)])
        ci_pr = boot_stat(lambda ix: partial_spearman(v[ix], fill[ix], [np.log1p(n[ix])]), len(v), rng, 300)
        # AUC within n-decile strata (V vs any filled)
        dec = np.digitize(np.log1p(n), np.quantile(np.log1p(n), np.arange(1, 10) / 10))
        h8.append(dict(Y=Y, n_pre=int(refY.sum()), n_post=int(post.sum()), frac_any_filled=anyf.mean(),
                       total_filled=int(fill.sum()), spearman_V_filled=rho, rho_CI_lo=ci_rho[0], rho_CI_hi=ci_rho[1],
                       AUC_V_anyfilled=a, AUC_CI_lo=lo, AUC_CI_hi=hi,
                       AUC_n_anyfilled=auc_fast(n[anyf], n[~anyf]), AUC_E_anyfilled=auc_fast(E[anyf], E[~anyf]),
                       AUC_negO_anyfilled=auc_fast(-O[anyf], -O[~anyf]),
                       AUC_V_anyfilled_strat_n_decile=strat_auc(v[anyf], v[~anyf], dec[anyf], dec[~anyf]),
                       partial_spearman_V_filled_ctrl_logn=pr, partial_CI_lo=ci_pr[0], partial_CI_hi=ci_pr[1],
                       spearman_V_fill_rate_of_absent=spearmanr(v[hn], haz[hn])[0],
                       spearman_n_filled=spearmanr(n, fill)[0], spearman_E_filled=spearmanr(E, fill)[0],
                       logit_a=VY["logit_a"], logit_b=VY["logit_b"]))
        print(h8[-1])
    pd.DataFrame(h8).to_csv(os.path.join(OUT, "e8_h8_filling.csv"), index=False)

    # 4. removed vs retained GNoME ------------------------------------------------------------
    add_auc("4_removed", "MP23 exp", "GNoME removed-only formulas", g_rem_only & ~ref_exp,
            "GNoME retained formulas", g_ret & ~ref_exp, V1)
    e1223 = gn[gn["in_gnome_1223"]].copy()
    e1223["V"] = V1["V"][e1223["f"].map(fi).values]
    e1223 = e1223[np.isfinite(e1223["V"])]
    rem = ~e1223["in_gnome_current"].values
    nel_e = e1223["nelements"].values
    a, lo, hi = auc_ci(e1223["V"].values[rem], e1223["V"].values[~rem], rng)
    rows_auc.append(dict(part="4_removed", reference="MP23 exp", positive="removed entries (formula-level V)",
                         negative="retained entries", n_pos=int(rem.sum()), n_neg=int((~rem).sum()),
                         AUC_V=a, CI_lo=lo, CI_hi=hi,
                         AUC_V_strat_n_elements=strat_auc(e1223["V"].values[rem], e1223["V"].values[~rem],
                                                          nel_e[rem], nel_e[~rem]),
                         mean_V_pos=e1223["V"].values[rem].mean(), mean_V_neg=e1223["V"].values[~rem].mean()))
    print(rows_auc[-1])

    aucs = pd.DataFrame(rows_auc)
    aucs.to_csv(os.path.join(OUT, "e8_auc.csv"), index=False)
    per = pd.DataFrame(dict(formula=F, n_elements=n_el, f_block=fblock, first_year=year,
                            in_mp23_exp=ref_exp, in_mp23_all=ref_all, gnome_1223=g1223, gnome_retained=g_ret,
                            gnome_removed_only=g_rem_only, gnome_new=g_new, p44=p44,
                            n=V1["n"], E_exp=V1["E"], O_exp=V1["O"], V_exp=V1["V"],
                            E_all=V2["E"], O_all=V2["O"], V_all=V2["V"],
                            **{f"V_year{Y}": VYs[Y] for Y in YEARS}))
    per = per[np.isfinite(per["V_exp"]) | np.isfinite(per["V_all"])]
    per.to_csv(os.path.join(OUT, "e8_per_formula.csv.gz"), index=False)
    summ["runtime_s"] = time.time() - t0
    pd.Series(summ).to_csv(os.path.join(OUT, "e8_summary.csv"), header=["value"])
    print(pd.Series(summ).to_string())
    print(aucs[["part", "reference", "positive", "negative", "n_pos", "n_neg", "AUC_V", "CI_lo", "CI_hi",
                "AUC_V_strat_n_elements", "AUC_n", "AUC_E"]].round(3).to_string())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
