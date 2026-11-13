"""E6: Analogue vacancy as a novelty signal distinct from MINov density.

H6: how many of a material's plausible substitution analogues are ABSENT from the reference,
relative to how many the calibration expects to be present, separates GNoME candidates from
MP materials better than density, is only weakly correlated with density, and flags GNoME
candidates that density calls "not novel".

Generator and calibration are E2's (imported from e2_denovelling): element-level isovalent
substitutions from the Hautier lambda table, per-site odds r, product over sites, Li fixed,
r >= 0.03; pi0(r) = sigmoid(a + b log r) fitted on whether MP materials' analogues are in MP.

Per material i with analogues j (own formula excluded):
  n  = number of analogues, E = sum_j pi0(r_j), O = number present in the reference
       (MP leave-one-out for MP materials; full MP for GNoME),
  V  = log((E + 0.5) / (O + 0.5))   [primary score; higher = more vacant]
  p  = Poisson-binomial mid-p, P(O' < O) + 0.5 P(O' = O), O' ~ PoiBin(pi0(r_j))  [flag]
V is an effect size (expected vs observed, shrunk by +0.5) that is comparable across
materials with very different n; p is a significance that grows with n (a material with
1000 analogues and O = 0 gets p ~ 1e-40), so it ranks mostly by n. V ranks, p flags.

Parts
  1. Reproduce E2 (70,902 MP analogues, logit parameters, GNoME in-MP counts).
  2. Scores with in-sample pi0, cross-fitted pi0 (random halves of MP; and halves grouped by
     anonymized formula, which every analogue shares with its parent, so the calibration
     never sees the evaluated material's substitution family), and a covariate-adjusted
     pi0(log r, n_sub, n_elements).
  3. (a) AUC GNoME vs MP of V, mid-p, density (LoStOP, ElMD) and baselines (n, E, anon
     formula unseen); stratified AUCs (n_elements, n analogues, anon seen, no f-block).
     (b) Spearman / partial Spearman V vs density. (c) vacancy of GNoME not novel by density.
     Elements: which substituted elements carry the missing expected mass.
  4. Falsification: random 50/50 MP splits into known K / pseudo-candidates P (pi0 refit
     on K); plus chemical-system-grouped splits.

Usage: uv run python experiments/e6_analogue_vacancy.py
"""

import os
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
import e2_denovelling as e2  # noqa: E402

OUT = os.path.join(REPO, "experiments", "results")
N_MP, N_GN = e2.N_MP, 44
N_CROSS = 10        # cross-fitting repeats
N_FALS = 20         # falsification splits
N_BOOT = 2000
SEED = 6
F_BLOCK = set("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split())

warnings.filterwarnings("ignore")


# Analogues ############################################################################


def build(formulas, odds, ref_index):
    """Flat analogue table: owner, formula, r, n_sub, mp index (-1 if not in MP), substituted elements."""
    from pymatgen.core import Composition
    rows, meta = [], []
    for i, f in enumerate(formulas):
        comp = Composition(f)
        own = comp.reduced_formula
        els = [e.symbol for e in comp.elements]
        an = e2.element_analogues(f, odds)
        n_raw = len(an)
        for a in an:
            if a["formula"] == own:   # degenerate swap that reproduces the parent
                continue
            rows.append((i, a["formula"], a["r"], a["n_sub"], ref_index.get(a["formula"], -1),
                         ",".join(a["mapping"].keys())))
        anon = comp.anonymized_formula
        meta.append(dict(owner=i, formula=f, reduced=own, n_el=len(els), elements=",".join(sorted(els)),
                         anon=anon, chemsys="-".join(sorted(els)), n_raw=n_raw,
                         f_block=bool(F_BLOCK & set(els))))
    A = pd.DataFrame(rows, columns=["owner", "formula", "r", "n_sub", "mp_idx", "subbed"])
    return A, pd.DataFrame(meta)


# Calibration ##########################################################################


def design(A, M, cov):
    X = np.log(A["r"].values)[:, None]
    if not cov:
        return X
    n_el = np.clip(M["n_el"].values[A["owner"].values], 2, 5)
    oh = np.stack([n_el == k for k in (3, 4, 5)], axis=1).astype(float)
    cols = [X, A["n_sub"].values[:, None] - 1, oh]
    if cov == "n":   # also the parent's (log) number of analogues
        cols.append(np.log(M["n_raw"].values[A["owner"].values])[:, None])
    return np.hstack(cols)


def fit_pi(A, M, hits, mask, cov=False):
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(design(A[mask], M, cov), hits[mask])
    return lambda AA: lr.predict_proba(design(AA, M, cov))[:, 1]


def present(A, ref_mask):
    """Analogue present in the reference (ref_mask over MP), never counting the owner itself."""
    idx = A["mp_idx"].values
    ok = idx >= 0
    out = np.zeros(len(A), bool)
    out[ok] = ref_mask[idx[ok]] & (idx[ok] != A["owner"].values[ok])
    return out


def pb_midp(p, O):
    """Poisson-binomial mid-p P(X < O) + 0.5 P(X = O), exact DP truncated at O."""
    dp = np.zeros(O + 1)
    dp[0] = 1.0
    for pj in p:
        dp[1:] = dp[1:] * (1 - pj) + dp[:-1] * pj
        dp[0] *= 1 - pj
    return dp[:O].sum() + 0.5 * dp[O]


def scores(A, pi, hit, n_owner, midp=True):
    own = A["owner"].values
    n = np.bincount(own, minlength=n_owner)
    E = np.bincount(own, weights=pi, minlength=n_owner)
    O = np.bincount(own, weights=hit.astype(float), minlength=n_owner).astype(int)
    var = np.bincount(own, weights=pi * (1 - pi), minlength=n_owner)
    V = np.log((E + 0.5) / (O + 0.5))
    out = dict(n=n, E=E, O=O, V=V, z=(E - O) / np.sqrt(np.maximum(var, 1e-12)))
    if midp:
        order = np.argsort(own, kind="stable")
        starts = np.r_[0, np.cumsum(n)]
        ps = pi[order]
        out["midp"] = np.array([pb_midp(ps[starts[i]:starts[i + 1]], O[i]) if n[i] else 0.5
                                for i in range(n_owner)])
    return out


# Statistics ###########################################################################


def auc(y, s):
    ok = np.isfinite(s)
    return roc_auc_score(y[ok], s[ok]) if len(set(y[ok])) == 2 else np.nan


def boot_auc(y, s, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = [auc(y[np.r_[a, b]], s[np.r_[a, b]]) for a, b in
         ((rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(N_BOOT))]
    return np.nanpercentile(v, [2.5, 97.5])


def strat_auc(y, s, strata):
    """Pair-weighted mean of within-stratum AUCs (strata with both classes)."""
    num = den = 0.0
    for g in np.unique(strata):
        m = strata == g
        npos, nneg = (y[m] == 1).sum(), (y[m] == 0).sum()
        if npos and nneg:
            num += auc(y[m], s[m]) * npos * nneg
            den += npos * nneg
    return num / den if den else np.nan


def partial_spearman(x, y, covs):
    rx, ry = rankdata(x), rankdata(y)
    C = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in covs])
    res = [v - C @ np.linalg.lstsq(C, v, rcond=None)[0] for v in (rx, ry)]
    return np.corrcoef(*res)[0, 1]


# Main #################################################################################


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    B = e2.load_basics()
    odds = e2.element_odds()
    from pymatgen.core import Composition
    mp_red = [Composition(f).reduced_formula for f in B["mp_formulas"]]
    ref_index = {f: i for i, f in enumerate(mp_red)}
    formulas = B["mp_formulas"] + B["gn_formulas"]
    A, M = build(formulas, odds, ref_index)
    N = len(formulas)
    is_gn = np.r_[np.zeros(N_MP, int), np.ones(N_GN, int)]
    M["source"] = np.where(is_gn, "GNoME", "MP")
    a_mp = A["owner"].values < N_MP
    all_mp = np.ones(N_MP, bool)

    # 1. reproduce E2 ---------------------------------------------------------------------
    n_raw_mp = int(M["n_raw"][:N_MP].sum())
    e2_hits_raw = []   # E2 definition: analogue formula in MP (no LOO, degenerate swaps kept)
    for f in B["mp_formulas"]:
        e2_hits_raw += [(a["r"], a["formula"] in ref_index) for a in e2.element_analogues(f, odds)]
    rr, hh = np.array(e2_hits_raw).T
    lr = LogisticRegression(C=1e6).fit(np.log(rr)[:, None], hh.astype(bool))
    cal_e2 = pd.read_csv(os.path.join(OUT, "e2_calibration.csv"))
    summ_e2 = pd.read_csv(os.path.join(OUT, "e2_analogue_summary.csv"))
    gn_inmp = [sum(a["formula"] in ref_index for a in e2.element_analogues(f, odds)) for f in B["gn_formulas"]]
    repro = dict(n_mp_analogues=n_raw_mp, e2_n=int(cal_e2["n"].sum()),
                 logit_a=lr.intercept_[0], e2_logit_a=cal_e2["logit_a"][0],
                 logit_b=lr.coef_[0, 0], e2_logit_b=cal_e2["logit_b"][0],
                 gn_analogues_in_mp=sum(gn_inmp), e2_gn_in_mp=int(summ_e2["n_elem_in_mp"].sum()),
                 gn_counts_match=bool(np.array_equal(
                     np.array(gn_inmp), summ_e2.set_index("candidate").loc[B["gn_formulas"], "n_elem_in_mp"].values)),
                 n_degenerate_self_analogues=int(n_raw_mp - a_mp.sum()))
    print("reproduction", repro)
    assert repro["n_mp_analogues"] == repro["e2_n"] and repro["gn_counts_match"]
    assert abs(repro["logit_a"] - repro["e2_logit_a"]) < 1e-3

    # 2. scores -----------------------------------------------------------------------------
    hit_full = present(A, all_mp)  # MP: leave-one-out; GNoME: full MP
    dens = {m: np.r_[B[m]["loo"], B[m]["dens"]] for m in e2.METRICS}
    for m in e2.METRICS:
        M[f"density_{m}"] = dens[m]
        M[f"novel10_{m}"] = dens[m] < B[m]["thr"][10]
        M[f"novel5_{m}"] = dens[m] < B[m]["thr"][5]
    anon_counts = M[:N_MP]["anon"].value_counts()
    M["anon_seen"] = [anon_counts.get(a, 0) - (i < N_MP) > 0 for i, a in enumerate(M["anon"])]
    cs_counts = M[:N_MP]["chemsys"].value_counts()
    M["chemsys_seen"] = [cs_counts.get(a, 0) - (i < N_MP) > 0 for i, a in enumerate(M["chemsys"])]

    variants = {}
    pi_in = fit_pi(A, M, hit_full, a_mp)(A)
    variants["insample"] = scores(A, pi_in, hit_full, N)
    pi_cov = None
    for kind in ("random", "anon_grouped", "random_cov", "random_covn"):
        reps = []
        for rep in range(N_CROSS):
            if kind == "anon_grouped":
                anons = M["anon"][:N_MP].unique()
                fold_of = dict(zip(anons, rng.integers(0, 2, len(anons))))
                fold = np.array([fold_of[a] for a in M["anon"][:N_MP]])
            else:
                fold = rng.permutation(np.arange(N_MP) % 2)
            pis = []
            for k in (0, 1):
                fit_mask = a_mp & (fold[np.minimum(A["owner"].values, N_MP - 1)] == k)
                pis.append(fit_pi(A, M, hit_full, fit_mask, cov={"random_cov": True, "random_covn": "n"}.get(kind, False))(A))
            own_fold = np.r_[fold, np.full(N_GN, -1)][A["owner"].values]
            # out-of-fold for MP; GNoME scored with the mean of the two fold calibrations
            pi = np.where(own_fold == 0, pis[1], np.where(own_fold == 1, pis[0], 0.5 * (pis[0] + pis[1])))
            reps.append(scores(A, pi, hit_full, N, midp=(rep == 0)))
            if kind == "random_cov" and rep == 0:
                pi_cov = pi
        variants[kind] = {k: (np.mean([r[k] for r in reps], axis=0) if k != "midp" else reps[0][k])
                          for k in reps[0]}
        variants[kind]["_aucs"] = [auc(is_gn, r["V"]) for r in reps]
    main_v = variants["random"]   # cross-fitted, primary
    for k in ("n", "E", "O", "V", "z", "midp"):
        M[k] = main_v[k]
    M["V_insample"] = variants["insample"]["V"]
    M["V_anon_grouped"] = variants["anon_grouped"]["V"]
    M["V_cov"] = variants["random_cov"]["V"]
    M["midp_cov"] = variants["random_cov"]["midp"]
    M["V_covn"] = variants["random_covn"]["V"]
    v95 = np.percentile(M["V"][:N_MP], 95)
    M["vacant_V95"] = M["V"] > v95
    M["vacant_p05"] = M["midp"] < 0.05
    M.drop(columns=["owner"]).to_csv(os.path.join(OUT, "e6_per_material.csv"), index=False)

    # 3a. AUCs ------------------------------------------------------------------------------
    y = is_gn
    nel = M["n_el"].values
    n_bins = np.digitize(np.log1p(M["n"].values), np.quantile(np.log1p(M["n"].values[:N_MP]), [.25, .5, .75]))
    n_dec = np.digitize(np.log1p(M["n"].values), np.quantile(np.log1p(M["n"].values[:N_MP]), np.arange(1, 10) / 10))
    anon_seen, fb = M["anon_seen"].values, M["f_block"].values
    score_defs = {
        "V (cross-fit)": main_v["V"], "V (in-sample)": variants["insample"]["V"],
        "V (anon-grouped cross-fit)": variants["anon_grouped"]["V"],
        "V (covariate pi0, cross-fit)": variants["random_cov"]["V"],
        "V (covariate + log n pi0, cross-fit)": variants["random_covn"]["V"],
        "-log midp (cross-fit)": -np.log(np.maximum(main_v["midp"], 1e-300)),
        "-log midp (covariate pi0)": -np.log(np.maximum(variants["random_cov"]["midp"], 1e-300)),
        "z (cross-fit)": main_v["z"],
        "-density LoStOP": -dens["lostop"], "-density ElMD": -dens["elmd"],
        "baseline: n analogues": M["n"].values.astype(float), "baseline: E": main_v["E"],
        "baseline: -O": -main_v["O"].astype(float), "baseline: n_elements": nel.astype(float),
        "baseline: anon formula unseen": (~anon_seen).astype(float),
        "baseline: chemsys unseen": (~M["chemsys_seen"].values).astype(float),
    }
    rows = []
    for name, s in score_defs.items():
        ci = boot_auc(y, s, rng)
        rows.append(dict(score=name, AUC=auc(y, s), CI_lo=ci[0], CI_hi=ci[1],
                         AUC_strat_n_elements=strat_auc(y, s, nel),
                         AUC_strat_n_analogues_quartile=strat_auc(y, s, n_bins),
                         AUC_strat_nel_x_nquartile=strat_auc(y, s, nel * 10 + n_bins),
                         AUC_strat_n_analogues_decile=strat_auc(y, s, n_dec),
                         AUC_top_n_decile_only=auc(y[n_dec == 9], s[n_dec == 9]),
                         AUC_anon_seen_only=auc(y[anon_seen], s[anon_seen]),
                         AUC_no_fblock=auc(y[~fb], s[~fb]),
                         n_GNoME_anon_seen=int(y[anon_seen].sum()), n_GNoME_no_fblock=int(y[~fb].sum()),
                         n_GNoME_top_n_decile=int(y[n_dec == 9].sum())))
    for kind in ("random", "anon_grouped", "random_cov", "random_covn"):
        a = variants[kind]["_aucs"]
        rows.append(dict(score=f"V ({kind}) AUC over {N_CROSS} cross-fit repeats", AUC=np.mean(a),
                         CI_lo=np.min(a), CI_hi=np.max(a)))
    aucs = pd.DataFrame(rows)
    aucs.to_csv(os.path.join(OUT, "e6_auc.csv"), index=False)
    print(aucs.round(3).to_string())

    # 3b. correlations ------------------------------------------------------------------------
    rows = []
    for m in e2.METRICS:
        for grp, mask in (("all", np.ones(N, bool)), ("MP", y == 0), ("GNoME", y == 1)):
            for vname in ("V", "V_cov", "V_covn", "midp"):
                x = M[vname].values[mask] if vname != "midp" else -np.log(np.maximum(M["midp"].values[mask], 1e-300))
                d = dens[m][mask]
                rho, p = spearmanr(x, d)
                rows.append(dict(metric=m, group=grp, score=vname if vname != "midp" else "-log midp",
                                 spearman=rho, p=p,
                                 partial_ctrl_n_nel=partial_spearman(x, d, [M["n"].values[mask], nel[mask]]),
                                 spearman_density_vs_n=spearmanr(d, M["n"].values[mask])[0],
                                 spearman_V_vs_n=spearmanr(x, M["n"].values[mask])[0]))
    corr = pd.DataFrame(rows)
    corr.to_csv(os.path.join(OUT, "e6_correlations.csv"), index=False)
    print(corr.round(3).to_string())

    # 3c. GNoME not novel by density ----------------------------------------------------------
    G = M[M["source"] == "GNoME"]
    rows = []
    for m in e2.METRICS:
        for q in (5, 10):
            nn = G[~G[f"novel{q}_{m}"]]
            rows.append(dict(metric=m, q=q, n_GNoME=len(G), n_novel_density=int(G[f"novel{q}_{m}"].sum()),
                             n_not_novel=len(nn), n_not_novel_vacant_V95=int(nn["vacant_V95"].sum()),
                             n_not_novel_vacant_p05=int(nn["vacant_p05"].sum()),
                             n_not_novel_vacant_both=int((nn["vacant_V95"] & nn["vacant_p05"]).sum()),
                             MP_frac_vacant_V95=M[:N_MP]["vacant_V95"].mean(),
                             MP_frac_vacant_p05=M[:N_MP]["vacant_p05"].mean(),
                             MP_not_novel_frac_vacant_V95=M[:N_MP][~M[:N_MP][f"novel{q}_{m}"]]["vacant_V95"].mean(),
                             V95_threshold=v95))
    tc = pd.DataFrame(rows)
    tc.to_csv(os.path.join(OUT, "e6_not_novel_vacant.csv"), index=False)
    print(tc.to_string())

    # elements: missing expected mass by substituted element (GNoME), f-block enrichment (MP)
    miss = ~hit_full & (A["owner"].values >= N_MP)
    contrib = {}
    for s_el, p in zip(A["subbed"].values[miss], pi_in[miss]):
        parts = s_el.split(",")
        for e in parts:
            contrib[e] = contrib.get(e, 0.0) + p / len(parts)
    tot = sum(contrib.values())
    el_rows = [dict(scope="GNoME missing expected mass by substituted element", element=e, value=v,
                    frac=v / tot) for e, v in sorted(contrib.items(), key=lambda t: -t[1])[:15]]
    Mp = M[:N_MP]
    top = Mp["V"] >= np.percentile(Mp["V"], 90)
    el_rows += [dict(scope="MP f-block fraction", element="all MP", value=Mp["f_block"].mean()),
                dict(scope="MP f-block fraction", element="MP top-10% V", value=Mp["f_block"][top].mean()),
                dict(scope="GNoME f-block fraction", element="all GNoME", value=G["f_block"].mean())]
    all_els = sorted({e for s in M["elements"] for e in s.split(",")} - {"Li"})
    for e in all_els:
        has = M["elements"].str.split(",").apply(lambda s: e in s).values
        mp_has, gn_has = has[:N_MP], has[N_MP:]
        if gn_has.sum():
            el_rows.append(dict(scope="per element: mean V (MP | GNoME)", element=e,
                                value=Mp["V"][mp_has].mean() if mp_has.any() else np.nan,
                                frac=G["V"][gn_has].mean(), n_mp=int(mp_has.sum()), n_gn=int(gn_has.sum()),
                                mp_frac_top10=top[mp_has].mean() if mp_has.any() else np.nan))
    elements = pd.DataFrame(el_rows)
    elements.to_csv(os.path.join(OUT, "e6_elements.csv"), index=False)
    print(elements.head(20).to_string())
    print(G.sort_values("V", ascending=False)[["formula", "n", "E", "O", "V", "midp", "anon_seen", "f_block",
                                               "novel10_lostop", "novel10_elmd"]].to_string())

    # 4. falsification ------------------------------------------------------------------------
    rows = []
    for kind in ("random", "chemsys_grouped"):
        for rep in range(N_FALS):
            if kind == "random":
                K = rng.permutation(np.arange(N_MP) % 2) == 0
            else:
                cs = M["chemsys"][:N_MP].unique()
                cs_fold = dict(zip(cs, rng.integers(0, 2, len(cs))))
                K = np.array([cs_fold[c] == 0 for c in M["chemsys"][:N_MP]])
            hitK = present(A, K)
            ownK = np.r_[K, np.zeros(N_GN, bool)][A["owner"].values]
            piK = fit_pi(A, M, hitK, a_mp & ownK)(A)
            sK = scores(A, piK, hitK, N, midp=False)
            P = ~K
            yKP = np.r_[np.zeros(K.sum()), np.ones(P.sum())]
            vKP = np.r_[sK["V"][:N_MP][K], sK["V"][:N_MP][P]]
            yKG = np.r_[np.zeros(K.sum()), np.ones(N_GN)]
            vKG = np.r_[sK["V"][:N_MP][K], sK["V"][N_MP:]]
            rows.append(dict(split=kind, rep=rep, n_K=int(K.sum()),
                             AUC_pseudo_vs_known=auc(yKP, vKP), AUC_GNoME_vs_known=auc(yKG, vKG),
                             mean_V_known=sK["V"][:N_MP][K].mean(), mean_V_pseudo=sK["V"][:N_MP][P].mean(),
                             mean_V_GNoME=sK["V"][N_MP:].mean(),
                             obs_over_exp_known=sK["O"][:N_MP][K].sum() / sK["E"][:N_MP][K].sum(),
                             obs_over_exp_pseudo=sK["O"][:N_MP][P].sum() / sK["E"][:N_MP][P].sum(),
                             obs_over_exp_GNoME=sK["O"][N_MP:].sum() / sK["E"][N_MP:].sum()))
    fal = pd.DataFrame(rows)
    fal.to_csv(os.path.join(OUT, "e6_falsification.csv"), index=False)
    print(fal.groupby("split").agg(["mean", "min", "max"]).T.round(3).to_string())

    summary = dict(repro, V95_threshold=v95,
                   MP_obs_over_exp=Mp["O"].sum() / Mp["E"].sum(), GNoME_obs_over_exp=G["O"].sum() / G["E"].sum(),
                   GNoME_E_total=G["E"].sum(), GNoME_O_total=int(G["O"].sum()),
                   GNoME_n_O0=int((G["O"] == 0).sum()), GNoME_anon_seen=int(G["anon_seen"].sum()),
                   MP_anon_seen_frac=Mp["anon_seen"].mean(), runtime_s=time.time() - t0)
    pd.Series(summary).to_csv(os.path.join(OUT, "e6_summary.csv"), header=["value"])
    print(pd.Series(summary).to_string())
    figure(M, B, v95, fal)
    print(f"total {time.time() - t0:.0f}s")


def figure(M, B, v95, fal):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    mp, gn = M[M["source"] == "MP"], M[M["source"] == "GNoME"]
    for ax, m, title in zip(axes[:2], e2.METRICS, ("LoStOP (structure)", "ElMD (composition)")):
        off = 1.0
        ax.scatter(mp[f"density_{m}"] + off, mp["V"], s=8, c="#a3a29c", alpha=0.5, lw=0, label="MP (leave-one-out)")
        ax.scatter(gn[f"density_{m}"] + off, gn["V"], s=26, c="#2a78d6", edgecolor="white", lw=0.8,
                   label="GNoME candidates")
        ax.axvline(B[m]["thr"][10] + off, color="#52514e", lw=1, ls="--", label="density novel (MP 10th pct)")
        ax.axhline(v95, color="#eb6834", lw=1, ls="--", label="vacant (MP 95th pct)")
        ax.set_xscale("log")
        ax.set_xlabel("MINov density + 1 (low = novel)")
        ax.set_ylabel("analogue vacancy V = log((E+0.5)/(O+0.5))")
        ax.set_title(title, loc="left")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.36, 0.0))
    ax = axes[2]
    r = fal[fal["split"] == "random"]
    bins = np.linspace(0.3, 1.0, 36)
    ax.hist(r["AUC_pseudo_vs_known"], bins=bins, color="#a3a29c", label="MP pseudo-candidates vs known")
    ax.hist(r["AUC_GNoME_vs_known"], bins=bins, color="#2a78d6", label="GNoME vs known")
    ax.axvline(0.5, color="#52514e", lw=1, ls=":")
    ax.set_xlabel("AUC of V (higher = looks more vacant)")
    ax.set_ylabel(f"random 50/50 MP splits (n={len(r)})")
    ax.set_title("Falsification", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT, "e6_vacancy_vs_density.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
