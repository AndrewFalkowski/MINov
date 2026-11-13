"""E2: How easily can a NOVEL candidate be de-novelled by plausible missing references?

Removing references only makes a candidate more novel, so the threat to a "novel" verdict
is ADDING materials that exist but are missing from the reference (MP). Li dataset,
reference = 1834 MP materials, candidates = 44 GNoME materials.

Novel := MINov density below the q-th percentile (q = 5, 10) of MP leave-one-out densities
(reference density minus each material's own self weight w(0) = 1). Candidate densities
are computed against the reference only (no self term), on the same footing.

Task 1  Insertion budget. A candidate with density rho and threshold T needs
        m*(d) = ceil((T - rho) / w(d)) added references at distance d to lose novelty.
        Compared with a 1-NN criterion (novel := NN distance above the 95th percentile of
        MP leave-one-out NN distances), which flips with ONE addition closer than that.
Task 2  Plausible neighbours. Substitution analogues of each candidate from the
        Hautier et al. data-mined substitution table bundled with pymatgen:
          * ionic model (SubstitutionPredictor, charge balanced) when oxidation states
            can be assigned (oxi_state_guesses, else BVAnalyzer) -- fails for most GNoME
            Li materials, which are intermetallics, so it is reported as a check only;
          * element model (main): the same lambda table restricted to isovalent pairs and
            summed over oxidation states, W(a, b) = sum_x exp(lambda(a^x, b^x)), so a swap
            keeps charge balance whatever the (unknown) oxidation states; a site's odds are
            r(b -> a) = W(a, b) / W(b, b), an analogue's odds are the product over sites.
            Li is kept fixed (the reference is Li materials), r >= R_MIN.
        Analogues whose reduced formula is in MP are dropped (already in the reference).
        Existence probability pi_j = min(1, s * pi0(r_j)), pi0 a logistic fit in log r to
        the fraction of MP materials' analogues that are also in MP; s scanned (s > 1
        allows for MP's incompleteness).
        Distances: ElMD exactly from the formula; LoStOP (a) "unrelaxed": substitution
        applied to the candidate's own structure and featurised (top K analogues by r,
        remaining analogues imputed with the candidate's mean weight); (b) "empirical":
        distances drawn from LoStOP distances of real MP pairs that differ by exactly one
        element substitution (relaxed, possibly different structure type).
        P(de-novelled) by Monte Carlo over independent existence of analogues.
Task 3  Masking: each candidate scored with the other 43 GNoME added to the reference
        (fixed MP profile/threshold, and a variant refitting profile and threshold).

Usage: uv run python experiments/e2_denovelling.py [--reuse]  (--reuse: load saved analogues)
"""

import json
import os
import sys
import time
import warnings
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO]
from MINOV.density import MIProfile  # noqa: E402
from MINOV.distance import get_elmd_dm, get_lostop_features  # noqa: E402

OUT = os.path.join(REPO, "experiments", "results")
MP_DIR = os.path.join(REPO, "data", "MP_Li_dataset")
GN_DIR = os.path.join(REPO, "data", "GNOME_Li_dataset")
N_MP = 1834
QS = (5, 10)
R_MIN = 0.03          # minimum analogue odds relative to the candidate itself
TOP_K = 20            # analogues per candidate featurised for LoStOP
N_MC = 4000
SCALES = (0.3, 1, 3, 10)   # multipliers on the fitted existence probability pi0(r)
SEED = 0
METRICS = ("lostop", "elmd")

warnings.filterwarnings("ignore")


# Reference set ########################################################################


def load_basics():
    mp_files = [f for f in os.listdir(MP_DIR) if f.endswith(".cif")]
    gn_files = [f for f in os.listdir(GN_DIR) if f.endswith(".cif")]
    assert len(mp_files) == N_MP and len(gn_files) == 44
    out = {"mp_formulas": [f[:-4] for f in mp_files], "gn_formulas": [f[:-4] for f in gn_files],
           "gn_files": gn_files}
    for m in METRICS:
        D = np.load(os.path.join(REPO, "precomputed", f"Li_{m}_dm.npy"))
        R, Q, G = D[:N_MP, :N_MP], D[N_MP:, :N_MP], D[N_MP:, N_MP:]
        prof = MIProfile(R)
        loo = prof.density(R) - prof.weight(0.0)
        nn_loo = np.sort(R + np.diag(np.full(N_MP, np.inf)), axis=1)[:, 0]
        out[m] = dict(R=R, Q=Q, G=G, D=D, prof=prof, loo=loo,
                      thr={q: np.percentile(loo, q) for q in QS},
                      dens=prof.density(Q), nn=Q.min(axis=1),
                      nn_thr=np.percentile(nn_loo, 95))
    return out


# Task 1: insertion budget #############################################################


def task1(B):
    rows, curves = [], []
    for m in METRICS:
        b = B[m]
        prof = b["prof"]
        grid = np.linspace(0, prof.cutoff, 101)
        wg = prof.weight(grid)
        for i, f in enumerate(B["gn_formulas"]):
            row = dict(metric=m, candidate=f, density=b["dens"][i], nn_dist=b["nn"][i],
                       nn_thr95=b["nn_thr"], nn_novel=b["nn"][i] > b["nn_thr"],
                       cutoff=prof.cutoff)
            for q in QS:
                gap = b["thr"][q] - b["dens"][i]
                row[f"thr{q}"] = b["thr"][q]
                row[f"novel{q}"] = gap > 0
                row[f"gap{q}"] = gap
                row[f"m_star_at_d0_{q}"] = int(np.ceil(gap)) if gap > 0 else 0
                # largest distance at which ONE addition flips it: w(d) >= gap
                ok = grid[wg >= gap] if gap > 0 else grid
                row[f"single_flip_dist{q}"] = ok.max() if gap > 0 and len(ok) else (
                    np.nan if gap > 0 else 0.0)
                if gap > 0:
                    with np.errstate(divide="ignore"):
                        mstar = np.where(wg > 0, np.ceil(gap / np.maximum(wg, 1e-300)), np.inf)
                    for d, ms in zip(grid, mstar):
                        curves.append(dict(metric=m, q=q, candidate=f, d=d, d_over_cutoff=d / prof.cutoff,
                                           w=prof.weight(d), m_star=ms))
            rows.append(row)
    t1 = pd.DataFrame(rows)
    t1.to_csv(os.path.join(OUT, "e2_insertion_budget.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(OUT, "e2_insertion_curves.csv"), index=False)
    return t1, pd.DataFrame(curves)


# Task 2: substitution models ##########################################################


def lambda_table():
    import pymatgen.analysis.structure_prediction as sp_mod
    with open(os.path.join(os.path.dirname(sp_mod.__file__), "data", "lambda.json")) as fh:
        return json.load(fh)


def element_odds():
    """odds[old] = [(new, r)], r = W(new, old) / W(old, old), W summed over isovalent species pairs."""
    from pymatgen.core import Species
    W = defaultdict(float)
    for s1, s2, lam in lambda_table():
        if "D1+" in (s1, s2):
            continue
        sp1, sp2 = Species.from_str(s1), Species.from_str(s2)
        if sp1.oxi_state != sp2.oxi_state:  # isovalent swaps only: keeps charge balance
            continue
        a, b = sp1.symbol, sp2.symbol
        W[(a, b)] += np.exp(lam)
        if a != b:
            W[(b, a)] += np.exp(lam)
    odds = defaultdict(list)  # old -> [(new, r)] sorted desc
    for (new, old), w in W.items():
        if new != old and W.get((old, old), 0) > 0 and new != "Li":
            r = w / W[(old, old)]
            if r >= R_MIN:
                odds[old].append((new, r))
    for k in odds:
        odds[k].sort(key=lambda t: -t[1])
    return odds


def element_analogues(formula, odds):
    """All Li-keeping element-substitution analogues with product odds >= R_MIN."""
    from pymatgen.core import Composition
    comp = Composition(formula)
    els = [e.symbol for e in comp.elements]
    opts = [[(e, 1.0)] + ([] if e == "Li" else odds.get(e, [])) for e in els]
    out = []

    def rec(i, chosen, r):
        if i == len(els):
            if len(set(chosen)) == len(chosen) and chosen != els:
                new = Composition({c: comp[e] for c, e in zip(chosen, els)})
                out.append(dict(formula=new.reduced_formula, r=r,
                                mapping={e: c for e, c in zip(els, chosen) if e != c},
                                n_sub=sum(e != c for e, c in zip(els, chosen))))
            return
        for c, ri in opts[i]:
            if r * ri < R_MIN:
                break
            rec(i + 1, chosen + [c], r * ri)

    rec(0, [], 1.0)
    # keep the highest-odds route to each reduced formula
    best = {}
    for a in out:
        if a["formula"] not in best or a["r"] > best[a["formula"]]["r"]:
            best[a["formula"]] = a
    return sorted(best.values(), key=lambda a: -a["r"])


def ionic_analogues(structure):
    """Charge-balanced analogues from pymatgen's SubstitutionPredictor, or (None, reason)."""
    from pymatgen.analysis.bond_valence import BVAnalyzer
    from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
    from pymatgen.core import Composition, Species
    comp = structure.composition.reduced_composition
    pred = SubstitutionPredictor(threshold=1e-3)
    tried = []
    guesses = comp.oxi_state_guesses(max_sites=-50)
    if guesses:
        tried.append(("oxi_state_guesses", guesses[0]))
    try:
        vals = BVAnalyzer().get_valences(structure)
        ox = {}
        for site, v in zip(structure, vals):
            ox.setdefault(site.specie.symbol, v)
        tried.append(("BVAnalyzer", ox))
    except Exception:
        pass
    if not tried:
        return None, "no oxidation states"
    reason = "species not in table"
    for method, ox in tried:
        if any(abs(v - round(v)) > 1e-6 for v in ox.values()):
            reason = "fractional oxidation states"
            continue
        species = {Species(e, int(round(v))): comp[e] for e, v in ox.items()}
        if not all(s in pred.p.species for s in species):
            continue
        scomp = Composition(species)
        p_id = pred.p.cond_prob_list(list(scomp), list(scomp))
        res = []
        for p in pred.composition_prediction(scomp, to_this_composition=False):
            sub = p["substitutions"]
            if all(k == v for k, v in sub.items()):
                continue
            new = Composition({sub[s].symbol: amt for s, amt in scomp.items()})
            if "Li" not in [e.symbol for e in new.elements]:
                continue
            res.append(dict(formula=new.reduced_formula, r=p["probability"] / p_id))
        best = {}
        for a in res:
            if a["formula"] not in best or a["r"] > best[a["formula"]]["r"]:
                best[a["formula"]] = a
        return (sorted(best.values(), key=lambda a: -a["r"]), method), None
    return None, reason


def calibrate(B, odds, mp_set):
    """Existence probability pi0(r) = sigmoid(a + b log r), fitted (max likelihood) on how often
    MP materials' own analogues are themselves in MP. MP is incomplete, so this is a lower
    bound on existence; the SCALES scan multiplies pi0 (capped at 1)."""
    from sklearn.linear_model import LogisticRegression
    rs, hits = [], []
    for f in B["mp_formulas"]:
        for a in element_analogues(f, odds):
            rs.append(a["r"])
            hits.append(a["formula"] in mp_set)
    rs, hits = np.array(rs), np.array(hits)
    lr = LogisticRegression(C=1e6).fit(np.log(rs)[:, None], hits)
    a, b = float(lr.intercept_[0]), float(lr.coef_[0, 0])

    def pi0(r):
        return 1 / (1 + np.exp(-(a + b * np.log(np.asarray(r, dtype=float)))))

    edges = np.array([R_MIN, 0.05, 0.1, 0.2, 0.4, 1.0, np.inf])
    tab = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rs >= lo) & (rs < hi)
        tab.append(dict(r_lo=lo, r_hi=hi, n=int(m.sum()), frac_in_mp=hits[m].mean() if m.any() else np.nan,
                        mean_r=rs[m].mean() if m.any() else np.nan,
                        fitted_pi=pi0(rs[m]).mean() if m.any() else np.nan))
    tab = pd.DataFrame(tab)
    tab["logit_a"], tab["logit_b"] = a, b
    tab["n_mp_formulas"] = len(B["mp_formulas"])
    tab.to_csv(os.path.join(OUT, "e2_calibration.csv"), index=False)
    return pi0, tab


def empirical_single_sub_pairs(B, odds):
    """LoStOP / ElMD distances of MP pairs that differ by exactly one (non-Li) element swap."""
    from pymatgen.core import Composition
    keys = defaultdict(list)
    for i, f in enumerate(B["mp_formulas"]):
        c = Composition(f).reduced_composition
        items = {e.symbol: c[e] for e in c.elements}
        for e in items:
            if e == "Li":
                continue
            key = tuple(sorted((k if k != e else "*", v) for k, v in items.items()))
            keys[key].append((i, e))
    rows = []
    for key, members in keys.items():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                (i, ei), (j, ej) = members[a], members[b]
                if ei == ej:
                    continue
                r = max([rr for nn, rr in odds.get(ei, []) if nn == ej] + [0.0])
                rows.append(dict(i=i, j=j, el_i=ei, el_j=ej, r=r,
                                 lostop=B["lostop"]["R"][i, j], elmd=B["elmd"]["R"][i, j]))
    df = pd.DataFrame(rows).drop_duplicates(["i", "j"])
    for m in METRICS:
        df[f"w_{m}"] = B[m]["prof"].weight(df[m].values)
    df.to_csv(os.path.join(OUT, "e2_mp_single_sub_pairs.csv"), index=False)
    return df


def build_analogues(B, structures, odds, mp_set):
    from pymatgen.core import Composition
    gn_set = {Composition(f).reduced_formula: f for f in B["gn_formulas"]}
    recs, info = [], []
    for ci, (f, s) in enumerate(zip(B["gn_formulas"], structures)):
        all_an = element_analogues(f, odds)
        keep = [a for a in all_an if a["formula"] not in mp_set]
        ion, why = ionic_analogues(s)
        ion_list = None
        if ion is not None:
            ion_list = [a for a in ion[0] if a["formula"] not in mp_set]
        info.append(dict(candidate=f, n_elem_analogues=len(all_an), n_elem_in_mp=len(all_an) - len(keep),
                         n_elem_kept=len(keep), n_elem_is_other_gnome=sum(a["formula"] in gn_set for a in keep),
                         ionic_ok=ion is not None, ionic_method=ion[1] if ion else None,
                         ionic_fail_reason=why, n_ionic_kept=len(ion_list) if ion_list is not None else np.nan,
                         n_ionic_in_mp=(len(ion[0]) - len(ion_list)) if ion else np.nan,
                         ionic_elem_overlap=(len({a["formula"] for a in ion_list} & {a["formula"] for a in keep})
                                             if ion_list is not None else np.nan)))
        for rank, a in enumerate(keep):
            recs.append(dict(cand_idx=ci, candidate=f, model="element", rank=rank, formula=a["formula"],
                             r=a["r"], n_sub=a["n_sub"], mapping=json.dumps(a["mapping"]),
                             featurise=rank < TOP_K, other_gnome=a["formula"] in gn_set))
        for rank, a in enumerate(ion_list or []):
            recs.append(dict(cand_idx=ci, candidate=f, model="ionic", rank=rank, formula=a["formula"],
                             r=a["r"], n_sub=np.nan, mapping="", featurise=False,
                             other_gnome=a["formula"] in gn_set))
    return pd.DataFrame(recs), pd.DataFrame(info)


def add_distances(B, A, structures, gn_feats):
    # ElMD: exact from formulas
    A["d_elmd"] = np.nan
    for ci, f in enumerate(B["gn_formulas"]):
        m = A["cand_idx"] == ci
        if m.any():
            A.loc[m, "d_elmd"] = get_elmd_dm([f], reference=list(A.loc[m, "formula"]))[0]
    A["w_elmd"] = B["elmd"]["prof"].weight(A["d_elmd"].values)
    # LoStOP: apply the element substitution to the candidate's own structure
    sel = A[(A["model"] == "element") & A["featurise"]]
    new_structs = []
    for _, a in sel.iterrows():
        s = structures[a["cand_idx"]].copy()
        s.replace_species(json.loads(a["mapping"]))
        new_structs.append(s)
    t = time.time()
    F = get_lostop_features(new_structs)
    print(f"featurised {len(new_structs)} analogues in {time.time() - t:.0f}s")
    A["d_lostop_unrelaxed"] = np.nan
    A.loc[sel.index, "d_lostop_unrelaxed"] = np.linalg.norm(F - gn_feats[sel["cand_idx"].values], axis=1)
    A["w_lostop_unrelaxed"] = B["lostop"]["prof"].weight(A["d_lostop_unrelaxed"].fillna(np.inf).values)
    A["w_lostop_unrelaxed_imputed"] = A["w_lostop_unrelaxed"]
    for ci in A["cand_idx"].unique():
        m = A["cand_idx"] == ci
        fm = m & A["featurise"]
        A.loc[m & ~A["featurise"], "w_lostop_unrelaxed_imputed"] = A.loc[fm, "w_lostop_unrelaxed"].mean() \
            if fm.any() else np.nan
    return A


def added_samples(pi, w_fixed, emp_w, rng):
    """Monte Carlo draws of the added density sum_j B_j w_j, B_j ~ Bernoulli(pi_j) independent.
    w_fixed: per-analogue weights, or None to draw each existing analogue's weight from emp_w."""
    if len(pi) == 0:
        return np.zeros(N_MC)
    exists = rng.random((N_MC, len(pi)), dtype=np.float32) < pi.astype(np.float32)
    if w_fixed is not None:
        return exists.astype(np.float32) @ w_fixed.astype(np.float32)
    counts = exists.sum(axis=1)
    draws = rng.choice(emp_w, size=(N_MC, max(1, counts.max())))
    return np.where(np.arange(draws.shape[1]) < counts[:, None], draws, 0.0).sum(axis=1)


def task2(B, A, pi0, pairs, rng):
    emp_w = pairs["w_lostop"].values
    rows = []
    for ci, f in enumerate(B["gn_formulas"]):
        for model in ("element", "ionic"):
            a = A[(A["cand_idx"] == ci) & (A["model"] == model)]
            if model == "ionic" and a.empty:
                continue
            for scale in SCALES:
                pi = np.minimum(1.0, scale * pi0(a["r"].values))
                variants = {"elmd": ("elmd", a["w_elmd"].values)}
                if model == "element":
                    variants["lostop_unrelaxed"] = ("lostop", a["w_lostop_unrelaxed_imputed"].fillna(0).values)
                variants["lostop_empirical"] = ("lostop", None)
                for vname, (m, w_fixed) in variants.items():
                    b = B[m]
                    dens = b["dens"][ci]
                    mean_w = w_fixed if w_fixed is not None else np.full(len(a), emp_w.mean())
                    row = dict(candidate=f, model=model, scale=scale, variant=vname,
                               metric=m, n_analogues=len(a), sum_pi=pi.sum(), density=dens,
                               expected_added=float(np.sum(pi * mean_w)))
                    novel_any = any(dens < b["thr"][q] for q in QS)
                    added = added_samples(pi, w_fixed, emp_w, rng) if novel_any else None
                    for q in QS:
                        thr = b["thr"][q]
                        row[f"novel{q}"] = dens < thr
                        row[f"P_stays_novel{q}"] = float(np.mean(dens + added < thr)) if dens < thr else 0.0
                    # 1-NN criterion: flips if any existing analogue is closer than the 95th pct NN distance
                    if vname == "elmd":
                        close = a["d_elmd"].values < b["nn_thr"]
                        row["P_nn_stays_novel"] = float(np.prod(1 - pi[close])) if b["nn"][ci] > b["nn_thr"] else 0.0
                    elif vname == "lostop_unrelaxed":
                        close = a["d_lostop_unrelaxed"].values < b["nn_thr"]
                        # non-featurised analogues: assume same closeness rate as featurised ones
                        feat = a["featurise"].values
                        rate = close[feat].mean() if feat.any() else 0.0
                        p_close = np.where(feat, close.astype(float), rate)
                        row["P_nn_stays_novel"] = float(np.prod(1 - pi * p_close)) if b["nn"][ci] > b["nn_thr"] else 0.0
                    else:
                        rate = np.mean(pairs["lostop"].values < b["nn_thr"])
                        row["P_nn_stays_novel"] = float(np.prod(1 - pi * rate)) if b["nn"][ci] > b["nn_thr"] else 0.0
                    row["nn_novel"] = b["nn"][ci] > b["nn_thr"]
                    rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "e2_fragility.csv"), index=False)
    return res


def rank_tests(res):
    """Does P(stays novel) rank novel candidates differently from raw density?"""
    rows = []
    sub = res[res["model"] == "element"]
    for (variant, scale), g in sub.groupby(["variant", "scale"]):
        for q in QS:
            n = g[g[f"novel{q}"]]
            rec = dict(variant=variant, scale=scale, q=q, n_novel=len(n),
                       mean_P_stays=n[f"P_stays_novel{q}"].mean() if len(n) else np.nan,
                       n_P_below_half=int((n[f"P_stays_novel{q}"] < 0.5).sum()))
            if len(n) >= 4 and n[f"P_stays_novel{q}"].nunique() > 1:
                rho, p = spearmanr(-n["density"], n[f"P_stays_novel{q}"])
                rec.update(spearman_negdensity_vs_Pstay=rho, p_value=p)
                # does fragility add information beyond density? partial out the margin
                margin = n["density"].values
                resid_rank = spearmanr(n["sum_pi"], n[f"P_stays_novel{q}"])[0]
                rec["spearman_sumpi_vs_Pstay"] = resid_rank
                top = max(1, len(n) // 4)
                by_d = set(n.nsmallest(top, "density")["candidate"])
                by_p = set(n.nlargest(top, f"P_stays_novel{q}")["candidate"])
                rec[f"top_quarter_overlap"] = len(by_d & by_p) / top
            rows.append(rec)
        # all 44: density vs expected post-addition density
        rho_all = spearmanr(g["density"], g["density"] + g["expected_added"])[0]
        rows.append(dict(variant=variant, scale=scale, q="all44", n_novel=len(g),
                         spearman_density_vs_density_plus_expected=rho_all))
    rt = pd.DataFrame(rows)
    rt.to_csv(os.path.join(OUT, "e2_rank_tests.csv"), index=False)
    return rt


# Task 3: masking ######################################################################


def task3(B):
    rows = []
    for m in METRICS:
        b = B[m]
        prof = b["prof"]
        Wg = prof.weight(b["G"])
        np.fill_diagonal(Wg, 0)
        dens_plus = b["dens"] + Wg.sum(axis=1)
        Gd = b["G"] + np.diag(np.full(44, np.inf))
        for i, f in enumerate(B["gn_formulas"]):
            # refit variant: reference = MP + the other 43 GNoME
            idx = np.r_[np.arange(N_MP), N_MP + np.delete(np.arange(44), i)]
            Rr = b["D"][np.ix_(idx, idx)]
            pr = MIProfile(Rr)
            loo = pr.density(Rr) - pr.weight(0.0)
            d_refit = pr.density(b["D"][N_MP + i, idx])
            nn_ext = np.sort(Rr + np.diag(np.full(len(idx), np.inf)), axis=1)[:, 0]
            row = dict(metric=m, candidate=f, density=b["dens"][i], density_plus_gnome=dens_plus[i],
                       added_by_gnome=dens_plus[i] - b["dens"][i],
                       n_gnome_within_cutoff=int((Wg[i] > 0).sum()),
                       nearest_gnome=B["gn_formulas"][int(np.argmin(Gd[i]))], nearest_gnome_dist=Gd[i].min(),
                       nn_novel=b["nn"][i] > b["nn_thr"],
                       nn_novel_plus_gnome=min(b["nn"][i], Gd[i].min()) > b["nn_thr"],
                       nn_novel_refit=min(b["nn"][i], Gd[i].min()) > np.percentile(nn_ext, 95))
            for q in QS:
                row[f"novel{q}"] = b["dens"][i] < b["thr"][q]
                row[f"novel{q}_plus_gnome"] = dens_plus[i] < b["thr"][q]
                row[f"novel{q}_refit"] = d_refit < np.percentile(loo, q)
                row[f"masked{q}"] = row[f"novel{q}"] and not row[f"novel{q}_plus_gnome"]
            rows.append(row)
    t3 = pd.DataFrame(rows)
    t3.to_csv(os.path.join(OUT, "e2_masking.csv"), index=False)
    return t3


# Figures ##############################################################################

COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def figures(B, t1, curves, res):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, m, title in zip(axes, METRICS, ("LoStOP (structure)", "ElMD (composition)")):
        c = curves[(curves["metric"] == m) & (curves["q"] == 10)]
        for k, (f, g) in enumerate(c.groupby("candidate")):
            g = g[np.isfinite(g["m_star"])]
            ax.step(g["d_over_cutoff"], g["m_star"], where="post", lw=1.2, color="#8a8984", alpha=0.6)
        ax.axvline(B[m]["nn_thr"] / B[m]["prof"].cutoff, color=COL[1], lw=2,
                   label="1-NN: one addition closer than this flips")
        ax.axhline(1, color="#52514e", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel("distance of added references / MINov cutoff")
        ax.set_ylabel("additions needed to de-novel, m*(d)")
        n = c["candidate"].nunique()
        ax.set_title(f"{title}: {n} novel GNoME (10th pct)", loc="left")
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e2_insertion_budget.png"), dpi=150)
    plt.close(fig)

    # P(stays novel at the 10th percentile) against the calibration scale s
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sub = res[res["model"] == "element"]
    g = sub[(sub["variant"] == "elmd") & sub["novel10"]]
    ax = axes[0]
    for _, c in g.groupby("candidate"):
        ax.plot(c["scale"], c["P_stays_novel10"], color="#8a8984", lw=1, alpha=0.6, marker="o", ms=3)
    mean = g.groupby("scale")["P_stays_novel10"].mean()
    ax.plot(mean.index, mean.values, color=COL[0], lw=2.5, label=f"MINov, mean of {g['candidate'].nunique()} novel")
    nn = sub[(sub["variant"] == "elmd") & sub["nn_novel"]].groupby("scale")["P_nn_stays_novel"].mean()
    ax.plot(nn.index, nn.values, color=COL[1], lw=2.5, label="1-NN criterion, mean")
    ax.set_title("ElMD (composition)", loc="left")
    ax = axes[1]
    for k, (cand, c) in enumerate(sub[(sub["metric"] == "lostop") & sub["novel10"]].groupby("candidate")):
        for v, ls in (("lostop_unrelaxed", "-"), ("lostop_empirical", "--")):
            cc = c[c["variant"] == v]
            ax.plot(cc["scale"], cc["P_stays_novel10"], color=COL[[0, 2, 6][k]], ls=ls, lw=2, marker="o", ms=3,
                    label=f"{cand} ({'unrelaxed' if v.endswith('unrelaxed') else 'empirical d'})")
    nn = sub[(sub["variant"] == "lostop_unrelaxed") & sub["nn_novel"]].groupby("scale")["P_nn_stays_novel"].mean()
    ax.plot(nn.index, nn.values, color=COL[1], lw=2.5, label="1-NN criterion, mean")
    ax.set_title("LoStOP (structure)", loc="left")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(SCALES, [str(x) for x in SCALES])
        ax.set_xlabel("existence-probability scale s (1 = MP-fitted)")
        ax.set_ylabel("P(stays novel), 10th-percentile threshold")
        ax.set_ylim(-0.05, 1.05)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    axes[1].legend(frameon=False, fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e2_fragility.png"), dpi=150)
    plt.close(fig)


# Main #################################################################################


def main():
    from pymatgen.core import Composition, Structure
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    B = load_basics()
    t1, curves = task1(B)
    print("Task 1 done")
    print(t1.groupby("metric")[["novel5", "novel10", "nn_novel"]].sum())

    structures = [Structure.from_file(os.path.join(GN_DIR, f)) for f in B["gn_files"]]
    gn_feats = get_lostop_features(structures)
    assert np.allclose(np.linalg.norm(gn_feats[:, None] - gn_feats[None], axis=-1), B["lostop"]["G"], atol=1e-8)

    mp_set = {Composition(f).reduced_formula for f in B["mp_formulas"]}
    odds = element_odds()
    pi0, cal = calibrate(B, odds, mp_set)
    print(cal)
    pairs = empirical_single_sub_pairs(B, odds)
    print(f"{len(pairs)} MP single-substitution pairs; mean w_lostop={pairs['w_lostop'].mean():.3f}, "
          f"mean w_elmd={pairs['w_elmd'].mean():.3f}")

    a_path, i_path = os.path.join(OUT, "e2_analogues.csv"), os.path.join(OUT, "e2_analogue_summary.csv")
    if "--reuse" in sys.argv and os.path.exists(a_path):
        A, info = pd.read_csv(a_path), pd.read_csv(i_path)
    else:
        A, info = build_analogues(B, structures, odds, mp_set)
        A = add_distances(B, A, structures, gn_feats)
        A.to_csv(a_path, index=False)
        info.to_csv(i_path, index=False)
    print(info.to_string())

    res = task2(B, A, pi0, pairs, rng)
    rt = rank_tests(res)
    print(rt.to_string())
    t3 = task3(B)
    print(t3[[c for c in t3.columns if "novel" in c or "masked" in c] + ["metric"]].groupby("metric").sum())
    figures(B, t1, curves, res)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
