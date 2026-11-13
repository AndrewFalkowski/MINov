"""E18: Does E15's reference-choice result replicate on Na compounds and ternary nitrides? (E15 + E15b analyses)

E15 (Li) showed that swapping the reference database changes 28-41% of MINov's ElMD calls on GNoME candidates
(25-36x the change from 30 years of growth) but ~2% of LoStOP calls. E18 repeats E15 and the E15b consensus /
margin analysis with the same scoring engine (imported from e15_reference_choice) on the two chemistries of E12.

Data: data/raw/processed_<chem>/ from build_chem_sets.py (U_index.csv, mp2023.csv) and e18_build_refs.py
  (e18_V_index.csv, e18_formula_index.csv, e18_formula_elmd.npy, e18_V_lostop_features.npy,
  e18_mp25new_gnome_matches.csv).

References (per chemistry; definitions mirror E15 / build_li_sets.py):
  A   R: all experimental (non-theoretical, non-deprecated) MP-2023 entries (not sampled: Na 3,065 < cap 3,500;
      nitrides 798), i.e. E12's full reference.
  B   A + N25_theo: entries new in MP-2025 (2025-09-25) that are theoretical (largely GNoME imports).
  C   A + G_new: a sample (<= 3,000) of current GNoME entries not in the Dec-2023 release (disjoint from G by id).
  D   all MP-2023 entries incl. theoretical (ElMD only, formula multiplicities); Af, Df = unique formulas.
  Xsub_k / Asub(X)_k: as E15 (10 draws, seed 15, size of A; R members only = removal-only baseline).
  LoStOP featurisation failures (if any) are dropped from the reference for both metrics.
Candidates: G (E12's GNoME Dec-2023 sample: Na 3,000, nitrides 770). Excluded for reference X if in X (same id,
  same gnome_id, or StructureMatcher match to a same-formula X entry; e18_mp25new_gnome_matches.csv). Pool "G" =
  eligible under both references of a pair; "G_all" = all G with self-copies left in.
Scores, threshold rule, decompositions, comparison and outputs: exactly E15's (e15.evaluate / e15.compare), plus
  E15b's consensus (LOO 5% call vs rank top 5/10/20% within the pool, references ABCD / ABC) and margin band (E13
  li2000 pooled c_cap90 / c_cap95). Growth = E12 forward test, pool a_G, Y = 2000 -> full R.
Sanity check (stage check): engine on A vs the stored E12 full-reference scores / thresholds / calls for G
  (e12_<chem>_fwd_candidates.csv.gz, Y = 2000, sF / thrF5 / novelF5); ElMD formula matrix vs U_elmd.npy; weighted
  engine vs explicit expansion on a subset of D.

Outputs (experiments/results/): e18_check.csv, e18_references.csv, e18_pairs.csv, e18_pairs_summary.csv,
  e18_comparison.csv, e18_<chem>_scores.npz, e18_consensus.csv, e18_margin.csv, e18_summary.csv (Li E15 numbers
  alongside), e18_call_changes.png (Li / Na / nitrides x ElMD / LoStOP).

Usage: uv run python experiments/e18_reference_choice_chems.py [--chems na nitride] [--stages check run cons summary fig]
  Single process; LoStOP distances computed on the fly from features (float64 -> float32 blocks); ElMD from the
  memory-mapped formula matrix.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
import e15_reference_choice as e15  # noqa: E402
from e13_margin_rule import percentile_of  # noqa: E402
from novelty_scores import SCORES  # noqa: E402

OUT = e15.OUT
NAMES = e15.NAMES
METRICS = e15.METRICS
Q = e15.Q
N_DRAWS = e15.N_DRAWS
SEED = e15.SEED
TOPS = (5, 10, 20)
SETS = {("elmd", "ABCD"): ["A", "B", "C", "D"], ("elmd", "ABC"): ["A", "B", "C"], ("lostop", "ABC"): ["A", "B", "C"]}
CHEM_LABEL = {"li": "Li", "na": "Na", "nitride": "Ternary nitrides"}
log = e15.log


def proc(chem):
    return os.path.join(REPO, "data", "raw", f"processed_{chem}")


def read_csv(path, **kw):
    """Keeps the formula "NaN" (sodium nitride) as a string; only empty cells are missing."""
    kw.setdefault("keep_default_na", False)
    kw.setdefault("na_values", [""])
    return pd.read_csv(path, **kw)


# Data ##################################################################################


class _Rows:
    def __init__(self, X, r):
        self.X, self.r = X, r

    def __getitem__(self, key):
        _, cols = key
        return cdist(self.X[self.r], self.X[np.asarray(cols)]).astype(np.float32)


class FeatDist:
    """Euclidean LoStOP distances on demand: Mx[rows][:, cols] as for a stored matrix (e15.ref_block/query_block)."""

    def __init__(self, X):
        self.X = X

    def __getitem__(self, r):
        return _Rows(self.X, np.asarray(r))


def load(chem):
    p = proc(chem)
    V = read_csv(os.path.join(p, "e18_V_index.csv"), dtype={"material_id": str, "gnome_id": str})
    fi = read_csv(os.path.join(p, "e18_formula_index.csv"))
    fmap = dict(zip(fi.reduced_formula, fi.formula_row))
    V["frow"] = V.reduced_formula.map(fmap).astype(int)
    X = np.load(os.path.join(p, "e18_V_lostop_features.npy"))
    V["lostop_ok"] = np.isfinite(X).all(1)
    mp = read_csv(os.path.join(p, "mp2023.csv"))
    mp["frow"] = mp.reduced_formula.map(fmap).astype(int)
    M = read_csv(os.path.join(p, "e18_mp25new_gnome_matches.csv"), dtype={"material_id_gnome": str})
    Mx = {"elmd": np.load(os.path.join(p, "e18_formula_elmd.npy"), mmap_mode="r"), "lostop": FeatDist(X)}
    return V, mp, M, Mx


def entries(V, mp):
    def from_v(mask):
        d = V[mask & V.lostop_ok]
        return pd.DataFrame(dict(uid=d.vrow.values, material_id=d.material_id.values, gnome_id=d.gnome_id.values,
                                 fkey=d.fkey.values, frow=d.frow.values, is_r23=(d.set == "R").values))
    A = from_v(V.set == "R")
    refs = {"A": A, "B": pd.concat([A, from_v(V.set == "N25_theo")], ignore_index=True),
            "C": pd.concat([A, from_v(V.set == "G_new")], ignore_index=True)}
    rids = set(A.material_id)
    vrow = dict(zip(V.material_id, V.vrow))
    refs["D"] = pd.DataFrame(dict(uid=[vrow.get(i, -1) if i in rids else -1 for i in mp.material_id],
                                  material_id=mp.material_id.values, gnome_id=np.nan, fkey=mp.fkey.values,
                                  frow=mp.frow.values, is_r23=mp.material_id.isin(rids).values))
    refs["Af"] = A.drop_duplicates("frow").reset_index(drop=True)
    refs["Df"] = refs["D"].drop_duplicates("frow").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    for Xn in ["B", "C", "D"]:
        for k in range(N_DRAWS):
            sub = refs[Xn].iloc[np.sort(rng.choice(len(refs[Xn]), len(A), replace=False))].reset_index(drop=True)
            refs[f"{Xn}sub{k}"] = sub
            refs[f"Asub({Xn}){k}"] = sub[sub.is_r23].reset_index(drop=True)
    return refs


def cand_rows(cands, metric):
    return cands.frow.values if metric == "elmd" else cands.vrow.values


def candidates(V):
    return V[V.set == "G"].reset_index(drop=True)


def loo_scores(Mx, r, m):
    D = e15.ref_block(Mx, r)
    prof = e15.pair_profile(D, m)
    out = {k: [] for k in NAMES}
    for s in range(0, len(r), e15.BLOCK):
        rr = np.arange(s, min(s + e15.BLOCK, len(r)))
        b = e15.scores_block(D[rr], m, prof, self_rows=rr, mult_self=m[rr])
        for k in NAMES:
            out[k].append(b[k])
    return {k: np.repeat(np.concatenate(v), m) for k, v in out.items()}


# Stage: check ##########################################################################


def stage_check(chems):
    rows = []
    for chem in chems:
        V, mp, _, Mx = load(chem)
        refs = entries(V, mp)
        cands = candidates(V)
        # ElMD formula matrix vs the stored entry matrix used by E12
        U_e = np.load(os.path.join(proc(chem), "U_elmd.npy"), mmap_mode="r")
        uv = V[V.urow >= 0]
        rng = np.random.default_rng(0)
        a = np.sort(rng.choice(len(uv), min(500, len(uv)), replace=False))
        F = np.asarray(Mx["elmd"][uv.frow.values[a]])[:, uv.frow.values]
        rows.append(dict(chem=chem, check="ElMD formula matrix vs U_elmd.npy (500 rows x all U)", metric="elmd",
                         max_abs_score_diff=float(np.max(np.abs(F - U_e[uv.urow.values[a]][:, uv.urow.values])))))
        e12 = read_csv(os.path.join(OUT, f"e12_{chem}_fwd_candidates.csv.gz"), dtype={"material_id": str})
        e12 = e12[(e12.Y == 2000) & (e12.pool == "a_G")]
        A = refs["A"]
        for metric in METRICS:
            r, m, mA = e15.ref_rows(A, metric)
            res = e15.evaluate(Mx[metric], r, m, mA, cand_rows(cands, metric))
            for k in NAMES:
                d = e12[(e12.metric == metric) & (e12.score == k)].set_index("material_id").loc[cands.material_id]
                mine = res["scores"][k]
                call = mine < res["thr"][k]
                rows.append(dict(chem=chem, check="A engine vs stored E12 full-reference (Y=2000, pool a_G)",
                                 metric=metric, score=k, n=len(d),
                                 max_abs_score_diff=float(np.max(np.abs(d.sF.values - mine))),
                                 thr_ref=float(d.thrF5.iloc[0]), thr_engine=res["thr"][k],
                                 n_novel_ref=int(d.novelF5.sum()), n_novel_engine=int(call.sum()),
                                 n_call_disagree=int((d.novelF5.values != call).sum())))
            log(f"check {chem} A {metric}")
        # weighted engine vs explicit expansion on a D subset with repeated formulas
        D = refs["D"]
        fr = D.frow.values
        vc = pd.Series(fr).value_counts()
        pick = np.flatnonzero(np.isin(fr, vc.index[vc > 3][:60]))
        sub = D.iloc[np.r_[pick, rng.choice(len(D), 300, replace=False)]].drop_duplicates("material_id")
        r, m, mA = e15.ref_rows(sub, "elmd")
        q = cands.frow.values[:400]
        res = e15.evaluate(Mx["elmd"], r, m, mA, q)
        ef = sub.frow.values
        Dref = np.asarray(Mx["elmd"][ef][:, ef], np.float64)
        np.fill_diagonal(Dref, 0.0)
        Dq = np.asarray(Mx["elmd"][q][:, ef], np.float64)
        loo_q = Dref.copy()
        np.fill_diagonal(loo_q, np.inf)
        for k in NAMES:
            s = SCORES[k](Dref, Dq)
            loo = SCORES[k](Dref, loo_q)
            rows.append(dict(chem=chem, check=f"weighted engine vs expanded ({len(sub)} entries, {len(r)} formulas)",
                             metric="elmd", score=k, n=len(q), max_abs_score_diff=float(np.max(np.abs(s - res["scores"][k]))),
                             thr_ref=float(np.percentile(loo, Q)), thr_engine=res["thr"][k],
                             n_novel_ref=int((s < np.percentile(loo, Q)).sum()),
                             n_novel_engine=int((res["scores"][k] < res["thr"][k]).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "e18_check.csv"), index=False)
    pd.set_option("display.width", 250)
    print(out.to_string(), flush=True)


# Stage: run ############################################################################


def run_chem(chem):
    V, mp, Mt, Mx = load(chem)
    refs = entries(V, mp)
    cands = candidates(V)
    isG = np.ones(len(cands), bool)
    R, rinfo = {}, []
    for name, ref in refs.items():
        ex = e15.excluded(cands, ref, Mt)
        fk = set(ref.fkey)
        for metric in e15.metrics_for(name):
            r, m, mA = e15.ref_rows(ref, metric)
            res = e15.evaluate(Mx[metric], r, m, mA, cand_rows(cands, metric))
            res["excluded"] = ex
            R[(name, metric)] = res
            row = dict(chem=chem, ref=name, metric=metric, n_entries=len(ref), n_unique_rows=len(r),
                       n_r23_members=int(mA.sum()), mi_cutoff=res["cutoff"], kde_bw_median_pair=res["median"],
                       n_pairs=res["n_pairs"], n_G_excluded=int(ex.sum()), n_G_same_fkey=int(cands.fkey.isin(fk).sum()),
                       exact_comp_novel_G_eligible=float((~cands.fkey[~ex].isin(fk)).mean()),
                       exact_comp_novel_G_all=float((~cands.fkey.isin(fk)).mean()))
            for k in NAMES:
                s = res["scores"][k]
                row[f"thr|{k}"] = res["thr"][k]
                row[f"thr_pop|{k}"] = res["thr_pop"][k]
                row[f"novel_G_eligible|{k}"] = float((s[~ex] < res["thr"][k]).mean())
                pc = res["pct"][k][~ex]
                row[f"near_thr_G_2.5to10pct|{k}"] = float(((pc >= Q / 2) & (pc < 2 * Q)).mean())
                row[f"median_loo_pct_G|{k}"] = float(np.median(pc))
            rinfo.append(row)
        log(f"{chem} ref {name}: {len(ref)} entries")
    loo = {}
    for metric in METRICS:
        r, m, _ = e15.ref_rows(refs["A"], metric)
        for k, v in loo_scores(Mx[metric], r, m).items():
            loo[f"A|{metric}|{k}|loo"] = v
    np.savez_compressed(os.path.join(OUT, f"e18_{chem}_scores.npz"),
                        cand_material_id=cands.material_id.values.astype(str),
                        **{f"{n}|{m}|{k}": R[(n, m)]["scores"][k] for (n, m) in R for k in NAMES},
                        **{f"{n}|{m}|excluded": R[(n, m)]["excluded"] for (n, m) in R}, **loo)

    pairs = list(e15.PAIRS_MAIN)
    for Xn in ["B", "C", "D"]:
        for k in range(N_DRAWS):
            pairs += [("A", f"{Xn}sub{k}"), (f"{Xn}sub{k}", Xn), ("A", f"Asub({Xn}){k}"),
                      (f"Asub({Xn}){k}", f"{Xn}sub{k}")]
    rows = []
    for r1, r2 in pairs:
        for metric in METRICS:
            if (r1, metric) not in R or (r2, metric) not in R:
                continue
            a, b = R[(r1, metric)], R[(r2, metric)]
            elig = ~a["excluded"] & ~b["excluded"]
            for pool, sel in [("G", elig), ("G_all", isG)]:
                if pool == "G_all" and not (a["excluded"].any() or b["excluded"].any()):
                    continue
                for k in NAMES:
                    tp = b["thr_pop"][k] if r1 == "A" else np.nan
                    d = e15.compare(a["scores"][k][sel], a["thr"][k], b["scores"][k][sel], b["thr"][k], tp)
                    d.update(chem=chem, ref1=r1, ref2=r2, metric=metric, score=k, pool=pool,
                             n_excluded=int(len(sel) - sel.sum()) if pool == "G" else 0)
                    rows.append(d)
    return pd.DataFrame(rinfo), pd.DataFrame(rows)


def stage_run(chems):
    infos, pairs = [], []
    for chem in chems:
        i, p = run_chem(chem)
        infos.append(i)
        pairs.append(p)
    rinfo = pd.concat(infos, ignore_index=True)
    rinfo.to_csv(os.path.join(OUT, "e18_references.csv"), index=False)
    P = pd.concat(pairs, ignore_index=True)
    cols = ["chem", "ref1", "ref2", "metric", "score", "pool", "n", "n_excluded"]
    P = P[cols + [c for c in P.columns if c not in cols]]
    P.to_csv(os.path.join(OUT, "e18_pairs.csv"), index=False)
    P["family"] = P.ref1.str.replace(r"\d+$", "", regex=True) + "->" + P.ref2.str.replace(r"\d+$", "", regex=True)
    num = [c for c in P.columns if c not in cols + ["family"]]
    g = P.groupby(["chem", "family", "metric", "score", "pool"])
    S = g[num].mean()
    S["n"] = g.n.mean()
    S["n_draws"] = g.size()
    S["call_change_min"] = g.call_change.min()
    S["call_change_max"] = g.call_change.max()
    S = S.reset_index()
    S.to_csv(os.path.join(OUT, "e18_pairs_summary.csv"), index=False)

    comp = []
    for chem in chems:
        c = read_csv(os.path.join(OUT, f"e12_{chem}_fwd_candidates.csv.gz"), dtype={"material_id": str})
        c = c[(c.pool == "a_G") & (c.Y == 2000)]
        for metric in METRICS:
            for k in NAMES:
                d = c[(c.metric == metric) & (c.score == k)]
                row = dict(chem=chem, metric=metric, score=k, growth_2000_to_full=(d.novelY5 != d.novelF5).mean())
                for fam in ["A->B", "A->C", "A->D", "Af->Df", "A->Bsub", "A->Csub", "A->Dsub", "Bsub->B", "Csub->C",
                            "Dsub->D", "A->Asub(B)", "A->Asub(C)", "A->Asub(D)"]:
                    x = S[(S.chem == chem) & (S.family == fam) & (S.metric == metric) & (S.score == k) & (S.pool == "G")]
                    row[fam] = x.call_change.iloc[0] if len(x) else np.nan
                x = S[(S.chem == chem) & (S.family == "A->B") & (S.metric == metric) & (S.score == k) & (S.pool == "G_all")]
                row["A->B (G_all, self-copies kept)"] = x.call_change.iloc[0] if len(x) else np.nan
                comp.append(row)
        ri = rinfo[rinfo.chem == chem].set_index(["ref", "metric"])
        exact = dict(chem=chem, metric="-", score="exact composition match", growth_2000_to_full=0.0)
        for fam, r2 in [("A->B", "B"), ("A->C", "C"), ("A->D", "D")]:
            exact[fam] = 1 - ri.loc[(r2, "elmd"), "exact_comp_novel_G_eligible"]
        exact["A->B (G_all, self-copies kept)"] = 1 - ri.loc[("B", "elmd"), "exact_comp_novel_G_all"]
        comp.append(exact)
    comp = pd.DataFrame(comp)
    comp.to_csv(os.path.join(OUT, "e18_comparison.csv"), index=False)
    pd.set_option("display.width", 250)
    print(comp.round(4).to_string(), flush=True)
    log("run done")


# Stage: consensus + margin (E15b) ######################################################


def top_call(s, pool_idx, p):
    k = max(1, int(round(p / 100 * len(pool_idx))))
    c = np.zeros(len(s), bool)
    c[pool_idx[np.argsort(s[pool_idx], kind="stable")[:k]]] = True
    return c


def stage_cons(chems):
    refs_all = read_csv(os.path.join(OUT, "e18_references.csv"))
    cal = read_csv(os.path.join(OUT, "e13_calibration.csv"))
    cal = cal[(cal.calibration == "li2000") & (cal.margin == "pct") & (cal.scope == "pooled")].set_index("score")
    crows, mrows = [], []
    for chem in chems:
        Z = np.load(os.path.join(OUT, f"e18_{chem}_scores.npz"), allow_pickle=False)
        refs = refs_all[refs_all.chem == chem].set_index(["ref", "metric"])
        elig = ~(Z["B|elmd|excluded"] | Z["C|elmd|excluded"] | Z["D|elmd|excluded"] | Z["B|lostop|excluded"])
        idx = np.flatnonzero(elig)
        for (metric, tag), Rs in SETS.items():
            for k in NAMES:
                S = {r: Z[f"{r}|{metric}|{k}"] for r in Rs}
                calls = {"LOO 5%": {r: S[r] < refs.loc[(r, metric), f"thr|{k}"] for r in Rs}}
                for p in TOPS:
                    calls[f"rank top {p}%"] = {r: top_call(S[r], idx, p) for r in Rs}
                for ctype, cc in calls.items():
                    C = {r: cc[r][idx] for r in Rs}
                    Mm = np.array([C[r] for r in Rs])
                    a = C["A"].mean()
                    row = dict(chem=chem, metric=metric, refs=tag, score=k, call=ctype, pool="G", n=len(idx),
                               novel_A=a, novel_all=Mm.all(0).mean(), novel_any=Mm.any(0).mean(),
                               novel_A_only=(C["A"] & ~Mm[1:].any(0)).mean(),
                               retention=Mm.all(0).mean() / a if a else np.nan)
                    for r in Rs[1:]:
                        row[f"change_A->{r}"] = (C["A"] != C[r]).mean()
                    crows.append(row)
        for metric in METRICS:
            for k in NAMES:
                sA = Z[f"A|{metric}|{k}"]
                thr = refs.loc[("A", metric), f"thr|{k}"]
                margin = percentile_of(sA, Z[f"A|{metric}|{k}|loo"]) - Q
                assert np.array_equal(margin < 0, sA < thr), "margin sign must match the A call"
                am = np.abs(margin[idx])
                for rule in ("c_cap90", "c_cap95"):
                    c = cal.loc[k, rule]
                    unc = am <= c
                    row = dict(chem=chem, metric=metric, score=k, pool="G", n=len(idx), rule=rule, c_pp=c,
                               uncertain_under_A=unc.mean(), novel_under_A=(margin[idx] < 0).mean())
                    for r in ["B", "C", "D"] if metric == "elmd" else ["B", "C"]:
                        ch = (sA[idx] < thr) != (Z[f"{r}|{metric}|{k}"][idx] < refs.loc[(r, metric), f"thr|{k}"])
                        row[f"n_change_A->{r}"] = int(ch.sum())
                        row[f"changes_in_band_A->{r}"] = unc[ch].mean() if ch.any() else np.nan
                        row[f"c90_needed_A->{r}"] = np.percentile(am[ch], 90) if ch.any() else np.nan
                        row[f"median_abs_margin_changes_A->{r}"] = np.median(am[ch]) if ch.any() else np.nan
                    mrows.append(row)
    cons = pd.DataFrame(crows)
    li = read_csv(os.path.join(OUT, "e15b_consensus.csv"))
    li = li[li.pool == "G"].assign(chem="li")
    cons = pd.concat([li, cons], ignore_index=True)[["chem"] + [c for c in cons.columns if c != "chem"]]
    cons.to_csv(os.path.join(OUT, "e18_consensus.csv"), index=False)
    marg = pd.DataFrame(mrows)
    lim = read_csv(os.path.join(OUT, "e15b_margin.csv"))
    lim = lim[lim.pool == "G"].assign(chem="li")
    marg = pd.concat([lim, marg], ignore_index=True)[["chem"] + [c for c in marg.columns if c != "chem"]]
    marg.to_csv(os.path.join(OUT, "e18_margin.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(cons.round(3).to_string(index=False))
    print(marg.round(3).to_string(index=False))


# Stage: summary (Li E15 alongside) #####################################################


def li_tables():
    S = read_csv(os.path.join(OUT, "e15_pairs_summary.csv"))
    S = S[S.pool != "P44"].assign(chem="li")
    comp = read_csv(os.path.join(OUT, "e15_comparison.csv")).assign(chem="li")
    ri = read_csv(os.path.join(OUT, "e15_references.csv")).assign(chem="li")
    return S, comp, ri


def stage_summary(chems):
    S = read_csv(os.path.join(OUT, "e18_pairs_summary.csv"))
    comp = read_csv(os.path.join(OUT, "e18_comparison.csv"))
    ri = read_csv(os.path.join(OUT, "e18_references.csv"))
    lS, lcomp, lri = li_tables()
    S, comp, ri = (pd.concat([lS, S], ignore_index=True), pd.concat([lcomp, comp], ignore_index=True),
                   pd.concat([lri, ri], ignore_index=True))
    rows = []
    for chem in ["li"] + list(chems):
        for metric in METRICS:
            for k in NAMES:
                g = comp[(comp.chem == chem) & (comp.metric == metric) & (comp.score == k)].iloc[0]
                a = ri[(ri.chem == chem) & (ri.ref == "A") & (ri.metric == metric)].iloc[0]
                for X in ["B", "C", "D"]:
                    d = S[(S.chem == chem) & (S.family == f"A->{X}") & (S.metric == metric) & (S.score == k) & (S.pool == "G")]
                    if not len(d):
                        continue
                    d = d.iloc[0]
                    sub = S[(S.chem == chem) & (S.family == f"A->{X}sub") & (S.metric == metric) & (S.score == k)
                            & (S.pool == "G")].iloc[0]
                    rows.append(dict(chem=chem, metric=metric, score=k, ref=X, n=int(d.n),
                                     novel_A=d.novel1, novel_X=d.novel2, call_change=d.call_change, cc_lo=d.cc_lo,
                                     cc_hi=d.cc_hi, n_novel_to_not=d.n_novel_to_not, n_not_to_novel=d.n_not_to_novel,
                                     size_matched=sub.call_change, size_matched_min=sub.call_change_min,
                                     size_matched_max=sub.call_change_max, subset_to_full=g[f"{X}sub->{X}"],
                                     removal_only=g[f"A->Asub({X})"], growth=g.growth_2000_to_full,
                                     ratio_to_growth=d.call_change / g.growth_2000_to_full if g.growth_2000_to_full
                                     else np.nan,
                                     spearman=d.spearman, top10_overlap=d.top10_overlap, jaccard=d.jaccard,
                                     rank_part=d.rank_part, thr_part=d.thr_part, score_part=d.get("score_part"),
                                     pop_thr_part=d.get("pop_thr_part"),
                                     median_pct_A=a[f"median_loo_pct_G|{k}"], near_thr_A=a[f"near_thr_G_2.5to10pct|{k}"],
                                     formula_only_Af_Df=g["Af->Df"] if X == "D" else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "e18_summary.csv"), index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    print(out.round(3).to_string(index=False))


# Stage: figure #########################################################################


def stage_fig(chems):
    import matplotlib.pyplot as plt
    import make_report_figures as mrf

    S = pd.concat([li_tables()[0], read_csv(os.path.join(OUT, "e18_pairs_summary.csv"))], ignore_index=True)
    S = S[S.pool == "G"]
    comp = pd.concat([li_tables()[1], read_csv(os.path.join(OUT, "e18_comparison.csv"))], ignore_index=True)
    ri = pd.concat([li_tables()[2], read_csv(os.path.join(OUT, "e18_references.csv"))], ignore_index=True)
    labels = {"MINov (exact)": "MINov", "1-NN distance": "1-NN distance",
              "KDE (median bw)": "Gaussian KDE, median bandwidth"}
    groups = [("A->B", "+ MP-2025\nadditions"), ("A->C", "+ other GNoME\npredictions"),
              ("A->D", "all MP-2023\n(incl. theoretical)"),
              ("A->Bsub", "MP-2025 mix,\nsize of A"), ("A->Csub", "GNoME mix,\nsize of A"),
              ("A->Dsub", "MP-2023 mix,\nsize of A"), ("growth", "growth\n2000 -> full")]
    chs = ["li"] + list(chems)
    fig, axes = plt.subplots(len(chs), 2, figsize=(13, 3.9 * len(chs)), gridspec_kw=dict(width_ratios=[7, 5]),
                             sharey="row")
    width = 0.26
    for row, chem in enumerate(chs):
        Sc, cc = S[S.chem == chem], comp[comp.chem == chem]
        a = ri[(ri.chem == chem) & (ri.ref == "A") & (ri.metric == "elmd")].iloc[0]
        ncand = int(Sc[(Sc.family == "A->B") & (Sc.metric == "elmd")].n.iloc[0])
        for ax, metric, mt in [(axes[row, 0], "elmd", "ElMD (composition)"), (axes[row, 1], "lostop", "LoStOP (structure)")]:
            gs = [g for g in groups if g[0] == "growth" or len(Sc[(Sc.family == g[0]) & (Sc.metric == metric)])]
            x = np.arange(len(gs))
            for i, (k, lab) in enumerate(labels.items()):
                vals, lo, hi = [], [], []
                for fam, _ in gs:
                    if fam == "growth":
                        vals.append(cc[(cc.metric == metric) & (cc.score == k)].growth_2000_to_full.iloc[0] * 100)
                        lo.append(0), hi.append(0)
                        continue
                    d = Sc[(Sc.family == fam) & (Sc.metric == metric) & (Sc.score == k)].iloc[0]
                    vals.append(d.call_change * 100)
                    lo.append((d.call_change - d.call_change_min) * 100)
                    hi.append((d.call_change_max - d.call_change) * 100)
                bars = ax.bar(x + (i - 1) * width, vals, width, color=mrf.SERIES[lab], label=lab,
                              edgecolor=mrf.SURFACE, linewidth=2)
                sub = [j for j, (fam, _) in enumerate(gs) if fam.endswith("sub")]
                if sub:
                    ax.errorbar(x[sub] + (i - 1) * width, np.array(vals)[sub],
                                yerr=[np.array(lo)[sub], np.array(hi)[sub]], fmt="none", ecolor=mrf.INK2,
                                elinewidth=1, capsize=2)
                for b, v, h in zip(bars, vals, hi):
                    ax.text(b.get_x() + b.get_width() / 2, v + h + 0.6, f"{v:.1f}", ha="center", va="bottom",
                            fontsize=6.5, color=mrf.INK2)
            ax.axvline(len(gs) - 1.5, color=mrf.CONTEXT, linewidth=1, linestyle="--")
            ax.set_xticks(x, [g[1] for g in gs], fontsize=8)
            ax.set_title(f"{CHEM_LABEL[chem]}, {mt}\nA = {int(a.n_entries):,} entries, n = {ncand:,} candidates "
                         f"(A -> B)", fontsize=10)
            if metric == "elmd":
                ax.set_ylabel("% of GNoME candidates\n(call changed)")
            ax.grid(axis="x", visible=False)
            ax.set_axisbelow(True)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=3, fontsize=9)
    fig.suptitle("Novelty calls that change when the reference is swapped from experimental MP-2023 (A) to another "
                 "database, by chemistry\nsize-matched bars: mean of 10 draws, whiskers min-max; growth = E7 (Li) / "
                 "E12 (Na, nitrides), Y = 2000",
                 x=0.01, ha="left", fontsize=10, color=mrf.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "e18_call_changes.png"), dpi=200)
    log("figure written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chems", nargs="+", default=["na", "nitride"])
    ap.add_argument("--stages", nargs="+", default=["check", "run", "cons", "summary", "fig"])
    a = ap.parse_args()
    for s in a.stages:
        {"check": stage_check, "run": stage_run, "cons": stage_cons, "summary": stage_summary,
         "fig": stage_fig}[s](a.chems)


if __name__ == "__main__":
    main()
