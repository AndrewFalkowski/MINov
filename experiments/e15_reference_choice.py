"""E15: Which database counts as "known"? Sensitivity of novelty calls to the choice of reference.

E7/E12 measured reference growth over time. Here time is fixed and the *content* of the reference
changes: the same GNoME candidates are scored against different databases that one could
reasonably call "known", and the novel/not-novel calls are compared.

Data: data/raw/processed (U_index.csv, U_elmd.npy, U_lostop.npy, formula_elmd.npy,
  formula_elmd_index.csv, mp2023_li.csv, mp25new_gnome_matches.csv, sets.json).

References (entries; ElMD uses the formula matrix with per-formula multiplicities, which is exactly
equal to the entry-level matrix because same-formula entries have ElMD 0 and identical rows):
  A   R23: 2,095 experimental MP-2023 Li entries (baseline of E1-E12).
  B   R23 + N25_theo: what MP-2025 adds (2,360 theoretical entries, mostly GNoME imports).
  C   R23 + G_new: a generator's other predictions (3,000 current-GNoME sample) treated as known.
  D   all 21,761 MP-2023 Li entries, experimental + theoretical (ElMD only; no LoStOP for them).
  Af, Df  A and D as unique formulas (multiplicity 1), to check that polymorph counts do not drive D.
  Xsub_k  (X in B, C, D; k < N_DRAWS) random subset of X with |A| = 2,095 entries: same size as A,
          different content. Asub(X)_k = the R23 entries of Xsub_k alone: removal-only baseline.
Candidates: G (3,000 GNoME Dec-2023 Li sample) and P44 (the paper's 44), reported separately.
  A candidate is excluded for reference X if it is itself in X: same material_id, same gnome_id, or
  a StructureMatcher match to an X entry of the same formula (mp25new_gnome_matches.csv); a P44 row
  without gnome_id falls back to same fkey. Pair comparisons use candidates eligible under both.
  "G_all" rows repeat the comparison on all 3,000 G with self-copies left in the reference (the
  framing of the exact-match MP-2025 number in E10).
Scores: MINov (exact), 1-NN distance, KDE (median bw), orientation higher = denser = less novel.
  Novel = score < 5th percentile of the reference's leave-one-out scores (E7/E12 rule). All
  parameters (MI cutoff, bandwidth, threshold) refit on each reference.
Per reference pair (R1 -> R2), metric, score: novel shares, call-change share (Wilson 95% CI),
  Jaccard of novel sets, Spearman of scores, top-10% overlap, and two decompositions:
  (i) rank vs threshold: H = the k1 lowest-scoring candidates under R2 (k1 = #novel under R1).
      rank_part = share differing between R1 and H (reordering only, same count);
      thr_part  = |k2 - k1| / n (count change only). total <= rank_part + thr_part, total >= thr_part.
  (ii) score vs threshold population (R1 = A only): P = R2 scores with the threshold taken from the
      LOO scores of R2's R23 members only. A->P = candidates' scores moved relative to the same known
      population; P->R2 = the threshold population changed (new members redefine "typical").
Also: MI cutoff and KDE bandwidth per reference, exact composition match per reference, and a
  comparison with reference growth (E7 forward test, G pool, 2000 -> full record).

Outputs (experiments/results/): e15_references.csv, e15_pairs.csv (all pairs incl. every draw),
  e15_pairs_summary.csv (draws aggregated), e15_p44.csv, e15_comparison.csv, e15_check.csv,
  e15_scores.npz, e15_call_changes.png.

Usage: uv run python experiments/e15_reference_choice.py [--stages check run fig]
  Single process, ~3 min. Max RSS ~3.3 GB, of which ~1.6 GB are clean memory-mapped matrix pages; the
  largest private arrays are reference D (9,551 x 9,551 float32) and its 45.6M sorted pair distances.
"""

import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
from novelty_scores import SCORES  # noqa: E402

PROC = os.path.join(REPO, "data", "raw", "processed")
OUT = os.path.join(REPO, "experiments", "results")
NAMES = ["MINov (exact)", "1-NN distance", "KDE (median bw)"]
METRICS = ["elmd", "lostop"]
Q = 5
TOP = 0.10
N_DRAWS = 10
SEED = 15
BLOCK = 256


def log(msg, t0=[time.perf_counter()]):  # noqa: B006
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS
    print(f"[{time.perf_counter() - t0[0]:7.1f}s peak {rss:6.0f} MB] {msg}", flush=True)


# Data #################################################################################


def load():
    U = pd.read_csv(os.path.join(PROC, "U_index.csv"))
    sets = json.load(open(os.path.join(PROC, "sets.json")))["sets"]
    mp = pd.read_csv(os.path.join(PROC, "mp2023_li.csv"))
    mp = mp[mp.material_id.isin(set(sets["R23_all"]))].reset_index(drop=True)
    fi = pd.read_csv(os.path.join(PROC, "formula_elmd_index.csv"))
    fmap = dict(zip(fi.reduced_formula, fi.formula_row))
    U["frow"] = U.reduced_formula.map(fmap).astype(int)
    mp["frow"] = mp.reduced_formula.map(fmap).astype(int)
    M = pd.read_csv(os.path.join(PROC, "mp25new_gnome_matches.csv"))
    return U, mp, M


def entries(U, mp):
    """Entry tables: uid (U row or -1), material_id, gnome_id, fkey, frow, is_r23."""
    def from_u(mask):
        d = U[mask]
        return pd.DataFrame(dict(uid=d.row.values, material_id=d.material_id.values, gnome_id=d.gnome_id.values,
                                 fkey=d.fkey.values, frow=d.frow.values, is_r23=(d.set == "R23").values))
    A = from_u(U.set == "R23")
    refs = {"A": A,
            "B": pd.concat([A, from_u(U.set == "N25_theo")], ignore_index=True),
            "C": pd.concat([A, from_u(U.set == "G_new")], ignore_index=True)}
    r23ids = set(A.material_id)
    urow = dict(zip(U.material_id, U.row))
    refs["D"] = pd.DataFrame(dict(uid=[urow.get(i, -1) for i in mp.material_id], material_id=mp.material_id.values,
                                  gnome_id=np.nan, fkey=mp.fkey.values, frow=mp.frow.values,
                                  is_r23=mp.material_id.isin(r23ids).values))
    refs["Af"] = A.drop_duplicates("frow").reset_index(drop=True)
    refs["Df"] = refs["D"].drop_duplicates("frow").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    for X in ["B", "C", "D"]:
        for k in range(N_DRAWS):
            sub = refs[X].iloc[np.sort(rng.choice(len(refs[X]), len(A), replace=False))].reset_index(drop=True)
            refs[f"{X}sub{k}"] = sub
            refs[f"Asub({X}){k}"] = sub[sub.is_r23].reset_index(drop=True)
    return refs


def metrics_for(name):
    return ["elmd"] if name.startswith(("D", "Af", "Asub(D)")) else METRICS


def excluded(cands, ref, M):
    """True where the candidate is itself in the reference."""
    ids = set(ref.material_id)
    gids = set(ref.gnome_id.dropna())
    sm = M[M.structure_match & M.material_id.isin(ids)]
    gids |= set(sm.material_id_gnome)
    fk = set(ref.fkey)
    g = cands.gnome_id
    out = cands.material_id.isin(ids) | g.isin(gids)
    out |= g.isna() & cands.fkey.isin(fk)
    return out.values


# Weighted scoring engine ##############################################################
# A reference is a set of unique matrix rows with multiplicities m (entries per row). For ElMD the
# rows are formulas; for LoStOP they are U rows with m = 1. Every quantity equals the entry-level
# computation of novelty_scores.SCORES on the expanded matrix (checked in stage "check").


def ref_block(Mx, rows):
    D = np.empty((len(rows), len(rows)), np.float32)
    for s in range(0, len(rows), 512):
        D[s:s + 512] = Mx[rows[s:s + 512]][:, rows]
    if not np.array_equal(D, D.T):
        D = ((D.astype(np.float64) + D.T) / 2).astype(np.float32)
    D[D == 0] = 0.0  # no negative zeros (bit-packing below assumes +0)
    np.fill_diagonal(D, 0.0)
    return D


def query_block(Mx, qrows, rows):
    out = np.empty((len(qrows), len(rows)), np.float32)
    for s in range(0, len(qrows), 512):
        out[s:s + 512] = Mx[qrows[s:s + 512]][:, rows]
    return out


def pair_profile(D, m):
    """Sorted weighted pair distances (entry level) streamed into: MI cutoff, weight function on the
    distinct thresholds up to the cutoff, and the median pair distance."""
    n = len(D)
    m = m.astype(np.uint64)
    n_within = int((m > 1).sum())
    packed = np.empty(n * (n - 1) // 2 + n_within, np.uint64)
    p = 0
    for i in range(n - 1):
        v = D[i, i + 1:].view(np.uint32).astype(np.uint64)
        packed[p:p + len(v)] = (v << np.uint64(32)) | (m[i] * m[i + 1:])
        p += len(v)
    w_in = m[m > 1] * (m[m > 1] - np.uint64(1)) // np.uint64(2)
    packed[p:] = w_in  # distance 0 -> high bits 0
    packed.sort()
    M_ent = int(m.sum())
    total = M_ent * (M_ent - 1) // 2
    pos_lo, pos_hi = (total - 1) // 2, total // 2
    med = [None, None]
    thr, F = [np.zeros(1, np.float32)], [np.zeros(1)]  # threshold 0 (MIProfile appends 0)
    cum = 0
    done = False
    CH = 4_000_000
    s = 0
    while s < len(packed):
        e = min(s + CH, len(packed))
        if e < len(packed):  # end the chunk on a value boundary so no distance is split
            e = int(np.searchsorted(packed, ((packed[e - 1] >> np.uint64(32)) + np.uint64(1)) << np.uint64(32)))
        c = packed[s:e]
        s = e
        v = (c >> np.uint64(32)).astype(np.uint32).view(np.float32)
        w = (c & np.uint64(0xFFFFFFFF)).astype(np.int64)
        cw = cum + np.cumsum(w)
        for j, pos in enumerate((pos_lo, pos_hi)):
            if med[j] is None and cw[-1] > pos:
                med[j] = float(v[np.searchsorted(cw, pos, side="right")])
        if not done:
            last = np.r_[v[1:] != v[:-1], True]  # last occurrence of each distinct value
            vu, fu = v[last], cw[last] / total
            k = np.searchsorted(fu, 0.5, side="right")  # keep F <= 0.5 plus the first F > 0.5
            if k < len(fu):
                vu, fu, done = vu[:k + 1], fu[:k + 1], True
            thr.append(vu)
            F.append(fu)
        cum = int(cw[-1])
    assert cum == total, (cum, total)
    del packed
    thr, F = np.concatenate(thr), np.concatenate(F)
    if len(thr) > 1 and thr[1] == 0:  # zero distances present: threshold 0 carries their fraction
        thr, F = thr[1:], F[1:]
    H = np.zeros_like(F)
    ok = (F > 0) & (F < 1)
    H[ok] = -(F[ok] * np.log2(F[ok]) + (1 - F[ok]) * np.log2(1 - F[ok]))
    ic = int(np.argmax(H))
    wf = 1 - H[:ic + 1] / H[ic]
    wf /= wf.max()
    return dict(thr=thr[:ic + 1], wf=wf, cutoff=float(thr[ic]), median=(med[0] + med[1]) / 2,
                n_pairs=total, n_thresholds_kept=ic + 1)


def mi_weight(d, prof):
    idx = np.searchsorted(prof["thr"], d, side="right") - 1
    return np.where(d <= prof["cutoff"], prof["wf"][np.clip(idx, 0, len(prof["wf"]) - 1)], 0.0)


def scores_block(Dq, m, prof, self_rows=None, mult_self=None):
    """Scores of query rows against the reference rows. self_rows: for LOO, the column index of each
    query's own row (its own entry is removed; same-row siblings, if m > 1, stay)."""
    mf = m.astype(np.float64)
    out = {}
    W = mi_weight(Dq, prof)
    out["MINov (exact)"] = W @ mf
    K = np.exp(-0.5 * (Dq.astype(np.float64) / prof["median"]) ** 2)
    out["KDE (median bw)"] = K @ mf
    if self_rows is None:
        out["1-NN distance"] = -Dq.min(axis=1).astype(np.float64)
    else:
        r = np.arange(len(Dq))
        out["MINov (exact)"] -= W[r, self_rows]  # remove one copy of self (w(0))
        out["KDE (median bw)"] -= 1.0
        Dm = Dq.copy()
        Dm[r, self_rows] = np.inf
        nn = Dm.min(axis=1).astype(np.float64)
        nn[mult_self > 1] = 0.0
        out["1-NN distance"] = -nn
    return out


def evaluate(Mx, rows, m, mA, qrows):
    D = ref_block(Mx, rows)
    prof = pair_profile(D, m)
    loo = {k: [] for k in NAMES}
    for s in range(0, len(rows), BLOCK):
        r = np.arange(s, min(s + BLOCK, len(rows)))
        b = scores_block(D[r], m, prof, self_rows=r, mult_self=m[r])
        for k in NAMES:
            loo[k].append(b[k])
    del D
    sq = {k: [] for k in NAMES}
    for s in range(0, len(qrows), BLOCK):
        b = scores_block(query_block(Mx, qrows[s:s + BLOCK], rows), m, prof)
        for k in NAMES:
            sq[k].append(b[k])
    res = dict(cutoff=prof["cutoff"], median=prof["median"], n_pairs=prof["n_pairs"], scores={}, thr={}, thr_pop={},
               pct={})
    for k in NAMES:
        L = np.sort(np.repeat(np.concatenate(loo[k]), m))
        res["scores"][k] = np.concatenate(sq[k])
        res["thr"][k] = float(np.percentile(L, Q))
        res["thr_pop"][k] = float(np.percentile(np.repeat(np.concatenate(loo[k]), mA), Q)) if mA.sum() else np.nan
        res["pct"][k] = 100 * np.searchsorted(L, res["scores"][k], side="right") / len(L)  # LOO percentile
    return res


def ref_rows(ref, metric):
    if metric == "elmd":
        u, inv = np.unique(ref.frow.values, return_inverse=True)
        return u, np.bincount(inv), np.bincount(inv, weights=ref.is_r23.values.astype(float)).astype(int)
    assert (ref.uid >= 0).all()
    return ref.uid.values, np.ones(len(ref), int), ref.is_r23.values.astype(int)


def matrices():
    return {"elmd": np.load(os.path.join(PROC, "formula_elmd.npy"), mmap_mode="r"),
            "lostop": np.load(os.path.join(PROC, "U_lostop.npy"), mmap_mode="r")}


def cand_rows(cands, metric):
    return cands.frow.values if metric == "elmd" else cands.row.values


# Stage: check ##########################################################################


def stage_check():
    U, mp, _ = load()
    refs = entries(U, mp)
    Mx = matrices()
    E = np.load(os.path.join(PROC, "U_elmd.npy"), mmap_mode="r")
    cands = U[U.set.isin(["G", "P44"])].reset_index(drop=True)
    rows = []
    e7 = pd.read_csv(os.path.join(OUT, "e7_t1_summary.csv")).set_index(["metric", "score"])
    A = refs["A"]
    for metric in METRICS:
        r, m, mA = ref_rows(A, metric)
        res = evaluate(Mx[metric], r, m, mA, cand_rows(cands, metric))
        Dmat = E if metric == "elmd" else Mx["lostop"]
        Dref = np.asarray(Dmat[np.ix_(A.uid.values, A.uid.values)], np.float64)
        Dref = (Dref + Dref.T) / 2
        np.fill_diagonal(Dref, 0.0)
        Dq = np.asarray(Dmat[np.ix_(cands.row.values, A.uid.values)], np.float64)
        loo_q = Dref.copy()
        np.fill_diagonal(loo_q, np.inf)
        g = (cands.set == "G").values
        for k in NAMES:
            s = SCORES[k](Dref, Dq)
            thr = np.percentile(SCORES[k](Dref, loo_q), Q)
            mine = res["scores"][k]
            rows.append(dict(check="A entry-level vs engine", metric=metric, score=k,
                             max_abs_score_diff=float(np.max(np.abs(s - mine))), thr_ref=thr, thr_engine=res["thr"][k],
                             n_novel_ref=int((s[g] < thr).sum()), n_novel_engine=int((mine[g] < res["thr"][k]).sum()),
                             n_novel_e7=int(e7.loc[(metric, k), "n_novel5"])))
        log(f"check A {metric}")
    # weighted engine vs explicit expansion on a small D subset with repeated formulas
    rng = np.random.default_rng(1)
    D = refs["D"]
    fr = D.frow.values
    vc = pd.Series(fr).value_counts()
    pick = np.flatnonzero(np.isin(fr, vc.index[vc > 3][:60]))
    sub = D.iloc[np.r_[pick, rng.choice(len(D), 300, replace=False)]].drop_duplicates("material_id")
    r, m, mA = ref_rows(sub, "elmd")
    q = cands.frow.values[:400]
    res = evaluate(Mx["elmd"], r, m, mA, q)
    ef = sub.frow.values
    F = Mx["elmd"]
    Dref = np.asarray(F[ef][:, ef], np.float64)
    np.fill_diagonal(Dref, 0.0)
    Dq = np.asarray(F[q][:, ef], np.float64)
    loo_q = Dref.copy()
    np.fill_diagonal(loo_q, np.inf)
    for k in NAMES:
        s = SCORES[k](Dref, Dq)
        loo = SCORES[k](Dref, loo_q)
        rows.append(dict(check=f"weighted engine vs expanded ({len(sub)} entries, {len(r)} formulas)", metric="elmd",
                         score=k, max_abs_score_diff=float(np.max(np.abs(s - res["scores"][k]))),
                         thr_ref=float(np.percentile(loo, Q)), thr_engine=res["thr"][k],
                         n_novel_ref=int((s < np.percentile(loo, Q)).sum()),
                         n_novel_engine=int((res["scores"][k] < res["thr"][k]).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "e15_check.csv"), index=False)
    print(out.to_string(), flush=True)


# Stage: run ############################################################################


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def compare(s1, t1, s2, t2, tp2=np.nan):
    n = len(s1)
    c1, c2 = s1 < t1, s2 < t2
    k1, k2 = int(c1.sum()), int(c2.sum())
    nf = int((c1 != c2).sum())
    lo, hi = wilson(nf, n)
    order = np.lexsort((s1, s2))
    H = np.zeros(n, bool)
    H[order[:k1]] = True
    kt = max(1, int(round(TOP * n)))
    t1s, t2s = set(np.argsort(s1, kind="stable")[:kt]), set(np.argsort(s2, kind="stable")[:kt])
    union = (c1 | c2).sum()
    d = dict(n=n, novel1=k1 / n, novel2=k2 / n, call_change=nf / n, cc_lo=lo, cc_hi=hi,
             n_novel_to_not=int((c1 & ~c2).sum()), n_not_to_novel=int((~c1 & c2).sum()),
             jaccard=(c1 & c2).sum() / union if union else np.nan, spearman=spearmanr(s1, s2)[0],
             top10_overlap=len(t1s & t2s) / kt, rank_part=float((c1 != H).mean()), thr_part=abs(k2 - k1) / n)
    if np.isfinite(tp2):
        P = s2 < tp2
        d.update(novel_pop=P.mean(), score_part=float((c1 != P).mean()), pop_thr_part=float((P != c2).mean()))
    return d


PAIRS_MAIN = [("A", "B"), ("A", "C"), ("A", "D"), ("Af", "Df"), ("B", "C")]


def stage_run():
    U, mp, Mt = load()
    refs = entries(U, mp)
    Mx = matrices()
    cands = U[U.set.isin(["G", "P44"])].reset_index(drop=True)
    isG = (cands.set == "G").values
    R = {}
    rinfo = []
    for name, ref in refs.items():
        ex = excluded(cands, ref, Mt)
        fk = set(ref.fkey)
        for metric in metrics_for(name):
            r, m, mA = ref_rows(ref, metric)
            res = evaluate(Mx[metric], r, m, mA, cand_rows(cands, metric))
            res["excluded"] = ex
            R[(name, metric)] = res
            row = dict(ref=name, metric=metric, n_entries=len(ref), n_unique_rows=len(r), n_r23_members=int(mA.sum()),
                       mi_cutoff=res["cutoff"], kde_bw_median_pair=res["median"], n_pairs=res["n_pairs"],
                       n_G_excluded=int(ex[isG].sum()), n_P44_excluded=int(ex[~isG].sum()),
                       n_G_same_fkey=int(cands.fkey[isG].isin(fk).sum()),
                       n_P44_same_fkey=int(cands.fkey[~isG].isin(fk).sum()),
                       exact_comp_novel_G_eligible=float((~cands.fkey[isG & ~ex].isin(fk)).mean()),
                       exact_comp_novel_G_all=float((~cands.fkey[isG].isin(fk)).mean()))
            for k in NAMES:
                s = res["scores"][k]
                row[f"thr|{k}"] = res["thr"][k]
                row[f"thr_pop|{k}"] = res["thr_pop"][k]
                row[f"novel_G_eligible|{k}"] = float((s[isG & ~ex] < res["thr"][k]).mean())
                pc = res["pct"][k][isG & ~ex]
                row[f"near_thr_G_2.5to10pct|{k}"] = float(((pc >= Q / 2) & (pc < 2 * Q)).mean())
                row[f"median_loo_pct_G|{k}"] = float(np.median(pc))
            rinfo.append(row)
        log(f"ref {name}: {len(ref)} entries")
    rinfo = pd.DataFrame(rinfo)
    rinfo.to_csv(os.path.join(OUT, "e15_references.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "e15_scores.npz"), cand_material_id=cands.material_id.values.astype(str),
                        cand_set=cands.set.values.astype(str),
                        **{f"{n}|{m}|{k}": R[(n, m)]["scores"][k] for (n, m) in R for k in NAMES},
                        **{f"{n}|{m}|excluded": R[(n, m)]["excluded"] for (n, m) in R})

    pairs = list(PAIRS_MAIN)
    for X in ["B", "C", "D"]:
        for k in range(N_DRAWS):
            pairs += [("A", f"{X}sub{k}"), (f"{X}sub{k}", X), ("A", f"Asub({X}){k}"), (f"Asub({X}){k}", f"{X}sub{k}")]
    rows = []
    for r1, r2 in pairs:
        for metric in METRICS:
            if (r1, metric) not in R or (r2, metric) not in R:
                continue
            a, b = R[(r1, metric)], R[(r2, metric)]
            elig = ~a["excluded"] & ~b["excluded"]
            for pool, sel in [("G", isG & elig), ("G_all", isG), ("P44", ~isG & elig)]:
                if pool == "G_all" and not (a["excluded"][isG].any() or b["excluded"][isG].any()):
                    continue
                for k in NAMES:
                    s1, s2 = a["scores"][k][sel], b["scores"][k][sel]
                    tp = b["thr_pop"][k] if r1 == "A" else np.nan
                    d = compare(s1, a["thr"][k], s2, b["thr"][k], tp)
                    d.update(ref1=r1, ref2=r2, metric=metric, score=k, pool=pool,
                             n_excluded=int((isG if pool != "P44" else ~isG).sum() - sel.sum()) if pool != "G_all" else 0)
                    rows.append(d)
    cols = ["ref1", "ref2", "metric", "score", "pool", "n", "n_excluded"]
    P = pd.DataFrame(rows)
    P = P[cols + [c for c in P.columns if c not in cols]]
    P.to_csv(os.path.join(OUT, "e15_pairs.csv"), index=False)

    # aggregate draws: pair family = names with the draw index stripped
    P["family"] = P.ref1.str.replace(r"\d+$", "", regex=True) + "->" + P.ref2.str.replace(r"\d+$", "", regex=True)
    num = [c for c in P.columns if c not in cols + ["family"]]
    g = P.groupby(["family", "metric", "score", "pool"])
    S = g[num].mean()
    S["n"] = g.n.mean()
    S["n_draws"] = g.size()
    S["call_change_min"] = g.call_change.min()
    S["call_change_max"] = g.call_change.max()
    S = S.reset_index()
    S.to_csv(os.path.join(OUT, "e15_pairs_summary.csv"), index=False)
    S[S.pool == "P44"].to_csv(os.path.join(OUT, "e15_p44.csv"), index=False)

    # comparison with reference growth (E7 forward, G pool) and exact match (E10)
    c = pd.read_csv(os.path.join(OUT, "e7_fwd2000_candidates.csv.gz"))
    c = c[c.pool == "a_G"]
    comp = []
    for metric in METRICS:
        for k in NAMES:
            d = c[(c.metric == metric) & (c.score == k)]
            row = dict(metric=metric, score=k, growth_2000_to_full=(d.novelY5 != d.novelF5).mean())
            for fam in ["A->B", "A->C", "A->D", "Af->Df", "A->Bsub", "A->Csub", "A->Dsub",
                        "Bsub->B", "Csub->C", "Dsub->D", "A->Asub(B)", "A->Asub(C)", "A->Asub(D)"]:
                x = S[(S.family == fam) & (S.metric == metric) & (S.score == k) & (S.pool == "G")]
                row[fam] = x.call_change.iloc[0] if len(x) else np.nan
            x = S[(S.family == "A->B") & (S.metric == metric) & (S.score == k) & (S.pool == "G_all")]
            row["A->B (G_all, self-copies kept)"] = x.call_change.iloc[0] if len(x) else np.nan
            comp.append(row)
    ri = rinfo.set_index(["ref", "metric"])
    exact = dict(metric="-", score="exact composition match", growth_2000_to_full=0.0)
    for fam, r2 in [("A->B", "B"), ("A->C", "C"), ("A->D", "D")]:
        exact[fam] = 1 - ri.loc[(r2, "elmd"), "exact_comp_novel_G_eligible"]
    exact["A->B (G_all, self-copies kept)"] = 1 - ri.loc[("B", "elmd"), "exact_comp_novel_G_all"]
    comp.append(exact)
    pd.DataFrame(comp).to_csv(os.path.join(OUT, "e15_comparison.csv"), index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(comp).round(4).to_string(), flush=True)
    log("run done")


# Stage: figure ##########################################################################


def stage_fig():
    import matplotlib.pyplot as plt
    import make_report_figures as mrf  # rcParams and palette

    S = pd.read_csv(os.path.join(OUT, "e15_pairs_summary.csv"))
    S = S[S.pool == "G"]
    comp = pd.read_csv(os.path.join(OUT, "e15_comparison.csv"))
    labels = {"MINov (exact)": "MINov", "1-NN distance": "1-NN distance",
              "KDE (median bw)": "Gaussian KDE, median bandwidth"}
    groups = [("A->B", "+ MP-2025\nadditions"), ("A->C", "+ other GNoME\npredictions"),
              ("A->D", "all MP-2023\n(incl. theoretical)"),
              ("A->Bsub", "MP-2025 mix,\nsize of A"), ("A->Csub", "GNoME mix,\nsize of A"),
              ("A->Dsub", "MP-2023 mix,\nsize of A"), ("growth", "growth\n2000 -> full\n(E7)")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[7, 5]))
    width = 0.26
    for ax, metric, title in [(axes[0], "elmd", "ElMD (composition)"), (axes[1], "lostop", "LoStOP (structure)")]:
        gs = [g for g in groups if g[0] == "growth" or len(S[(S.family == g[0]) & (S.metric == metric)])]
        x = np.arange(len(gs))
        for i, (k, lab) in enumerate(labels.items()):
            vals, lo, hi = [], [], []
            for fam, _ in gs:
                if fam == "growth":
                    v = comp[(comp.metric == metric) & (comp.score == k)].growth_2000_to_full.iloc[0] * 100
                    vals.append(v), lo.append(0), hi.append(0)
                    continue
                d = S[(S.family == fam) & (S.metric == metric) & (S.score == k)].iloc[0]
                vals.append(d.call_change * 100)
                lo.append((d.call_change - d.call_change_min) * 100)
                hi.append((d.call_change_max - d.call_change) * 100)
            bars = ax.bar(x + (i - 1) * width, vals, width, color=mrf.SERIES[lab], label=lab,
                          edgecolor=mrf.SURFACE, linewidth=2)
            sub = [j for j, (fam, _) in enumerate(gs) if fam.endswith("sub")]
            if sub:
                ax.errorbar(x[sub] + (i - 1) * width, np.array(vals)[sub], yerr=[np.array(lo)[sub], np.array(hi)[sub]],
                            fmt="none", ecolor=mrf.INK2, elinewidth=1, capsize=2)
            top = max(max(vals), 1)
            for b, v, h in zip(bars, vals, hi):
                ax.text(b.get_x() + b.get_width() / 2, v + h + 0.012 * top * 3, f"{v:.1f}", ha="center",
                        va="bottom", fontsize=7, color=mrf.INK2)
        ax.axvline(len(gs) - 1.5, color=mrf.CONTEXT, linewidth=1, linestyle="--")
        ax.set_xticks(x, [g[1] for g in gs], fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("% of GNoME candidates (call changed)")
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=3, fontsize=9)
    fig.suptitle("Novelty calls that change when the reference is swapped from experimental MP-2023 (A) to another "
                 "database; size-matched bars: mean of 10 draws, whiskers min-max",
                 x=0.01, ha="left", fontsize=10, color=mrf.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(OUT, "e15_call_changes.png"), dpi=200)
    log("figure written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["check", "run", "fig"])
    a = ap.parse_args()
    for s in a.stages:
        {"check": stage_check, "run": stage_run, "fig": stage_fig}[s]()


if __name__ == "__main__":
    main()
