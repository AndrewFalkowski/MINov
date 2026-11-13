"""E7: Removal fragility at scale, and forward (time-split) tests of C1 and C4.

Data: data/raw/processed (U_index.csv, U_elmd.npy, U_lostop.npy, r23_first_year.csv).
  R23 = 2095 experimental MP-2023 Li entries (polymorphs kept), G = 3000 GNoME Li entries.

Task 1 (scale replication of E4). Reference R23, candidates G. Novel = score below the q-th
  percentile (q = 5, 10) of reference leave-one-out scores; everything refit on the reference.
  Scores: MINov (exact), 1-NN distance, KDE (median bw); metrics ElMD and LoStOP.
  - novel counts (all 3000);
  - m-out-of-n rank intervals (m/n = 0.5, DRAWS subsamples), widths, tiers (all 3000);
  - greedy certified removals (E4's greedy, imported) on a random subsample of N_GREEDY_T1
    candidates: non-novel -> novel, and swap with the next-less-novel neighbour in the full
    3000-candidate ranking.
Task 2 (forward test of C1). Reference_Y = R23 with first_year <= Y. Pools: (a) G, (b) R23 rows
  first reported after Y, scored at Y as if new. Future = full R23 (pool b: minus the candidate
  and its same-composition rows first reported after Y). At Y: call, rank (within pool),
  m-out-of-n interval, removal fragility = certified greedy removals to flip the call (non-novel ->
  novel with E4's greedy; novel -> not novel with the same greedy driven by the threshold) and to
  swap with each adjacent neighbour in the pool ranking (min of the two). Realized: call flip and
  |rank change| between Y and future. PASS if Spearman(-removals, realized) has a bootstrap CI
  excluding 0 and beats the interval width (paired bootstrap of the difference).
Task 3 (forward test of C4). At Y, analogues (E2's element substitution model) and existence
  calibration pi0 refit on Reference_Y only; P(stays novel) for candidates novel at Y (MINov,
  ElMD exact; LoStOP via E2's empirical single-substitution distances on Reference_Y). Realized =
  still novel against the future reference. AUC vs the density margin; reliability; global scale s.

Usage: uv run python experiments/e7_fragility_forward.py [--stages t1 fwd c4 report] [--workers 8]
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
import e4_rank_uncertainty as e4  # noqa: E402
from novelty_scores import SCORES  # noqa: E402

PROC = os.path.join(REPO, "data", "raw", "processed")
OUT = os.path.join(REPO, "experiments", "results")
METRICS = ["elmd", "lostop"]
NAMES = ["MINov (exact)", "1-NN distance", "KDE (median bw)"]
QS = (5, 10)
DRAWS = 100
RATIO = 0.5
N_GREEDY_T1 = 400        # task 1 swap subsample of G
N_CALL_T1 = 120          # task 1 call-flip subsample (first N_CALL_T1 of the swap subsample)
N_FWD = {2000: 200, 1990: 100, 2005: 100}   # greedy subsample per pool in the forward test
YEARS = [2000, 1990, 2005]
FWD_NAMES = {2000: ["MINov (exact)", "1-NN distance"], 1990: ["MINov (exact)"], 2005: ["MINov (exact)"]}
SEED = 0
N_BOOT = 2000


# Data #################################################################################


def index():
    U = pd.read_csv(os.path.join(PROC, "U_index.csv"))
    fy = pd.read_csv(os.path.join(PROC, "r23_first_year.csv"))[["row", "first_year"]]
    U = U.merge(fy, on="row", how="left")
    return U


def block(metric, rows_a, rows_b):
    D = np.load(os.path.join(PROC, f"U_{metric}.npy"), mmap_mode="r")
    return np.asarray(D[np.ix_(rows_a, rows_b)], dtype=np.float64)


# Worker contexts: reference block + candidate-to-reference block ########################

CTX = {}


def init_worker(specs):
    """specs: list of (key, metric, ref_rows, cand_rows)."""
    for key, metric, ref_rows, cand_rows in specs:
        Dref = block(metric, ref_rows, ref_rows)
        Dref = (Dref + Dref.T) / 2  # exact symmetry for MIProfile(exact)
        np.fill_diagonal(Dref, 0.0)
        Dc = block(metric, cand_rows, ref_rows)
        n = len(Dref)
        a, b = np.triu_indices(n, k=1)
        v = Dref[a, b]
        o = np.argsort(v, kind="stable")
        CTX[key] = (Dref, Dc, (a[o], b[o], v[o]))


def job_matrix(key, cands):
    Dref, Dc, pairs = CTX[key]
    n = len(Dref)
    D = np.zeros((n + len(cands),) * 2)
    D[:n, :n] = Dref
    D[n:, :n] = Dc[cands]
    D[:n, n:] = Dc[cands].T
    e4.N_MP = n
    e4.CAP = n // 2
    e4._DM = {key: D}
    e4._PAIRS = {id(D): pairs}
    return D


def greedy_denovel(key, name, i):
    """Novel candidate i -> not novel by removals (only via the threshold falling below its score).
    Same loop as e4.greedy (batches, refits, certification); effects evaluated for every alive ref."""
    dm = e4.dm_of(key)
    t0 = time.perf_counter()
    st = e4.State(dm, name, need_loo=True).refit(np.ones(e4.N_MP, bool))

    def margin(s):
        return s.tau - s.s[i]

    def effects(s):
        al = np.flatnonzero(s.alive)
        return al, (s.tau_after(al) - s.tau) - s.delta_c([i], al)[0]

    m0 = margin(st)
    removed, last_refit, n_refits, n_false = [], 0, 0, 0
    status = "censored"
    while True:
        if margin(st) <= 0:
            if not st.fresh:
                st.refit(st.alive)
                n_refits += 1
                last_refit = len(removed)
                if margin(st) > 0:
                    n_false += 1
                    continue
            status = "flipped"
            break
        if len(removed) >= e4.CAP:
            break
        rs, eff = effects(st)
        b = min(max(1, int(0.05 * len(removed))), e4.CAP - len(removed))
        pick = np.argsort(eff, kind="stable")[:b]
        pick = pick[eff[pick] < 0]
        if len(pick) == 0:
            if not st.fresh:
                st.refit(st.alive)
                n_refits += 1
                last_refit = len(removed)
                continue
            status = "stuck"
            break
        st.remove(rs[pick])
        removed.extend(rs[pick].tolist())
        if len(removed) - last_refit >= max(5, 0.1 * last_refit):
            st.refit(st.alive)
            n_refits += 1
            last_refit = len(removed)
    cert = np.nan
    if status == "flipped":
        s, tau = e4.exact_eval(dm, st.alive, name)
        cert = bool(s[i] >= tau)
    return dict(margin0=m0, n_removed=len(removed) if status == "flipped" else np.nan, status=status,
                certified=cert, n_refits=n_refits, n_false_flips=n_false,
                seconds=time.perf_counter() - t0)


def run_job(job):
    """job = (key, name, task, ci, cj): task in novel / denovel / swap (make cj score below ci)."""
    key, name, task, ci, cj = job
    cands = [ci] if cj < 0 else [ci, cj]
    job_matrix(key, cands)
    if task == "denovel":
        r = greedy_denovel(key, name, 0)
    else:
        r = e4.greedy(key, name, task, 0, None if cj < 0 else 1)
        r = {k: r[k] for k in ("margin0", "n_removed", "status", "certified", "n_refits",
                               "n_false_flips", "seconds")}
    r.update(key=key, score=name, task=task, i=ci, j=cj)
    return r


def run_draw(job):
    key, names, seed = job
    Dref, Dc, _ = CTX[key]
    n = len(Dref)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, int(round(RATIO * n)), replace=False))
    ref = Dref[np.ix_(idx, idx)]
    q = Dc[:, idx]
    loo_q = ref.copy()
    np.fill_diagonal(loo_q, np.inf)
    out = {}
    for name in names:
        s = SCORES[name](ref, q)
        loo = SCORES[name](ref, loo_q)
        out[name] = (s.astype(np.float64), {qq: np.percentile(loo, qq) for qq in QS})
    return out


def full_eval(Dref, Dq, name, Dq_mask=None):
    """Scores of queries + LOO thresholds on the reference; Dq_mask: True = drop that ref for that query."""
    q = Dq if Dq_mask is None else np.where(Dq_mask, np.inf, Dq)
    s = SCORES[name](Dref, q)
    loo_q = Dref.copy()
    np.fill_diagonal(loo_q, np.inf)
    loo = SCORES[name](Dref, loo_q)
    return s, {qq: np.percentile(loo, qq) for qq in QS}, loo


def rank_intervals(draws, name, cand_sel, q=5):
    S = np.array([d[name][0][cand_sel] for d in draws])
    R = rankdata(S, method="average", axis=1)
    lo, hi = np.percentile(R, 5, axis=0), np.percentile(R, 95, axis=0)
    pn = np.mean([d[name][0][cand_sel] < d[name][1][q] for d in draws], axis=0)
    return lo, hi, np.median(R, axis=0), pn


# Statistics ###########################################################################


def boot_rho(x, y, rng, x2=None):
    """Spearman rho(x, y) with percentile bootstrap CI; with x2 also rho(x2, y) and the difference."""
    ok = np.isfinite(x) & np.isfinite(y) & (np.isfinite(x2) if x2 is not None else True)
    x, y = x[ok], y[ok]
    x2 = x2[ok] if x2 is not None else None
    n = len(x)
    r1 = spearmanr(x, y)[0]
    r2 = spearmanr(x2, y)[0] if x2 is not None else np.nan
    b1, bd = [], []
    for _ in range(N_BOOT):
        k = rng.integers(0, n, n)
        a = spearmanr(x[k], y[k])[0]
        b1.append(a)
        if x2 is not None:
            bd.append(a - spearmanr(x2[k], y[k])[0])
    out = dict(n=n, rho=r1, rho_lo=np.nanpercentile(b1, 2.5), rho_hi=np.nanpercentile(b1, 97.5))
    if x2 is not None:
        out.update(rho_ref=r2, diff=r1 - r2, diff_lo=np.nanpercentile(bd, 2.5), diff_hi=np.nanpercentile(bd, 97.5))
    return out


def auc(score, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, score) if 0 < y.sum() < len(y) else np.nan


def boot_auc(s1, y, rng, s2=None):
    ok = np.isfinite(s1) & (np.isfinite(s2) if s2 is not None else True)
    s1, y = s1[ok], y[ok]
    s2 = s2[ok] if s2 is not None else None
    a1 = auc(s1, y)
    a2 = auc(s2, y) if s2 is not None else np.nan
    b1, bd = [], []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(y), len(y))
        if not 0 < y[k].sum() < len(k):
            continue
        u = auc(s1[k], y[k])
        b1.append(u)
        if s2 is not None:
            bd.append(u - auc(s2[k], y[k]))
    out = dict(n=len(y), n_pos=int(y.sum()), auc=a1, auc_lo=np.percentile(b1, 2.5), auc_hi=np.percentile(b1, 97.5))
    if s2 is not None:
        out.update(auc_ref=a2, diff=a1 - a2, diff_lo=np.percentile(bd, 2.5), diff_hi=np.percentile(bd, 97.5))
    return out


# Task 1 ###############################################################################


def task1(U, workers):
    t0 = time.perf_counter()
    r23 = U.index[U.set == "R23"].values
    g = U.index[U.set == "G"].values
    specs = [(f"t1|{m}", m, r23, g) for m in METRICS]
    rng = np.random.default_rng(SEED)
    sub_perm = rng.choice(len(g), N_GREEDY_T1, replace=False)
    sub, sub_call = np.sort(sub_perm), set(sub_perm[:N_CALL_T1].tolist())
    init_worker(specs)
    with Pool(workers, initializer=init_worker, initargs=(specs,)) as pool:
        draws = {}
        jobs = [(f"t1|{m}", NAMES, (SEED, k, d)) for k, m in enumerate(METRICS) for d in range(DRAWS)]
        res = pool.map(run_draw, jobs, chunksize=2)
        for jb, r in zip(jobs, res):
            draws.setdefault(jb[0], []).append(r)
        print(f"t1 draws {time.perf_counter() - t0:.0f}s", flush=True)
        rows, summ, gjobs = [], [], []
        for m in METRICS:
            key = f"t1|{m}"
            Dref, Dc, _ = CTX[key]
            for name in NAMES:
                s, thr, _ = full_eval(Dref, Dc, name)
                r0 = rankdata(s, method="average")
                lo, hi, med, pn = rank_intervals(draws[key], name, np.arange(len(g)))
                comp, lead = e4.tiers(lo, hi, r0)
                order = np.argsort(s, kind="stable")
                pos = np.empty(len(s), int)
                pos[order] = np.arange(len(s))
                subset = set(sub.tolist())
                for c in range(len(g)):
                    rows.append(dict(metric=m, score=name, cand=c, material_id=U.material_id.values[g[c]],
                                     full_score=s[c], full_rank=r0[c], novel5=s[c] < thr[5], novel10=s[c] < thr[10],
                                     rank_lo=lo[c], rank_hi=hi[c], width=hi[c] - lo[c], p_novel5=pn[c],
                                     in_greedy_sub=c in subset, tier_comp=comp[c], tier_lead=lead[c]))
                summ.append(dict(metric=m, score=name, n_cand=len(g), n_novel5=int((s < thr[5]).sum()),
                                 n_novel10=int((s < thr[10]).sum()), median_width=np.median(hi - lo),
                                 median_width_frac=np.median(hi - lo) / len(g), max_width=np.max(hi - lo),
                                 n_tiers_comp=len(np.unique(comp)), n_tiers_lead=len(np.unique(lead)),
                                 n_decision_stable5=int(np.sum((pn >= 0.9) | (pn <= 0.1)))))
                for c in sub:
                    if s[c] >= thr[5] and c in sub_call:
                        gjobs.append((key, name, "novel", int(c), -1))
                    if pos[c] + 1 < len(s):
                        gjobs.append((key, name, "swap", int(c), int(order[pos[c] + 1])))
        ranks = pd.DataFrame(rows)
        ranks.to_csv(os.path.join(OUT, "e7_t1_rank_intervals.csv.gz"), index=False)
        gjobs.sort(key=lambda j: (j[2] != "novel", j[1] == "1-NN distance"))
        print(f"t1: {len(gjobs)} greedy jobs", flush=True)
        out = []
        for k, r in enumerate(pool.imap_unordered(run_job, gjobs)):
            out.append(r)
            if (k + 1) % 250 == 0:
                print(f"  t1 greedy {k + 1}/{len(gjobs)} {time.perf_counter() - t0:.0f}s", flush=True)
    rem = pd.DataFrame(out)
    rem["metric"] = rem.key.str.split("|").str[1]
    # score gap to the swapped neighbour
    gap = []
    for r in rem.itertuples():
        d = ranks[(ranks.metric == r.metric) & (ranks.score == r.score)]
        gap.append(abs(d.full_score.values[r.j] - d.full_score.values[r.i]) if r.j >= 0 else np.nan)
    rem["score_gap"] = gap
    rem.to_csv(os.path.join(OUT, "e7_t1_removals.csv"), index=False)
    summ = pd.DataFrame(summ)
    cap = len(r23) // 2
    for k, sm in summ.iterrows():
        a = rem[(rem.metric == sm.metric) & (rem.score == sm.score)]
        nv, sw = a[a.task == "novel"], a[a.task == "swap"]
        fill = lambda d: np.where(d.status == "flipped", d.n_removed, cap + 1)  # noqa: E731
        summ.loc[k, "n_novel_jobs"] = len(nv)
        summ.loc[k, "median_removals_to_novel"] = np.median(fill(nv)) if len(nv) else np.nan
        summ.loc[k, "censored_novel"] = int((nv.status != "flipped").sum())
        summ.loc[k, "median_removals_to_swap"] = np.median(fill(sw))
        summ.loc[k, "frac_swap_le1"] = np.mean(fill(sw) <= 1)
        summ.loc[k, "rho_swap_vs_gap"] = spearmanr(fill(sw), sw.score_gap)[0]
        summ.loc[k, "cert_fail"] = int((a.certified == False).sum())  # noqa: E712
    summ.to_csv(os.path.join(OUT, "e7_t1_summary.csv"), index=False)
    print(summ.to_string(), flush=True)
    print(f"task 1 {time.perf_counter() - t0:.0f}s", flush=True)


# Task 2: forward test of C1 #############################################################


def forward_setup(U, Y):
    r23 = U.index[U.set == "R23"].values
    fyear = U.first_year.values
    refY = r23[fyear[r23] <= Y]
    poolb = r23[fyear[r23] > Y]
    g = U.index[U.set == "G"].values
    return r23, refY, g, poolb


def future_scores(U, Y, metric, name, r23, g, poolb):
    """Future scores/thresholds on full R23. Pool b: drop the candidate and its same-fkey rows
    first reported after Y."""
    Dref = block(metric, r23, r23)
    Dref = (Dref + Dref.T) / 2
    np.fill_diagonal(Dref, 0.0)
    sa, thr, loo = full_eval(Dref, block(metric, g, r23), name)
    fk = U.fkey.values
    fy = U.first_year.values
    mask = (fk[poolb][:, None] == fk[r23][None, :]) & (fy[r23][None, :] > Y)
    sb, _, _ = full_eval(Dref, block(metric, poolb, r23), name, mask)
    return sa, sb, thr


def forward(U, Y, workers, names):
    t0 = time.perf_counter()
    r23, refY, g, poolb = forward_setup(U, Y)
    cands = np.r_[g, poolb]
    ng = len(g)
    pools = {"a_G": np.arange(ng), "b_R23new": np.arange(ng, len(cands))}
    specs = [(f"fwd{Y}|{m}", m, refY, cands) for m in METRICS]
    init_worker(specs)
    rng = np.random.default_rng(SEED + Y)
    subs = {p: np.sort(rng.choice(ix, min(N_FWD[Y], len(ix)), replace=False)) for p, ix in pools.items()}
    rows, gjobs = [], []
    with Pool(workers, initializer=init_worker, initargs=(specs,)) as pool:
        jobs = [(f"fwd{Y}|{m}", NAMES, (SEED, Y, k, d)) for k, m in enumerate(METRICS) for d in range(DRAWS)]
        draws = {}
        for jb, r in zip(jobs, pool.map(run_draw, jobs, chunksize=2)):
            draws.setdefault(jb[0], []).append(r)
        print(f"fwd{Y} draws {time.perf_counter() - t0:.0f}s", flush=True)
        for m in METRICS:
            key = f"fwd{Y}|{m}"
            Dref, Dc, _ = CTX[key]
            for name in NAMES:
                sY, thrY, _ = full_eval(Dref, Dc, name)
                sa, sb, thrF = future_scores(U, Y, m, name, r23, g, poolb)
                sF = np.r_[sa, sb]
                for p, ix in pools.items():
                    rY = rankdata(sY[ix], method="average")
                    rF = rankdata(sF[ix], method="average")
                    lo, hi, med, pn = rank_intervals(draws[key], name, ix)
                    order = np.argsort(sY[ix], kind="stable")
                    pos = np.empty(len(ix), int)
                    pos[order] = np.arange(len(ix))
                    subset = set(subs[p].tolist())
                    for k, c in enumerate(ix):
                        nb_lo = ix[order[pos[k] - 1]] if pos[k] > 0 else -1     # next more novel
                        nb_hi = ix[order[pos[k] + 1]] if pos[k] + 1 < len(ix) else -1  # next less novel
                        gaps = [abs(sY[c] - sY[n]) for n in (nb_lo, nb_hi) if n >= 0]
                        rows.append(dict(Y=Y, metric=m, score=name, pool=p, cand=c,
                                         material_id=U.material_id.values[cands[c]], fkey=U.fkey.values[cands[c]],
                                         sY=sY[c], sF=sF[c], thrY5=thrY[5], thrY10=thrY[10], thrF5=thrF[5],
                                         thrF10=thrF[10], novelY5=sY[c] < thrY[5], novelF5=sF[c] < thrF[5],
                                         novelY10=sY[c] < thrY[10], novelF10=sF[c] < thrF[10],
                                         rankY=rY[k], rankF=rF[k], rank_lo=lo[k], rank_hi=hi[k],
                                         rank_med=med[k], p_novel5=pn[k], min_gap=min(gaps),
                                         nb_more=nb_lo, nb_less=nb_hi, in_sub=c in subset))
                    if name not in names:
                        continue
                    for c in subs[p]:
                        k = np.searchsorted(ix, c)
                        task = "denovel" if sY[c] < thrY[5] else "novel"
                        gjobs.append((key, name, task, int(c), -1))
                        if pos[k] + 1 < len(ix):
                            gjobs.append((key, name, "swap", int(c), int(ix[order[pos[k] + 1]])))
                        if pos[k] > 0:
                            gjobs.append((key, name, "swap", int(ix[order[pos[k] - 1]]), int(c)))
        gjobs = list(dict.fromkeys(gjobs))
        gjobs.sort(key=lambda j: (j[2] == "swap", j[1] == "1-NN distance"))
        print(f"fwd{Y}: {len(gjobs)} greedy jobs", flush=True)
        out = []
        for k, r in enumerate(pool.imap_unordered(run_job, gjobs)):
            out.append(r)
            if (k + 1) % 250 == 0:
                print(f"  fwd{Y} greedy {k + 1}/{len(gjobs)} {time.perf_counter() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    rem = pd.DataFrame(out)
    rem["metric"] = rem.key.str.split("|").str[1]
    rem["Y"] = Y
    rem.to_csv(os.path.join(OUT, f"e7_fwd{Y}_removals.csv"), index=False)
    # attach fragility per candidate
    cap = len(refY) // 2
    rem["n_fill"] = np.where(rem.status == "flipped", rem.n_removed, cap + 1)
    df["call_removals"] = np.nan
    df["swap_removals"] = np.nan
    for (m, name), a in rem.groupby(["metric", "score"]):
        sel = (df.metric == m) & (df.score == name)
        cl = a[a.task != "swap"].set_index("i").n_fill
        df.loc[sel, "call_removals"] = df.loc[sel, "cand"].map(cl).values
        sw = a[a.task == "swap"]
        best = pd.concat([sw[["i", "n_fill"]].rename(columns={"i": "c"}),
                          sw[["j", "n_fill"]].rename(columns={"j": "c"})]).groupby("c").n_fill.min()
        df.loc[sel, "swap_removals"] = df.loc[sel, "cand"].map(best).values
    df.to_csv(os.path.join(OUT, f"e7_fwd{Y}_candidates.csv.gz"), index=False)
    print(f"fwd{Y} {time.perf_counter() - t0:.0f}s", flush=True)


def c1_tests(Ys):
    rng = np.random.default_rng(SEED)
    rows = []
    for Y in Ys:
        df = pd.read_csv(os.path.join(OUT, f"e7_fwd{Y}_candidates.csv.gz"))
        for (m, name, p), d in df.groupby(["metric", "score", "pool"]):
            n = len(d)
            width = (d.rank_hi - d.rank_lo).values
            drank = np.abs(d.rankF - d.rankY).values
            flip = (d.novelY5 != d.novelF5).values.astype(int)
            cover = np.mean((d.rankF >= d.rank_lo) & (d.rankF <= d.rank_hi))
            # intervals re-centred at the Y rank (tests width, not the subsample's bias)
            cover_c = np.mean(np.abs(d.rankF - d.rankY) <= width / 2)
            base = dict(Y=Y, metric=m, score=name, pool=p, n_pool=n, n_novelY5=int(d.novelY5.sum()),
                        n_novelF5=int(d.novelF5.sum()), n_flip5=int(flip.sum()),
                        n_flip_novel_to_not=int((d.novelY5 & ~d.novelF5).sum()),
                        median_abs_drank=np.median(drank), median_width=np.median(width),
                        coverage90=cover, coverage90_centred=cover_c,
                        rho_rankY_rankF=spearmanr(d.rankY, d.rankF)[0])
            rows.append(dict(base, test="summary"))
            s = d[d.in_sub & d.swap_removals.notna()]
            if len(s) < 20:
                continue
            ds = np.abs(s.rankF - s.rankY).values
            w = (s.rank_hi - s.rank_lo).values
            fr = -s.swap_removals.values
            rows.append(dict(base, test="rank: -swap_removals vs |drank| (ref = width)", **boot_rho(fr, ds, rng, w)))
            rows.append(dict(base, test="rank: width vs |drank|", **boot_rho(w, ds, rng)))
            rows.append(dict(base, test="rank: -min_gap vs |drank|", **boot_rho(-s.min_gap.values, ds, rng)))
            rows.append(dict(base, test="rank: -swap_removals vs |drank| | ref = -min_gap",
                             **boot_rho(fr, ds, rng, -s.min_gap.values)))
            f = (s.novelY5 != s.novelF5).values.astype(int)
            cr = -s.call_removals.values
            inst = 1 - np.abs(2 * s.p_novel5.values - 1)
            marg = -np.abs(np.log(s.sY.values.clip(1e-12) if name != "1-NN distance" else -s.sY.values)
                           - np.log(s.thrY5.values if name != "1-NN distance" else -s.thrY5.values))
            if 0 < f.sum() < len(f):
                rows.append(dict(base, test="call: -call_removals vs flip AUC (ref = subsample instability)",
                                 **boot_auc(cr, f, rng, inst)))
                rows.append(dict(base, test="call: -call_removals vs flip AUC (ref = -|log margin|)",
                                 **boot_auc(cr, f, rng, marg)))
                rows.append(dict(base, test="call: rho(-call_removals, flip) (ref = instability)",
                                 **boot_rho(cr, f.astype(float), rng, inst)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "e7_c1_tests.csv"), index=False)
    return t


# Task 3: forward test of C4 #############################################################


def c4(U, Y, scales=(0.1, 0.3, 1, 3, 10)):
    import e2_denovelling as e2
    from pymatgen.core import Composition
    from sklearn.linear_model import LogisticRegression
    from MINOV.density import MIProfile
    from MINOV.distance import get_elmd_dm
    import contextlib
    import io

    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED + Y)
    df = pd.read_csv(os.path.join(OUT, f"e7_fwd{Y}_candidates.csv.gz"))
    df = df[df.score == "MINov (exact)"]
    r23, refY, g, poolb = forward_setup(U, Y)
    cands = np.r_[g, poolb]
    rf = {}

    def red(f):
        if f not in rf:
            rf[f] = Composition(f).reduced_formula
        return rf[f]

    formY = sorted({red(f) for f in U.formula.values[refY]})
    setY = set(formY)
    odds = e2.element_odds()
    # calibration on Reference_Y only
    rs, hits = [], []
    for f in formY:
        for a in e2.element_analogues(f, odds):
            rs.append(a["r"])
            hits.append(a["formula"] in setY)
    rs, hits = np.array(rs), np.array(hits)
    lr = LogisticRegression(C=1e6).fit(np.log(rs)[:, None], hits)
    ca, cb = float(lr.intercept_[0]), float(lr.coef_[0, 0])

    def pi0(r):
        return 1 / (1 + np.exp(-(ca + cb * np.log(np.asarray(r, float)))))

    # same calibration on full R23 (in-sample check of drift)
    print(f"c4 Y={Y}: pi0 logit a={ca:.2f} b={cb:.2f}, {len(rs)} analogues of {len(formY)} formulas, "
          f"hit rate {hits.mean():.3f}", flush=True)

    # empirical single-substitution LoStOP distances among Reference_Y rows
    D_l = block("lostop", refY, refY)
    prof = {}
    for m in METRICS:
        Dr = block(m, refY, refY)
        Dr = (Dr + Dr.T) / 2
        np.fill_diagonal(Dr, 0.0)
        prof[m] = MIProfile(Dr)
    keys = {}
    for i, f in enumerate(U.formula.values[refY]):
        c = Composition(f).reduced_composition
        items = {e.symbol: c[e] for e in c.elements}
        for e in items:
            if e != "Li":
                keys.setdefault(tuple(sorted((k if k != e else "*", v) for k, v in items.items())), []).append((i, e))
    emp = []
    for mem in keys.values():
        for x in range(len(mem)):
            for y in range(x + 1, len(mem)):
                if mem[x][1] != mem[y][1]:
                    emp.append(D_l[mem[x][0], mem[y][0]])
    emp_w = prof["lostop"].weight(np.array(emp))
    print(f"c4 Y={Y}: {len(emp)} empirical single-substitution LoStOP pairs, mean w {emp_w.mean():.3f}", flush=True)

    rows = []
    for m in METRICS:
        d = df[(df.metric == m) & df.novelY5]
        for r in d.itertuples():
            f = red(U.formula.values[cands[r.cand]])
            an = [a for a in e2.element_analogues(f, odds) if a["formula"] not in setY and a["formula"] != f]
            rr = np.array([a["r"] for a in an])
            if m == "elmd" and len(an):
                with contextlib.redirect_stdout(io.StringIO()):
                    dd = get_elmd_dm([f], reference=[a["formula"] for a in an])[0]
                w = prof[m].weight(dd)
            else:
                w = None
            gap = r.thrY5 - r.sY
            rec = dict(Y=Y, metric=m, pool=r.pool, cand=r.cand, material_id=r.material_id, formula=f,
                       sY=r.sY, thrY5=r.thrY5, margin=gap, rel_margin=gap / r.thrY5, n_analogues=len(an),
                       survived=bool(r.novelF5), sF=r.sF, thrF5=r.thrF5, rank_width=r.rank_hi - r.rank_lo,
                       p_novel5_sub=r.p_novel5)
            for s in scales:
                pi = np.minimum(1.0, s * pi0(rr)) if len(an) else np.zeros(0)
                added = e2.added_samples(pi, w, emp_w, rng)
                rec[f"P_stay_s{s}"] = float(np.mean(r.sY + added < r.thrY5))
                rec[f"E_added_s{s}"] = float(np.mean(added))
            rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, f"e7_c4_Y{Y}.csv"), index=False)
    print(f"c4 Y={Y}: {time.perf_counter() - t0:.0f}s", flush=True)
    return res


def c4_tests(Ys, scales=(0.1, 0.3, 1, 3, 10)):
    rng = np.random.default_rng(SEED)
    rows, rel = [], []
    for Y in Ys:
        res = pd.read_csv(os.path.join(OUT, f"e7_c4_Y{Y}.csv"))
        for (m, p), d in list(res.groupby(["metric", "pool"])) + [((m, "all"), d) for m, d in res.groupby("metric")]:
            y = d.survived.values.astype(int)
            base = dict(Y=Y, metric=m, pool=p, n_novelY=len(d), n_survived=int(y.sum()), surv_rate=y.mean(),
                        mean_P_stay_s1=d["P_stay_s1"].mean())
            # global scale: minimise log loss
            ll = {}
            for s in scales:
                pp = d[f"P_stay_s{s}"].clip(1e-3, 1 - 1e-3).values
                ll[s] = -np.mean(y * np.log(pp) + (1 - y) * np.log(1 - pp))
            base.update({f"logloss_s{s}": v for s, v in ll.items()}, best_scale=min(ll, key=ll.get))
            if 0 < y.sum() < len(y):
                base.update({f"{k}_vs_margin": v for k, v in boot_auc(d["P_stay_s1"].values, y, rng,
                                                                        d.margin.values).items()})
                base["auc_rel_margin"] = auc(d.rel_margin.values, y)
                base["auc_sub_pnovel"] = auc(d.p_novel5_sub.values, y)
                # P adds information beyond the margin? logistic with both
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import cross_val_predict
                X1 = np.log(d.margin.values.clip(1e-9))[:, None]
                X2 = np.column_stack([X1[:, 0], np.log(d["P_stay_s1"].values.clip(1e-3)),
                                      np.log1p(d.n_analogues.values)])
                for nm, X in (("cv_auc_margin", X1), ("cv_auc_margin_plus_P", X2)):
                    pr = cross_val_predict(LogisticRegression(), X, y, cv=5, method="predict_proba")[:, 1]
                    base[nm] = auc(pr, y)
            rows.append(base)
            if p == "all":
                for s in (1, base["best_scale"]):
                    pp = d[f"P_stay_s{s}"].values
                    bins = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0001])
                    for lo, hi in zip(bins[:-1], bins[1:]):
                        k = (pp >= lo) & (pp < hi)
                        if k.any():
                            rel.append(dict(Y=Y, metric=m, scale=s, p_lo=lo, p_hi=hi, n=int(k.sum()),
                                            mean_pred=pp[k].mean(), realized=y[k].mean()))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "e7_c4_tests.csv"), index=False)
    pd.DataFrame(rel).to_csv(os.path.join(OUT, "e7_c4_reliability.csv"), index=False)
    return t, pd.DataFrame(rel)


def figure(Ys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rel = pd.read_csv(os.path.join(OUT, "e7_c4_reliability.csv"))
    df = pd.read_csv(os.path.join(OUT, "e7_fwd2000_candidates.csv.gz"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax = axes[0]
    d = df[(df.score == "MINov (exact)") & df.in_sub & df.swap_removals.notna()]
    for m, c in (("elmd", "#2a78d6"), ("lostop", "#eb6834")):
        x = d[d.metric == m]
        npool = x.pool.map(df[(df.score == "MINov (exact)") & (df.metric == m)].pool.value_counts())
        ax.scatter(x.swap_removals, np.abs(x.rankF - x.rankY) / npool,
                   s=6, alpha=0.5, color=c, label=m)
    ax.set_xscale("log")
    ax.set_xlabel("removals to swap an adjacent rank at Y=2000")
    ax.set_ylabel("|rank change| / pool size, Y -> 2023")
    ax.legend(frameon=False)
    ax = axes[1]
    for m, c in (("elmd", "#2a78d6"), ("lostop", "#eb6834")):
        x = d[d.metric == m]
        ax.scatter(x.rank_hi - x.rank_lo, np.abs(x.rankF - x.rankY), s=6, alpha=0.5, color=c)
    ax.set_xlabel("90% m-out-of-n rank interval width")
    ax.set_ylabel("|rank change|")
    ax = axes[2]
    for (Y, m, s), x in rel.groupby(["Y", "metric", "scale"]):
        if Y != 2000:
            continue
        ax.plot(x.mean_pred, x.realized, "o-", label=f"{m}, s={s:g}")
    ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls=":")
    ax.set_xlabel("predicted P(stays novel) at Y=2000")
    ax.set_ylabel("realized survival vs 2023 reference")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e7_forward.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["t1", "fwd", "c4", "report"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--years", nargs="+", type=int, default=YEARS)
    args = ap.parse_args()
    U = index()
    if "t1" in args.stages:
        task1(U, args.workers)
    if "fwd" in args.stages:
        for Y in args.years:
            forward(U, Y, args.workers, FWD_NAMES[Y])
    if "c4" in args.stages:
        for Y in args.years:
            c4(U, Y)
    if "report" in args.stages:
        with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 3):
            print(c1_tests([Y for Y in args.years if os.path.exists(os.path.join(OUT, f"e7_fwd{Y}_candidates.csv.gz"))]).to_string())
            t, rel = c4_tests([Y for Y in args.years if os.path.exists(os.path.join(OUT, f"e7_c4_Y{Y}.csv"))])
            print(t.to_string())
            print(rel.to_string())
        figure(args.years)


if __name__ == "__main__":
    main()
