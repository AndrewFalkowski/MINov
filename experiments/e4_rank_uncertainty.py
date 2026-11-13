"""E4: Rank uncertainty and removal fragility of novelty scores for the GNoME Li candidates.

Data: precomputed Li LoStOP / ElMD distance matrices. Rows 0..1833 = Materials Project
(reference), rows 1834..1877 = the 44 GNoME candidates (queries, never part of the reference).
Scores (experiments/novelty_scores.py, HIGHER = LESS novel): MINov (exact), KDE (median bw),
1-NN distance, kNN (paper). "Novel" = score below the 5th percentile of the reference
leave-one-out (LOO) scores; every parameter and the threshold are refit on the reference.

Part 1 (rank uncertainty): m-out-of-n subsampling of MP without replacement (m/n = 0.5, 0.8),
refit every score on each subsample, rank the 44 candidates (rank 1 = most novel). Gives 90%
rank intervals, P(top-5), P(top-10), P(novel) and tiers (groups of overlapping intervals).

Part 2 (removal fragility): greedy removal of MP references with refitting.
  (a) non-novel candidate -> novel;
  (b) adjacent pair in the full-data ranking -> swapped;
  (c) novel candidate -> overtaken by one of the 5 most novel non-novel candidates.
Each step removes the reference(s) with the largest exact effect on the margin under the
current fitted parameters (score parameters AND threshold held fixed between refits; the
effect on the threshold is evaluated exactly for a shortlist). The parameters and threshold are
refit every max(5, 10% of removals) and whenever the fixed-parameter margin crosses zero; a
crossing only counts once it survives a refit, and the final removal set is certified by an
independent exact recomputation with the functions in novelty_scores.py. Counts are greedy
upper bounds on the minimum; batches of max(1, 5% of removals so far) are taken per step.

Part 3 (H4c): correlation of fragility with density and the MINov effective number of
neighbours n_eff = (sum w)^2 / sum w^2; partial Spearman correlations controlling for density.

If no single removal improves the margin right after a refit (a tie plateau, e.g. several
equidistant nearest neighbours), the references nearest to the candidate that must lose density
are removed instead.

Usage: uv run python experiments/e4_rank_uncertainty.py [--draws 200] [--workers 8] [--reuse]
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
from MINOV.density import MIProfile, _binary_entropy  # noqa: E402
from novelty_scores import SCORES, paper_k  # noqa: E402

OUT = os.path.join(REPO, "experiments", "results")
N_MP = 1834
METRICS = ["lostop", "elmd"]
NAMES = ["MINov (exact)", "KDE (median bw)", "1-NN distance", "kNN (paper)"]
RATIOS = [0.5, 0.8]
Q = 5.0  # novelty threshold percentile
CAP = N_MP // 2  # removal budget (censored beyond)
SHORTLIST = 100
N_OVERTAKERS = 5

_DM = {}


def dm_of(metric):
    if metric not in _DM:
        _DM[metric] = np.load(os.path.join(REPO, "precomputed", f"Li_{metric}_dm.npy"))
    return _DM[metric]


def gnome_names():
    path = os.path.join(REPO, "data", "GNOME_Li_dataset")
    return [f[:-4] for f in os.listdir(path) if ".cif" in f]


def exact_eval(dm, alive, name):
    """Independent exact recomputation with novelty_scores.SCORES: (candidate scores, threshold)."""
    idx = np.flatnonzero(alive)
    ref = dm[np.ix_(idx, idx)]
    s = SCORES[name](ref, dm[N_MP:, idx])
    loo_q = ref.copy()
    np.fill_diagonal(loo_q, np.inf)
    return s, np.percentile(SCORES[name](ref, loo_q), Q)


# Fitted state with fast fixed-parameter removal effects ###############################


_PAIRS = {}


def sorted_pairs(dm):
    """MP pair distances (upper triangle) sorted once per matrix: (row, col, value)."""
    key = id(dm)
    if key not in _PAIRS:
        a, b = np.triu_indices(N_MP, k=1)
        v = dm[a, b]
        o = np.argsort(v)
        _PAIRS[key] = (a[o], b[o], v[o])
    return _PAIRS[key]


def fit_minov(dm, alive, want_pairs):
    """Exact MINov weight function on the alive references (identical to MIProfile.weight),
    and optionally the weights of all alive pairs as (row, col, weight)."""
    A, B, V = sorted_pairs(dm)
    keep = alive[A] & alive[B]
    sv = V[keep]
    npairs = len(sv)
    ends = np.append(np.flatnonzero(np.diff(sv) > 0) + 1, npairs)  # 'right' count per distinct value
    n0 = np.searchsorted(sv, 0.0, side="right")
    t = np.append(0.0, sv[ends - 1])
    h_t = _binary_entropy(np.append(n0, ends) / npairs)
    imax = np.argmax(h_t)
    cutoff, hmax = t[imax], h_t[imax]
    w0 = 1 - h_t[0] / hmax

    def w_of_f(f, d):
        w = (1 - _binary_entropy(f) / hmax) / w0
        w[d > cutoff] = 0.0
        return w

    def weight(x):
        return w_of_f(np.searchsorted(sv, x, side="right") / npairs, x)

    if not want_pairs:
        return weight, None
    cnt = np.repeat(ends, np.diff(np.append(0, ends)))
    return weight, (A[keep], B[keep], w_of_f(cnt / npairs, sv))


class State:
    """Scores of candidates (s) and LOO scores of MP (loo) under parameters fit on `alive`."""

    def __init__(self, dm, name, need_loo):
        self.dm, self.name, self.need_loo = dm, name, need_loo
        self.kernel = name in ("MINov (exact)", "KDE (median bw)")

    def refit(self, alive):
        self.alive = alive.copy()
        self.fresh = True
        idx = np.flatnonzero(alive)
        dm = self.dm
        cq = dm[N_MP:, idx]
        nc = len(dm) - N_MP
        if self.kernel:
            if self.name == "MINov (exact)":
                weight, wpairs = fit_minov(dm, alive, self.need_loo)
            else:
                ref = dm[np.ix_(idx, idx)]
                h = np.median(ref[np.triu_indices(len(idx), k=1)])

                def weight(x):
                    return np.exp(-0.5 * (x / h) ** 2)
            self.Kc = np.zeros((nc, N_MP))
            self.Kc[:, idx] = weight(cq)
            self.s = self.Kc.sum(1)
            if self.need_loo:
                self.Km = np.zeros((N_MP, N_MP))
                if self.name == "MINov (exact)":
                    a, b, w = wpairs
                    self.Km[a, b] = w
                    self.Km[b, a] = w
                else:
                    wref = weight(ref)
                    np.fill_diagonal(wref, 0.0)
                    self.Km[np.ix_(idx, idx)] = wref
                self.loo = self.Km.sum(1)
        else:
            ref = dm[np.ix_(idx, idx)]
            self.k = 1 if self.name == "1-NN distance" else paper_k(len(idx))
            self.Dc = np.full((nc, N_MP), np.inf)
            self.Dc[:, idx] = cq
            if self.name == "kNN (paper)":
                self.Dc[self.Dc == 0] = np.inf  # paper drops zero distances
            self.cstat = self._stats(self.Dc)
            self.s = self._f(self.cstat[0])
            if self.need_loo:
                self.Dm = np.full((N_MP, N_MP), np.inf)
                self.Dm[np.ix_(idx, idx)] = ref
                if self.name == "kNN (paper)":
                    self.Dm[self.Dm == 0] = np.inf
                np.fill_diagonal(self.Dm, np.inf)
                self.mstat = self._stats(self.Dm)
                self.loo = self._f(self.mstat[0])
        if self.need_loo:
            self.tau = np.percentile(self.loo[self.alive], Q)
        return self

    # kNN helpers: per row (sum of k nearest, k-th nearest, (k+1)-th nearest)
    def _stats(self, D):
        k = self.k
        part = np.partition(D, [k - 1, k], axis=1)
        return [part[:, :k].sum(1), part[:, k - 1].copy(), part[:, k].copy()]

    def _f(self, sumk):
        mean = sumk / self.k
        return -mean if self.name == "1-NN distance" else 1.0 / (mean + 1e-10)

    def _delta_knn(self, D, stat, rows, rs):
        sumk, dk, dk1 = (a[rows] for a in stat)
        sub = D[np.ix_(rows, rs)] if rows is not None else D[:, rs]
        new = sumk[:, None] + np.where(sub <= dk[:, None], dk1[:, None] - sub, 0.0)
        return self._f(new) - self._f(sumk)[:, None]

    def delta_c(self, rows, rs):
        """Change in candidate scores (len(rows), len(rs)) from removing each r alone."""
        rows = np.asarray(rows)
        if self.kernel:
            return -self.Kc[np.ix_(rows, rs)]
        return self._delta_knn(self.Dc, self.cstat, rows, rs)

    def tau_after(self, rs):
        """Threshold after removing each r alone (parameters fixed), exact percentile."""
        al = np.flatnonzero(self.alive)
        if self.kernel:
            dl = -self.Km[np.ix_(al, rs)]
        else:
            dl = self._delta_knn(self.Dm, self.mstat, al, rs)
        A = (self.loo[al][:, None] + dl).T  # (R, n_alive)
        pos = np.searchsorted(al, rs)
        A[np.arange(len(rs)), pos] = np.inf  # r leaves the pool
        n = len(al) - 1
        v = Q / 100 * (n - 1)
        lo = int(np.floor(v))
        fr = v - lo
        part = np.partition(A, [lo, lo + 1], axis=1)
        return part[:, lo] + fr * (part[:, lo + 1] - part[:, lo])

    def remove(self, rs):
        self.alive[rs] = False
        self.fresh = False
        if self.kernel:
            self.s = self.s - self.Kc[:, rs].sum(1)
            self.Kc[:, rs] = 0.0
            if self.need_loo:
                self.loo = self.loo - self.Km[:, rs].sum(1)
                self.Km[:, rs] = 0.0
        else:
            for D, stat in [(self.Dc, self.cstat)] + ([(self.Dm, self.mstat)] if self.need_loo else []):
                hit = np.flatnonzero((D[:, rs] <= stat[2][:, None]).any(1))
                D[:, rs] = np.inf
                if len(hit):
                    new = self._stats(D[hit])
                    for a, b in zip(stat, new):
                        a[hit] = b
            self.s = self._f(self.cstat[0])
            if self.need_loo:
                self.loo = self._f(self.mstat[0])
        if self.need_loo:
            self.tau = np.percentile(self.loo[self.alive], Q)


# Greedy removal with refit ############################################################


def greedy(metric, name, kind, i, j=None):
    """kind 'novel': make candidate i novel. 'swap'/'overtake': make candidate j score below i."""
    dm = dm_of(metric)
    t0 = time.perf_counter()
    st = State(dm, name, need_loo=(kind == "novel")).refit(np.ones(N_MP, bool))

    if kind == "novel":
        def margin(s):
            return s.s[i] - s.tau

        def effects(s):
            al = np.flatnonzero(s.alive)
            dsc = s.delta_c([i], al)[0]
            short = np.union1d(al[np.argsort(dsc, kind="stable")[:SHORTLIST]],
                               al[np.argsort(s.loo[al], kind="stable")[:SHORTLIST]])
            return short, s.delta_c([i], short)[0] - (s.tau_after(short) - s.tau)
    else:
        def margin(s):
            return s.s[j] - s.s[i]

        def effects(s):
            al = np.flatnonzero(s.alive)
            d = s.delta_c([j, i], al)
            return al, d[0] - d[1]

    m0 = margin(st)
    removed, last_refit, n_refits, n_false, first_fixed, n_plateau = [], 0, 0, 0, None, 0
    status = "censored"
    while True:
        if margin(st) < 0:
            if first_fixed is None:
                first_fixed = len(removed)
            if not st.fresh:
                st.refit(st.alive)
                n_refits += 1
                last_refit = len(removed)
                if margin(st) >= 0:
                    n_false += 1
                    continue
            status = "flipped"
            break
        if len(removed) >= CAP:
            break
        rs, eff = effects(st)
        b = min(max(1, int(0.05 * len(removed))), CAP - len(removed))
        pick = np.argsort(eff, kind="stable")[:b]
        pick = pick[eff[pick] < 0]
        if len(pick) == 0:
            if not st.fresh:
                st.refit(st.alive)
                n_refits += 1
                last_refit = len(removed)
                continue
            # no single removal helps (tie plateau, e.g. several equidistant nearest
            # neighbours): remove the references nearest to the candidate that must lose density
            target = i if kind == "novel" else j
            al = np.flatnonzero(st.alive)
            rs = al[np.argsort(dm[N_MP + target, al], kind="stable")]
            pick = np.arange(b)
            n_plateau += 1
        st.remove(rs[pick])
        removed.extend(rs[pick].tolist())
        if len(removed) - last_refit >= max(5, 0.1 * last_refit):
            st.refit(st.alive)
            n_refits += 1
            last_refit = len(removed)

    cert = np.nan
    if status == "flipped":  # independent exact check with novelty_scores
        s, tau = exact_eval(dm, st.alive, name)
        cert = bool(s[i] < tau) if kind == "novel" else bool(s[j] < s[i])
    return dict(metric=metric, score=name, kind=kind, i=i, j=-1 if j is None else j,
                margin0=m0, n_removed=len(removed) if status == "flipped" else np.nan,
                status=status, certified=cert, first_fixed_flip=first_fixed,
                n_refits=n_refits, n_false_flips=n_false, n_plateau_steps=n_plateau, seconds=time.perf_counter() - t0)


def run_greedy(job):
    return greedy(*job)


# Subsampling ##########################################################################


def run_draw(job):
    metric, ratio, draw, seed = job
    dm = dm_of(metric)
    rng = np.random.default_rng(seed)
    alive = np.zeros(N_MP, bool)
    alive[rng.choice(N_MP, int(round(ratio * N_MP)), replace=False)] = True
    out = []
    for name in NAMES:
        s, tau = exact_eval(dm, alive, name)
        out.append(dict(metric=metric, ratio=ratio, draw=draw, score=name, s=s, novel=s < tau))
    return out


def tiers(lo, hi, rank0):
    """Component tiers: connected groups of overlapping intervals. Leader tiers: walk down the
    full-data ranking, opening a new tier when an interval no longer overlaps the tier leader's."""
    order = np.argsort(lo, kind="stable")
    comp = np.empty(len(lo), int)
    c, reach = 0, -np.inf
    for k in order:
        if lo[k] > reach:
            c += 1
        reach = max(reach, hi[k])
        comp[k] = c
    lead = np.empty(len(lo), int)
    t, leader = 0, None
    for k in np.argsort(rank0, kind="stable"):
        if leader is None or lo[k] > hi[leader]:
            t, leader = t + 1, k
        lead[k] = t
    return comp, lead


# Main #################################################################################


def check_state():
    """The fast state must reproduce novelty_scores exactly on the full and a reduced reference."""
    rng = np.random.default_rng(1)
    for metric in METRICS:
        dm = dm_of(metric)
        for alive in [np.ones(N_MP, bool), rng.random(N_MP) < 0.7]:
            for name in NAMES:
                st = State(dm, name, True).refit(alive)
                s, tau = exact_eval(dm, alive, name)
                assert np.allclose(st.s, s, rtol=1e-9, atol=1e-12), (metric, name)
                assert np.isclose(st.tau, tau, rtol=1e-9), (metric, name, st.tau, tau)
    print("state check passed", flush=True)


def part1(pool, draws, seed):
    jobs = [(m, r, d, (seed, mi, ri, d)) for mi, m in enumerate(METRICS)
            for ri, r in enumerate(RATIOS) for d in range(draws)]
    t = time.perf_counter()
    res = [x for out in pool.imap_unordered(run_draw, jobs, chunksize=4) for x in out]
    print(f"part 1: {len(jobs)} subsamples in {time.perf_counter() - t:.0f}s", flush=True)

    names = gnome_names()
    rows, summ = [], []
    for metric in METRICS:
        dm = dm_of(metric)
        for name in NAMES:
            s_full, tau_full = exact_eval(dm, np.ones(N_MP, bool), name)
            r_full = rankdata(s_full, method="average")
            for ratio in RATIOS:
                sel = [x for x in res if x["metric"] == metric and x["ratio"] == ratio and x["score"] == name]
                S = np.array([x["s"] for x in sel])
                R = rankdata(S, method="average", axis=1)
                Rmin = rankdata(S, method="min", axis=1)
                lo, hi = np.percentile(R, 5, axis=0), np.percentile(R, 95, axis=0)
                comp, lead = tiers(lo, hi, r_full)
                # pairwise separability: P(i ranked more novel than j) >= 0.95 or <= 0.05
                P = (S[:, :, None] < S[:, None, :]).mean(0)
                iu = np.triu_indices(len(s_full), k=1)
                sep = np.mean((P[iu] >= 0.95) | (P[iu] <= 0.05))
                for c in range(len(s_full)):
                    rows.append(dict(metric=metric, score=name, ratio=ratio, cand=c, name=names[c],
                                     full_score=s_full[c], full_rank=r_full[c],
                                     full_novel=bool(s_full[c] < tau_full),
                                     rank_lo=lo[c], rank_med=np.median(R[:, c]), rank_hi=hi[c],
                                     width=hi[c] - lo[c], p_top5=np.mean(Rmin[:, c] <= 5),
                                     p_top10=np.mean(Rmin[:, c] <= 10),
                                     p_novel=np.mean([x["novel"][c] for x in sel]),
                                     tier_comp=comp[c], tier_lead=lead[c]))
                csz = np.bincount(comp)[1:]
                lsz = np.bincount(lead)[1:]
                pn = np.mean([x["novel"] for x in sel], axis=0)
                summ.append(dict(metric=metric, score=name, ratio=ratio, draws=len(sel),
                                 median_width=np.median(hi - lo), max_width=np.max(hi - lo),
                                 n_tiers_comp=len(csz), tier_sizes_comp="/".join(map(str, csz)),
                                 n_tiers_lead=len(lsz), tier_sizes_lead="/".join(map(str, lsz)),
                                 frac_sep_pairs=sep,
                                 n_sure_top5=int(np.sum(np.mean(Rmin <= 5, 0) >= 0.9)),
                                 n_sure_top10=int(np.sum(np.mean(Rmin <= 10, 0) >= 0.9)),
                                 n_novel_full=int(np.sum(s_full < tau_full)),
                                 n_decision_stable=int(np.sum((pn >= 0.9) | (pn <= 0.1)))))
    return pd.DataFrame(rows), pd.DataFrame(summ)


def part2(pool):
    jobs, pairs = [], []
    for metric in METRICS:
        dm = dm_of(metric)
        for name in NAMES:
            s, tau = exact_eval(dm, np.ones(N_MP, bool), name)
            order = np.argsort(s, kind="stable")  # most novel first
            novel = np.flatnonzero(s < tau)
            nonnovel = order[~np.isin(order, novel)]
            jobs += [(metric, name, "novel", int(c)) for c in nonnovel]
            jobs += [(metric, name, "swap", int(a), int(b)) for a, b in zip(order[:-1], order[1:])]
            for c in novel:
                for c2 in nonnovel[:N_OVERTAKERS]:
                    jobs.append((metric, name, "overtake", int(c), int(c2)))
    # heavy jobs first
    jobs.sort(key=lambda j: (j[2] != "novel", j[1] != "MINov (exact)"))
    t = time.perf_counter()
    res = []
    for k, r in enumerate(pool.imap_unordered(run_greedy, jobs)):
        res.append(r)
        if (k + 1) % 100 == 0:
            print(f"  part 2: {k + 1}/{len(jobs)} jobs, {time.perf_counter() - t:.0f}s", flush=True)
    df = pd.DataFrame(res).rename(columns={"kind": "task"})
    print(f"part 2: {len(jobs)} greedy runs in {time.perf_counter() - t:.0f}s", flush=True)
    return df


def minov_neff(metric):
    dm = dm_of(metric)
    w = MIProfile(dm[:N_MP, :N_MP]).weight(dm[N_MP:, :N_MP])
    sw = w.sum(1)
    return sw, np.where(sw > 0, sw**2 / np.maximum((w**2).sum(1), 1e-300), 0.0)


def partial_spearman(y, x, z):
    """Spearman correlation of y and x after linearly removing rank(z) from both ranks."""
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(z)
    ry, rx, rz = (rankdata(a[ok]) for a in (y, x, z))
    Z = np.column_stack([np.ones(ok.sum()), rz])
    res = [v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0] for v in (ry, rx)]
    r = np.corrcoef(*res)[0, 1]
    n = ok.sum()
    tstat = r * np.sqrt((n - 3) / max(1e-12, 1 - r**2))
    from scipy.stats import t as tdist
    return r, 2 * tdist.sf(abs(tstat), n - 3), n


def part3(ranks, rem):
    rows = []
    for metric in METRICS:
        dens, neff = minov_neff(metric)
        for name in NAMES:
            a = rem[(rem.metric == metric) & (rem.score == name)]
            meas = {}
            # (a) removals to become novel (censored -> CAP+1 as a rank-preserving fill)
            nv = a[a.task == "novel"]
            y = np.full(len(dens), np.nan)
            y[nv.i.values] = np.where(nv.status == "flipped", nv.n_removed, CAP + 1)
            meas["removals_to_novel"] = y
            # (b) swap cost per candidate: min over its adjacent swaps
            sw = a[a.task == "swap"]
            y = np.full(len(dens), np.nan)
            for r in sw.itertuples():
                v = r.n_removed if r.status == "flipped" else CAP + 1
                for c in (r.i, r.j):
                    y[c] = v if np.isnan(y[c]) else min(y[c], v)
            meas["min_adjacent_swap"] = y
            for ratio in RATIOS:
                rr = ranks[(ranks.metric == metric) & (ranks.score == name) & (ranks.ratio == ratio)]
                y = np.full(len(dens), np.nan)
                y[rr.cand.values] = rr.width.values
                meas[f"rank_width_m{ratio}"] = y
            for k, y in meas.items():
                ok = np.isfinite(y)
                if ok.sum() < 6:
                    continue
                rd = spearmanr(y[ok], dens[ok])[0]
                rn = spearmanr(y[ok], neff[ok])[0]
                rdn = spearmanr(dens[ok], neff[ok])[0]
                pr, pp, n = partial_spearman(y, neff, dens)
                pr2, pp2, _ = partial_spearman(y, dens, neff)
                rows.append(dict(metric=metric, score=name, measure=k, n=n, rho_density=rd, rho_neff=rn,
                                 rho_density_neff=rdn, partial_neff_given_density=pr, p_partial_neff=pp,
                                 partial_density_given_neff=pr2, p_partial_density=pp2))
    return pd.DataFrame(rows)


def plots(ranks, rem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"MINov (exact)": "#2a78d6", "KDE (median bw)": "#1baf7a",
           "1-NN distance": "#eb6834", "kNN (paper)": "#eda100"}
    # Figure 1: 90% rank intervals (m/n = 0.5), candidates sorted by full-data rank
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True, sharey=True)
    for a, metric in enumerate(METRICS):
        for b, name in enumerate(NAMES):
            ax = axes[a, b]
            d = ranks[(ranks.metric == metric) & (ranks.score == name) & (ranks.ratio == 0.5)]
            d = d.sort_values("full_rank")
            y = np.arange(1, len(d) + 1)
            ax.hlines(y, d.rank_lo, d.rank_hi, color=col[name], lw=2)
            ax.plot(d.full_rank, y, "o", ms=3, color="#333333")
            nov = d.full_novel.values
            ax.plot(d.full_rank[nov], y[nov], "o", ms=5, mfc="none", mec="#c0392b", mew=1)
            nt = d.tier_comp.nunique()
            ax.set_title(f"{metric} | {name}\n{nt} overlap tiers", fontsize=9)
            ax.invert_yaxis()
            ax.grid(alpha=0.25, lw=0.5)
            if b == 0:
                ax.set_ylabel("candidate (full-data order)")
            if a == 1:
                ax.set_xlabel("rank (1 = most novel)")
    fig.suptitle("GNoME candidates: 90% rank intervals under 50%-of-MP subsampling (red ring = novel on full MP)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e4_rank_intervals.png"), dpi=150)
    plt.close(fig)

    # Figure 2: removal counts by task and score
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    tasks = [("novel", "(a) non-novel -> novel"), ("swap", "(b) adjacent pair swap"),
             ("overtake", "(c) novel overtaken")]
    rng = np.random.default_rng(0)
    for ax, (task, title) in zip(axes, tasks):
        for k, metric in enumerate(METRICS):
            for b, name in enumerate(NAMES):
                d = rem[(rem.task == task) & (rem.metric == metric) & (rem.score == name)]
                if task == "overtake":  # cheapest overtaker per novel candidate
                    d = d.groupby("i").n_removed.min().to_frame()
                    d["status"] = np.where(d.n_removed.notna(), "flipped", "censored")
                if len(d) == 0:
                    continue
                v = np.where(d.status == "flipped", d.n_removed, CAP * 1.4)
                x = k * 5 + b + rng.uniform(-0.25, 0.25, len(v))
                ax.scatter(x, v, s=10, color=col[name], alpha=0.7, lw=0,
                           label=name if (k == 0 and task == "novel") else None)
                ax.hlines(np.median(v), k * 5 + b - 0.35, k * 5 + b + 0.35, color="#333333", lw=1.5)
        ax.axhline(CAP * 1.4, color="#999999", lw=0.5, ls=":")
        ax.set_yscale("log")
        ax.set_xticks([1.5, 6.5])
        ax.set_xticklabels(METRICS)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5, axis="y")
    axes[0].set_ylabel(f"greedy removals (certified; top row = >{CAP}, censored)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e4_removal_fragility.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--reuse", action="store_true",
                        help="reuse part 1 and re-run only greedy runs that stopped on a tie plateau")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not args.skip_check:
        check_state()

    with Pool(args.workers) as pool:
        if args.reuse:
            ranks = pd.read_csv(os.path.join(OUT, "e4_rank_intervals.csv"))
            tier_summary = pd.read_csv(os.path.join(OUT, "e4_tier_summary.csv"))
            rem = pd.read_csv(os.path.join(OUT, "e4_removals.csv"))
            # the greedy is deterministic, so only runs that stopped on a tie plateau change
            redo = rem[rem.status == "stuck"]
            jobs = [(r.metric, r.score, r.task, int(r.i)) + (() if r.j < 0 else (int(r.j),))
                    for r in redo.itertuples()]
            new = pd.DataFrame(pool.map(run_greedy, jobs)).rename(columns={"kind": "task"})
            print(f"re-ran {len(jobs)} plateau-stopped greedy runs", flush=True)
            rem = pd.concat([rem[rem.status != "stuck"], new], ignore_index=True)
        else:
            ranks, tier_summary = part1(pool, args.draws, args.seed)
            ranks.to_csv(os.path.join(OUT, "e4_rank_intervals.csv"), index=False)
            tier_summary.to_csv(os.path.join(OUT, "e4_tier_summary.csv"), index=False)
            rem = part2(pool)
    rem.to_csv(os.path.join(OUT, "e4_removals.csv"), index=False)
    h4c = part3(ranks, rem)
    h4c.to_csv(os.path.join(OUT, "e4_h4c_correlations.csv"), index=False)
    plots(ranks, rem)

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        print(tier_summary)
        g = rem.assign(cens=rem.status != "flipped")
        print(g.groupby(["task", "metric", "score"]).agg(
            n=("i", "size"), median=("n_removed", "median"), min=("n_removed", "min"),
            max=("n_removed", "max"), censored=("cens", "sum"),
            le5=("n_removed", lambda v: int((v <= 5).sum())),
            cert_fail=("certified", lambda v: int((v == False).sum())),  # noqa: E712
            false_flips=("n_false_flips", "sum")))
        print(h4c)


if __name__ == "__main__":
    main()
