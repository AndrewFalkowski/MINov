"""E12: Do the Li forward-test conclusions replicate in other chemistries (Na, ternary nitrides)?

Data: data/raw/processed_<chem>/ from experiments/build_chem_sets.py (R = experimental MP-2023 entries with a
first-report year, G = GNoME Dec-2023 sample; U_elmd.npy, U_lostop.npy float32).

Forward test (E7 / E10 recipe), Y = 2000 (1990, 2005 as checks):
  Reference_Y = R rows first reported <= Y. Pools: (a) G; (b) R rows first reported after Y, scored at Y as if new.
  Future = full R; for pool b the candidate and its same-fkey rows first reported after Y are excluded (E7 rule);
  for exact match also the "keep siblings" variant (only the candidate itself excluded, E10).
  - call flips, 5% leave-one-out threshold: MINov (exact), 1-NN distance, Gaussian KDE (median bandwidth),
    on ElMD and LoStOP;
  - exact match: composition (same fkey), structure (same fkey + StructureMatcher defaults), prototype (same
    anonymized formula + StructureMatcher(comparator=FrameworkComparator()), default tolerances);
  - coverage of nominal 90% m-out-of-n rank intervals (m/n = 0.5, DRAWS draws of Reference_Y), ranks within pool;
  - year-split AUC: R rows first reported after Y (scored vs Reference_Y) vs rows reported <= Y (leave-one-out
    within Reference_Y); P(later scored more novel). Measures newness / local isolation, not global novelty.

Machine limits: at most 3 worker processes (structure matching only); matrices read with mmap_mode="r".

Usage: uv run python experiments/e12_new_chemistries.py --chem na|nitride [--stages fwd match analyze]
       uv run python experiments/e12_new_chemistries.py --stages combine fig
"""

import argparse
import gzip
import json
import os
import resource
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
from novelty_scores import SCORES  # noqa: E402

OUT = os.path.join(REPO, "experiments", "results")
METRICS = ["elmd", "lostop"]
NAMES = ["MINov (exact)", "1-NN distance", "KDE (median bw)"]
SHORT = {"MINov (exact)": "MINov", "1-NN distance": "1-NN", "KDE (median bw)": "median KDE"}
YEARS = [2000, 1990, 2005]
DRAWS = 100
RATIO = 0.5
Q = 5
SEED = 0
N_BOOT = 500
WORKERS = 3
CHEM_LABEL = {"li": "Li", "na": "Na", "nitride": "Ternary nitrides"}

warnings.filterwarnings("ignore")


def read_csv(path, **kw):
    """pd.read_csv that keeps the formula "NaN" (sodium nitride) as a string; only empty cells are missing."""
    kw.setdefault("keep_default_na", False)
    kw.setdefault("na_values", [""])
    return pd.read_csv(path, **kw)


def log(msg, t0=[time.perf_counter()]):  # noqa: B006
    print(f"[{time.perf_counter() - t0[0]:7.0f}s] {msg}", flush=True)


def proc(chem):
    return os.path.join(REPO, "data", "raw", f"processed_{chem}")


def peak_mb():
    return dict(peak_rss_self_MB=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
                peak_rss_children_MB=round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20, 1))


def load_index(chem):
    U = read_csv(os.path.join(proc(chem), "U_index.csv"), dtype={"material_id": str})
    if "lostop_ok" in U:
        U = U[U.lostop_ok]  # rows without LoStOP features are dropped from every analysis
    return U


def block(chem, metric, a, b):
    D = np.load(os.path.join(proc(chem), f"U_{metric}.npy"), mmap_mode="r")
    return np.asarray(D[np.ix_(a, b)], dtype=np.float64)


def ref_block(chem, metric, rows):
    D = block(chem, metric, rows, rows)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    return D


def evaluate(Dref, Dq, name, mask=None):
    """Query scores (higher = denser = less novel) + 5% leave-one-out threshold + LOO reference scores."""
    q = Dq if mask is None else np.where(mask, np.inf, Dq)
    s = SCORES[name](Dref, q)
    loo_q = Dref.copy()
    np.fill_diagonal(loo_q, np.inf)
    loo = SCORES[name](Dref, loo_q)
    return s, np.percentile(loo, Q), loo


def auc(neg, pos):
    """P(pos scored more novel = LOWER density score than neg); ties half."""
    r = rankdata(-np.concatenate([neg, pos]))
    n0, n1 = len(neg), len(pos)
    return (r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


# Forward test: density / NN scores ######################################################


def stage_fwd(chem, years):
    U = load_index(chem)
    rows_all = U.row.values
    is_r = (U.set == "R").values
    fy = U.first_year.values
    fk = U.fkey.values
    r = np.flatnonzero(is_r)
    g = np.flatnonzero(~is_r)
    cand_rows, auc_rows, meta = [], [], {}
    for Y in years:
        t0 = time.perf_counter()
        refY = r[fy[r] <= Y]
        poolb = r[fy[r] > Y]
        cands = np.r_[g, poolb]
        pools = {"a_G": np.arange(len(g)), "b_Rnew": np.arange(len(g), len(cands))}
        # future pool-b mask: same fkey and first reported after Y (includes the candidate itself)
        maskF = (fk[poolb][:, None] == fk[r][None, :]) & (fy[r][None, :] > Y)
        for k, m in enumerate(METRICS):
            Dref = ref_block(chem, m, rows_all[refY])
            Dc = block(chem, m, rows_all[cands], rows_all[refY])
            Dfull = ref_block(chem, m, rows_all[r])
            Dg_full = block(chem, m, rows_all[g], rows_all[r])
            Db_full = block(chem, m, rows_all[poolb], rows_all[r])
            n = len(Dref)
            draws = {name: [] for name in NAMES}
            for d in range(DRAWS):
                rng = np.random.default_rng((SEED, Y, k, d))
                idx = np.sort(rng.choice(n, int(round(RATIO * n)), replace=False))
                ref, q = Dref[np.ix_(idx, idx)], Dc[:, idx]
                for name in NAMES:
                    draws[name].append(SCORES[name](ref, q))
            for name in NAMES:
                sY, thrY, looY = evaluate(Dref, Dc, name)
                sa, thrF, _ = evaluate(Dfull, Dg_full, name)
                sb, _, _ = evaluate(Dfull, Db_full, name, maskF)
                sF = np.r_[sa, sb]
                S = np.array(draws[name])
                for p, ix in pools.items():
                    if len(ix) == 0:
                        continue
                    rY = rankdata(sY[ix])
                    rF = rankdata(sF[ix])
                    R = rankdata(S[:, ix], axis=1)
                    lo, hi = np.percentile(R, 5, axis=0), np.percentile(R, 95, axis=0)
                    for j, c in enumerate(ix):
                        cand_rows.append(dict(chem=chem, Y=Y, metric=m, score=name, pool=p,
                                              material_id=U.material_id.values[cands[c]], fkey=fk[cands[c]],
                                              sY=sY[c], sF=sF[c], thrY5=thrY, thrF5=thrF, novelY5=sY[c] < thrY,
                                              novelF5=sF[c] < thrF, rankY=rY[j], rankF=rF[j], rank_lo=lo[j],
                                              rank_hi=hi[j]))
                # year-split AUC: later (pool b at Y) vs earlier (LOO within Reference_Y)
                s_post = sY[pools["b_Rnew"]]
                if len(s_post) and len(looY):
                    a = auc(looY, s_post)
                    rng = np.random.default_rng((SEED, Y, k, 99))
                    B = [auc(looY[rng.integers(0, len(looY), len(looY))],
                             s_post[rng.integers(0, len(s_post), len(s_post))]) for _ in range(N_BOOT)]
                    newf = ~np.isin(fk[poolb], fk[refY])
                    auc_rows.append(dict(chem=chem, Y=Y, metric=m, score=name, n_pre=len(looY), n_post=len(s_post),
                                         auc=a, auc_lo=np.quantile(B, 0.025), auc_hi=np.quantile(B, 0.975),
                                         n_post_newformula=int(newf.sum()),
                                         auc_newformula=auc(looY, s_post[newf]) if newf.any() else np.nan))
            del Dref, Dc, Dfull, Dg_full, Db_full
        meta[Y] = dict(n_ref=len(refY), n_G=len(g), n_poolb=len(poolb), seconds=round(time.perf_counter() - t0, 1))
        log(f"{chem} fwd Y={Y}: {meta[Y]}")
    pd.DataFrame(cand_rows).to_csv(os.path.join(OUT, f"e12_{chem}_fwd_candidates.csv.gz"), index=False)
    pd.DataFrame(auc_rows).to_csv(os.path.join(OUT, f"e12_{chem}_year_auc.csv"), index=False)
    update_meta(chem, {"fwd": dict(years=meta, **peak_mb())})


def update_meta(chem, upd):
    path = os.path.join(OUT, f"e12_{chem}_meta.json")
    m = json.load(open(path)) if os.path.exists(path) else {}
    m.update(upd)
    json.dump(m, open(path, "w"), indent=1, default=str)


# Exact match ###########################################################################


def stage_match(chem, budget_min):
    import e10_exact_match as e10
    t0 = time.perf_counter()
    U = load_index(chem)
    r = U[U.set == "R"]
    g = U[U.set == "G"]
    need_g = g[g.anon.isin(set(r.anon))].material_id.tolist()
    with gzip.open(os.path.join(proc(chem), "R_structures.json.gz"), "rt") as f:
        mp = json.load(f)
    items = [(k, "mp", mp[k]) for k in r.material_id]
    del mp
    with gzip.open(os.path.join(proc(chem), "G_cifs.json.gz"), "rt") as f:
        cif = json.load(f)
    items += [(k, "cif", cif[k]) for k in need_g]
    del cif
    log(f"{chem}: reducing {len(items)} structures ({len(need_g)} G with a same-anon R row)")
    with Pool(WORKERS) as pool:
        res = pool.map(e10.reduce_one, items, chunksize=20)
    del items
    red = {k: d for k, d, _, _ in res}
    nred = {k: n for k, _, n, _ in res}
    log(f"reduced in {time.perf_counter() - t0:.0f}s")
    jobs = []
    rf = r.groupby("fkey").material_id.apply(list)
    for ids in rf:
        jobs += [("struct", a, b, 1.0) for a in ids for b in ids if a != b]
    for a, f in zip(g.material_id, g.fkey):  # G -> R, same composition (normally none: GNoME formulas are new)
        jobs += [("struct", a, b, 1.0) for b in rf.get(f, [])]
    n_struct = len(jobs)
    an = r.groupby("anon").material_id.apply(list)
    ridx = {k: i for i, k in enumerate(r.material_id)}
    anon_of = dict(zip(U.material_id, U.anon))
    proto, n_pre = [], 0
    for a in list(r.material_id) + need_g:
        for b in an.get(anon_of[a], []):
            if a == b or (a in ridx and ridx[b] <= ridx[a]):
                continue
            n_pre += 1
            if nred[a] == nred[b]:
                proto.append((a, b))
    proto.sort(key=lambda p: nred[p[0]])
    jobs += [("proto", a, b, 1.0) for a, b in proto]
    log(f"{n_struct} struct fits, {len(proto)} proto pairs ({n_pre} same-anon before site-count filter)")
    chunks = [jobs[i:i + 40] for i in range(0, len(jobs), 40)]
    out, done, cut = [], 0, False
    deadline = t0 + budget_min * 60
    with Pool(WORKERS, initializer=e10.init_pairs, initargs=(red,)) as pool:
        it = pool.imap(e10.fit_chunk, chunks)
        for _ in range(len(chunks)):
            out += it.next()
            done += 1
            if done % 200 == 0:
                log(f"  {done}/{len(chunks)} chunks")
            if time.perf_counter() > deadline:
                cut = True
                log(f"BUDGET reached after {done}/{len(chunks)} chunks")
                pool.terminate()
                break
    df = pd.DataFrame(out, columns=["kind", "a", "b", "scale", "match", "seconds"])
    df.to_csv(os.path.join(OUT, f"e12_{chem}_pairs.csv.gz"), index=False)
    update_meta(chem, {"match": dict(n_reduced=len(red), n_struct_fits=n_struct, n_proto_pairs=len(proto),
                                     n_proto_same_anon_pairs=n_pre, n_fitted=len(df), n_jobs=len(jobs),
                                     budget_cut=cut, n_timeouts=int((df.match < 0).sum()),
                                     seconds=round(time.perf_counter() - t0, 1), **peak_mb())})


def match_sets(U, pairs):
    r = U[U.set == "R"]
    sets = {"comp": {}}
    byf = r.groupby("fkey").material_id.apply(set)
    for m, f in zip(U.material_id, U.fkey):
        sets["comp"][m] = byf.get(f, set()) - {m}
    for kind, d in pairs[pairs.match == 1].groupby("kind"):
        dd = {}
        for a, b in zip(d.a, d.b):
            dd.setdefault(a, set()).add(b)
            if kind == "proto":
                dd.setdefault(b, set()).add(a)
        sets[kind] = dd
    sets.setdefault("struct", {})
    sets.setdefault("proto", {})
    return sets


def exact_calls(U, sets, Y):
    fy = dict(zip(U.material_id, U.first_year))
    fk = dict(zip(U.material_id, U.fkey))
    r = U[U.set == "R"]
    pools = {"a_G": U[U.set == "G"].material_id.tolist(), "b_Rnew": r[r.first_year > Y].material_id.tolist()}
    rows = []
    for notion, d in sets.items():
        for pool, ids in pools.items():
            for c in ids:
                ms = d.get(c, set()) - {c}
                novelY = not any(fy[x] <= Y for x in ms)
                if pool == "a_G":
                    novelF = keepF = not ms
                else:
                    novelF = not any(not (fk[x] == fk[c] and fy[x] > Y) for x in ms)
                    keepF = not ms
                rows.append(dict(Y=Y, notion=notion, pool=pool, material_id=c, novelY=novelY, novelF=novelF,
                                 novelF_keep=keepF))
    return pd.DataFrame(rows)


def summarize(d, nY, nF):
    flip = d[nY] != d[nF]
    return dict(n=len(d), novel_rate_Y=d[nY].mean(), novel_rate_F=d[nF].mean(), flip_rate=flip.mean(),
                n_flip=int(flip.sum()), n_novel_to_not=int((d[nY] & ~d[nF]).sum()),
                n_not_to_novel=int((~d[nY] & d[nF]).sum()))


def stage_analyze(chem, years):
    U = load_index(chem)
    pairs = read_csv(os.path.join(OUT, f"e12_{chem}_pairs.csv.gz"), dtype={"a": str, "b": str})
    sets = match_sets(U, pairs)
    cand = read_csv(os.path.join(OUT, f"e12_{chem}_fwd_candidates.csv.gz"), dtype={"material_id": str})
    flips, allc = [], []
    for Y in years:
        c = exact_calls(U, sets, Y)
        allc.append(c)
        for (notion, pool), d in c.groupby(["notion", "pool"], sort=False):
            flips.append(dict(chem=chem, Y=Y, pool=pool, family="exact match", score=notion, metric="-",
                              **summarize(d, "novelY", "novelF")))
            if pool == "b_Rnew":
                flips.append(dict(chem=chem, Y=Y, pool="b_Rnew_keep_siblings", family="exact match", score=notion,
                                  metric="-", **summarize(d, "novelY", "novelF_keep")))
        for (score, m, pool), d in cand[cand.Y == Y].groupby(["score", "metric", "pool"]):
            cov = ((d.rankF >= d.rank_lo) & (d.rankF <= d.rank_hi)).mean()
            flips.append(dict(chem=chem, Y=Y, pool=pool, family="density / NN", score=SHORT[score], metric=m,
                              **summarize(d, "novelY5", "novelF5"), coverage90=cov,
                              median_width_frac=np.median(d.rank_hi - d.rank_lo) / len(d)))
    pd.concat(allc).to_csv(os.path.join(OUT, f"e12_{chem}_exact_calls.csv.gz"), index=False)
    ft = pd.DataFrame(flips)
    ft.to_csv(os.path.join(OUT, f"e12_{chem}_flip_table.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        print(ft[ft.Y == 2000].to_string())


# Combine with Li and plot ##############################################################


def li_table():
    """Li rows from E10 (flips, exact match) and E7 (coverage), E9 (year AUC), relabelled."""
    ft = read_csv(os.path.join(OUT, "e10_flip_table.csv"))
    ft = ft[ft.score.isin(["comp", "struct@1x", "proto@1x", "MINov", "1-NN", "median KDE"])].copy()
    ft["score"] = ft.score.str.replace("@1x", "", regex=False)
    ft["pool"] = ft.pool.str.replace("R23new", "Rnew", regex=False)
    ft.insert(0, "chem", "li")
    cov = []
    for Y in YEARS:
        e7 = read_csv(os.path.join(OUT, f"e7_fwd{Y}_candidates.csv.gz"))
        for (score, m, pool), d in e7.groupby(["score", "metric", "pool"]):
            cov.append(dict(Y=Y, score=SHORT[score], metric=m, pool=pool.replace("R23new", "Rnew"),
                            coverage90=((d.rankF >= d.rank_lo) & (d.rankF <= d.rank_hi)).mean(),
                            median_width_frac=np.median(d.rank_hi - d.rank_lo) / len(d)))
    ft = ft.merge(pd.DataFrame(cov), on=["Y", "score", "metric", "pool"], how="left")
    a = read_csv(os.path.join(OUT, "e9_discovery_auc.csv"))
    a = a[(a.ref == "loo") & (a.comparison == "post") & a.score.isin(["MI q=0.5", "kNN p=0"])].copy()
    a["score"] = a.score.map({"MI q=0.5": "MINov", "kNN p=0": "1-NN"})
    a = a.rename(columns={"n_neg": "n_pre", "n_pos": "n_post"})[["Y", "metric", "score", "n_pre", "n_post", "auc",
                                                                 "auc_lo", "auc_hi"]]
    a.insert(0, "chem", "li")
    return ft, a


def stage_combine(chems):
    ft_li, auc_li = li_table()
    fts = [ft_li] + [read_csv(os.path.join(OUT, f"e12_{c}_flip_table.csv")) for c in chems]
    aucs = [auc_li]
    for c in chems:
        a = read_csv(os.path.join(OUT, f"e12_{c}_year_auc.csv"))
        a["score"] = a.score.map(SHORT)
        aucs.append(a)
    ft = pd.concat(fts, ignore_index=True)
    au = pd.concat(aucs, ignore_index=True)
    ft.to_csv(os.path.join(OUT, "e12_flip_table.csv"), index=False)
    au.to_csv(os.path.join(OUT, "e12_year_auc.csv"), index=False)
    # compact wide table: one row per (Y, quantity), one column per chemistry
    rows = []
    for Y in YEARS:
        for _, x in ft[(ft.Y == Y)].iterrows():
            lab = f"{x.score}" + ("" if x.metric == "-" else f" {x.metric}")
            rows.append(dict(Y=Y, quantity="flip %", pool=x.pool, score=lab, chem=x.chem, value=100 * x.flip_rate,
                             n=x.n))
            if pd.notna(x.get("coverage90", np.nan)):
                rows.append(dict(Y=Y, quantity="coverage90 %", pool=x.pool, score=lab, chem=x.chem,
                                 value=100 * x.coverage90, n=x.n))
        for _, x in au[(au.Y == Y) & au.score.isin(["MINov", "1-NN"])].iterrows():
            rows.append(dict(Y=Y, quantity="year AUC", pool="later vs earlier", score=f"{x.score} {x.metric}",
                             chem=x.chem, value=x.auc, n=x.n_post))
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index=["Y", "quantity", "pool", "score"], columns="chem", values="value").reset_index()
    wide.to_csv(os.path.join(OUT, "e12_summary.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.precision", 2):
        print(wide[wide.Y == 2000].to_string())


def stage_fig(chems):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
    SERIES = {"MINov": "#2a78d6", "1-NN": "#eb6834", "median KDE": "#1baf7a"}
    EXACT = {"comp": "#52514e", "struct": "#8a8984", "proto": "#b5b4ae"}
    LABEL = {"MINov": "MINov", "1-NN": "1-NN distance", "median KDE": "Gaussian KDE, median bandwidth",
             "comp": "composition match", "struct": "structure match", "proto": "prototype match"}
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })
    ft = read_csv(os.path.join(OUT, "e12_flip_table.csv"))
    ft = ft[ft.Y == 2000]
    order = ["li"] + list(chems)
    panels = [("elmd", "ElMD (composition distance)", list(SERIES)), ("lostop", "LoStOP (structure distance)",
                                                                       list(SERIES)),
              ("-", "Exact match (\"not in the reference\")", list(EXACT))]
    pools = [("a_G", "GNoME candidates"), ("b_Rnew", "Later discoveries")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharey="row")
    x = np.arange(len(order))
    w = 0.26
    for i, (pool, ptitle) in enumerate(pools):
        for j, (metric, mtitle, scores) in enumerate(panels):
            ax = axes[i, j]
            for k, s in enumerate(scores):
                vals, ns = [], []
                for ch in order:
                    d = ft[(ft.chem == ch) & (ft.pool == pool) & (ft.score == s) & (ft.metric == metric)]
                    vals.append(d.flip_rate.iloc[0] * 100 if len(d) else np.nan)
                    ns.append(int(d.n.iloc[0]) if len(d) else 0)
                col = SERIES.get(s, EXACT.get(s))
                bars = ax.bar(x + (k - 1) * w, vals, w, color=col, label=LABEL[s], edgecolor=SURFACE, linewidth=2)
                for b, v in zip(bars, vals):
                    if np.isfinite(v):
                        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", va="bottom",
                                fontsize=7.5, color=INK2)
            nn = {ch: int(ft[(ft.chem == ch) & (ft.pool == pool)].n.iloc[0]) for ch in order}
            ax.set_xticks(x, [f"{CHEM_LABEL[ch]}\n(n = {nn[ch]:,})" for ch in order])
            ax.set_title(f"{ptitle}: {mtitle}" if j == 0 else mtitle, fontsize=10)
            ax.grid(axis="x", visible=False)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel("% of candidates whose call flipped")
            if i == 0:
                ax.legend(frameon=False, loc="upper left", fontsize=8)
        axes[i, 0].set_title(ptitle + "\n" + panels[0][1], fontsize=10)
    ymax = np.nanmax(ft[ft.score.isin(list(SERIES) + list(EXACT))].flip_rate) * 100
    for ax in axes.ravel():
        ax.set_ylim(0, ymax * 1.15 + 1)
    fig.suptitle("Forward test by chemistry: reference = experimental compounds known by 2000; "
                 "future = full MP-2023 experimental record\nNovel = below the 5% leave-one-out threshold; "
                 "exact match uses pymatgen StructureMatcher defaults",
                 x=0.01, ha="left", fontsize=9.5, color=INK2)
    keep = ft[(ft.pool == "b_Rnew_keep_siblings") & (ft.score == "comp")].set_index("chem").flip_rate * 100
    fig.text(0.01, 0.005, "Later discoveries: composition / structure match cannot flip by construction (later "
             "same-composition entries leave the future reference). Kept, composition match flips: "
             + ", ".join(f"{CHEM_LABEL[c]} {keep[c]:.1f}%" for c in order if c in keep) + ".",
             fontsize=8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.025, 1, 0.94))
    fig.savefig(os.path.join(OUT, "e12_flip_rates.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chem", choices=["na", "nitride"])
    ap.add_argument("--stages", nargs="+", default=["fwd", "match", "analyze"])
    ap.add_argument("--years", nargs="+", type=int, default=YEARS)
    ap.add_argument("--budget-min", type=float, default=20)
    ap.add_argument("--chems", nargs="+", default=["na", "nitride"])
    args = ap.parse_args()
    t = time.perf_counter()
    if "fwd" in args.stages:
        stage_fwd(args.chem, args.years)
    if "match" in args.stages:
        stage_match(args.chem, args.budget_min)
    if "analyze" in args.stages:
        stage_analyze(args.chem, args.years)
    if "combine" in args.stages:
        stage_combine(args.chems)
    if "fig" in args.stages:
        stage_fig(args.chems)
    if args.chem:
        update_meta(args.chem, {f"run_{'_'.join(args.stages)}_seconds": round(time.perf_counter() - t, 1)})


if __name__ == "__main__":
    main()
