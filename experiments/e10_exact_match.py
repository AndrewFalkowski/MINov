"""E10: Fragility of binary exact-match novelty ("not found in the reference") under reference growth.

Same forward test as E7: Reference_Y = R23 rows first reported <= Y (Y = 2000; 1990, 2005), future = full R23.
Pools: (a) G (3,000 GNoME Li, Dec-2023); (b) R23 rows first reported after Y. For pool b the future reference
excludes the candidate and its same-fkey rows first reported after Y (E7's rule); a variant "b_keep" excludes only
the candidate itself (later same-composition siblings stay).

Exact-match notions (novel iff no reference row matches):
  comp        same fkey (canonical reduced composition)
  struct@s    same fkey and pymatgen StructureMatcher fit, ltol/stol/angle_tol = s x (0.2, 0.3, 5),
              primitive_cell=True, scale=True (defaults otherwise), s in 0.5, 1, 2
  proto@s     same anonymized formula and StructureMatcher(comparator=FrameworkComparator()) fit
              (species-agnostic framework match, the substitution-derived notion of Negishi & Walsh)
Reduced (primitive + Niggli) structures are computed once and passed with skip_structure_reduction=True, which is
the same computation as the default fit; pairs whose reduced site counts differ are skipped because the matcher
returns no match for them (allow_subset=False, attempt_supercell=False). Proto pairs are fitted once per
unordered pair; struct pairs in both orders (candidate -> reference).

Also: composition match for all GNoME Dec-2023 Li formulas; comparison with E7's MINov / 1-NN / median-KDE calls.

Usage: uv run python experiments/e10_exact_match.py [--stages match analyze fig] [--workers 4]
"""

import argparse
import gzip
import json
import os
import resource
import signal
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(REPO, "data", "raw", "processed")
OUT = os.path.join(REPO, "experiments", "results")
YEARS = (2000, 1990, 2005)
TOLS = (0.5, 1.0, 2.0)
BASE_TOL = (0.2, 0.3, 5.0)
PAIR_TIMEOUT = 20  # s per fit; timeouts count as "no match" and are reported
SCORES = {"MINov (exact)": "MINov", "1-NN distance": "1-NN", "KDE (median bw)": "median KDE"}


def log(msg, t0=[time.perf_counter()]):  # noqa: B006
    print(f"[{time.perf_counter() - t0[0]:7.0f}s] {msg}", flush=True)


def index():
    from pymatgen.core import Composition
    U = pd.read_csv(os.path.join(PROC, "U_index.csv"), usecols=["row", "material_id", "set", "fkey", "nsites"])
    fy = pd.read_csv(os.path.join(PROC, "r23_first_year.csv"))[["row", "first_year"]]
    U = U[U.set.isin(["R23", "G"])].merge(fy, on="row", how="left").reset_index(drop=True)
    U["anon"] = [Composition(f).anonymized_formula for f in U.fkey]
    return U


# Matching (workers) ###################################################################


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout


def reduce_one(item):
    """(material_id, kind, payload) -> (material_id, reduced structure dict, n reduced sites, seconds)."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    mid, kind, payload = item
    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = Structure.from_dict(payload) if kind == "mp" else Structure.from_str(payload, fmt="cif")
    s.remove_oxidation_states()
    r = StructureMatcher._get_reduced_structure(s, True, True)
    return mid, r.as_dict(), len(r), time.perf_counter() - t


S = {}


def init_pairs(red):
    from pymatgen.core import Structure
    signal.signal(signal.SIGALRM, _alarm)
    for k, d in red.items():
        S[k] = Structure.from_dict(d)


def fit_chunk(chunk):
    """chunk: list of (kind, a, b, scale) -> list of (kind, a, b, scale, match 0/1/-1 timeout, seconds)."""
    from pymatgen.analysis.structure_matcher import FrameworkComparator, StructureMatcher
    out = []
    for kind, a, b, sc in chunk:
        lt, st, at = (x * sc for x in BASE_TOL)
        kw = dict(ltol=lt, stol=st, angle_tol=at, primitive_cell=True, scale=True)
        if kind == "proto":
            kw["comparator"] = FrameworkComparator()
        sm = StructureMatcher(**kw)
        t = time.perf_counter()
        signal.alarm(PAIR_TIMEOUT)
        try:
            m = int(bool(sm.fit(S[a], S[b], skip_structure_reduction=True)))
        except _Timeout:
            m = -1
        finally:
            signal.alarm(0)
        out.append((kind, a, b, sc, m, time.perf_counter() - t))
    return out


def stage_match(U, workers, proto_tols, budget_min):
    t0 = time.perf_counter()
    r23 = U[U.set == "R23"]
    g = U[U.set == "G"]
    need_g = g[g.anon.isin(set(r23.anon))].material_id.tolist()
    need_mp = set(r23.material_id)
    with gzip.open(os.path.join(PROC, "mp2023_li_structures.json.gz"), "rt") as f:
        mp = json.load(f)
    items = [(k, "mp", mp[k]) for k in r23.material_id if k in need_mp]
    del mp
    with gzip.open(os.path.join(PROC, "gnome_li_cifs.json.gz"), "rt") as f:
        cif = json.load(f)
    items += [(k, "cif", cif[k]) for k in need_g]
    del cif
    log(f"reducing {len(items)} structures ({len(need_g)} G with a same-anon R23 row)")
    with Pool(workers) as pool:
        res = pool.map(reduce_one, items, chunksize=20)
    del items
    red = {k: d for k, d, _, _ in res}
    nred = {k: n for k, _, n, _ in res}
    log(f"reduced in {time.perf_counter() - t0:.0f}s (max {max(r[3] for r in res):.1f}s)")

    # pair lists
    mid = U.material_id.values
    jobs = []
    fk = r23.groupby("fkey").material_id.apply(list)
    for ids in fk:
        for a in ids:
            for b in ids:
                if a != b:
                    jobs += [("struct", a, b, s) for s in TOLS]
    n_struct = len(jobs)
    an = r23.groupby("anon").material_id.apply(list)
    proto_pairs, n_pre = [], 0
    ridx = {k: i for i, k in enumerate(r23.material_id)}
    anon_of = dict(zip(mid, U.anon))
    for a in list(r23.material_id) + need_g:
        for b in an.get(anon_of[a], []):
            if a == b or (a in ridx and ridx[b] <= ridx[a]):
                continue  # unordered among R23
            n_pre += 1
            if nred[a] == nred[b]:
                proto_pairs.append((a, b))
    # cheapest first so a budget cut drops the largest cells
    proto_pairs.sort(key=lambda p: nred[p[0]])
    jobs += [("proto", a, b, s) for s in proto_tols for a, b in proto_pairs]
    log(f"{n_struct} struct fits, {len(proto_pairs)} proto pairs x {len(proto_tols)} tols "
        f"({n_pre} same-anon pairs before the reduced-site-count filter)")
    chunks = [jobs[i:i + 40] for i in range(0, len(jobs), 40)]
    out, done = [], 0
    deadline = t0 + budget_min * 60
    with Pool(workers, initializer=init_pairs, initargs=(red,)) as pool:
        it = pool.imap(fit_chunk, chunks)
        for k in range(len(chunks)):
            out += it.next()
            done += 1
            if done % 100 == 0:
                log(f"  {done}/{len(chunks)} chunks")
            if time.perf_counter() > deadline:
                log(f"BUDGET reached after {done}/{len(chunks)} chunks; remaining pairs not fitted")
                pool.terminate()
                break
    df = pd.DataFrame(out, columns=["kind", "a", "b", "scale", "match", "seconds"])
    df.to_csv(os.path.join(OUT, "e10_pairs.csv.gz"), index=False)
    rus = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS
    ruc = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
    meta = dict(n_reduced=len(red), n_struct_fits=n_struct, n_proto_pairs=len(proto_pairs),
                n_proto_same_anon_pairs=n_pre, proto_tols=list(proto_tols), n_fitted=len(df),
                n_jobs=len(jobs), n_timeouts=int((df.match < 0).sum()),
                seconds=time.perf_counter() - t0, peak_rss_parent_MB=rus, peak_rss_max_child_MB=ruc,
                fit_seconds_by_kind=df.groupby(["kind", "scale"]).seconds.sum().round(1).to_dict().__repr__())
    json.dump(meta, open(os.path.join(OUT, "e10_match_meta.json"), "w"), indent=1)
    log(json.dumps(meta, indent=1))


# Analysis #############################################################################


def match_sets(U, pairs):
    """{notion: {candidate material_id: set of matching R23 material_ids}}."""
    r23 = U[U.set == "R23"]
    sets = {"comp": {}}
    byf = r23.groupby("fkey").material_id.apply(set)
    for m, f in zip(U.material_id, U.fkey):
        sets["comp"][m] = byf.get(f, set()) - {m}
    for (kind, sc), d in pairs.groupby(["kind", "scale"]):
        name = f"{kind}@{sc:g}x"
        d = d[d.match == 1]
        dd = {}
        for a, b in zip(d.a, d.b):
            dd.setdefault(a, set()).add(b)
            if kind == "proto":
                dd.setdefault(b, set()).add(a)
        sets[name] = dd
    return sets


def calls(U, sets, Y):
    """Per candidate and notion: novel at Y and in the future (E7 rule; pool b also siblings-kept variant)."""
    fy = dict(zip(U.material_id, U.first_year))
    fk = dict(zip(U.material_id, U.fkey))
    r23 = U[U.set == "R23"]
    pools = {"a_G": U[U.set == "G"].material_id.tolist(), "b_R23new": r23[r23.first_year > Y].material_id.tolist()}
    rows = []
    for notion, d in sets.items():
        for pool, ids in pools.items():
            for c in ids:
                ms = d.get(c, set()) - {c}
                novelY = not any(fy[r] <= Y for r in ms)
                if pool == "a_G":
                    novelF = keepF = not ms
                else:
                    novelF = not any(not (fk[r] == fk[c] and fy[r] > Y) for r in ms)
                    keepF = not ms
                rows.append(dict(Y=Y, notion=notion, pool=pool, material_id=c, novelY=novelY, novelF=novelF,
                                 novelF_keep=keepF))
    return pd.DataFrame(rows)


def auc(score, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, score) if 0 < y.sum() < len(y) else np.nan


def summarize(d, novY, novF):
    flip = d[novY] != d[novF]
    return dict(n=len(d), novel_rate_Y=d[novY].mean(), novel_rate_F=d[novF].mean(), flip_rate=flip.mean(),
                n_flip=int(flip.sum()), n_novel_to_not=int((d[novY] & ~d[novF]).sum()),
                n_not_to_novel=int((~d[novY] & d[novF]).sum()))


def stage_analyze(U):
    pairs = pd.read_csv(os.path.join(OUT, "e10_pairs.csv.gz"))
    sets = match_sets(U, pairs)
    allc, flips, agree = [], [], []
    for Y in YEARS:
        c = calls(U, sets, Y)
        allc.append(c)
        for (notion, pool), d in c.groupby(["notion", "pool"], sort=False):
            flips.append(dict(Y=Y, pool=pool, family="exact match", score=notion, metric="-",
                              **summarize(d, "novelY", "novelF")))
            if pool == "b_R23new":
                flips.append(dict(Y=Y, pool="b_R23new_keep_siblings", family="exact match", score=notion,
                                  metric="-", **summarize(d, "novelY", "novelF_keep")))
        e7 = pd.read_csv(os.path.join(OUT, f"e7_fwd{Y}_candidates.csv.gz"),
                         usecols=["metric", "score", "pool", "material_id", "sY", "novelY5", "novelF5"])
        for (score, m, pool), d in e7.groupby(["score", "metric", "pool"]):
            flips.append(dict(Y=Y, pool=pool, family="density / NN", score=SCORES[score], metric=m,
                              **summarize(d, "novelY5", "novelF5")))
            for notion in ("comp", "struct@1x", "proto@1x", "proto@0.5x", "proto@2x"):
                lab = c[(c.notion == notion) & (c.pool == pool)].set_index("material_id")
                if not len(lab):
                    continue
                x = d.set_index("material_id").join(lab[["novelY"]], how="inner")
                y = x.novelY.values.astype(int)
                nov = -x.sY.values  # scores are oriented higher = denser = less novel
                ex = x.novelY.values
                agree.append(dict(Y=Y, pool=pool, exact=notion, score=SCORES[score], metric=m, n=len(x),
                                  n_exact_novel=int(ex.sum()), n_score_novel=int(x.novelY5.sum()),
                                  frac_exact_novel_called_novel=x.novelY5[ex].mean() if ex.any() else np.nan,
                                  frac_exact_known_called_novel=x.novelY5[~ex].mean() if (~ex).any() else np.nan,
                                  frac_score_novel_exact_novel=ex[x.novelY5.values].mean()
                                  if x.novelY5.any() else np.nan,
                                  auc_novelty_vs_exact_novel=auc(nov, y),
                                  spearman_novelty_vs_exact_novel=spearmanr(nov, y)[0] if 0 < y.sum() < len(y)
                                  else np.nan))
    pd.concat(allc).to_csv(os.path.join(OUT, "e10_calls.csv.gz"), index=False)
    ft = pd.DataFrame(flips)
    ft.to_csv(os.path.join(OUT, "e10_flip_table.csv"), index=False)
    ag = pd.DataFrame(agree)
    ag.to_csv(os.path.join(OUT, "e10_agreement.csv"), index=False)

    # all GNoME Dec-2023 formulas, composition match; plus reference-choice swap (formula flags)
    gl = pd.read_csv(os.path.join(PROC, "gnome_li.csv"))
    gl = gl[gl.in_gnome_1223].drop_duplicates("fkey")
    r23 = U[U.set == "R23"]
    rows = []
    for Y in YEARS:
        fY = set(r23[r23.first_year <= Y].fkey)
        nY = ~gl.fkey.isin(fY)
        nF = ~gl.fkey.isin(set(r23.fkey))
        rows.append(dict(Y=Y, reference_Y=f"R23 <= {Y}", reference_F="R23 (full)", n=len(gl),
                         novel_rate_Y=nY.mean(), novel_rate_F=nF.mean(), flip_rate=(nY != nF).mean(),
                         n_flip=int((nY != nF).sum())))
    for col, name in (("f_in_mp23_any", "MP-2023 all (incl. theoretical)"), ("f_in_mp25_exp", "MP-2025 experimental"),
                      ("f_in_mp25_any", "MP-2025 all (incl. GNoME imports)")):
        nF = ~gl[col].astype(bool)
        nY = ~gl.fkey.isin(set(r23.fkey))
        rows.append(dict(Y=np.nan, reference_Y="R23 (full)", reference_F=name, n=len(gl), novel_rate_Y=nY.mean(),
                         novel_rate_F=nF.mean(), flip_rate=(nY != nF).mean(), n_flip=int((nY != nF).sum())))
    gt = pd.DataFrame(rows)
    gt.to_csv(os.path.join(OUT, "e10_gnome_all_formulas.csv"), index=False)

    # tolerance alone: R23 leave-one-out novelty vs the rest of R23 (no time split)
    tol = []
    for notion, d in sets.items():
        ids = r23.material_id
        tol.append(dict(notion=notion, n=len(ids),
                        novel_rate_R23_loo=np.mean([not (d.get(m, set()) - {m}) for m in ids])))
    tt = pd.DataFrame(tol)
    tt.to_csv(os.path.join(OUT, "e10_tolerance_r23_loo.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 4):
        print(ft.to_string())
        print(ag[ag.exact.isin(["comp", "struct@1x", "proto@1x"])].to_string())
        print(gt.to_string())
        print(tt.to_string())


# Figure ###############################################################################


def stage_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
    SERIES = {"MINov": "#2a78d6", "1-NN": "#eb6834", "median KDE": "#1baf7a"}
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })
    ft = pd.read_csv(os.path.join(OUT, "e10_flip_table.csv"))
    ft = ft[ft.Y == 2000]
    rows = [(s, m, f"{s}, {'ElMD' if m == 'elmd' else 'LoStOP'}", SERIES[s], "///" if m == "lostop" else None)
            for s in SERIES for m in ("elmd", "lostop")]
    greys = {"comp": "#52514e", "struct": "#8a8984", "proto": "#b5b4ae"}
    names = {"comp": "composition match", "struct": "structure match", "proto": "prototype match"}
    for e in ("comp", "struct@0.5x", "struct@1x", "struct@2x", "proto@0.5x", "proto@1x", "proto@2x"):
        k = e.split("@")[0]
        lab = names[k] + ("" if k == "comp" else f", {e.split('@')[1]} tol")
        rows.append((e, "-", lab, greys[k], None))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    y = np.arange(len(rows))[::-1].astype(float)
    y[6:] -= 0.6  # gap between density / NN scores and exact-match variants
    for ax, (pool, title) in zip(axes, [("a_G", "GNoME candidates (n = 3,000)"),
                                        ("b_R23new", "Later discoveries (n = 753)")]):
        d = ft[ft.pool == pool]
        keep = ft[ft.pool == "b_R23new_keep_siblings"]
        for yy, (s, m, lab, col, hatch) in zip(y, rows):
            v = d[(d.score == s) & (d.metric == m)].flip_rate.iloc[0] * 100
            ax.barh(yy, v, 0.75, color=col, hatch=hatch, edgecolor=SURFACE, linewidth=1.5)
            txt = f"{v:.1f}"
            end = v
            if pool == "b_R23new" and m == "-":
                kv = keep[keep.score == s].flip_rate.iloc[0] * 100
                if abs(kv - v) > 0.05:
                    ax.plot([kv, kv], [yy - 0.38, yy + 0.38], color=INK, lw=2, solid_capstyle="butt", zorder=3)
                    txt = f"{v:.1f}  ({kv:.1f})"
                    end = max(v, kv)
                if v == 0 and s in ("comp", "struct@0.5x", "struct@1x", "struct@2x"):
                    txt += "  by construction"
            elif pool == "a_G" and v == 0:
                txt += "  (no same-composition reference)"
            ax.text(end + 0.25, yy, txt, va="center", fontsize=8, color=INK2)
        ax.set_yticks(y, [r[2] for r in rows])
        ax.set_title(title)
        ax.set_xlim(0, 23)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        ax.set_xlabel("% of candidates whose novelty call flipped")
    handles = [Patch(facecolor="#ffffff", edgecolor=INK2, label="ElMD (composition distance)"),
               Patch(facecolor="#ffffff", edgecolor=INK2, hatch="///", label="LoStOP (structure distance)"),
               plt.Line2D([], [], color=INK, lw=2, label="exact match with later same-composition entries "
                                                          "kept in the future reference (value in brackets)")]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, loc="lower left", ncol=3, bbox_to_anchor=(0.01, 0.0))
    fig.suptitle("Forward test: reference = experimental Li compounds known by 2000; future = full record (2023)",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(os.path.join(OUT, "e10_flip_rates.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["match", "analyze", "fig"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--proto-tols", nargs="+", type=float, default=[1.0])
    ap.add_argument("--budget-min", type=float, default=35)
    args = ap.parse_args()
    U = index()
    if "match" in args.stages:
        stage_match(U, min(args.workers, 4), args.proto_tols, args.budget_min)
    if "analyze" in args.stages:
        stage_analyze(U)
    if "fig" in args.stages:
        stage_fig()


if __name__ == "__main__":
    main()
