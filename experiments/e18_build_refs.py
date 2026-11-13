"""E18 inputs: the extra reference sets of E15 (Li) built for Na compounds and ternary nitrides.

Mirrors experiments/build_li_sets.py (N25_theo, G_new, R23_all, MP-2025 vs GNoME structure matches) on top of the
sets of experiments/build_chem_sets.py (data/raw/processed_<chem>/: R = experimental MP-2023 entries, G = GNoME
Dec-2023 sample; U_index.csv). Existing files there are only read; every output has an "e18_" prefix (git-ignored).

Sets (chemistry filter = build_chem_sets.in_chem):
  N25_theo  entries of MP-2025 (2025-09-25, non-deprecated) whose material_id is not a non-deprecated MP-2023 entry
            and that are theoretical (largely GNoME imports). As for Li, the few new *experimental* entries (N25) and
            entries switched theoretical -> experimental (T25) are counted but not added.
  G_new     current GNoME entries (stable_materials_summary.csv) not in the Dec-2023 (12_1_23) release, i.e.
            disjoint from the candidates by id; sampled to CAP = 3,000 with np.random.default_rng(18) on the pool
            ordered by material_id (Li drew G_new from the same rng after G; a fresh seed is used here).
  D (all)   all non-deprecated MP-2023 entries incl. theoretical = existing mp2023.csv (ElMD only).
  V         = U rows (R then G, same order as U_index.csv) + N25_theo + G_new; e18_V_index.csv (vrow = row of
            e18_V_lostop_features.npy; urow = U row or -1).

Stages (all by default):
  mp25    parse MP-2025 summary -> e18_mp2025.csv, e18_mp25new_structures.json.gz (new entries only)
  gnome   current-GNoME new entries -> e18_gnome_new.csv, e18_Gnew_cifs.json.gz (sampled ones)
  sets    e18_V_index.csv, e18_formula_index.csv (unique reduced formulas of mp2023.csv and V), stats
  elmd    e18_formula_elmd.npy: formula x formula ElMD, float32, written row by row into a memmap with
          MINOV.distance._elmd_vectors/_elmd_row (identical arithmetic to get_elmd_dm)
  lostop  LoStOP features (get_lostop_features, n_jobs=3, after remove_oxidation_states) for N25_theo + G_new,
          cached in e18_lostop_cache/; R and G features read from the existing lostop_cache/ (float64, the values
          U_lostop.npy was computed from). -> e18_V_lostop_features.npy (float64, NaN rows = failures).
          If more than LOSTOP_MAX new structures, N25_theo and G_new are subsampled (rng 18) to that total.
  match   StructureMatcher (defaults) of every new MP-2025 entry against G candidates of the same formula ->
          e18_mp25new_gnome_matches.csv (as build_li_sets.stage_match; used to exclude imported candidates).

Machine limits: at most 3 worker processes; matrices float32 and written in place.
Usage: uv run python experiments/e18_build_refs.py <na|nitride> [stage ...]
"""

import ast
import glob
import gzip
import json
import os
import re
import sys
import time
import warnings
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
from build_chem_sets import fkey, in_chem, quick_token, read_csv  # noqa: E402

RAW = os.path.join(REPO, "data", "raw")
MP25 = os.path.join(RAW, "mp", "collections", "2025-09-25")
GNOME_DIR = os.path.join(RAW, "gnome", "gnome_data")
N_JOBS = 3
CAP = 3000
SEED = 18
CHUNK = 600
LOSTOP_MAX = 12000  # ~0.15 s / structure with 3 workers -> ~30 min
MP_FIELDS = ["material_id", "formula_pretty", "chemsys", "nelements", "nsites", "theoretical", "deprecated",
             "energy_above_hull", "is_stable"]

warnings.filterwarnings("ignore")


def log(msg, t0=[time.perf_counter()]):  # noqa: B006
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"[{time.perf_counter() - t0[0]:7.1f}s peak {rss:6.0f} MB] {msg}", flush=True)


class Ctx:
    def __init__(self, chem):
        self.chem = chem
        self.dir = os.path.join(RAW, f"processed_{chem}")

    def p(self, name):
        return os.path.join(self.dir, name)

    def stats(self):
        return json.load(open(self.p("e18_stats.json"))) if os.path.exists(self.p("e18_stats.json")) else {}

    def save_stats(self, upd):
        s = self.stats()
        s.update(upd)
        json.dump(s, open(self.p("e18_stats.json"), "w"), indent=1,
                  default=lambda x: x.item() if hasattr(x, "item") else str(x))


# 1. MP-2025 ###########################################################################


def _parse(args):
    path, chem, old_ids = args
    rows, structs = [], {}
    tok = quick_token(chem)
    with gzip.open(path, "rt") as f:
        for line in f:
            if tok not in line:
                continue
            r = json.loads(line)
            if not in_chem(chem, r["elements"]):
                continue
            row = {k: r.get(k) for k in MP_FIELDS}
            row["composition_reduced"] = json.dumps(r["composition_reduced"])
            row["icsd_ids"] = ";".join((r.get("database_IDs") or {}).get("icsd", []) or [])
            rows.append(row)
            if not r["deprecated"] and r["material_id"] not in old_ids:
                structs[r["material_id"]] = r["structure"]
    return rows, structs


def stage_mp25(c):
    t = time.time()
    mp23 = read_csv(c.p("mp2023.csv"))
    old = frozenset(mp23.material_id)
    files = sorted(glob.glob(os.path.join(MP25, "summary", "**", "*.jsonl.gz"), recursive=True))
    with Pool(N_JOBS) as pool:
        res = pool.map(_parse, [(f, c.chem, old) for f in files], chunksize=1)
    df = pd.DataFrame([row for r in res for row in r[0]])
    structs = {k: v for r in res for k, v in r[1].items()}
    n_dep = int(df.deprecated.sum())
    df = df[~df.deprecated].drop(columns="deprecated").drop_duplicates("material_id")
    keys = [fkey(json.loads(x)) for x in df.composition_reduced]
    df["fkey"], df["reduced_formula"], df["anon"] = [k[0] for k in keys], [k[1] for k in keys], [k[2] for k in keys]
    df["new"] = ~df.material_id.isin(old)
    df = df.sort_values("material_id").reset_index(drop=True)
    df.to_csv(c.p("e18_mp2025.csv"), index=False)
    with gzip.open(c.p("e18_mp25new_structures.json.gz"), "wt") as f:
        json.dump({m: structs[m] for m in df.loc[df.new, "material_id"]}, f)
    both = mp23.merge(df, on="material_id", suffixes=("_23", "_25"))
    new = df[df.new]
    f23_exp = set(mp23.loc[~mp23.theoretical, "fkey"])
    c.save_stats({"mp25": dict(files=len(files), deprecated_dropped=n_dep, entries=len(df), new=len(new),
                               new_theoretical=int(new.theoretical.sum()), new_experimental=int((~new.theoretical).sum()),
                               new_formulas=int(new.fkey.nunique()),
                               new_formula_not_in_mp23=int((~new.fkey.isin(set(mp23.fkey))).sum()),
                               new_formula_in_mp23_exp=int(new.fkey.isin(f23_exp).sum()),
                               switched_theo_to_exp=int((both.theoretical_23 & ~both.theoretical_25).sum()),
                               switched_exp_to_theo=int((~both.theoretical_23 & both.theoretical_25).sum()),
                               mp23_absent_or_deprecated_in_25=int(len(old - set(df.material_id))),
                               seconds=round(time.time() - t, 1))})
    log(f"mp25 {c.stats()['mp25']}")


# 2. Current GNoME, new since Dec 2023 ##################################################


def stage_gnome(c):
    t = time.time()
    cur = read_csv(os.path.join(GNOME_DIR, "stable_materials_summary.csv"), dtype={"MaterialId": str},
                   usecols=["MaterialId", "Composition", "Reduced Formula", "Elements", "NSites",
                            "Decomposition Energy Per Atom"])
    cur = cur[cur.Elements.map(lambda s: in_chem(c.chem, ast.literal_eval(s)))]
    leg_ids = set(read_csv(c.p("gnome_1223.csv"), dtype={"material_id": str}).material_id)
    new = cur[~cur.MaterialId.isin(leg_ids)].copy()
    with zipfile.ZipFile(os.path.join(GNOME_DIR, "by_id.zip")) as z:
        names = set(z.namelist())
    new["has_structure"] = [f"by_id/{m}.CIF" in names for m in new.MaterialId]
    keys = [fkey(x) for x in new.Composition]
    g = pd.DataFrame(dict(material_id=new.MaterialId.values, composition=new.Composition.values,
                          gnome_reduced_formula=new["Reduced Formula"].values, fkey=[k[0] for k in keys],
                          reduced_formula=[k[1] for k in keys], anon=[k[2] for k in keys],
                          nsites=new.NSites.values, has_structure=new.has_structure.values))
    g = g.sort_values("material_id").reset_index(drop=True)
    pool = g.loc[g.has_structure, "material_id"].tolist()
    rng = np.random.default_rng(SEED)
    sampled = len(pool) > CAP
    pick = [pool[i] for i in sorted(rng.choice(len(pool), CAP, replace=False))] if sampled else pool
    g["in_G_new"] = g.material_id.isin(set(pick))
    g.to_csv(c.p("e18_gnome_new.csv"), index=False)
    with zipfile.ZipFile(os.path.join(GNOME_DIR, "by_id.zip")) as z:
        cifs = {m: z.read(f"by_id/{m}.CIF").decode() for m in pick}
    with gzip.open(c.p("e18_Gnew_cifs.json.gz"), "wt") as f:
        json.dump(cifs, f)
    U = read_csv(c.p("U_index.csv"), dtype={"material_id": str})
    gf = set(U.loc[U.set == "G", "fkey"])
    c.save_stats({"gnome_new": dict(current_in_chem=len(cur), new_since_1223=len(g), with_structure=len(pool),
                                    G_new=len(pick), sampled=sampled,
                                    G_new_formula_shared_with_G=int(g[g.in_G_new].fkey.isin(gf).sum()),
                                    seconds=round(time.time() - t, 1))})
    log(f"gnome_new {c.stats()['gnome_new']}")


# 3. Sets ##############################################################################


def stage_sets(c):
    U = read_csv(c.p("U_index.csv"), dtype={"material_id": str})
    mp25 = read_csv(c.p("e18_mp2025.csv"))
    gn = read_csv(c.p("e18_gnome_new.csv"), dtype={"material_id": str})
    mp23 = read_csv(c.p("mp2023.csv"))
    n25 = mp25[mp25.new & mp25.theoretical]
    gnew = gn[gn.in_G_new]
    V = pd.concat([
        pd.DataFrame(dict(urow=U.row.values, material_id=U.material_id.values, set=U.set.values,
                          reduced_formula=U.reduced_formula.values, fkey=U.fkey.values,
                          gnome_id=np.where(U.set == "G", U.material_id, ""))),
        pd.DataFrame(dict(urow=-1, material_id=n25.material_id.values, set="N25_theo",
                          reduced_formula=n25.reduced_formula.values, fkey=n25.fkey.values, gnome_id="")),
        pd.DataFrame(dict(urow=-1, material_id=gnew.material_id.values, set="G_new",
                          reduced_formula=gnew.reduced_formula.values, fkey=gnew.fkey.values,
                          gnome_id=gnew.material_id.values)),
    ], ignore_index=True)
    assert V.material_id.is_unique
    V.insert(0, "vrow", np.arange(len(V)))
    V.to_csv(c.p("e18_V_index.csv"), index=False)
    forms = sorted(set(mp23.reduced_formula) | set(V.reduced_formula))
    pd.DataFrame(dict(formula_row=np.arange(len(forms)), reduced_formula=forms)).to_csv(
        c.p("e18_formula_index.csv"), index=False)
    c.save_stats({"sets": dict(V=len(V), **{f"V_{k}": int(v) for k, v in V.set.value_counts().items()},
                               D_entries=len(mp23), D_formulas=int(mp23.reduced_formula.nunique()),
                               formulas=len(forms))})
    log(f"sets {c.stats()['sets']}")


# 4. ElMD ##############################################################################


def stage_elmd(c):
    from MINOV.distance import _elmd_row, _elmd_vectors
    t = time.time()
    forms = read_csv(c.p("e18_formula_index.csv")).reduced_formula.tolist()
    vec = _elmd_vectors(forms)
    n = len(forms)
    out = np.lib.format.open_memmap(c.p("e18_formula_elmd.npy"), mode="w+", dtype=np.float32, shape=(n, n))
    for i in range(n):
        out[i] = _elmd_row(vec[i] - vec).astype(np.float32)
        if i % 2000 == 0:
            log(f"elmd row {i}/{n}")
    out.flush()
    del out
    c.save_stats({"elmd": dict(formulas=n, seconds=round(time.time() - t, 1))})
    log(f"elmd {c.stats()['elmd']}")


# 5. LoStOP ############################################################################


def load_new_structures(c, ids):
    from pymatgen.core import Structure
    need = set(ids)
    found = {}
    with gzip.open(c.p("e18_mp25new_structures.json.gz"), "rt") as f:
        d = json.load(f)
    for m in need & set(d):
        found[m] = Structure.from_dict(d[m])
    del d
    rest = need - set(found)
    if rest:
        with gzip.open(c.p("e18_Gnew_cifs.json.gz"), "rt") as f:
            cifs = json.load(f)
        for m in rest:
            found[m] = Structure.from_str(cifs[m], fmt="cif")
        del cifs
    for s in found.values():
        s.remove_oxidation_states()  # MP-2025 dicts carry oxidation_state=None (see build_li_sets)
    return [found[m] for m in ids]


def read_cache(path_glob):
    feats, failed = {}, set()
    for path in sorted(glob.glob(path_glob)):
        z = np.load(path, allow_pickle=True)
        feats.update(zip(z["ids"].tolist(), z["X"]))
        failed.update(z["failed"].tolist())
    return feats, failed


def stage_lostop(c):
    from MINOV.distance import get_lostop_features
    t = time.time()
    V = read_csv(c.p("e18_V_index.csv"), dtype={"material_id": str, "gnome_id": str})
    new_ids = V.loc[V.urow < 0, "material_id"].tolist()
    sub = False
    if len(new_ids) > LOSTOP_MAX:
        rng = np.random.default_rng(SEED)
        new_ids = [new_ids[i] for i in sorted(rng.choice(len(new_ids), LOSTOP_MAX, replace=False))]
        sub = True
    cache = c.p("e18_lostop_cache")
    os.makedirs(cache, exist_ok=True)
    feats, failed = read_cache(os.path.join(cache, "*.npz"))
    todo = [m for m in new_ids if m not in feats and m not in failed]
    log(f"lostop: {len(new_ids)} new ids, {len(todo)} to featurise")
    for k in range(0, len(todo), CHUNK):
        chunk = todo[k:k + CHUNK]
        t1 = time.time()
        structs = load_new_structures(c, chunk)
        order = np.argsort([-len(s) for s in structs], kind="stable")
        chunk, structs = [chunk[i] for i in order], [structs[i] for i in order]
        bad = []
        while True:
            keep = [i for i in range(len(chunk)) if i not in set(bad)]
            try:
                X = get_lostop_features([structs[i] for i in keep], n_jobs=N_JOBS)
                break
            except RuntimeError as e:
                local = [int(x) for x in re.search(r"indices \[(.*?)\]", str(e)).group(1).split(",")]
                bad += [keep[i] for i in local]
                log(f"  failures: {[chunk[i] for i in bad]}")
        ok_ids, bad_ids = [chunk[i] for i in keep], [chunk[i] for i in bad]
        np.savez(os.path.join(cache, f"V_{int(time.time() * 1000)}.npz"), ids=np.array(ok_ids), X=X,
                 failed=np.array(bad_ids, dtype=object))
        feats.update(zip(ok_ids, X))
        failed.update(bad_ids)
        log(f"  chunk {k // CHUNK + 1}/{-(-len(todo) // CHUNK)}: {len(chunk)} structures, "
            f"{sum(len(s) for s in structs)} sites, {time.time() - t1:.0f}s")
    old, _ = read_cache(c.p(os.path.join("lostop_cache", "*.npz")))
    feats.update({m: old[m] for m in V.loc[V.urow >= 0, "material_id"]})
    dim = len(next(iter(feats.values())))
    X = np.full((len(V), dim), np.nan)
    ok = np.array([m in feats for m in V.material_id])
    X[ok] = np.stack([feats[m] for m in V.material_id[ok]])
    np.save(c.p("e18_V_lostop_features.npy"), X)
    # agreement with the stored U matrix (R/G rows)
    from scipy.spatial.distance import cdist
    U_l = np.load(c.p("U_lostop.npy"), mmap_mode="r")
    u = V.urow.values[V.urow.values >= 0]
    rng = np.random.default_rng(0)
    a = np.sort(rng.choice(len(u), min(500, len(u)), replace=False))
    diff = float(np.max(np.abs(cdist(X[a], X[:len(u)]).astype(np.float32) - U_l[u[a]][:, u])))
    c.save_stats({"lostop": dict(new_requested=int((V.urow < 0).sum()), new_featurised=len(new_ids),
                                 subsampled=sub, n_failed=int(len(set(new_ids) & failed)),
                                 failed=sorted(set(new_ids) & failed), n_ok=int(ok.sum()),
                                 max_abs_diff_vs_U_lostop=diff, seconds=round(time.time() - t, 1))})
    log(f"lostop {c.stats()['lostop']}")


# 6. MP-2025 new entries vs G candidates ###############################################


def _fit(args):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    a, b, da, cb = args
    sa = Structure.from_dict(da)
    sa.remove_oxidation_states()
    sb = Structure.from_str(cb, fmt="cif")
    try:
        return a, b, bool(StructureMatcher().fit(sa, sb))
    except Exception:
        return a, b, False


def stage_match(c):
    t = time.time()
    mp25 = read_csv(c.p("e18_mp2025.csv"))
    U = read_csv(c.p("U_index.csv"), dtype={"material_id": str})
    G = U[U.set == "G"]
    new = mp25[mp25.new]
    pairs = new[["material_id", "fkey", "theoretical"]].merge(
        G[["material_id", "fkey"]].rename(columns={"material_id": "material_id_gnome"}), on="fkey")
    with gzip.open(c.p("e18_mp25new_structures.json.gz"), "rt") as f:
        d = json.load(f)
    with gzip.open(c.p("G_cifs.json.gz"), "rt") as f:
        cifs = json.load(f)
    jobs = [(a, b, d[a], cifs[b]) for a, b in zip(pairs.material_id, pairs.material_id_gnome)]
    del d, cifs
    with Pool(N_JOBS) as pool:
        res = pool.map(_fit, jobs, chunksize=10)
    m = dict(((a, b), r) for a, b, r in res)
    pairs["structure_match"] = [m[(a, b)] for a, b in zip(pairs.material_id, pairs.material_id_gnome)]
    pairs.to_csv(c.p("e18_mp25new_gnome_matches.csv"), index=False)
    s = pairs[pairs.structure_match]
    c.save_stats({"match": dict(pairs_same_formula=len(pairs), matched_pairs=len(s),
                                G_with_same_formula_new25=int(pairs.material_id_gnome.nunique()),
                                G_matched=int(s.material_id_gnome.nunique()),
                                G_matched_by_theoretical=int(s[s.theoretical].material_id_gnome.nunique()),
                                seconds=round(time.time() - t, 1))})
    log(f"match {c.stats()['match']}")


STAGES = {"mp25": stage_mp25, "gnome": stage_gnome, "sets": stage_sets, "elmd": stage_elmd, "lostop": stage_lostop,
          "match": stage_match}

if __name__ == "__main__":
    c = Ctx(sys.argv[1])
    for name in sys.argv[2:] or list(STAGES):
        t = time.time()
        STAGES[name](c)
        c.save_stats({f"timing_{name}": round(time.time() - t, 1)})
        log(f"== {c.chem} stage {name} done")
