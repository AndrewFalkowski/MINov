"""Build reference / candidate sets and distance matrices for other chemistries (same recipe as Li).

Chemistries:
  na       non-deprecated MP-2023 entries containing Na
  nitride  ternary nitrides: exactly 3 elements, one of them N, none of O, H, C

For each chemistry (outputs in data/raw/processed_<chem>/, git-ignored):
  R  = experimental (theoretical False) MP-2023 entries, all polymorphs, first_year = earliest year field in the
       MP provenance references (the MP paper, Jain 2013, is always among them, so entries without a literature
       year get 2013; counted in stats). Sampled to R_CAP (seed 0) if larger.
  G  = GNoME Dec-2023 (legacy 12_1_23) entries of the chemistry still present in by_id.zip; sampled to G_CAP
       (seed 0, pool ordered by material_id, as for Li).
  U  = R then G. U_index.csv (row <-> row/column of U_elmd.npy, U_lostop.npy, float32, symmetric),
       R_first_year.csv (all R before sampling), R_structures.json.gz, G_cifs.json.gz, stats.json.

Stages: mp year gnome sets elmd lostop (default: all). `checkli` recomputes first years for the Li R23 entries and
compares them with data/raw/processed/r23_first_year.csv.

Machine limits: at most N_JOBS = 3 worker processes; files are streamed; matrices are float32.

Usage: uv run python experiments/build_chem_sets.py <na|nitride> [stage ...]
"""

import ast
import glob
import gzip
import json
import os
import re
import resource
import sys
import time
import warnings
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO]
RAW = os.path.join(REPO, "data", "raw")
MP23 = os.path.join(RAW, "mp", "collections", "2023-11-01")
GNOME_DIR = os.path.join(RAW, "gnome", "gnome_data")
N_JOBS = 3
R_CAP = 3500
G_CAP = 3000
SEED = 0
CHUNK = 600
NON_NITRIDE = {"O", "H", "C"}

warnings.filterwarnings("ignore")


def read_csv(path, **kw):
    """pd.read_csv that keeps the formula "NaN" (sodium nitride) as a string; only empty cells are missing."""
    kw.setdefault("keep_default_na", False)
    kw.setdefault("na_values", [""])
    return pd.read_csv(path, **kw)


def in_chem(chem, elements):
    els = set(elements)
    if chem == "na":
        return "Na" in els
    if chem == "nitride":
        return len(els) == 3 and "N" in els and not (els & NON_NITRIDE)
    if chem == "li":
        return "Li" in els
    raise ValueError(chem)


def quick_token(chem):
    return {"na": '"Na"', "nitride": '"N"', "li": '"Li"'}[chem]


class Ctx:
    def __init__(self, chem):
        self.chem = chem
        self.out = os.path.join(RAW, f"processed_{chem}")
        os.makedirs(self.out, exist_ok=True)

    def p(self, name):
        return os.path.join(self.out, name)

    def stats(self):
        return json.load(open(self.p("stats.json"))) if os.path.exists(self.p("stats.json")) else {}

    def save_stats(self, upd):
        s = self.stats()
        s.update(upd)
        json.dump(s, open(self.p("stats.json"), "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))


_FKEY = {}


def fkey(formula):
    from pymatgen.core import Composition
    if isinstance(formula, dict):
        formula = "".join(f"{el}{amt:g}" for el, amt in formula.items())
    k = _FKEY.get(formula)
    if k is None:
        c = Composition(formula).reduced_composition
        k = (c.alphabetical_formula, c.reduced_formula, c.anonymized_formula)
        _FKEY[formula] = k
    return k


def peak_mb():
    return dict(peak_rss_self_MB=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
                peak_rss_children_MB=round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20, 1))


# 1. MP summary ##########################################################################

MP_FIELDS = ["material_id", "formula_pretty", "chemsys", "nelements", "nsites", "theoretical", "deprecated",
             "energy_above_hull", "is_stable"]


def _parse_summary(args):
    path, chem = args
    rows, structs, n = [], {}, 0
    tok = quick_token(chem)
    with gzip.open(path, "rt") as f:
        for line in f:
            n += 1
            if tok not in line:
                continue
            r = json.loads(line)
            if not in_chem(chem, r["elements"]):
                continue
            row = {k: r.get(k) for k in MP_FIELDS}
            row["composition_reduced"] = json.dumps(r["composition_reduced"])
            row["icsd_ids"] = ";".join((r.get("database_IDs") or {}).get("icsd", []) or [])
            rows.append(row)
            if not r["deprecated"] and not r["theoretical"]:
                structs[r["material_id"]] = r["structure"]
    return rows, structs, n


def summary_files(chem):
    files = sorted(glob.glob(os.path.join(MP23, "summary", "**", "*.jsonl.gz"), recursive=True))
    if chem == "nitride":
        files = [f for f in files if "nelements=3" in f]
    return files


def stage_mp(c):
    t = time.time()
    files = summary_files(c.chem)
    with Pool(N_JOBS) as pool:
        res = pool.map(_parse_summary, [(f, c.chem) for f in files], chunksize=4)
    df = pd.DataFrame([row for r in res for row in r[0]])
    structs = {k: v for r in res for k, v in r[1].items()}
    n_dep = int(df["deprecated"].sum())
    df = df[~df["deprecated"]].drop(columns="deprecated").drop_duplicates("material_id")
    keys = [fkey(json.loads(x)) for x in df["composition_reduced"]]
    df["fkey"] = [k[0] for k in keys]
    df["reduced_formula"] = [k[1] for k in keys]
    df["anon"] = [k[2] for k in keys]
    df = df.sort_values("material_id").reset_index(drop=True)
    df.to_csv(c.p("mp2023.csv"), index=False)
    exp = df[~df["theoretical"]]
    with gzip.open(c.p("R_structures.json.gz"), "wt") as f:
        json.dump({m: structs[m] for m in exp["material_id"]}, f)
    c.save_stats({"mp": dict(files=len(files), records_scanned=sum(r[2] for r in res), deprecated_dropped=n_dep,
                             entries=len(df), experimental=len(exp), theoretical=int(df["theoretical"].sum()),
                             formulas_experimental=int(exp["fkey"].nunique()), seconds=round(time.time() - t, 1))})
    print(c.stats()["mp"], flush=True)


# 2. First-report years from provenance ##################################################

YEAR_RE = re.compile(r"""year\s*=\s*[{"]?\s*(\d{4})""", re.IGNORECASE)


def _parse_prov(args):
    path, ids = args
    out = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            m = re.search(r'"material_id":\s*"([^"]+)"', line)
            if not m or m.group(1) not in ids:
                continue
            r = json.loads(line)
            years = [int(y) for ref in (r.get("references") or []) for y in YEAR_RE.findall(ref)]
            lit = [int(y) for ref in (r.get("references") or []) if "Jain2013" not in ref
                   for y in YEAR_RE.findall(ref)]
            out[r["material_id"]] = (min(years) if years else np.nan, max(years) if years else np.nan,
                                     len(years), len(lit))
    return out


def first_years(ids):
    files = sorted(glob.glob(os.path.join(MP23, "provenance", "**", "*.jsonl.gz"), recursive=True))
    ids = frozenset(ids)
    with Pool(N_JOBS) as pool:
        res = pool.map(_parse_prov, [(f, ids) for f in files], chunksize=8)
    d = {k: v for r in res for k, v in r.items()}
    return pd.DataFrame([dict(material_id=m, first_year=v[0], last_year=v[1], n_year_refs=v[2],
                              n_literature_years=v[3]) for m, v in d.items()])


def stage_year(c):
    t = time.time()
    df = read_csv(c.p("mp2023.csv"))
    exp = df[~df["theoretical"]]
    fy = first_years(exp["material_id"]).merge(exp[["material_id", "formula_pretty", "fkey"]], on="material_id",
                                               how="right")
    fy.to_csv(c.p("R_first_year.csv"), index=False)
    c.save_stats({"year": dict(n=len(fy), missing=int(fy.first_year.isna().sum()),
                               no_literature_year=int((fy.n_literature_years == 0).sum()),
                               median=float(fy.first_year.median()), min=float(fy.first_year.min()),
                               max=float(fy.first_year.max()),
                               n_le_1990=int((fy.first_year <= 1990).sum()), n_le_2000=int((fy.first_year <= 2000).sum()),
                               n_le_2005=int((fy.first_year <= 2005).sum()), seconds=round(time.time() - t, 1))})
    print(c.stats()["year"], flush=True)


def stage_checkli():
    ref = read_csv(os.path.join(RAW, "processed", "r23_first_year.csv"))
    fy = first_years(ref["material_id"]).merge(ref, on="material_id", suffixes=("", "_li"))
    for col in ["first_year", "last_year", "n_year_refs"]:
        print(col, "agree", int((fy[col] == fy[col + "_li"]).sum()), "of", len(ref), flush=True)


# 3. GNoME ###############################################################################


def stage_gnome(c):
    t = time.time()
    cols = ["MaterialId", "Composition", "Reduced Formula", "Elements", "NSites", "Space Group Number",
            "Decomposition Energy Per Atom"]
    leg = read_csv(os.path.join(GNOME_DIR, "legacy_versions", "12_1_23", "stable_materials_summary.csv"),
                      usecols=cols, dtype={"MaterialId": str})
    leg = leg[leg["Elements"].map(lambda s: in_chem(c.chem, ast.literal_eval(s)))].copy()
    cur_ids = set(read_csv(os.path.join(GNOME_DIR, "stable_materials_summary.csv"), usecols=["MaterialId"],
                              dtype={"MaterialId": str})["MaterialId"])
    with zipfile.ZipFile(os.path.join(GNOME_DIR, "by_id.zip")) as z:
        names = set(z.namelist())
    leg["in_gnome_current"] = leg["MaterialId"].isin(cur_ids)
    leg["has_structure"] = [f"by_id/{m}.CIF" in names for m in leg["MaterialId"]]
    keys = [fkey(x) for x in leg["Composition"]]
    g = pd.DataFrame({"material_id": leg["MaterialId"].values, "composition": leg["Composition"].values,
                      "gnome_reduced_formula": leg["Reduced Formula"].values,
                      "fkey": [k[0] for k in keys], "reduced_formula": [k[1] for k in keys],
                      "anon": [k[2] for k in keys],
                      "nelements": leg["Elements"].map(lambda s: len(ast.literal_eval(s))).values,
                      "nsites": leg["NSites"].values, "spacegroup_number": leg["Space Group Number"].values,
                      "decomposition_energy_per_atom": leg["Decomposition Energy Per Atom"].values,
                      "in_gnome_current": leg["in_gnome_current"].values,
                      "has_structure": leg["has_structure"].values})
    g = g.sort_values("material_id").reset_index(drop=True)
    g.to_csv(c.p("gnome_1223.csv"), index=False)
    pool = g.loc[g["has_structure"], "material_id"].tolist()
    rng = np.random.default_rng(SEED)
    if len(pool) > G_CAP:
        pick = rng.choice(len(pool), G_CAP, replace=False)
        G = [pool[i] for i in sorted(pick)]
    else:
        G = pool
    cifs = {}
    with zipfile.ZipFile(os.path.join(GNOME_DIR, "by_id.zip")) as z:
        for m in G:
            cifs[m] = z.read(f"by_id/{m}.CIF").decode()
    with gzip.open(c.p("G_cifs.json.gz"), "wt") as f:
        json.dump(cifs, f)
    json.dump({"G": G}, open(c.p("G_ids.json"), "w"))
    c.save_stats({"gnome": dict(dec2023_entries=len(g), dec2023_formulas=int(g.fkey.nunique()),
                                with_structure=len(pool), removed_from_current=int((~g.in_gnome_current).sum()),
                                G=len(G), G_sampled=len(pool) > G_CAP, seconds=round(time.time() - t, 1))})
    print(c.stats()["gnome"], flush=True)


# 4. Sets / index ########################################################################


def stage_sets(c):
    mp = read_csv(c.p("mp2023.csv"))
    fy = read_csv(c.p("R_first_year.csv"))
    g = read_csv(c.p("gnome_1223.csv"), dtype={"material_id": str})
    G = json.load(open(c.p("G_ids.json")))["G"]
    exp = mp[~mp["theoretical"]].merge(fy[["material_id", "first_year", "n_literature_years"]], on="material_id")
    exp = exp.sort_values("material_id").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    r_sampled = len(exp) > R_CAP
    if r_sampled:
        exp = exp.iloc[np.sort(rng.choice(len(exp), R_CAP, replace=False))].reset_index(drop=True)
    gi = g.set_index("material_id").loc[G].reset_index()
    rows = pd.concat([
        pd.DataFrame(dict(material_id=exp.material_id, source="mp", set="R", formula=exp.formula_pretty,
                          reduced_formula=exp.reduced_formula, fkey=exp.fkey, anon=exp.anon,
                          nelements=exp.nelements, nsites=exp.nsites, icsd_ids=exp.icsd_ids,
                          energy=exp.energy_above_hull, first_year=exp.first_year,
                          n_literature_years=exp.n_literature_years)),
        pd.DataFrame(dict(material_id=gi.material_id, source="gnome", set="G", formula=gi.gnome_reduced_formula,
                          reduced_formula=gi.reduced_formula, fkey=gi.fkey, anon=gi.anon, nelements=gi.nelements,
                          nsites=gi.nsites, icsd_ids="", energy=gi.decomposition_energy_per_atom,
                          first_year=np.nan, n_literature_years=np.nan)),
    ], ignore_index=True)
    rows["f_in_R"] = rows.fkey.isin(set(mp.loc[~mp.theoretical, "fkey"]))
    rows["f_in_mp_any"] = rows.fkey.isin(set(mp.fkey))
    rows.insert(0, "row", np.arange(len(rows)))
    rows.to_csv(c.p("U_index.csv"), index=False)
    r = rows[rows.set == "R"]
    c.save_stats({"sets": dict(R_full=int((~mp.theoretical).sum()), R=len(r), R_sampled=r_sampled, R_cap=R_CAP,
                               G=int((rows.set == "G").sum()), U=len(rows),
                               R_formulas=int(r.fkey.nunique()),
                               R_le_1990=int((r.first_year <= 1990).sum()), R_le_2000=int((r.first_year <= 2000).sum()),
                               R_le_2005=int((r.first_year <= 2005).sum()),
                               G_formula_in_R=int(rows[rows.set == "G"].f_in_R.sum()),
                               G_formula_in_mp_any=int(rows[rows.set == "G"].f_in_mp_any.sum()))})
    print(c.stats()["sets"], flush=True)


# 5. ElMD ################################################################################


def stage_elmd(c):
    from MINOV.distance import get_elmd_dm
    t = time.time()
    idx = read_csv(c.p("U_index.csv"))
    uniq = sorted(set(idx.reduced_formula))
    dm = get_elmd_dm(uniq).astype(np.float32)
    pos = {f: i for i, f in enumerate(uniq)}
    p = idx.reduced_formula.map(pos).to_numpy()
    np.save(c.p("U_elmd.npy"), dm[np.ix_(p, p)])
    c.save_stats({"elmd": dict(unique_formulas=len(uniq), seconds=round(time.time() - t, 1), **peak_mb())})
    print(c.stats()["elmd"], flush=True)


# 6. LoStOP ##############################################################################


def load_structures(c, ids):
    from pymatgen.core import Structure
    need = set(ids)
    found = {}
    with gzip.open(c.p("R_structures.json.gz"), "rt") as f:
        d = json.load(f)
    for m in need:
        if m in d:
            found[m] = Structure.from_dict(d[m])
    del d
    rest = need - set(found)
    if rest:
        with gzip.open(c.p("G_cifs.json.gz"), "rt") as f:
            cifs = json.load(f)
        for m in rest:
            found[m] = Structure.from_str(cifs[m], fmt="cif")
        del cifs
    for s in found.values():
        s.remove_oxidation_states()  # the oxidation-state fix used for Li (harmless for 2023 dicts and CIFs)
    return [found[m] for m in ids]


def stage_lostop(c):
    from scipy.spatial.distance import pdist, squareform
    from MINOV.distance import get_lostop_features
    t = time.time()
    idx = read_csv(c.p("U_index.csv"), dtype={"material_id": str})
    ids = idx.material_id.tolist()
    cache = c.p("lostop_cache")
    os.makedirs(cache, exist_ok=True)
    feats, failed = {}, set()
    for path in sorted(glob.glob(os.path.join(cache, "*.npz"))):
        z = np.load(path, allow_pickle=True)
        feats.update(zip(z["ids"].tolist(), z["X"]))
        failed.update(z["failed"].tolist())
    todo = [m for m in ids if m not in feats and m not in failed]
    print(f"{len(ids)} ids, {len(todo)} to featurise", flush=True)
    for k in range(0, len(todo), CHUNK):
        chunk = todo[k:k + CHUNK]
        t1 = time.time()
        structs = load_structures(c, chunk)
        order = np.argsort([-len(s) for s in structs], kind="stable")
        chunk = [chunk[i] for i in order]
        structs = [structs[i] for i in order]
        bad = []
        while True:
            keep = [i for i in range(len(chunk)) if i not in set(bad)]
            try:
                X = get_lostop_features([structs[i] for i in keep], n_jobs=N_JOBS)
                break
            except RuntimeError as e:
                local = [int(x) for x in re.search(r"indices \[(.*?)\]", str(e)).group(1).split(",")]
                bad += [keep[i] for i in local]
                print(f"  failures: {[chunk[i] for i in bad]}", flush=True)
        ok_ids = [chunk[i] for i in keep]
        bad_ids = [chunk[i] for i in bad]
        np.savez(os.path.join(cache, f"U_{int(time.time() * 1000)}.npz"), ids=np.array(ok_ids), X=X,
                 failed=np.array(bad_ids, dtype=object))
        feats.update(zip(ok_ids, X))
        failed.update(bad_ids)
        print(f"  chunk {k // CHUNK + 1}/{-(-len(todo) // CHUNK)}: {len(chunk)} structures, "
              f"{sum(len(s) for s in structs)} sites, {time.time() - t1:.0f}s", flush=True)
    ok = np.array([m in feats for m in ids])
    dim = len(next(iter(feats.values())))
    X = np.full((len(ids), dim), np.nan)
    X[ok] = np.stack([feats[m] for m in np.array(ids)[ok]])
    np.save(c.p("U_lostop_features.npy"), X.astype(np.float32))
    D = np.full((len(ids), len(ids)), np.nan, dtype=np.float32)
    D[np.ix_(ok, ok)] = squareform(pdist(X[ok])).astype(np.float32)
    np.save(c.p("U_lostop.npy"), D)
    idx["lostop_ok"] = ok
    idx.to_csv(c.p("U_index.csv"), index=False)
    c.save_stats({"lostop": dict(n_ok=int(ok.sum()), failed=sorted(failed), seconds=round(time.time() - t, 1),
                                 **peak_mb())})
    print(c.stats()["lostop"], flush=True)


STAGES = {"mp": stage_mp, "year": stage_year, "gnome": stage_gnome, "sets": stage_sets, "elmd": stage_elmd,
          "lostop": stage_lostop}

if __name__ == "__main__":
    chem = sys.argv[1]
    if chem == "checkli":
        stage_checkli()
        sys.exit()
    c = Ctx(chem)
    timings = {}
    for name in sys.argv[2:] or list(STAGES):
        t = time.time()
        STAGES[name](c)
        timings[name] = round(time.time() - t, 1)
        c.save_stats({f"timing_{name}": timings[name]})
        print(f"== {chem} stage {name}: {timings[name]}s", flush=True)
