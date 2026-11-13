"""Build Li-containing reference / forward / GNoME sets and their distance matrices.

Inputs (downloaded by experiments/fetch_data.py into data/raw/, git-ignored):
  * Materials Project summary releases 2023-11-01 and 2025-09-25 (jsonl.gz)
  * GNoME stable_materials_summary.csv (current and legacy 12_1_23), by_id.zip (CIFs),
    external_data/mp_snapshot_summary.csv (the MP snapshot GNoME compared against)

Outputs go to data/raw/processed/ (see the docstring of `write_readme` / README.txt there).

Stages (all by default; pass names to run a subset, later stages read earlier outputs):
  mp      MP 2023 / 2025 Li tables + structures
  gnome   GNoME Li table (flags) + CIFs for the Li subset
  sets    set definitions, sanity checks, the index table of the union U
  elmd    U x U ElMD matrix (and R23_all ElMD on unique formulas)
  lostop  LoStOP features (cached per material, chunked) and the U x U LoStOP matrix
  lostop_r23all  LoStOP features for all MP 2023 Li entries (theoretical included); no matrix
  match   StructureMatcher of N25 / T25 entries against GNoME entries with the same formula

Usage: uv run python experiments/build_li_sets.py [stage ...]
"""

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
sys.path[:0] = [REPO]
RAW = os.path.join(REPO, "data", "raw")
OUT = os.path.join(RAW, "processed")
MP_DIR = os.path.join(RAW, "mp", "collections")
RELEASES = {"2023": "2023-11-01", "2025": "2025-09-25"}
GNOME_DIR = os.path.join(RAW, "gnome")
SAMPLE_CAP = 3000
SEED = 0
CHUNK = 800

warnings.filterwarnings("ignore")


def out(name):
    return os.path.join(OUT, name)


def save_json(obj, name):
    with open(out(name), "w") as f:
        json.dump(obj, f, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))


def load_json(name):
    with open(out(name)) as f:
        return json.load(f)


def save_gz_json(obj, name):
    with gzip.open(out(name), "wt") as f:
        json.dump(obj, f)


def load_gz_json(name):
    with gzip.open(out(name), "rt") as f:
        return json.load(f)


_FKEY_CACHE = {}


def fkey(formula):
    """Canonical reduced-composition key (alphabetical, explicit amounts), e.g. 'Li1 O2 Co1'."""
    from pymatgen.core import Composition

    if isinstance(formula, dict):
        formula = "".join(f"{el}{amt:g}" for el, amt in formula.items())
    key = _FKEY_CACHE.get(formula)
    if key is None:
        comp = Composition(formula).reduced_composition
        key = (comp.alphabetical_formula, comp.reduced_formula)
        _FKEY_CACHE[formula] = key
    return key


# 1. Materials Project ##################################################################

MP_FIELDS = ["material_id", "formula_pretty", "chemsys", "nelements", "nsites", "theoretical",
             "deprecated", "energy_above_hull", "is_stable"]


def _parse_mp_file(path):
    rows, structures, n_all, n_li_dep = [], {}, 0, 0
    with gzip.open(path, "rt") as f:
        for line in f:
            n_all += 1
            if '"Li"' not in line:
                continue
            r = json.loads(line)
            if "Li" not in r["elements"]:
                continue
            row = {k: r.get(k) for k in MP_FIELDS}
            row["composition_reduced"] = json.dumps(r["composition_reduced"])
            row["icsd_ids"] = ";".join((r.get("database_IDs") or {}).get("icsd", []) or [])
            row["batch_id"] = (r.get("builder_meta") or {}).get("batch_id")
            rows.append(row)
            if not r["deprecated"]:
                structures[r["material_id"]] = r["structure"]
            else:
                n_li_dep += 1
    return rows, structures, n_all


def stage_mp():
    stats = {}
    for rel, folder in RELEASES.items():
        t = time.time()
        files = sorted(glob.glob(os.path.join(MP_DIR, folder, "summary", "**", "*.jsonl.gz"), recursive=True))
        with Pool(8) as pool:
            results = pool.map(_parse_mp_file, files, chunksize=1)
        rows = [row for r in results for row in r[0]]
        structures = {k: v for r in results for k, v in r[1].items()}
        n_all = sum(r[2] for r in results)
        df = pd.DataFrame(rows)
        n_li_dep = int(df["deprecated"].sum())
        dup = df["material_id"].duplicated().sum()
        df = df[~df["deprecated"]].drop(columns="deprecated").drop_duplicates("material_id")
        keys = [fkey(json.loads(c)) for c in df["composition_reduced"]]
        df["fkey"] = [k[0] for k in keys]
        df["reduced_formula"] = [k[1] for k in keys]
        df = df.sort_values("material_id", key=lambda s: s.map(_mp_sort_key)).reset_index(drop=True)
        df.to_csv(out(f"mp{rel}_li.csv"), index=False)
        save_gz_json({m: structures[m] for m in df["material_id"]}, f"mp{rel}_li_structures.json.gz")
        stats[rel] = {
            "files": len(files), "records_all_elements": n_all, "li_deprecated_dropped": n_li_dep,
            "li_duplicate_ids_dropped": int(dup), "li_all": len(df),
            "li_experimental": int((~df["theoretical"]).sum()),
            "li_theoretical": int(df["theoretical"].sum()),
            "li_formulas_all": int(df["fkey"].nunique()),
            "li_formulas_experimental": int(df.loc[~df["theoretical"], "fkey"].nunique()),
            "seconds": round(time.time() - t, 1),
        }
        print(rel, stats[rel], flush=True)
    save_json(stats, "stats_mp.json")


def _mp_sort_key(mid):
    prefix, _, num = mid.rpartition("-")
    return (prefix, int(num) if num.isdigit() else 0)


# 2. GNoME ##############################################################################


def _li_rows(path):
    df = pd.read_csv(path, index_col=0)
    return df, df[df["Elements"].str.contains("'Li'", regex=False)].copy()


def stage_gnome():
    t = time.time()
    _, cur = _li_rows(os.path.join(GNOME_DIR, "gnome_data", "stable_materials_summary.csv"))
    legacy_all, leg = _li_rows(os.path.join(GNOME_DIR, "gnome_data", "legacy_versions", "12_1_23",
                                            "stable_materials_summary.csv"))
    snap = pd.read_csv(os.path.join(GNOME_DIR, "external_data", "mp_snapshot_summary.csv"), index_col=0)
    snap_keys = {fkey(c)[0] for c in snap["Composition"]}

    # Rows: union of current and 12_1_23 Li entries; values from current where present.
    cur["in_gnome_current"] = True
    leg_only = leg[~leg["MaterialId"].astype(str).isin(set(cur["MaterialId"].astype(str)))].copy()
    leg_only["in_gnome_current"] = False
    both = pd.concat([cur, leg_only], ignore_index=True)
    keys = [fkey(c) for c in both["Composition"]]
    g = pd.DataFrame({
        "material_id": both["MaterialId"].astype(str).values,
        "composition": both["Composition"].values,
        "gnome_reduced_formula": both["Reduced Formula"].values,
        "fkey": [k[0] for k in keys], "reduced_formula": [k[1] for k in keys],
        "nelements": both["Elements"].map(lambda s: len(eval(s))).values,
        "nsites": both["NSites"].values, "spacegroup_number": both["Space Group Number"].values,
        "decomposition_energy_per_atom": both["Decomposition Energy Per Atom"].values,
        "decomposition_energy_per_atom_mp": both["Decomposition Energy Per Atom MP"].values,
        "is_train": both["Is Train"].values,
        "in_gnome_current": both["in_gnome_current"].values,
    })
    legacy_ids = set(legacy_all["MaterialId"].astype(str))
    g["in_gnome_1223"] = g["material_id"].isin(legacy_ids)
    mp = {rel: pd.read_csv(out(f"mp{rel}_li.csv")) for rel in RELEASES}
    for rel, df in mp.items():
        g[f"f_in_mp{rel[2:]}_any"] = g["fkey"].isin(set(df["fkey"]))
        g[f"f_in_mp{rel[2:]}_exp"] = g["fkey"].isin(set(df.loc[~df["theoretical"], "fkey"]))
    g["f_in_gnome_mp_snapshot"] = g["fkey"].isin(snap_keys)
    g = g.sort_values("material_id").reset_index(drop=True)
    g.to_csv(out("gnome_li.csv"), index=False)

    cifs = {}
    with zipfile.ZipFile(os.path.join(GNOME_DIR, "gnome_data", "by_id.zip")) as z:
        names = set(z.namelist())
        for mid in g["material_id"]:
            name = f"by_id/{mid}.CIF"
            if name in names:
                cifs[mid] = z.read(name).decode()
    save_gz_json(cifs, "gnome_li_cifs.json.gz")
    g["has_structure"] = g["material_id"].isin(set(cifs))
    g.to_csv(out("gnome_li.csv"), index=False)

    c, l = g["in_gnome_current"], g["in_gnome_1223"]
    stats = {
        "gnome_li_rows_union": len(g), "gnome_current_li": int(c.sum()), "gnome_1223_li": int(l.sum()),
        "current_li_in_1223": int((c & l).sum()),
        "current_li_new_after_1223": int((c & ~l).sum()),
        "1223_li_removed_from_current": int((l & ~c).sum()),
        "1223_li_removed_formula_still_in_current": int(
            (l & ~c & g["fkey"].isin(set(g.loc[c, "fkey"]))).sum()),
        "current_li_formulas": int(g.loc[c, "fkey"].nunique()),
        "1223_li_formulas": int(g.loc[l, "fkey"].nunique()),
        "cifs_extracted": len(cifs), "cifs_missing": len(g) - len(cifs),
        "cifs_missing_current": int((c & ~g["has_structure"]).sum()),
        "seconds": round(time.time() - t, 1),
    }
    for sub, mask in [("1223", l), ("1223_kept", l & c), ("1223_removed", l & ~c), ("new", c & ~l)]:
        s = g[mask]
        for col in ["f_in_mp23_any", "f_in_mp23_exp", "f_in_mp25_any", "f_in_mp25_exp", "f_in_gnome_mp_snapshot"]:
            stats[f"{sub}_{col}"] = int(s[col].sum())
        # appear in MP 2025 but not in MP 2023 (by formula)
        stats[f"{sub}_formula_new_in_mp25_any"] = int((s["f_in_mp25_any"] & ~s["f_in_mp23_any"]).sum())
        stats[f"{sub}_formula_new_in_mp25_exp"] = int((s["f_in_mp25_exp"] & ~s["f_in_mp23_exp"]).sum())
    print(stats, flush=True)
    save_json(stats, "stats_gnome.json")


# 3. Sets, sanity checks, index table ###################################################


def stage_sets():
    mp23 = pd.read_csv(out("mp2023_li.csv"))
    mp25 = pd.read_csv(out("mp2025_li.csv"))
    g = pd.read_csv(out("gnome_li.csv"), dtype={"material_id": str})
    stats = {}

    ids23 = set(mp23["material_id"])
    new25 = mp25[~mp25["material_id"].isin(ids23)].copy()
    f23_any, f23_exp = set(mp23["fkey"]), set(mp23.loc[~mp23["theoretical"], "fkey"])
    for name, s in [("new2025_all", new25), ("new2025_exp", new25[~new25["theoretical"]]),
                    ("new2025_theo", new25[new25["theoretical"]])]:
        stats[name] = {
            "entries": len(s), "formulas": int(s["fkey"].nunique()),
            "entries_formula_not_in_mp23": int((~s["fkey"].isin(f23_any)).sum()),
            "entries_formula_in_mp23_theo_only": int((s["fkey"].isin(f23_any) & ~s["fkey"].isin(f23_exp)).sum()),
            "entries_formula_in_mp23_exp": int(s["fkey"].isin(f23_exp).sum()),
            "entries_formula_not_in_mp23_exp": int((~s["fkey"].isin(f23_exp)).sum()),
        }
    both = mp23.merge(mp25, on="material_id", suffixes=("_23", "_25"))
    switched = both[both["theoretical_23"] & ~both["theoretical_25"]]
    stats["shared_ids"] = len(both)
    stats["switched_theo_to_exp"] = len(switched)
    stats["switched_exp_to_theo"] = int((~both["theoretical_23"] & both["theoretical_25"]).sum())
    stats["switched_theo_to_exp_formula_not_in_mp23_exp"] = int((~switched["fkey_25"].isin(f23_exp)).sum())
    stats["formula_changed_shared_ids"] = int((both["fkey_23"] != both["fkey_25"]).sum())
    stats["ids_2023_absent_or_deprecated_in_2025"] = len(ids23 - set(mp25["material_id"]))
    stats["ids_2023_exp_absent_or_deprecated_in_2025"] = len(
        set(mp23.loc[~mp23["theoretical"], "material_id"]) - set(mp25["material_id"]))
    exp25_ids = set(mp25.loc[~mp25["theoretical"], "material_id"])
    stats["mp25_exp_not_in_R23"] = len(exp25_ids - set(mp23.loc[~mp23["theoretical"], "material_id"]))

    # Sanity checks
    mp_files = sorted(glob.glob(os.path.join(REPO, "data", "MP_Li_dataset", "*.cif")))
    pub = {fkey(os.path.basename(f)[:-4])[0] for f in mp_files}
    stats["sanity_mp_li_dataset_files"] = len(mp_files)
    stats["sanity_mp_li_dataset_unique_fkeys"] = len(pub)
    stats["sanity_mp_li_dataset_in_R23_formulas"] = len(pub & f23_exp)
    stats["sanity_mp_li_dataset_in_mp23_any_formulas"] = len(pub & f23_any)
    stats["sanity_R23_formulas_not_in_dataset"] = len(f23_exp - pub)
    stats["sanity_dataset_formulas_not_in_R23"] = sorted(
        os.path.basename(f)[:-4] for f in mp_files if fkey(os.path.basename(f)[:-4])[0] not in f23_exp)
    gn_files = sorted(glob.glob(os.path.join(REPO, "data", "GNOME_Li_dataset", "*.cif")))
    g1223 = g[g["in_gnome_1223"]]
    gcur = g[g["in_gnome_current"]]
    gk = {fkey(os.path.basename(f)[:-4])[0]: os.path.basename(f)[:-4] for f in gn_files}
    stats["sanity_gnome_li_dataset_files"] = len(gn_files)
    stats["sanity_gnome_li_dataset_in_1223_li"] = sum(k in set(g1223["fkey"]) for k in gk)
    stats["sanity_gnome_li_dataset_in_current_li"] = sum(k in set(gcur["fkey"]) for k in gk)
    stats["sanity_gnome_li_dataset_in_1223_and_current"] = sum(
        k in set(g1223.loc[g1223["in_gnome_current"], "fkey"]) for k in gk)
    stats["sanity_gnome_li_dataset_not_in_1223"] = [v for k, v in gk.items() if k not in set(g1223["fkey"])]
    stats["sanity_gnome_li_dataset_in_1223_removed_from_current"] = [
        v for k, v in gk.items() if k in set(g1223["fkey"]) and k not in set(gcur["fkey"])]
    stats["sanity_gnome_li_dataset_in_current_only"] = [
        v for k, v in gk.items() if k not in set(g1223["fkey"]) and k in set(gcur["fkey"])]
    stats["sanity_gnome_li_dataset_in_mp23_exp"] = [v for k, v in gk.items() if k in f23_exp]
    stats["sanity_gnome_li_dataset_in_mp25_exp"] = [
        v for k, v in gk.items() if k in set(mp25.loc[~mp25["theoretical"], "fkey"])]
    stats["sanity_gnome_li_dataset_ids_1223"] = {
        v: g1223.loc[g1223["fkey"] == k, "material_id"].tolist() for k, v in gk.items()}

    # Sets
    rng = np.random.default_rng(SEED)
    R23 = mp23.loc[~mp23["theoretical"], "material_id"].tolist()
    N25 = new25.loc[~new25["theoretical"], "material_id"].tolist()
    N25_all = new25["material_id"].tolist()
    T25 = switched["material_id"].tolist()
    # G pool: 12_1_23 Li entries that still have a structure (by_id.zip only holds current ids)
    G_full = g.loc[g["in_gnome_1223"] & g["has_structure"], "material_id"].tolist()
    G_removed = g.loc[g["in_gnome_1223"] & ~g["in_gnome_current"], "material_id"].tolist()
    Gn_full = g.loc[~g["in_gnome_1223"] & g["in_gnome_current"], "material_id"].tolist()
    # The paper's 44 GNoME Li candidates, with structures from data/GNOME_Li_dataset
    P44 = [f"paper44:{os.path.basename(f)[:-4]}" for f in gn_files]
    p44_gnome_ids = {f"paper44:{v}": ";".join(g1223.loc[g1223["fkey"] == k, "material_id"])
                     for k, v in gk.items()}

    def sample(ids):
        if len(ids) <= SAMPLE_CAP:
            return list(ids), False
        pick = rng.choice(len(ids), SAMPLE_CAP, replace=False)
        return [ids[i] for i in sorted(pick)], True

    G, g_sampled = sample(G_full)
    G_new, gn_sampled = sample(Gn_full)
    keep_n25_all = len(N25) < 500
    sets = {"R23": R23, "N25": N25, "T25": T25, "G": G, "G_new": G_new, "P44": P44,
            "G_full": G_full, "G_new_full": Gn_full, "G_1223_removed": G_removed, "N25_all": N25_all,
            "R23_all": mp23["material_id"].tolist()}
    save_json({"sets": sets, "G_sampled": g_sampled, "G_new_sampled": gn_sampled, "seed": SEED,
               "cap": SAMPLE_CAP, "N25_all_in_U": keep_n25_all, "P44_gnome_ids_1223": p44_gnome_ids,
               "note": ("G/G_new: rng=np.random.default_rng(0); G drawn first, then G_new; "
                        "rng.choice(len(pool), 3000, replace=False) on pools ordered by material_id. "
                        "G_1223_removed have no structure in by_id.zip. P44 rows use the local CIFs.")},
              "sets.json")
    stats["set_sizes"] = {k: len(v) for k, v in sets.items()}
    stats["G_sampled"], stats["G_new_sampled"], stats["N25_all_in_U"] = g_sampled, gn_sampled, keep_n25_all

    # Index table of U (MP rows first, then GNoME G, G_new, then the paper's 44)
    u_sets = ["R23", "N25", "T25"] + (["N25_all"] if keep_n25_all else []) + ["G", "G_new", "P44"]
    mp_ids = list(dict.fromkeys(R23 + N25 + T25 + (N25_all if keep_n25_all else [])))
    mp23i, mp25i = mp23.set_index("material_id"), mp25.set_index("material_id")
    sR23, sN25, sT25 = set(R23), set(N25), set(T25)
    rows = []
    for m in mp_ids:
        rel = "2023" if m in sR23 else "2025"
        r = (mp23i if rel == "2023" else mp25i).loc[m]
        primary = "R23" if m in sR23 else "N25" if m in sN25 else "T25" if m in sT25 else "N25_theo"
        rows.append({"material_id": m, "source": "mp", "release": rel, "set": primary,
                     "formula": r["formula_pretty"], "reduced_formula": r["reduced_formula"],
                     "fkey": r["fkey"], "nelements": r["nelements"], "nsites": r["nsites"],
                     "theoretical": bool(r["theoretical"]), "icsd_ids": r["icsd_ids"],
                     "mp_batch_id": r["batch_id"], "energy": r["energy_above_hull"],
                     "gnome_id": "", "in_gnome_1223": False, "in_gnome_current": False})
    gi = g.set_index("material_id")
    for m in G + G_new:
        r = gi.loc[m]
        rows.append({"material_id": m, "source": "gnome",
                     "release": "gnome_1223" if r["in_gnome_1223"] else "gnome_current",
                     "set": "G" if r["in_gnome_1223"] else "G_new",
                     "formula": r["gnome_reduced_formula"], "reduced_formula": r["reduced_formula"],
                     "fkey": r["fkey"], "nelements": r["nelements"], "nsites": r["nsites"],
                     "theoretical": True, "icsd_ids": "", "mp_batch_id": "",
                     "energy": r["decomposition_energy_per_atom"], "gnome_id": m,
                     "in_gnome_1223": bool(r["in_gnome_1223"]), "in_gnome_current": bool(r["in_gnome_current"])})
    for m in P44:
        name = m.split(":", 1)[1]
        k, red = fkey(name)
        gid = p44_gnome_ids[m]
        r = gi.loc[gid.split(";")[0]] if gid else None
        from pymatgen.core import Composition
        comp = Composition(name)
        rows.append({"material_id": m, "source": "gnome_paper44", "release": "paper",
                     "set": "P44", "formula": name, "reduced_formula": red, "fkey": k,
                     "nelements": len(comp.elements),
                     "nsites": len(_paper44_structure(name)),
                     "theoretical": True, "icsd_ids": "", "mp_batch_id": "",
                     "energy": r["decomposition_energy_per_atom"] if r is not None else np.nan,
                     "gnome_id": gid, "in_gnome_1223": bool(gid),
                     "in_gnome_current": bool(r is not None and r["in_gnome_current"])})
    idx = pd.DataFrame(rows)
    for s in u_sets:
        idx[f"in_{s}"] = idx["material_id"].isin(set(sets[s]))
    # GNoME rows that are one of the paper's 44 (same GNoME id)
    p44_ids = {i for v in p44_gnome_ids.values() for i in v.split(";") if i}
    idx["is_paper44"] = idx["in_P44"] | idx["material_id"].isin(p44_ids)
    f25_any, f25_exp = set(mp25["fkey"]), set(mp25.loc[~mp25["theoretical"], "fkey"])
    idx["f_in_mp23_any"] = idx["fkey"].isin(f23_any)
    idx["f_in_mp23_exp"] = idx["fkey"].isin(f23_exp)
    idx["f_in_mp25_any"] = idx["fkey"].isin(f25_any)
    idx["f_in_mp25_exp"] = idx["fkey"].isin(f25_exp)
    idx["f_in_gnome_1223"] = idx["fkey"].isin(set(g.loc[g["in_gnome_1223"], "fkey"]))
    idx["f_in_gnome_current"] = idx["fkey"].isin(set(g.loc[g["in_gnome_current"], "fkey"]))
    idx.insert(0, "row", np.arange(len(idx)))
    idx.to_csv(out("U_index.csv"), index=False)
    stats["U_size"] = len(idx)
    stats["U_by_set"] = idx["set"].value_counts().to_dict()

    # Cross-overlaps of interest
    n25 = idx[idx["in_N25"]]
    stats["N25_formula_in_gnome_1223_li"] = int(n25["f_in_gnome_1223"].sum())
    stats["N25_formula_in_gnome_current_li"] = int(n25["f_in_gnome_current"].sum())
    stats["N25_formula_in_gnome_1223_li_and_not_in_mp23_exp"] = int((n25["f_in_gnome_1223"] & ~n25["f_in_mp23_exp"]).sum())
    stats["T25_formula_in_gnome_1223_li"] = int(idx.loc[idx["in_T25"], "f_in_gnome_1223"].sum())
    na = new25
    stats["N25_all_formula_in_gnome_1223_li"] = int(na["fkey"].isin(set(g.loc[g["in_gnome_1223"], "fkey"])).sum())
    stats["N25_all_formula_in_gnome_current_li"] = int(na["fkey"].isin(set(g.loc[g["in_gnome_current"], "fkey"])).sum())
    print(json.dumps({k: v for k, v in stats.items() if not k.startswith("sanity_gnome_li_dataset_ids")},
                     indent=1, default=str)[:6000], flush=True)
    save_json(stats, "stats_sets.json")


# 4. ElMD ###############################################################################


def stage_elmd():
    from MINOV.distance import get_elmd_dm

    t = time.time()
    idx = pd.read_csv(out("U_index.csv"), dtype={"material_id": str})
    # ElMD depends only on the reduced composition: compute on unique formulas, then expand.
    uniq = sorted(set(idx["reduced_formula"]))
    dm_f = get_elmd_dm(uniq).astype(np.float32)
    pos = {f: i for i, f in enumerate(uniq)}
    p = idx["reduced_formula"].map(pos).to_numpy()
    np.save(out("U_elmd.npy"), dm_f[np.ix_(p, p)])
    timings = {"elmd_U_seconds": round(time.time() - t, 1), "elmd_U_unique_formulas": len(uniq)}

    # R23_all (MP 2023 Li, theoretical included) + U on unique formulas
    t = time.time()
    mp23 = pd.read_csv(out("mp2023_li.csv"))
    uniq_all = sorted(set(mp23["reduced_formula"]) | set(uniq))
    np.save(out("formula_elmd.npy"), get_elmd_dm(uniq_all).astype(np.float32))
    pd.DataFrame({"formula_row": np.arange(len(uniq_all)), "reduced_formula": uniq_all}).to_csv(
        out("formula_elmd_index.csv"), index=False)
    timings["elmd_formula_seconds"] = round(time.time() - t, 1)
    timings["elmd_formula_n"] = len(uniq_all)
    print(timings, flush=True)
    save_json(timings, "stats_elmd.json")


# 5. LoStOP #############################################################################


def _paper44_structure(name):
    from pymatgen.core import Structure

    return Structure.from_file(os.path.join(REPO, "data", "GNOME_Li_dataset", f"{name}.cif"))


def load_structures(ids):
    """Pymatgen Structures for MP (2023 then 2025), GNoME and paper44:<formula> ids."""
    from pymatgen.core import Structure

    found = {m: _paper44_structure(m.split(":", 1)[1]) for m in ids if m.startswith("paper44:")}
    need = set(ids) - set(found)
    for name in ["mp2023_li_structures.json.gz", "mp2025_li_structures.json.gz"]:
        d = load_gz_json(name)
        for m in need - set(found):
            if m in d:
                found[m] = Structure.from_dict(d[m])
        del d
    rest = need - set(found)
    if rest:
        cifs = load_gz_json("gnome_li_cifs.json.gz")
        for m in rest:
            found[m] = Structure.from_str(cifs[m], fmt="cif")
    # MP 2025 dicts carry explicit oxidation_state=None (Species with oxi None), which breaks
    # CrystalNN's radius lookup. Strip to plain Elements, as in the 2023 dicts, GNoME CIFs and
    # the published CIFs.
    for s in found.values():
        s.remove_oxidation_states()
    return [found[m] for m in ids]


def featurise(ids, tag):
    """LoStOP features for ids, cached per chunk in lostop_cache/; returns {id: feature or None}."""
    from MINOV.distance import get_lostop_features

    cache_dir = out("lostop_cache")
    os.makedirs(cache_dir, exist_ok=True)
    feats, failed = {}, {}
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        z = np.load(path, allow_pickle=True)
        feats.update(zip(z["ids"].tolist(), z["X"]))
        failed.update({i: "cached failure" for i in z["failed"].tolist()})
    todo = [m for m in ids if m not in feats and m not in failed]
    print(f"[{tag}] {len(ids)} ids, {len(ids) - len(todo)} cached, {len(todo)} to featurise", flush=True)
    for c in range(0, len(todo), CHUNK):
        chunk = todo[c:c + CHUNK]
        t = time.time()
        structs = load_structures(chunk)
        # big structures first for load balance
        order = np.argsort([-len(s) for s in structs], kind="stable")
        chunk = [chunk[i] for i in order]
        structs = [structs[i] for i in order]
        bad = []
        while True:
            keep = [i for i in range(len(chunk)) if i not in set(bad)]
            try:
                X = get_lostop_features([structs[i] for i in keep])
                break
            except RuntimeError as e:
                local = [int(x) for x in re.search(r"indices \[(.*?)\]", str(e)).group(1).split(",")]
                bad += [keep[i] for i in local]
                print(f"  failures: {[chunk[i] for i in bad]}; re-running chunk without them", flush=True)
        ok_ids = [chunk[i] for i in keep]
        bad_ids = [chunk[i] for i in bad]
        np.savez(os.path.join(cache_dir, f"{tag}_{int(time.time() * 1000)}.npz"), ids=np.array(ok_ids),
                 X=X, failed=np.array(bad_ids, dtype=object))
        feats.update(zip(ok_ids, X))
        failed.update({i: "no site featurised" for i in bad_ids})
        n_sites = sum(len(s) for s in structs)
        print(f"  [{tag}] chunk {c // CHUNK + 1}/{-(-len(todo) // CHUNK)}: {len(chunk)} structures, "
              f"{n_sites} sites, {time.time() - t:.0f}s", flush=True)
    return {m: feats.get(m) for m in ids}, {m: failed[m] for m in ids if m in failed}


def stage_lostop():
    from scipy.spatial.distance import pdist, squareform

    t = time.time()
    idx = pd.read_csv(out("U_index.csv"), dtype={"material_id": str})
    ids = idx["material_id"].tolist()
    feats, failed = featurise(ids, "U")
    ok = np.array([feats[m] is not None for m in ids])
    dim = next(f for f in feats.values() if f is not None).shape[0]
    X = np.full((len(ids), dim), np.nan, dtype=np.float64)
    X[ok] = np.stack([feats[m] for m in np.array(ids)[ok]])
    np.save(out("U_lostop_features.npy"), X.astype(np.float32))
    pd.Series(ids, name="material_id").to_csv(out("U_lostop_features_ids.csv"), index=False)
    t_feat = time.time() - t
    D = np.full((len(ids), len(ids)), np.nan, dtype=np.float32)
    D[np.ix_(ok, ok)] = squareform(pdist(X[ok], metric="euclidean")).astype(np.float32)
    np.save(out("U_lostop.npy"), D)
    idx["lostop_ok"] = ok
    idx.to_csv(out("U_index.csv"), index=False)
    stats = {"U_featurise_seconds": round(t_feat, 1), "U_total_seconds": round(time.time() - t, 1),
             "U_failed": failed, "U_n_ok": int(ok.sum()), "feature_dim": dim}

    print({k: v for k, v in stats.items()}, flush=True)
    save_json(stats, "stats_lostop.json")


def stage_lostop_r23all():
    """LoStOP features for every MP 2023 Li entry (theoretical included), rows aligned to
    mp2023_li.csv (NaN rows where featurisation failed). No distance matrix: use cdist."""
    t = time.time()
    ids_all = pd.read_csv(out("mp2023_li.csv"))["material_id"].tolist()
    feats, failed = featurise(ids_all, "R23all")
    dim = next(f for f in feats.values() if f is not None).shape[0]
    X = np.full((len(ids_all), dim), np.nan, dtype=np.float32)
    for k, m in enumerate(ids_all):
        if feats[m] is not None:
            X[k] = feats[m]
    np.save(out("R23_all_lostop_features.npy"), X)
    pd.Series(ids_all, name="material_id").to_csv(out("R23_all_lostop_features_ids.csv"), index=False)
    stats = {"R23_all_seconds": round(time.time() - t, 1), "R23_all_n": len(ids_all),
             "R23_all_failed": failed}
    print(stats, flush=True)
    save_json(stats, "stats_lostop_r23all.json")


# 6. Structure matching N25 / T25 vs GNoME ##############################################


def stage_match():
    from pymatgen.analysis.structure_matcher import StructureMatcher

    t = time.time()
    idx = pd.read_csv(out("U_index.csv"), dtype={"material_id": str})
    mp25 = pd.read_csv(out("mp2025_li.csv"))
    g = pd.read_csv(out("gnome_li.csv"), dtype={"material_id": str})
    new_ids = set(mp25["material_id"]) - set(pd.read_csv(out("mp2023_li.csv"))["material_id"])
    sets = load_json("sets.json")["sets"]
    targets = mp25[mp25["material_id"].isin(new_ids | set(sets["T25"]))]
    pairs = targets.merge(g.loc[g["has_structure"], ["material_id", "fkey", "in_gnome_1223"]], on="fkey",
                          suffixes=("", "_gnome"))
    sm = StructureMatcher()
    s_mp = dict(zip(pairs["material_id"].unique(), load_structures(list(pairs["material_id"].unique()))))
    s_g = dict(zip(pairs["material_id_gnome"].unique(), load_structures(list(pairs["material_id_gnome"].unique()))))
    pairs["structure_match"] = [sm.fit(s_mp[a], s_g[b]) for a, b in zip(pairs["material_id"], pairs["material_id_gnome"])]
    pairs = pairs[["material_id", "material_id_gnome", "fkey", "theoretical", "in_gnome_1223", "structure_match"]]
    pairs.to_csv(out("mp25new_gnome_matches.csv"), index=False)
    m = pairs[pairs["structure_match"]]
    stats = {
        "pairs_same_formula": len(pairs),
        "matched_pairs": len(m),
        "new25_entries_matching_a_gnome_structure_all": int(m["material_id"].nunique()),
        "new25_exp_entries_matching_gnome": int(m.loc[~m["theoretical"] & m["material_id"].isin(set(sets["N25"])), "material_id"].nunique()),
        "T25_entries_matching_gnome": int(m.loc[m["material_id"].isin(set(sets["T25"])), "material_id"].nunique()),
        "gnome_1223_matched_by_new25_any": int(m.loc[m["in_gnome_1223"], "material_id_gnome"].nunique()),
        "gnome_1223_matched_by_new25_exp_or_T25": int(m.loc[m["in_gnome_1223"] & ~m["theoretical"], "material_id_gnome"].nunique()),
        "gnome_new_matched_by_new25_any": int(m.loc[~m["in_gnome_1223"], "material_id_gnome"].nunique()),
        "seconds": round(time.time() - t, 1),
    }
    print(stats, flush=True)
    save_json(stats, "stats_match.json")


STAGES = {"mp": stage_mp, "gnome": stage_gnome, "sets": stage_sets, "elmd": stage_elmd,
          "lostop": stage_lostop, "lostop_r23all": stage_lostop_r23all, "match": stage_match}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    args = sys.argv[1:]
    timings = load_json("timings.json") if os.path.exists(out("timings.json")) else {}
    for name in args or list(STAGES):
        t = time.time()
        STAGES[name]()
        timings[name] = round(time.time() - t, 1)
        save_json(timings, "timings.json")
        print(f"== stage {name}: {timings[name]}s", flush=True)
