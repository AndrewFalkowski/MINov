"""E1: How stable are novelty rankings when the reference set shrinks?

Setup (Li dataset, precomputed LoStOP and ElMD distances):
  * Queries (fixed, never part of the reference):
      - GNoME:     the 44 GNoME Li materials
      - in-family: MP materials whose chemical system keeps >= 1 other member in the pool
      - held-out family: every member of randomly chosen multi-member chemical systems,
        removed from the pool, so their whole family is missing from the reference
  * Reference pool: all remaining MP materials.
  * The reference is reduced to a fraction f of the pool, either
      - uniform: each material kept independently with probability f, or
      - by system: each chemical system kept (with all its members) with probability f,
    and every score is refit on the reduced reference.

Reported per score, mode and fraction (mean and sd over draws):
  * spearman / kendall: rank agreement of query scores with the full-pool scores
  * top5 / top10: overlap of the 5% / 10% most novel queries with the full-pool set
  * auc: how well the score ranks held-out-family queries as more novel than
    in-family queries (informativeness; a constant score would be stable but useless)
plus a split-half check: Spearman between scores from two disjoint halves of the pool.

Usage: uv run python experiments/e1_reference_stability.py [--draws 30] [--seed 0]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
from novelty_scores import SCORES  # noqa: E402
OUT = os.path.join(REPO, "experiments", "results")
FRACTIONS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
N_IN_FAMILY = 150
N_HELD_OUT = 150


def chemical_systems():
    files = [f for f in os.listdir(os.path.join(REPO, "data", "MP_Li_dataset")) if f.endswith(".cif")]
    return np.array(["-".join(sorted(e.symbol for e in Composition(f[:-4]).elements)) for f in files])


def build_split(systems, rng):
    """Choose held-out families and in-family queries; the rest of MP is the reference pool."""
    names, counts = np.unique(systems, return_counts=True)
    multi = rng.permutation(names[counts >= 2])
    held_systems, n = [], 0
    for s in multi:
        if n >= N_HELD_OUT:
            break
        held_systems.append(s)
        n += np.sum(systems == s)
    held_out = np.flatnonzero(np.isin(systems, held_systems))

    # one in-family query per system that keeps >= 1 other member in the pool
    eligible = [s for s in names[counts >= 2] if s not in held_systems]
    in_family = np.array([rng.choice(np.flatnonzero(systems == s))
                          for s in rng.choice(eligible, N_IN_FAMILY, replace=False)])
    pool = np.setdiff1d(np.arange(len(systems)), np.concatenate([held_out, in_family]))
    return pool, in_family, held_out


def topk_overlap(a, b, frac):
    k = max(1, int(round(frac * len(a))))
    return len(set(np.argsort(a)[:k]) & set(np.argsort(b)[:k])) / k


def evaluate(dm, ref, queries, labels, full):
    """Scores for all queries against reference `ref`, compared with full-pool scores."""
    ref_dm = dm[np.ix_(ref, ref)]
    query_dm = dm[np.ix_(queries, ref)]
    rows = {}
    for name, score in SCORES.items():
        s = score(ref_dm, query_dm)
        row = {"auc": roc_auc_score(labels[labels >= 0], -s[labels >= 0])}
        if full is not None:
            row.update(spearman=spearmanr(s, full[name])[0], kendall=kendalltau(s, full[name])[0],
                       top5=topk_overlap(s, full[name], 0.05), top10=topk_overlap(s, full[name], 0.10))
        rows[name] = (s, row)
    return rows


def run_metric(metric, systems, draws, rng):
    dm = np.load(os.path.join(REPO, "precomputed", f"Li_{metric}_dm.npy"))
    n_mp = len(systems)
    pool, in_family, held_out = build_split(systems, rng)
    gnome = np.arange(n_mp, len(dm))
    queries = np.concatenate([gnome, in_family, held_out])
    labels = np.concatenate([np.full(len(gnome), -1), np.zeros(len(in_family)), np.ones(len(held_out))])
    print(f"\n[{metric}] pool {len(pool)}, queries: {len(gnome)} GNoME, {len(in_family)} in-family, "
          f"{len(held_out)} held-out family ({len(np.unique(systems[held_out]))} systems)", flush=True)

    full_eval = evaluate(dm, pool, queries, labels, None)
    full = {name: s for name, (s, _) in full_eval.items()}
    full_auc = {name: r["auc"] for name, (_, r) in full_eval.items()}
    records = [dict(metric=metric, mode="full", fraction=1.0, draw=0, score=n, auc=full_auc[n],
                    spearman=1.0, kendall=1.0, top5=1.0, top10=1.0) for n in SCORES]

    pool_systems = systems[pool]
    unique_systems = np.unique(pool_systems)
    for mode in ["uniform", "system"]:
        for f in FRACTIONS:
            t = time.perf_counter()
            for d in range(draws):
                if mode == "uniform":
                    ref = pool[rng.random(len(pool)) < f]
                else:
                    kept = unique_systems[rng.random(len(unique_systems)) < f]
                    ref = pool[np.isin(pool_systems, kept)]
                if len(ref) < 10:
                    continue
                for name, (_, row) in evaluate(dm, ref, queries, labels, full).items():
                    records.append(dict(metric=metric, mode=mode, fraction=f, draw=d, score=name,
                                        n_ref=len(ref), **row))
            print(f"  {mode:7s} f={f:<4} {time.perf_counter() - t:6.1f}s", flush=True)

    # split-half: two disjoint halves of the pool, no full-pool reference involved
    for d in range(draws):
        perm = rng.permutation(pool)
        a = evaluate(dm, perm[: len(pool) // 2], queries, labels, None)
        b = evaluate(dm, perm[len(pool) // 2:], queries, labels, None)
        for name in SCORES:
            records.append(dict(metric=metric, mode="split-half", fraction=0.5, draw=d, score=name,
                                spearman=spearmanr(a[name][0], b[name][0])[0],
                                top10=topk_overlap(a[name][0], b[name][0], 0.10)))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    systems = chemical_systems()
    records = []
    for metric in ["lostop", "elmd"]:
        records += run_metric(metric, systems, args.draws, rng)

    os.makedirs(OUT, exist_ok=True)
    df = pd.DataFrame(records)
    path = os.path.join(OUT, "e1_reference_stability.csv")
    df.to_csv(path, index=False)
    print(f"\nwrote {path} ({len(df)} rows)")

    summary = (df[df["mode"].isin(["uniform", "system"])]
               .groupby(["metric", "mode", "score", "fraction"])[["spearman", "top10", "auc"]].mean()
               .unstack("fraction"))
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.precision", 3):
        for col in ["spearman", "top10", "auc"]:
            print(f"\n=== {col} (mean over draws) ===")
            print(summary[col])
        print("\n=== full-pool AUC and split-half agreement ===")
        full = df[df["mode"] == "full"].set_index(["metric", "score"])["auc"].rename("AUC (full pool)")
        split = (df[df["mode"] == "split-half"].groupby(["metric", "score"])[["spearman", "top10"]].mean()
                 .add_prefix("split-half "))
        print(pd.concat([full, split], axis=1))


if __name__ == "__main__":
    main()
