"""Novelty scores evaluated for query materials against a reference set.

Every function takes
    ref_dm   : (n_ref, n_ref) distances among the reference materials
    query_dm : (n_query, n_ref) distances from each query material to each reference
and returns one score per query on a common orientation: HIGHER = DENSER = LESS NOVEL.
Any parameter (bandwidth, k, cutoff) is fit on the reference set only, so it is refit
whenever the reference set changes.

The KDE and kNN baselines follow `methods_analysis.ipynb` from the paper.
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from MINOV.density import MIProfile


def _pairs(ref_dm):
    return ref_dm[np.triu_indices(len(ref_dm), k=1)]


def paper_k(n_ref):
    """k used for kNN and LOF in the paper: Silverman-style rule, k = (4 / (3n))^(1/5) * n."""
    return max(1, int((4 / (3 * n_ref)) ** (1 / 5) * n_ref))


# MINov ################################################################################


def minov_exact(ref_dm, query_dm):
    return MIProfile(ref_dm).density(query_dm)


def minov_grid(ref_dm, query_dm):
    return MIProfile(ref_dm, method="grid").density(query_dm)


# KDE ##################################################################################


def _silverman_factor(ref_dm):
    """The paper's 'global bandwidth': Silverman factor x std of reference pair distances."""
    pairs = _pairs(ref_dm)
    return (0.75 * len(pairs)) ** (-1 / 5) * np.std(pairs, ddof=1)


def kde_paper(ref_dm, query_dm):
    """
    KDE as implemented in the paper: for each query, a 1D Gaussian KDE of its non-zero
    distances, reflected about 0 and evaluated at 0. The paper passes the 'global
    bandwidth' to scipy's gaussian_kde as `bw_method`, which scipy treats as a factor on
    each row's own standard deviation, so the effective bandwidth differs per query.
    Closed form of that computation (matches scipy to floating-point precision).
    """
    factor = _silverman_factor(ref_dm)
    out = np.empty(len(query_dm))
    for i, row in enumerate(query_dm):
        d = row[row > 0]
        sigma = factor * np.sqrt(2 * np.sum(d**2) / (2 * len(d) - 1))  # std of reflected data
        out[i] = np.mean(np.exp(-0.5 * (d / sigma) ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    return out


def kde_global(ref_dm, query_dm):
    """Gaussian kernel sum with one bandwidth for all queries (the paper's stated intent)."""
    h = _silverman_factor(ref_dm)
    return np.exp(-0.5 * (query_dm / h) ** 2).sum(axis=1)


def kde_median(ref_dm, query_dm):
    """Control: Gaussian kernel sum with bandwidth = median reference pair distance."""
    h = np.median(_pairs(ref_dm))
    return np.exp(-0.5 * (query_dm / h) ** 2).sum(axis=1)


# Nearest neighbours ###################################################################


def knn_paper(ref_dm, query_dm):
    """kNN as in the paper: 1 / mean of the k nearest non-zero distances, k by the paper's rule."""
    k = paper_k(len(ref_dm))
    out = np.empty(len(query_dm))
    for i, row in enumerate(query_dm):
        out[i] = 1.0 / (np.mean(np.sort(row[row > 0])[:k]) + 1e-10)
    return out


def nn_distance(ref_dm, query_dm, k=1):
    """Negative mean distance to the k nearest reference materials (common novelty check)."""
    return -np.mean(np.sort(query_dm, axis=1)[:, :k], axis=1)


def lof(ref_dm, query_dm):
    """Local outlier factor fit on the reference set (novelty mode), k by the paper's rule."""
    k = min(paper_k(len(ref_dm)), len(ref_dm) - 1)
    model = LocalOutlierFactor(n_neighbors=k, metric="precomputed", novelty=True)
    model.fit(ref_dm)
    return model.score_samples(query_dm)


SCORES = {
    "MINov (exact)": minov_exact,
    "MINov (grid)": minov_grid,
    "KDE (paper)": kde_paper,
    "KDE (global bw)": kde_global,
    "KDE (median bw)": kde_median,
    "kNN (paper)": knn_paper,
    "1-NN distance": lambda r, q: nn_distance(r, q, k=1),
    "5-NN distance": lambda r, q: nn_distance(r, q, k=5),
    "LOF": lof,
}
