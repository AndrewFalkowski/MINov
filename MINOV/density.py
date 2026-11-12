# functions for computing the mutual information density of points
import numpy as np
from scipy.interpolate import interp1d


def get_MI_profile(distance_matrix, n_thresholds=200):
    """
    Find optimal distance cutoff using mutual information analysis.

    Args:
        distance_matrix (np.ndarray): Square distance matrix
        n_thresholds: Number of threshold values to test

    Returns:
        float: cutoff distance
        float: MI profile
    """
    distance_matrix = np.asarray(distance_matrix)

    # Generate threshold values
    thresholds = np.linspace(0, np.max(distance_matrix), n_thresholds)
    mi_values = np.zeros(n_thresholds)

    # Each pair i < j is one sample (x, y) with x = [d_ij <= t], y = [d_ji <= t].
    # Count the 2x2 joint table for every threshold at once by sorting the pair
    # distances and counting how many fall at or below each threshold.
    triu_indices = np.triu_indices_from(distance_matrix, k=1)
    upper = distance_matrix[triu_indices]
    lower = distance_matrix[triu_indices[1], triu_indices[0]]
    n_pairs = len(upper)

    def count_leq(values):
        return np.searchsorted(np.sort(values), thresholds, side="right")

    if np.array_equal(upper, lower):  # symmetric: x == y for every pair
        n_x = n_y = n_11 = count_leq(upper)
    else:
        n_x, n_y = count_leq(upper), count_leq(lower)
        n_11 = count_leq(np.maximum(upper, lower))  # NaN propagates, never counted
    n_10 = n_x - n_11
    n_01 = n_y - n_11
    n_00 = n_pairs - n_10 - n_01 - n_11

    for idx in range(n_thresholds):

        # Joint distribution
        joint_dist = np.array(
            [[n_00[idx], n_01[idx]], [n_10[idx], n_11[idx]]], dtype=float
        )
        joint_dist /= n_pairs

        # Calculate marginal distributions
        p_x = joint_dist.sum(axis=1)
        p_y = joint_dist.sum(axis=0)

        # Calculate mutual information
        valid_mask = joint_dist > 0
        mi_terms = np.zeros_like(joint_dist)
        mi_terms[valid_mask] = joint_dist[valid_mask] * np.log2(
            joint_dist[valid_mask] / (p_x[:, None] * p_y[None, :])[valid_mask]
        )
        mi_values[idx] = np.sum(mi_terms)

    # get the index of the maximum MI value
    cutoff_idx = np.argmax(mi_values)

    return thresholds[cutoff_idx], mi_values


def compute_MI_density(distance_matrix, cutoff, mi_profile, thresholds=None):
    """
    Compute mutual information density based on a distance matrix and MI profile.

    Args:
        distance_matrix : (np.ndarray) Matrix of distances between points
        cutoff : (float) Distance cutoff beyond which MI decay becomes zero
        mi_profile : (np.ndarray) Mutual information profile to be used for density calculation
        thresholds : (np.ndarray, optional) Distances at which `mi_profile` was evaluated.
            Defaults to the thresholds `get_MI_profile` uses for `distance_matrix`,
            i.e. np.linspace(0, max(distance_matrix), len(mi_profile)). Pass them
            explicitly when `distance_matrix` is not the matrix the profile came from.

    Returns:
        np.ndarray : Density values for each point in the distance matrix
        np.ndarray : Weight function for the MI profile
    """

    # The weight function is defined on the distances the MI profile was evaluated at
    if thresholds is None:
        thresholds = np.linspace(0, np.max(distance_matrix), len(mi_profile))
    x_interp = np.asarray(thresholds)

    # Compute normalized MI decay profile
    weight_function = _weight_function(x_interp, cutoff, mi_profile)

    # Create forward interpolator (distance -> decay)
    decay_interpolator = interp1d(
        x_interp, weight_function, bounds_error=False, fill_value=0.0
    )

    # Compute densities using vectorized operations
    densities = np.sum(decay_interpolator(distance_matrix), axis=1)

    return densities, weight_function


def _weight_function(thresholds, cutoff, mi_profile):
    """Normalized MI decay: 1 at zero MI, falling to 0 at the cutoff, 0 beyond it."""
    weight_function = 1 - mi_profile / np.max(mi_profile)
    weight_function[thresholds > cutoff] = 0.0
    return weight_function / np.max(weight_function)


def _binary_entropy(p):
    """Binary entropy in bits, with H(0) = H(1) = 0."""
    h = np.zeros_like(p, dtype=float)
    m = (p > 0) & (p < 1)
    h[m] = -(p[m] * np.log2(p[m]) + (1 - p[m]) * np.log2(1 - p[m]))
    return h


class MIProfile:
    """
    MI profile, cutoff and weight function learned from a reference distance matrix.

    Creating an MIProfile computes everything from the reference set; `density` then
    scores any distances against it: the reference materials themselves, a batch of
    new materials, or a single new material.

    Methods:
        "exact" (default): evaluates the MI profile at every distinct reference pair
            distance using the paper's d <= t rule, so there are no thresholds to
            choose. For a symmetric distance matrix the 2x2 joint distribution is
            diagonal and the MI at threshold t reduces to the binary entropy of the
            fraction of pairs with d <= t. Requires a symmetric matrix.
        "grid": the published method. The MI profile is evaluated on `n_thresholds`
            evenly spaced thresholds from 0 to the largest distance and the weight
            function is linearly interpolated between them. Reproduces the results in
            the paper and the demo notebooks.

    Args:
        distance_matrix (np.ndarray): Square distance matrix of the reference materials
        method (str): "exact" or "grid"
        n_thresholds (int): Number of thresholds, used by method="grid" only

    Attributes:
        method (str): Method used
        cutoff (float): Distance at which the MI profile peaks; weights are 0 beyond it
        thresholds (np.ndarray): Distances at which the profile is evaluated
        mi_profile (np.ndarray): MI at each threshold
        weight_function (np.ndarray): Weight at each threshold
    """

    def __init__(self, distance_matrix, method="exact", n_thresholds=200):
        distance_matrix = np.asarray(distance_matrix, dtype=float)
        if len(distance_matrix) < 2:
            raise ValueError("Need a distance matrix of at least two materials.")

        if method == "exact":
            triu_indices = np.triu_indices_from(distance_matrix, k=1)
            pairs = distance_matrix[triu_indices]
            if not np.array_equal(pairs, distance_matrix.T[triu_indices]):
                raise ValueError(
                    'method="exact" requires a symmetric distance matrix; '
                    'use method="grid" for asymmetric matrices.'
                )
            pairs = np.sort(pairs)
            thresholds = np.unique(np.append(0.0, pairs))
            fraction_within = np.searchsorted(pairs, thresholds, side="right") / len(pairs)
            mi_profile = _binary_entropy(fraction_within)
            cutoff = thresholds[np.argmax(mi_profile)]
        elif method == "grid":
            cutoff, mi_profile = get_MI_profile(distance_matrix, n_thresholds=n_thresholds)
            thresholds = np.linspace(0, np.max(distance_matrix), n_thresholds)
        else:
            raise ValueError(f"Unknown method: '{method}' -> Choose 'exact' or 'grid'.")

        self.method = method
        self.cutoff = cutoff
        self.thresholds = thresholds
        self.mi_profile = mi_profile
        self.weight_function = _weight_function(thresholds, cutoff, mi_profile)
        if method == "grid":
            self._interpolator = interp1d(
                thresholds, self.weight_function, bounds_error=False, fill_value=0.0
            )

    def weight(self, distances):
        """Weight of each distance (same shape as `distances`)."""
        distances = np.asarray(distances, dtype=float)
        if self.method == "grid":
            return self._interpolator(distances)
        # step function: each distance takes the weight of the largest threshold <= it
        idx = np.searchsorted(self.thresholds, distances, side="right") - 1
        return np.where(idx >= 0, self.weight_function[np.maximum(idx, 0)], 0.0)

    def density(self, distances, include_self=False):
        """
        MI density: sum of weights over the last axis of `distances`.

        Args:
            distances (np.ndarray): Distances to the reference materials, either
                (n_reference,) for one material or (n_materials, n_reference) for
                several. Passing the reference distance matrix itself gives the
                reference densities; there each material's zero distance to itself
                is included.
            include_self (bool): Add each material's own zero-distance contribution.
                Use this for new materials so their densities are on the same footing
                as the reference densities.

        Returns:
            float or np.ndarray: Density of each material
        """
        density = np.sum(self.weight(distances), axis=-1)
        if include_self:
            density = density + self.weight(0.0)
        return density
