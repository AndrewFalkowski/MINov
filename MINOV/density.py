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


def compute_MI_density(distance_matrix, cutoff, mi_profile):
    """
    Compute mutual information density based on a distance matrix and MI profile.

    Args:
        distance_matrix : (np.ndarray) Matrix of distances between points
        cutoff : (float) Distance cutoff beyond which MI decay becomes zero
        mi_profile : (np.ndarray) Mutual information profile to be used for density calculation

    Returns:
        np.ndarray : Density values for each point in the distance matrix
        np.ndarray : Weight function for the MI profile
    """

    # Create interpolation space
    x_interp = np.linspace(
        np.min(distance_matrix), np.max(distance_matrix), len(mi_profile)
    )

    # Compute normalized MI decay profile
    weight_function = 1 - mi_profile / np.max(mi_profile)
    weight_function[x_interp > cutoff] = 0.0
    weight_function = weight_function / np.max(weight_function)

    # Create forward interpolator (distance -> decay)
    decay_interpolator = interp1d(
        x_interp, weight_function, bounds_error=False, fill_value=0.0
    )

    # Compute densities using vectorized operations
    densities = np.sum(decay_interpolator(distance_matrix), axis=1)

    return densities, weight_function
