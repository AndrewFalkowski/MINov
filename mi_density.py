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
    # Generate threshold values
    thresholds = np.linspace(
        np.min(distance_matrix), np.max(distance_matrix), n_thresholds
    )
    mi_values = np.zeros(n_thresholds)

    # Get upper triangle indices
    triu_indices = np.triu_indices_from(distance_matrix, k=1)

    for idx, threshold in enumerate(thresholds):

        # Create binary relationship matrix
        relation_matrix = (distance_matrix <= threshold).astype(int)

        # Calculate joint distribution
        pairs = np.stack(
            (
                relation_matrix[triu_indices],
                relation_matrix[triu_indices[1], triu_indices[0]],
            ),
            axis=1,
        )
        joint_dist = np.zeros((2, 2))
        np.add.at(joint_dist, (pairs[:, 0], pairs[:, 1]), 1)

        # Normalize
        joint_dist /= len(pairs)

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


def get_sigma_for_decay(cutoff, target_decay):
    """
    Calculate sigma parameter needed to achieve target gaussian
    decay value at cutoff.
    """
    return cutoff / np.sqrt(-2 * np.log(target_decay))


def compute_mi_density(distance_matrix, cutoff, mi_profile):
    """
    Compute mutual information density based on a distance matrix and MI profile.

    Args:
        distance_matrix : (np.ndarray) Matrix of distances between points
        cutoff : (float) Distance cutoff beyond which MI decay becomes zero
        mi_profile : (np.ndarray) Mutual information profile to be used for density calculation

    Returns:
        np.ndarray : Density values for each point in the distance matrix
    """

    # Create interpolation space
    x_interp = np.linspace(
        np.min(distance_matrix), np.max(distance_matrix), len(mi_profile)
    )

    # Compute normalized MI decay profile
    mi_decay = 1 - mi_profile / np.max(mi_profile)
    mi_decay[x_interp > cutoff] = 0.0

    # Create forward interpolator (distance -> decay)
    decay_interpolator = interp1d(
        x_interp, mi_decay, bounds_error=False, fill_value=0.0
    )

    # Compute densities using vectorized operations
    densities = np.sum(decay_interpolator(distance_matrix), axis=1)

    return densities


def compute_gaussian_density(distance_matrix, cutoff, end_decay=0.1):
    """Calculate density scores using a Gaussian kernel with cutoff.

    Args:
        distance_matrix (ndarray): Distance matrix
        cutoff (float): Maximum distance threshold
        end_decay (float, optional): Cutoff decay value. Defaults to 0.1.

    Returns:
        ndarray: Density values for each point in the distance matrix
    """

    sigma = get_sigma_for_decay(cutoff, end_decay)
    pdf = np.exp(-0.5 * (distance_matrix / sigma) ** 2)
    pdf[distance_matrix > cutoff] = 0
    densities = np.sum(pdf, axis=1)
    return densities