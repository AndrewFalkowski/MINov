from MINOV.distance import get_lostop_dm, get_elmd_dm
from MINOV.density import MIProfile
import numpy as np
import pandas as pd


def compute_MI_novelty(
    data,
    compute_metrics=None,
    precomputed_metrics=None,
    external_metrics=None,
    n_thresholds=200,
    data_dir=None,
    data_prefix=None,
    method="exact",
):
    """
    Compute novelty/density metrics for structures using MI-based density analysis.

    Args:
        data (pd.DataFrame): DataFrame containing structures and/or formulas
        compute_metrics (list): List of metric names to compute (e.g., ["lostop", "elmd"])
        precomputed_metrics (dict): Dict mapping metric names to file names
            e.g., {"lostop": "lostop_dm.npy"} - will be loaded from data_dir
        external_metrics (dict): Dict mapping custom metric names to distance matrices
            e.g., {"elmd": elmd_dm}
        n_thresholds (int): Number of thresholds for method="grid"
        data_dir (str): Directory for loading/saving distance matrices
        data_prefix (str): Prefix for saving distance matrices
        method (str): "exact" (default) or "grid" (the published method, used by the
            demo notebooks to reproduce the paper). See MIProfile.

    Returns:
        tuple: (DataFrame with density columns, dict of MI data). For each metric the
            MI data holds "tau", "cutoff", "mi_profile", "weight_function" and the
            fitted "profile" (an MIProfile, usable to score new materials).
    """
    # Initialize results
    df = data.copy()
    mi_data = {}
    distance_matrices = {}

    # Process precomputed metrics
    if precomputed_metrics:
        for metric_name, file_name in precomputed_metrics.items():
            if data_dir:
                path = f"{data_dir}/{file_name}"
            else:
                path = file_name

            distance_matrices[metric_name] = np.load(path)

    # Process metrics to compute
    if compute_metrics:
        for metric_name in compute_metrics:
            if metric_name == "lostop":
                dm = get_lostop_dm(data["structure"].values)
            elif metric_name == "elmd":
                dm = get_elmd_dm(data["formula"].values)
            else:
                raise ValueError(
                    f"Unknown metric: '{metric_name}' -> Choose from 'lostop' or 'elmd'."
                )

            distance_matrices[metric_name] = dm

            # Save if requested
            if data_dir:
                np.save(
                    f"{data_dir}/{data_prefix+'_' if data_prefix is not None else ''}{metric_name}_dm.npy",
                    dm,
                )

    # Add custom matrices
    if external_metrics:
        for metric_name, dm in external_metrics.items():
            distance_matrices[metric_name] = dm

    # Compute density for each distance matrix
    for metric_name, dm in distance_matrices.items():
        profile = MIProfile(dm, method=method, n_thresholds=n_thresholds)
        df[f"{metric_name}_density"] = profile.density(dm)
        mi_data[metric_name] = {
            "tau": profile.thresholds,
            "cutoff": profile.cutoff,
            "mi_profile": profile.mi_profile,
            "weight_function": profile.weight_function,
            "profile": profile,
        }
        print(f"MI cutoff for {metric_name}: {profile.cutoff}")

    return df, mi_data
