import numpy as np
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed
from tqdm import tqdm

from matminer.featurizers.site import CrystalNNFingerprint
from ElM2D import ElM2D

# LoStOP Distance ######################################################################


def process_structure(idx, structure, preset="ops", verbose=True):
    """Process a single structure to generate LoStOP fingerprints.

    Args:
        idx (int): Index of the structure being processed
        structure (Structure): Pymatgen structure object
        preset (str): Preset configuration for CrystalNNFingerprint
        verbose (bool): Whether to print processing messages

    Returns:
        tuple: (Fingerprint array for the structure, Feature labels)
    """
    cnnf = CrystalNNFingerprint.from_preset(preset)
    site_feats = []
    site_fails = 0
    n_sites = len(structure)

    for site in range(n_sites):
        try:
            feat = cnnf.featurize(structure, site)
            site_feats.append(feat)
        except Exception as e:
            if verbose:
                print(
                    f"Failed on site {site} of structure {idx} ({structure.formula}): {str(e)}"
                )
            site_fails += 1

    if site_fails > 0 and verbose:
        print(
            f"Failed to featurize {site_fails}/{n_sites} sites for structure {idx} ({structure.formula})"
        )

    if site_feats:
        site_feats_array = np.array(site_feats)
        stats = {
            "mean": np.mean(site_feats_array, axis=0),
            "std_dev": np.std(site_feats_array, axis=0),
            "minimum": np.min(site_feats_array, axis=0),
            "maximum": np.max(site_feats_array, axis=0),
        }
        structure_feat = np.concatenate(list(stats.values()))
        labels = [
            f"{stat}_{label}"
            for stat in stats.keys()
            for label in cnnf.feature_labels()
        ]
        return structure_feat, labels
    else:
        if verbose:
            print(f"No features calculated for structure {idx} ({structure.formula})")
        return None, None


def calculate_fingerprints(structures, preset="ops", n_jobs=-1):
    """
    Calculates structure fingerprints in parallel using joblib.

    Args:
        structures: List of pymatgen Structure objects
        preset: Preset to use for CrystalNNFingerprint
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        tuple: (list of fingerprints, feature labels)
    """
    indices = list(range(len(structures)))

    iterator = tqdm(
        zip(indices, structures),
        total=len(structures),
        desc="Generating LoStOP DM",
    )

    # Process structures in parallel
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_structure)(idx, structure, preset=preset, verbose=True)
        for idx, structure in iterator
    )

    # Separate fingerprints and labels
    structure_feats, labels_list = zip(*results)

    # Filter out None values from failed calculations
    structure_feats = [feat for feat in structure_feats if feat is not None]

    # Use labels from first successful calculation
    labels = next((lab for lab in labels_list if lab is not None), None)

    return structure_feats, labels


def get_lostop_dm(structures: list, precomputed: str = None):
    """
    Calculate LoStOP distance matrix with expanded statistics.

    Args:
        structures: List of pymatgen Structure objects
        precomputed: Path to precomputed features file (optional)

    Returns:
        numpy.ndarray: Distance matrix
    """
    print("\nCalculating LoStOP distance matrix...")

    if precomputed is None:
        structure_feats, labels = calculate_fingerprints(structures)
        structure_feats = np.array(structure_feats)
    else:
        structure_feats = np.loadtxt(precomputed, delimiter=",", skiprows=1)
        labels = None

    try:
        distance_matrix = squareform(pdist(structure_feats, metric="euclidean"))
    except ValueError as e:
        print(f"Error calculating distance matrix: {e}")
        distance_matrix = structure_feats

    return distance_matrix


# ElMD Distance ########################################################################


def get_elmd_dm(formulas: list):
    """
    Calculate ElMD distance matrix for chemical formulas.

    Args:
        formulas: List of chemical formula strings

    Returns:
        numpy.ndarray: Distance matrix
    """
    print("\nCalculating ElMD distance matrix...")
    mapper = ElM2D(verbose=False)
    mapper.fit(formulas)
    return mapper.dm
