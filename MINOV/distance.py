import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from joblib import Parallel, delayed
from tqdm import tqdm

from matminer.featurizers.site import CrystalNNFingerprint
from ElMD import ElMD

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


def get_lostop_features(structures, preset="ops", n_jobs=-1):
    """
    Calculate LoStOP fingerprints (statistics of CrystalNN site fingerprints) in parallel.

    Save the result (e.g. with np.save) to reuse a reference set's fingerprints
    without featurizing it again; see `get_lostop_dm(..., reference=...)`.

    Args:
        structures: List of pymatgen Structure objects
        preset: Preset to use for CrystalNNFingerprint
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        numpy.ndarray: Fingerprints, one row per structure

    Raises:
        RuntimeError: If no site of a structure could be featurized. Such structures
            would otherwise silently drop out and misalign the rows of the result.
    """
    results = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
        delayed(process_structure)(idx, structure, preset=preset, verbose=True)
        for idx, structure in enumerate(structures)
    )
    results = list(
        tqdm(results, total=len(structures), desc="Generating LoStOP fingerprints")
    )

    failed = [idx for idx, (feat, _) in enumerate(results) if feat is None]
    if failed:
        raise RuntimeError(
            f"LoStOP featurization failed for {len(failed)} structure(s) at indices "
            f"{failed}. Remove them from the input so rows stay aligned with your data."
        )
    return np.array([feat for feat, _ in results])


def get_lostop_dm(structures: list, reference=None, precomputed: str = None):
    """
    Calculate LoStOP distance matrix with expanded statistics.

    Args:
        structures: List of pymatgen Structure objects
        reference: Optional reference set, either a list of pymatgen Structure
            objects or their fingerprints from `get_lostop_features`. When given,
            returns distances from each structure to each reference structure.
        precomputed: Path to precomputed features file for `structures` (optional)

    Returns:
        numpy.ndarray: (n, n) distance matrix, or (n, n_reference) if `reference` is given
    """
    print("\nCalculating LoStOP distance matrix...")

    if precomputed is None:
        features = get_lostop_features(structures)
    else:
        features = np.loadtxt(precomputed, delimiter=",", skiprows=1)

    if reference is None:
        return squareform(pdist(features, metric="euclidean"))

    if not isinstance(reference, np.ndarray):
        reference = get_lostop_features(reference)
    return cdist(features, reference, metric="euclidean")


# ElMD Distance ########################################################################


def get_elmd_dm(formulas: list, reference: list = None):
    """
    Calculate ElMD distance matrix for chemical formulas.

    On the modified Pettifor scale, ElMD between two compositions is
    sum(|cumsum(a - b)|) over their Pettifor-ordered fractional composition
    vectors. Each row of pairs is evaluated with numpy; cumsum is used for both
    sums so additions happen in the same sequential order as ElMD's own kernel.

    Args:
        formulas: List of chemical formula strings
        reference: Optional list of reference formula strings. When given, returns
            distances from each formula to each reference formula.

    Returns:
        numpy.ndarray: (n, n) distance matrix, or (n, n_reference) if `reference` is given
    """
    print("\nCalculating ElMD distance matrix...")
    vectors = _elmd_vectors(formulas)

    if reference is not None:
        reference_vectors = _elmd_vectors(reference)
        distance_matrix = np.zeros((len(vectors), len(reference_vectors)))
        for i, vector in enumerate(vectors):
            distance_matrix[i] = _elmd_row(vector - reference_vectors)
        return distance_matrix

    n = len(vectors)
    distance_matrix = np.zeros((n, n))
    for i in range(n - 1):
        row = _elmd_row(vectors[i] - vectors[i + 1 :])
        distance_matrix[i, i + 1 :] = row
        distance_matrix[i + 1 :, i] = row
    return distance_matrix


def _elmd_vectors(formulas):
    """Fractional compositions ordered along the modified Pettifor scale."""
    return np.array(
        [ElMD(formula, metric="mod_petti").feature_vector for formula in formulas],
        dtype=np.float64,
    )


def _elmd_row(diff):
    """ElMD for each row of composition-vector differences."""
    return np.cumsum(np.abs(np.cumsum(diff, axis=1)), axis=1)[:, -1]
