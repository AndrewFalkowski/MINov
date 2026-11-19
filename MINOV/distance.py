import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from joblib import Parallel, delayed
from tqdm import tqdm

from matminer.featurizers.site import CrystalNNFingerprint
from pymatgen.core import Composition

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


# PDD Distance #########################################################################

# Pointwise Distance Distributions (Widdowson & Kurlin, "Resolving the data ambiguity
# for periodic crystals", NeurIPS 2022), computed with the `average-minimum-distance`
# package. Each atom gets a row of its sorted distances to its k nearest neighbours in
# the infinite crystal; rows are weighted by atom fraction, and two crystals are
# compared by the earth mover's distance between their rows. The distance is invariant
# to the choice of unit cell, rotation and translation, and continuous under atomic
# displacements. Atoms are treated as identical points, so it ignores chemistry.


def _pdd_periodic_set(structure, unit_density=True):
    import amd

    if unit_density:
        # one atom per unit volume, so lattice scale (a chemical effect) drops out
        structure = structure.copy()
        structure.scale_lattice(len(structure))
    return amd.PeriodicSet(motif=structure.cart_coords, cell=structure.lattice.matrix)


def get_pdd_features(structures, k=100, unit_density=True):
    """
    Calculate the Pointwise Distance Distribution (PDD) of each structure.

    PDDs have one row per distinct atomic environment, so they differ in shape between
    structures. Pickle the list (or save it with np.save(..., allow_pickle=True)) to
    reuse a reference set's PDDs; see `get_pdd_dm(..., reference=...)`.

    Args:
        structures: List of pymatgen Structure objects
        k (int): Number of nearest neighbours per atom
        unit_density (bool): Rescale each structure to one atom per unit volume before
            computing distances, so structures that differ only in lattice scale (for
            example NaCl and KCl) are identical

    Returns:
        list of numpy.ndarray: One PDD per structure; column 0 holds row weights
    """
    import amd

    return [
        amd.PDD(_pdd_periodic_set(structure, unit_density), k)
        for structure in tqdm(structures, desc="Generating PDDs")
    ]


def _as_pdds(items, k, unit_density):
    items = list(items)
    if items and isinstance(items[0], np.ndarray):
        return items
    return get_pdd_features(items, k=k, unit_density=unit_density)


def get_pdd_dm(structures, reference=None, k=100, unit_density=True, n_jobs=None):
    """
    Calculate the chemistry-free PDD earth mover's distance matrix.

    Args:
        structures: List of pymatgen Structure objects, or their PDDs from
            `get_pdd_features`
        reference: Optional reference set, as Structures or PDDs. When given, returns
            distances from each structure to each reference structure.
        k (int): Number of nearest neighbours per atom. Must match any precomputed PDDs.
        unit_density (bool): Rescale to one atom per unit volume (see
            `get_pdd_features`). Must match any precomputed PDDs.
        n_jobs (int): Parallel workers for the pairwise distances (None for serial,
            -1 for all cores)

    Returns:
        numpy.ndarray: (n, n) distance matrix, or (n, n_reference) if `reference` is given
    """
    import amd
    from joblib import cpu_count

    if n_jobs is not None and n_jobs < 0:
        n_jobs = max(1, cpu_count() + 1 + n_jobs)
    # loky, like the LoStOP featurizer: amd's default multiprocessing backend fails
    # in scripts without a __main__ guard on macOS
    parallel = dict(n_jobs=n_jobs, backend="loky")

    print("\nCalculating PDD distance matrix...")
    pdds = _as_pdds(structures, k, unit_density)
    if reference is None:
        return squareform(amd.PDD_pdist(pdds, **parallel))
    return amd.PDD_cdist(pdds, _as_pdds(reference, k, unit_density), **parallel)


# ElMD Distance ########################################################################

# Modified Pettifor scale: elements ordered by chemical similarity.
# Glawe et al., "The optimal one dimensional periodic table: a modified Pettifor
# chemical scale from data mining", New J. Phys. 18, 093011 (2016).
MOD_PETTIFOR = """
He Ne Ar Kr Xe Rn Fr Cs Rb K Na Li Ra Ba Sr Ca Eu Yb Lu Tm Y Er Ho Dy Tb Gd Sm Pm Nd Pr
Ce La Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Sc Zr Hf Ti Ta Nb V Cr Mo W Re Tc Os Ru
Ir Rh Pt Pd Au Ag Cu Ni Co Fe Mn Mg Zn Cd Hg Be Al Ga In Tl Pb Sn Ge Si B C N P As Sb Bi Po
Te Se S O At I Br Cl F H
""".split()
_PETTIFOR_INDEX = {element: i for i, element in enumerate(MOD_PETTIFOR)}
_PETTIFOR_INDEX.update({"D": _PETTIFOR_INDEX["H"], "T": _PETTIFOR_INDEX["H"]})


def get_elmd_dm(formulas: list, reference: list = None):
    """
    Calculate ElMD distance matrix for chemical formulas.

    The Element Movers Distance (Hargreaves et al., Chem. Mater. 32, 10610 (2020))
    on the modified Pettifor scale. On this 1D scale the earth mover's distance
    between two compositions is sum(|cumsum(a - b)|) over their fractional
    compositions ordered along the scale. Both sums use cumsum, so additions are
    strictly sequential and results match the ElMD package exactly.

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
    vectors = np.zeros((len(formulas), len(MOD_PETTIFOR)))
    for row, formula in zip(vectors, formulas):
        composition = Composition(formula)
        total = sum(composition.values(), 0.0)
        for element, amount in composition.items():
            symbol = element.symbol
            if symbol not in _PETTIFOR_INDEX:
                raise ValueError(
                    f"Element {symbol} in '{formula}' is not on the modified Pettifor scale."
                )
            row[_PETTIFOR_INDEX[symbol]] += amount / total
    return vectors


def _elmd_row(diff):
    """ElMD for each row of composition-vector differences."""
    return np.cumsum(np.abs(np.cumsum(diff, axis=1)), axis=1)[:, -1]
