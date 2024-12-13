import numpy as np
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed
from tqdm import tqdm

from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from ElM2D import ElM2D
from ElMD import ElMD
import gridrdf

from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint

# LoStOP Distance ######################################################################

import numpy as np
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed
from tqdm import tqdm

from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from ElM2D import ElM2D
from ElMD import ElMD
import gridrdf

from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint

import warnings

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
        delayed(process_structure)(idx, structure, preset=preset, verbose=False)
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


# GRID-RDF Distance ####################################################################


def calculate_grid_representations(
    structures: list,
    maximum_grid_distance=10,
    bin_size=0.1,
    broadening=0.1,
    number_of_shells=100,
):
    """
    Calculate grid representations for a list of structures using RDF analysis.

    Args:
        structures: List of structure objects to analyze
        maximum_grid_distance: Maximum distance for the grid calculation (default: 10)
        bin_size: Size of bins for discretization (default: 0.1)
        broadening: Broadening parameter for RDF calculation (default: 0.1)
        number_of_shells: Number of neighbor shells to consider (default: 100)

    Returns:
        list: List of grid representations for each structure
    """
    # Find all neighbours and cutoffs
    neighbours, cutoffs = gridrdf.extendRDF.find_all_neighbours(
        structures,
        num_neighbours=number_of_shells,
        cutoff=None,
        return_limits=True,
        dryrun=False,
    )

    # Adjust the cutoff to accommodate all nearest neighbours
    max_dist = max(maximum_grid_distance, np.round(cutoffs[1], 1))

    if max_dist != maximum_grid_distance:
        print(
            f"Maximum distance has been updated to {max_dist} to account for {number_of_shells} neighbours"
        )

    # Calculate grid representations for each structure
    grid_representations = []

    for i, struct in enumerate(structures):
        grid_rep = gridrdf.extendRDF.calculate_rdf(
            struct,
            neighbours[i],
            rdf_type="grid",
            max_dist=max_dist,
            bin_width=bin_size,
            smearing=broadening,
            normed=True,
            broadening_method="convolve",
            return_sparse=False,
        )
        grid_representations.append(grid_rep)

    return grid_representations


def get_gridrdf_dm(structures: list):
    """
    Calculate the grid-RDF distance matrix for a list of structures.

    Args:
        structures: List of structure objects

    Returns:
        numpy.ndarray: Distance matrix
    """
    print("\nCalculating GRID-RDF distance matrix...")
    grid_representations = calculate_grid_representations(structures)

    distance_matrix = gridrdf.earth_mover_distance.super_fast_EMD_matrix(
        grid_representations, bin_width=0.1
    )

    return distance_matrix
