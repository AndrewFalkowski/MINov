# Mutual Information-Informed Novelty Estimation Of Materials Along Chemical And Structural Axes
[![DOI](https://zenodo.org/badge/902081319.svg)](https://doi.org/10.5281/zenodo.15609549) 

A parameter-free method for estimating material novelty is introduced, leveraging mutual information to analyze inter-material distances along chemical and structural axes. This approach derives data-driven weight functions from the mutual information profile, enabling the computation of quantitative novelty scores based on local density without requiring preset cutoffs. The methodology is validated on diverse materials datasets, demonstrating its effectiveness in identifying and differentiating chemical and structural novelty to guide materials discovery.

For detailed methodology, validation results, and theoretical background, see the peer-reviewed paper: "[Mutual Information-Informed Novelty Estimation Of Materials Along Chemical And Structural Axes](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d5dd00167f)" published in Digital Discovery (2025).


## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/AndrewFalkowski/MINov.git
cd MINov

# Create the environment (installs Python 3.10 and all pinned dependencies)
uv sync

# Launch the notebooks
uv run jupyter lab
```

Environments are managed with [uv](https://docs.astral.sh/uv/). To install only the core dependencies (without Jupyter and plotting libraries), use `uv sync --no-default-groups`.

### Dependencies

All required libraries (numpy, pandas, scipy, matminer, pymatgen, ElMD, plus scikit-learn, matplotlib and Jupyter for the notebooks) are pinned in `pyproject.toml` to the versions used for the published results, and fully locked in `uv.lock`.

## Usage

Core functionality is provided through the scripts contained in the `MINOV` folder. Novelty can be copmuted over a dataframe of pymatgen structure objects by calling the `compute_MI_novelty` function as shown in the code snippet below. Variables isolating specific distance metrics, loading precomputed internal or external distance metrics, and specifying saving paths are provided.

```python
from MINOV.novelty import compute_MI_novelty

# Where 'mat_data' is a pandas DataFrame with 'structure' and 'formula' columns
# containing pymatgen structure objects and formula strings, respectively

data, mi_data = compute_MI_novelty(
    data = mat_data, # df of pymatgen structure objects and formulae
    compute_metrics = ['lostop'], # list of distance metrics to compute
    precomputed_metrics={"elmd": "perovskite_dataset_elmd_dm.npy"}, # load precomputed
    data_dir="precomputed", # path to folder with precomputed metrics
    data_prefix="perovskite_dataset", # prefix for labeling purposes
)

# outputs:
# data - a df containing material information and computed densities for each metric
# mi_data - a dictionary with computed MI profile data for each metric 
```

### Scoring new materials against a reference set

`MIProfile` learns the MI profile, cutoff and weight function from a reference distance matrix. Its `density` method then scores any distances against that reference: the reference materials themselves, a batch of new materials, or a single material.

```python
from MINOV.density import MIProfile

profile = MIProfile(reference_dm)   # e.g. distances among known materials
profile.cutoff                      # distance at which the MI profile peaks

profile.density(reference_dm)                         # densities of the reference materials
profile.density(new_to_reference, include_self=True)  # (n_new, n_reference) -> one density per new material
profile.density(one_to_reference, include_self=True)  # (n_reference,) -> a single density
```

`include_self=True` adds each new material's contribution to its own density, so new and reference densities are directly comparable. `compute_MI_novelty` also returns the fitted profile for each metric as `mi_data[metric]["profile"]`.

### Exact and grid methods

By default the MI profile is computed exactly: mutual information is evaluated at every distinct pairwise distance in the reference set (using the paper's `d <= t` rule), so there is no threshold grid to choose. The method used in the paper evaluates the profile on a grid of `n_thresholds` evenly spaced thresholds and interpolates between them. It is available with `method="grid"`:

```python
compute_MI_novelty(..., method="grid", n_thresholds=200)
MIProfile(reference_dm, method="grid", n_thresholds=200)
```

The demo notebooks use `method="grid"` to reproduce the published results. The two methods agree closely: on the datasets in this repository, density rankings have a Spearman correlation of at least 0.999, and densities differ by at most 0.05% on the Li dataset. Differences are largest for small datasets with many tied distances (up to ~7% for ElMD on the perovskite dataset, where only 3% of pairwise distances are distinct).

Further usage examples are available within the jupyter notebooks described below.

## Demonstrations on Materials Datasets

Three Jupyter notebooks demonstrate the application of this method:

1. `perovskite_novelty.ipynb`: Shows the method applied to a controlled dataset containing three distinct perovskite cyrstal systems: cubic, tetragonal, and orthorhombic. The data for this notebook is available in `data/perovskite_dataset`.

2. `diverse_novelty.ipynb`: Demonstrates the method using a structurally diverse dataset of materials with varying degrees of similarity. Shows how the method distinguishes between different types of novelty. The data for this notebook is available in `data/diverse_dataset`.

3. `Li_novelty.ipynb`: Applies the method to analyze some lithium-containing compounds from the GNOME dataset relative to known materials in the Materials Project database. The data for this notebook is available in `data/MP_Li_dataset` and `data/GNOME_Li_dataset`.

**NOTE:** The GNOME and MP datasets are pulled from v2023.11.1 of the database. As the database has since changed, we provide all structure files used in the analysis in the `/data` folder.

## Citing MINov

```
@article{falkowski2025mutual,
  title={Mutual Information Informed Novelty Estimation of Materials Along Chemical and Structural Axes},
  author={Falkowski, Andrew R and Sparks, Taylor D},
  journal={Digital Discovery},
  year={2025},
  publisher={RSC}
}
```
