# Mutual Information-Informed Novelty Estimation Of Materials Along Chemical And Structural Axes

A parameter-free method for estimating material novelty is introduced, leveraging mutual information to analyze inter-material distances along chemical and structural axes. This approach derives data-driven weight functions from the mutual information profile, enabling the computation of quantitative novelty scores based on local density without requiring preset cutoffs. The methodology is validated on diverse materials datasets, demonstrating its effectiveness in identifying and differentiating chemical and structural novelty to guide materials discovery. The manuscript is currently under review and subject to change.

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/AndrewFalkowski/MINov.git
cd MINOV

# Create and activate conda environment
conda env create -f environment.yml
conda activate MINOV
```
### Dependencies
All required libraries (numpy, pandas, scipy, scikit-learn, matplotlib, matminer, pymatgen) and their versions are specified in `environment.yml`.

## Demonstrations on Materials Datasets

Three Jupyter notebooks demonstrate the application of this method:

1. `perovskite_novelty.ipynb`: Shows the method applied to a controlled dataset containing three distinct perovskite cyrstal systems: cubic, tetragonal, and orthorhombic. 

2. `diverse_novelty.ipynb`: Demonstrates the method using a structurally diverse dataset of materials with varying degrees of similarity. Shows how the method distinguishes between different types of novelty.

3. `Li_novelty.ipynb`: Applies the method to analyze some lithium-containing compounds from the GNOME dataset relative to known materials in the Materials Project database.

**NOTE:** The GNOME and MP datasets are pulled from v2023.11.1 of the database. As the database, particularly with regards to GNOME offerings, has changed since, we provide all structure files used in the analysis.

## Usage

The core functionality is through the scripts contained in the `MINOV` folder. You can apply the method to an dataframe of materials structures by calling the `compute_MI_novelty` function as below:

```python
from MINOV.novelty import compute_MI_novelty

data, mi_data = compute_MI_novelty(
    data = data, # df of pymatgen structure objects
    compute_metrics = ['lostop'], # list of distance metrics to compute
    precomputed_metrics={"elmd": "perovskite_dataset_elmd_dm.npy"}, # precomputed data
    data_dir="precomputed", # path to folder with precomputed metrics
    data_prefix="perovskite_dataset", # prefix for labeling purposes
)
```

Further usage examples are available within the example jupyter notebooks mention in the preceeding section. 

## Citation

```
@article{falkowski2024mutual,
  title={Mutual Information Informed Novelty Estimation of Materials Along Chemical and Structural Axes},
  author={Falkowski, Andrew and Sparks, Taylor},
  year={2024}
}
```
