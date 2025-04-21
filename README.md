# Materials Novelty Estimation

A parameter-free approach for estimating materials novelty along chemical and structural axes using mutual information-informed density functions. This method computes material density in a way that balances the local and global datastructure to provide a simple means of assessing novelty. The manuscript (`NoveltyMatRevC.pdf`) is still a work and progress and some changes are expected as it moves towards publication.

If you have feedback or ideas for improvement, I would love to hear it!

## Demonstrations on Materials Datasets

Three Jupyter notebooks demonstrate the application of this method:

1. `perovskite_novelty.ipynb`: Shows the method applied to a controlled dataset containing three distinct perovskite cyrstal systems: cubic, tetragonal, and orthorhombic. 

2. `diverse_novelty.ipynb`: Demonstrates the method using a structurally diverse dataset of materials with varying degrees of similarity. Shows how the method distinguishes between different types of novelty.

3. `Li_novelty.ipynb`: Applies the method to analyze some lithium-containing compounds from the GNOME dataset relative to known materials in the Materials Project database.

**NOTE** The GNOME and MP datasets are pulled from v2023.11.1 of the database. As the database, particularly with regards to GNOME offerings, has changed since, we provide all structure files used in the analysis.

## Usage

The core functionality is through the scripts contained in the `MINOV` folder. One can apply the method to an dataframe of materials structures by calling the `compute_MI_novelty` function as below:

```python
from MINOV.novelty import compute_MI_novelty

data, mi_data = compute_MI_novelty(
    data = data, # data is assumed to be a df with columns "formula" and "structure"
    compute_metrics = ['lostop'], # compute the lostop distances
    precomputed_metrics={"elmd": "perovskite_dataset_elmd_dm.npy"}, # pull precomputed elmd distances
    data_dir="precomputed", # folder containing precomputed distance matrices
    data_prefix="perovskite_dataset", # prefix for labeling purposes
)
```

## Citation

```
@article{falkowski2024mutual,
  title={Mutual Information Informed Novelty Estimation of Materials Along Chemical and Structural Axes},
  author={Falkowski, Andrew and Sparks, Taylor},
  year={2024}
}
```
