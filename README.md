# Materials Novelty Estimation

A parameter-free approach for estimating materials novelty along chemical and structural axes using mutual information-informed density functions. This method computes material density in a way that balances the local and global datastructure to provide a simple means of assessing novelty. The manuscript (`NoveltyMatRevC.pdf`) is still a work and progress and some changes are expected as it moves towards publication.

If you have feedback or ideas for improvement, I would love to hear it!

## Usage Examples

Two Jupyter notebooks demonstrate the application of this method:

1. `control_novelty.ipynb`: Demonstrates the method using a control dataset of materials with varying degrees of similarity. Shows how the method distinguishes between different types of novelty.

2. `Li_novelty.ipynb`: Applies the method to analyze lithium-containing compounds from the GNOME dataset relative to known materials in the Materials Project database.

**NOTE** that the MP materials are not provided with this repository, but can be downloaded through the `get_mp_data.py` file.

The core functionality is provided through three main modules:

- `distance.py`: Distance calculations using ElMD and LoStOP metrics
- `mi_density.py`: Mutual information analysis and density calculations
- `utils.py`: Data processing and visualization utilities

## Citation

```
Falkowski, A. R., & Sparks, T. D. (2024). Mutual Information Informed Novelty Estimation of
Materials Along Chemical and Structural Axes.
```
