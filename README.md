# Spacetime Model

This repository is inspired from the work of Tamara M. Davis and Charles H. Lineweaver in "Expanding Confusion: common misconceptions of cosmological horizons and the superluminal expansion of the universe". *[see on arXiv](https://arxiv.org/abs/astro-ph/0310808)*

The code of the repository is meant to reproduce the first figures of the paper
describing the expading universe and the different cosmological horizons.

Notions of scale factor, Hubble sphere, particle horizon, event horizon, proper distances,
comoving distances and conformal time are necessary to understand the plots
provided here.

Due to an impletation as close as possible to the general equations,
the computation might take some time (4 minutes at most with an average laptop).
Saving methods are provided to avoid recomputing the same figures over and over again.
Those saving occurs in the `saved` folder.

Plots are always saved to the root directory of the repository (or from
wherever the script is run).

The code is written in Python and uses the following libraries:
- numpy
- matplotlib
- scipy