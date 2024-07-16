# pyem
A collection of Python modules and command-line utilities for electron microscopy of biological samples.

Documentation for the programs can be found in their usage text, comments in code, and in the Wiki of this repository.

The entire collection is licensed under the terms of the GNU Public License, version 3 (GPLv3).

# How to cite

Please cite the pyem DOI: [10.5281/zenodo.3576630](https://doi.org/10.5281/zenodo.3576630).

For example, the [formatting for a Nature journal](https://www.nature.com/nature/for-authors/formatting-guide) is:

Asarnow, D., Palovcak, E., Cheng, Y. UCSF pyem v0.5. Zenodo https://doi.org/10.5281/zenodo.3576630 (2019)

# Installation

Install pyem to any conda environment (Python >= 3.9) from conda-forge:

```
conda install -c conda-forge pyem
```

I recommend using [miniforge](https://github.com/conda-forge/miniforge) (and mamba), in which case the channel argument should be dropped:

```
mamba install pyem
```

For development pyem can be installed with an egg-link and adding pyem.cli programs to the $PATH:

```
mamba create -n pyem python=3.11
mamba activate pyem
mamba install numba numpy scipy matplotlib seaborn pandas pathos pyfftw healpy natsort starfile ipython
git clone https://github.com/asarnow/pyem.git
cd pyem
pip install --no-dependencies -e .
export PATH=$(realpath pyem/cli):$PATH
```

# Exporting from cryoSPARC to Relion

The most popular feature of pyem is exporting particle metadata from cryoSPARC.
Detailed [instructions](https://github.com/asarnow/pyem/wiki/Export-from-cryoSPARC-v2) can be found in the project wiki.

The TL;DR for using Relion Class3D is that you will mirror the particle stacks in your Relion project, and use
`csparc2star.py Px/Jy/last_iter_particles.cs Px/Jy/passthrouh_particles.cs Px_Jy_particles.star --inverty`
to create a particle data file.

(C) 2016-2024 Daniel Asarnow
