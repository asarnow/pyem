# UCSF PyEM
UCSF PyEM is a collection of Python modules and command-line utilities for electron microscopy of biological samples.

The entire collection is licensed under the terms of the GNU Public License, version 3 (GPLv3).

Copyright information is listed within each individual file. Current copyright holders include:
 * Eugene Palovcak (UCSF)
 * Daniel Asarnow (UCSF)

Documentation for the programs can be found in their usage text, comments in code, and in the Wiki of this repository.

## Programs
 1. `projection_subtraction.py` - Perform projection subtraction using per-particle FRC normalization.
 + `recenter.py` - Recenter particles on the center-of-mass of corresponding 2D class-averages.
 + `angdist.py` - Graph angular distributions on polar scatter plots. Supports particle subset selection.
 + `pyem/star.py` - Alter .star files. Supports dropping arbitrary fields, Euler angles, etc.
 + `project.py` - Project a map according to .star file entries (angles, CTF, etc.).
 + `csparc2star.py` - Convert Cryosparc metadata files to Relion .star format.

## Library modules
 1. `pyem/mrc.py` - Simple, standalone MRC I/O functions.
 + `pyem/star.py` - Parse and write .star files. Uses pandas.DataFrame as a backend.

## Other files
 1. `activate` - Place in `EMAN2/bin` to turn EMAN2 into a Python virtual environment.


(C) 2016 Daniel Asarnow  
University of California, San Francisco
