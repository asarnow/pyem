#!/usr/bin/env python
# Copyright (C) 2019 Daniel Asarnow
# University of California, San Francisco
#
# Simple program for converting CTFFIND4 output to micrograph .star file.
# See help text and README file for more information.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# *_ctfEstimation.txt sample:
# # Output from CTFFind version 4.1.10, run on 2019-05-29 19:35:36
# # Input file: Runs/000126_ProtCTFFind/tmp/mic_0126/20190529_hABCA4_DA-52-4_99_0009_aligned_mic.mrc ; Number of micrographs: 1
# # Pixel size: 1.140 Angstroms ; acceleration voltage: 200.0 keV ; spherical aberration: 2.70 mm ; amplitude contrast: 0.10
# # Box size: 512 pixels ; min. res.: 22.8 Angstroms ; max. res.: 2.9 Angstroms ; min. def.: 5000.0 um; max. def. 40000.0 um
# # Columns: #1 - micrograph number; #2 - defocus 1 [Angstroms]; #3 - defocus 2; #4 - azimuth of astigmatism; #5 - additional phase shift [radians]; #6 - cross correlation; #7 - spacing (in Angstroms) up to which CTF rings were fit successfully
# 1.000000 16793.875000 15208.728516 -66.871671 0.000000 -0.029152 9.327273
import glob
import os.path
import pandas as pd
import sys
from pyem import star


def main(args):
    data = []
    if os.path.isdir(args.input[0]):
        flist = glob.glob(os.path.join(args.input[0], "*_ctfEstimation.txt"))
    else:
        flist = args.input
    for fn in flist:
        row = {}
        with open(fn, 'r') as f:
                lines = f.readlines()
                g = lines[1].lstrip("# Input file:").split(" ;")[0]
                if args.apix is None:
                    args.apix = float(lines[2].lstrip("# Pixel size:").split("Angstrom")[0])
                tok = lines[-1].split()
                if args.path is None:
                    row[star.Relion.MICROGRAPH_NAME] = g
                else:
                    row[star.Relion.MICROGRAPH_NAME] = os.path.join(args.path, os.path.basename(g))
                row[star.Relion.DEFOCUSU] = float(tok[1])
                row[star.Relion.DEFOCUSV] = float(tok[2])
                row[star.Relion.DEFOCUSANGLE] = float(tok[3])
                row[star.Relion.PHASESHIFT] = float(tok[4])
                row[star.Relion.CTFFIGUREOFMERIT] = float(tok[5])
                row[star.Relion.CTFMAXRESOLUTION] = float(tok[6])
                row[star.Relion.MAGNIFICATION] = 10000
                row[star.Relion.DETECTORPIXELSIZE] = args.apix
        data.append(row)
    df = pd.DataFrame(data)
    if not args.no_sort:
        df = star.sort_records(df, inplace=True)
    df = star.sort_fields(df, inplace=True)
    star.write_star(args.output, df)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="*")
    parser.add_argument("output")
    parser.add_argument("--path", "-p", help="New path prepended to micrograph basenames", type=str)
    parser.add_argument("--no-sort", "-n", help="Preserve input filename order", action="store_true")
    parser.add_argument("--apix", help="Override pixel size (Angstroms)", type=float)
    sys.exit(main(parser.parse_args()))

