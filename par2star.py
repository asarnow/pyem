#!/usr/bin/env python2.7
# Copyright (C) 2017 Daniel Asarnow
# University of California, San Francisco
#
# Simple program for converting Frealign PAR files to Relion STAR files.
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
from __future__ import print_function
import pandas as pd
import sys
from pyem.metadata import parse_par
from pyem.metadata import par2star
from pyem.star import write_star


def main(args):
    par = [parse_par(inp) for inp in args.input]
    if len(par) > 1:
        idx = par[:,:,"OCC"].idxmax(axis=1)
    else:
        par = par[0]

    if args.data_path is not None:
        par["Input particle images"] = args.data_path
    if args.voltage is not None:
        par["Beam energy (keV)"] = args.voltage
    if args.cs is not None:
        par["Spherical aberration (mm)"] = args.cs
    if args.ac is not None:
        par["Amplitude contrast"] = args.ac
    if args.apix is not None:
        par["Pixel size of images (A)"] = args.apix
    star = par2star(par)
    write_star(args.output, star, reindex=True)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input Frealign .par file", nargs="+")
    parser.add_argument("output", help="Output Relion .star file")
    parser.add_argument("--ac", help="Amplitude contrast", type=float)
    parser.add_argument("--apix", "--angpix", help="Pixel size in Angstroms", type=float)
    parser.add_argument("--cs", help="Spherical abberation", type=float)
    parser.add_argument("--data-path", help="Alternate path for particle stack")
    parser.add_argument("--voltage", "--kv", "-v", help="Acceleration voltage (kV)", type=float)
    sys.exit(main(parser.parse_args()))

