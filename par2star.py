#!/usr/bin/env python3
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
import logging
import pandas as pd
import numpy as np
import sys
from pyem import metadata
from pyem import star


def main(args):
    log = logging.getLogger('root')
    hdlr = logging.StreamHandler(sys.stdout)
    log.addHandler(hdlr)
    log.setLevel(logging.getLevelName(args.loglevel.upper()))

    dfs = [metadata.parse_fx_par(fn) for fn in args.input]
    n = dfs[0].shape[0]
    if not np.all(np.array([df.shape[0] for df in dfs]) == n):
        log.error("Input files are not aligned!")
        return 1
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df["CLASS"] = np.repeat(np.arange(1, len(dfs) + 1), n)

    if args.min_occ:
        df = df[df["OCC"] >= args.min_occ]
    
    df = df.sort_values(by="OCC")
    df = df.drop_duplicates("C", keep="last")
    df = df.sort_values(by="C")

    df = metadata.par2star(df, data_path=args.stack, apix=args.apix, cs=args.cs,
                           ac=args.ac, kv=args.voltage, invert_eulers=args.invert_eulers)
    
    if args.cls is not None:
        df = star.select_classes(df, args.cls)

    star.write_star(args.output, df)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input Frealign .par file", nargs="*")
    parser.add_argument("output", help="Output Relion .star file")
    parser.add_argument("--stack", "-s", help="Particle stack path")
    parser.add_argument("--apix", "--angpix", help="Pixel size in Angstroms", type=float)
    parser.add_argument("--ac", help="Amplitude contrast", type=float)
    parser.add_argument("--cs", help="Spherical abberation", type=float)
    parser.add_argument("--voltage", "--kv", "-v", help="Acceleration voltage (kV)", type=float)
    parser.add_argument("--min-occ", help="Minimum occupancy for inclusion in output", type=float)
    parser.add_argument("--class", "-c", help="Classes to preserve in output", action="append", dest="cls")
    parser.add_argument("--relion", help=argparse.SUPPRESS, action="store_true")
    parser.add_argument("--invert-eulers", help="Invert Euler angles (generally unnecessary)", action="store_true")
    parser.add_argument("--loglevel", "-l", type=str, default="WARNING", help="Logging level and debug output")
    sys.exit(main(parser.parse_args()))

