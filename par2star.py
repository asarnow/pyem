#!/usr/bin/env python
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
    df.reset_index(inplace=True)

    if args.min_score is not None:
        if args.min_score < 1:
            args.min_score = np.percentile(df["SCORE"], (1 - args.min_score) * 100)
        df = df.loc[df["SCORE"] >= args.min_score]

    if args.merge is not None:
        dfo = star.parse_star(args.merge)
        args.apix = star.calculate_apix(dfo)
        args.cs = dfo.iloc[0][star.Relion.CS]
        args.ac = dfo.iloc[0][star.Relion.AC]
        args.voltage = dfo.iloc[0][star.Relion.VOLTAGE]
        df = metadata.par2star(df, data_path=args.stack, apix=args.apix, cs=args.cs,
                               ac=args.ac, kv=args.voltage, invert_eulers=args.invert_eulers)
        if args.stack is None:
            df[star.UCSF.IMAGE_INDEX] = dfo[star.UCSF.IMAGE_INDEX]
            df[star.UCSF.IMAGE_PATH] = dfo[star.UCSF.IMAGE_PATH]
        key = [star.UCSF.IMAGE_INDEX, star.UCSF.IMAGE_PATH]
        fields = star.Relion.MICROGRAPH_COORDS + [star.UCSF.IMAGE_ORIGINAL_INDEX, star.UCSF.IMAGE_ORIGINAL_PATH] + [star.Relion.OPTICSGROUP] + star.Relion.OPTICSGROUPTABLE
        df = star.smart_merge(df, dfo, fields=fields, key=key)
        if args.revert_original:
            df = star.revert_original(df, inplace=True)
    else:
        df = metadata.par2star(df, data_path=args.stack, apix=args.apix, cs=args.cs,
                               ac=args.ac, kv=args.voltage, invert_eulers=args.invert_eulers)

    if args.cls is not None:
        df = star.select_classes(df, args.cls)

    df = star.check_defaults(df, inplace=True)
    df = star.compatible(df, relion2=args.relion2, inplace=True)
    star.write_star(args.output, df, optics=(not args.relion2))
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input Frealign .par file", nargs="*")
    parser.add_argument("output", help="Output Relion .star file")
    parser.add_argument("--merge", "-m", help="Merge this .star file")
    parser.add_argument("--stack", "-s", help="Particle stack path")
    parser.add_argument("--apix", "--angpix", help="Pixel size in Angstroms", type=float)
    parser.add_argument("--ac", "-ac", help="Amplitude contrast", type=float)
    parser.add_argument("--cs", "-cs", help="Spherical abberation", type=float)
    parser.add_argument("--voltage", "--kv", "-kv", help="Acceleration voltage (kV)", type=float)
    parser.add_argument("--min-occ", help="Minimum occupancy for inclusion in output", type=float)
    parser.add_argument("--min-score", help="Minimum score (or percentile if < 1) for inclusion in output", type=float)
    parser.add_argument("--class", "-c", help="Classes to preserve in output", action="append", dest="cls")
    parser.add_argument("--relion2", "-r2", help="Write Relion2 compatible STAR file", action="store_true")
    parser.add_argument("--revert-original", "-v", help="Swap ImageName and ImageOriginalName before writing",
                        action="store_true")
    parser.add_argument("--invert-eulers", help="Invert Euler angles (generally unnecessary)", action="store_true")
    parser.add_argument("--loglevel", "-l", type=str, default="WARNING", help="Logging level and debug output")
    sys.exit(main(parser.parse_args()))

