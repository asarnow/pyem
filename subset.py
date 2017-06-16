#!/usr/bin/env python
# Copyright (C) 2017 Daniel Asarnow
# University of California, San Francisco
#
# Program for subsetting and resampling EM data.
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
import sys
from pyem.star import parse_star
from pyem.star import write_star

def main(args):
    star = parse_star(args.input, keep_index=False)

    if args.cls is not None:
        clsfields = [f for f in star.columns if "ClassNumber" in f]
        if len(clsfields) == 0:
            print("No class labels found")
            return 1
        ind = star[clsfields[0]].isin(args.cls)
        if not np.any(ind):
            print("Specified class has no members")
            return 1
        star = star.loc[ind]

    if args.max_astigmatism is not None:
        astigmatism = star["rlnDefocusU"] - star["rlnDefocusV"]
        ind = astigmatism <= args.max_astigmatism
        star = star.loc[ind]

    if args.max_resolution is not None:
        if "rlnFinalResolution" in star.columns:
            ind = star["rlnFinalResolution"] <= args.max_resolution
        elif "rlnCtfMaxResolution" in star.columns:
            ind = star["rlnCtfMaxResolution"] <= args.max_resolution
        else:
            print("No CTF resolution field found in input")
            return 1
        star = star.loc[ind]

    if args.min_ctf_fom is not None:
        ind = star["rlnCtfFigureOfMerit"] >= args.min_ctf_fom
        star = star.loc[ind]

    if args.subsample is not None:
        if args.subsample < 1:
            args.subsample = np.max(np.round(args.subsample * star.shape[0]), 1)
        if args.bootstrap > 1:
            inds = np.random.choice(star.shape[0], size=(np.int(args.subsample), star.shape[0]/np.int(args.subsample)), replace=True)
        else:
            star = star.sample(np.int(args.subsample), random_state=args.seed)

    write_star(args.output, star)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", help="Keep this class in output, may be passed multiple times",
                        action="append", type=int, dest="cls")
    parser.add_argument("--max-astigmatism", help="Maximum astigmatism (defocus difference) in Angstroms",
                        type=float)
    parser.add_argument("--max-resolution", help="Maximum CTF resolution in Angstroms",
                        type=float)
    parser.add_argument("--min-ctf-fom", help="Minimum CTF figure-of-merit",
                        type=float)
    parser.add_argument("--seed", help="Seed for random number generators",
                        type=int)
    parser.add_argument("--subsample", help="Randomly subsample particles",
                        type=float, metavar="N")
    parser.add_argument("--bootstrap", help="Sample --subsample particles N times, with replacement",
                        type=int, metavar="N")
    parser.add_argument("input", help="Input .star file")
    parser.add_argument("output", help="Output .star file")
    sys.exit(main(parser.parse_args()))
