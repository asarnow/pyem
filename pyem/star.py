#!/usr/bin/env python
# Copyright (C) 2016 Daniel Asarnow
# University of California, San Francisco
#
# Simple program for parsing and altering Relion .star files.
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
import sys
import re
import os.path
import numpy as np
import pandas as pd
from util import cent2edge


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

    if args.info:
        if "rlnClassNumber" in star.columns:
            clsuniq = star["rlnClassNumber"].unique()
            clshist = np.histogram(star["rlnClassNumber"], bins=cent2edge(np.array(sorted(clsuniq))))[0]
            print("%d particles, %d classes, %.3f +/- %.3f particles per class" %
                    (star.shape[0], len(clsuniq), np.mean(clshist), np.std(clshist)))
        else:
            print("%d particles" % star.shape[0])

    if args.drop_angles:
        ang_fields = [f for f in star.columns if "Tilt" in f or "Psi" in f or "Rot" in f]
        star.drop(ang_fields, axis=1, inplace=True, errors="ignore")

    if args.drop_containing is not None:
        containing_fields = [f for q in args.drop_containing for f in star.columns if q in f]
        if args.invert:
            containing_fields = list(set(star.columns) - set(containing_fields))
        star.drop(containing_fields, axis=1, inplace=True, errors="ignore")

    if args.offset_group is not None:
        groupnum_fields = [f for f in star.columns if "GroupNumber" in f]
        star[groupnum_fields] += args.offset_group

    if args.subsample is not None:
        if args.subsample < 1:
            args.subsample = np.max(np.round(args.subsample * star.shape[0]), 1)
        star = star.sample(np.int(args.subsample), random_state=args.seed)

    if args.pick:
        fields = ["rlnCoordinateX", "rlnCoordinateY", "rlnAnglePsi", "rlnClassNumber", "rlnAutopickFigureOfMerit", "rlnMicrographName"]
        containing_fields = [f for q in fields for f in star.columns if q in f]
        containing_fields = list(set(star.columns) - set(containing_fields))
        star.drop(containing_fields, axis=1, inplace=True, errors="ignore")

    if args.split_micrographs:
        gb = star.groupby("rlnMicrographName")
        for g in gb:
            g[1].drop("rlnMicrographName", axis=1, inplace=True, errors="ignore")
            write_star(os.path.join(args.output, os.path.basename(g[0])[:-4]) + args.suffix, g[1])
        return 0
        
    if args.output is not None:
        write_star(args.output, star)
    return 0


def parse_star(starfile, keep_index=True):
    headers = []
    foundheader = False
    ln = 0
    with open(starfile, 'rU') as f:
        for l in f:
            if l.startswith("_rln"):
                foundheader = True
                lastheader = True
                if keep_index:
                    head = l.rstrip()
                else:
                    head = l.split('#')[0].rstrip().lstrip('_')
                headers.append(head)
            else:
                lastheader = False
            if foundheader and not lastheader:
                break
            ln += 1
    star = pd.read_table(starfile, skiprows=ln, delimiter='\s+', header=None)
    star.columns = headers
    return star


def write_star(starfile, star, reindex=True):
    indexed = re.search("#\d+$", star.columns[0]) is not None  # Check first column for '#N' index.
    with open(starfile, 'w') as f:
        f.write('\n')
        f.write("data_images" + '\n')
        f.write('\n')
        f.write("loop_" + '\n')
        for i in range(len(star.columns)):
            if reindex and not indexed:  # No index present, append new, consecutive indices to each header line.
                line = star.columns[i] + " #%d \n" % (i + 1)
            elif reindex and indexed:  # Replace existing indices with new, consecutive indices.
                line = star.columns[i].split("#")[0].rstrip() + " #%d \n" % (i + 1)
            else:  # Use DataFrame column labels literally.
                line = star.columns[i] + " \n"
            line = line if line.startswith('_') else '_' + line
            f.write(line)
    star.to_csv(starfile, mode='a', sep=' ', header=False, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", help="Keep this class in output, may be passed multiple times",
                        action="append", type=int, dest="cls")
    parser.add_argument("--drop-angles", help="Drop tilt, psi and rot angles from output",
                        action="store_true")
    parser.add_argument("--drop-containing",
                        help="Drop fields containing string from output, may be passed multiple times",
                        action="append")
    parser.add_argument("--info", help="Print information about initial file",
                        action="store_true")
    parser.add_argument("--invert", help="Invert field match conditions",
                        action="store_true")
    parser.add_argument("--pick", help="Only keep fields output by Gautomatch",
                        action="store_true")
    parser.add_argument("--offset-group", help="Add fixed offset to group number",
                        type=int)
    parser.add_argument("--seed", help="Seed for random number generators",
                        type=int)
    parser.add_argument("--split-micrographs", help="Write separate output file for each micrograph",
                        action="store_true")
    parser.add_argument("--subsample", help="Randomly subsample particles",
                        type=float, metavar="N")
    parser.add_argument("--suffix", help="Suffix for multiple output files",
                        type=str, default="")
    parser.add_argument("input", help="Input .star file")
    parser.add_argument("output", help="Output .star file",
                        default=None, nargs="?")
    sys.exit(main(parser.parse_args()))

