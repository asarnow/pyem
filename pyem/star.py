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
import json
from math import modf
from util import rot2euler

COORDS = ["rlnCoordinateX", "rlnCoordinateY"]
ORIGINS = ["rlnOriginX", "rlnOriginY"]
ANGLES = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]


def main(args):
    star = parse_star(args.input, keep_index=False)
    
    otherstar = None

    if args.cls is not None:
        star = select_classes(star, args.cls)

    if args.info:
        print("%d particles" % star.shape[0])
        print("%f A/px" % calculate_apix(star))
        if "rlnMicrographName" in star.columns:
            mgraphcnt = star["rlnMicrographName"].value_counts()
            print("%d micrographs, %.3f +/- %.3f particles per micrograph" % 
                    (len(mgraphcnt), np.mean(mgraphcnt), np.std(mgraphcnt)))
        if "rlnClassNumber" in star.columns:
            clscnt = star["rlnClassNumber"].value_counts()
            print("%d classes, %.3f +/- %.3f particles per class" %
                    (len(clscnt), np.mean(clscnt), np.std(clscnt)))
        return 0

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

    if args.subsample_micrographs is not None:
        if args.bootstrap is not None:
            print("Only particle sampling allows bootstrapping")
            return 1
        mgraphs = star["rlnMicrographName"].unique()
        if args.subsample_micrographs < 1:
            args.subsample_micrographs = max(np.round(args.subsample_micrographs * len(mgraphs)), 1)
        ind = np.random.choice(len(mgraphs), size=args.subsample_micrographs, replace=False)
        mask = star["rlnMicrographName"].isin(mgraphs[ind])
        if args.auxout is not None:
            otherstar = star.loc[~mask]
        star = star.loc[mask]

    if args.subsample is not None and args.suffix == "":
        if args.subsample < 1:
            args.subsample = np.int(max(np.round(args.subsample * star.shape[0]), 1))
        ind = np.random.choice(star.shape[0], size=args.subsample, replace=False)
        mask = star.index.isin(ind)
        if args.auxout is not None:
            otherstar = star.loc[~mask]
        star = star.loc[mask]

    if args.copy_angles is not None:
        angle_star = parse_star(args.copy_angles, keep_index=False)
        ang_fields = [f for f in star.columns if "Tilt" in f or "Psi" in f or "Rot" in f]
        star[ang_fields] = angle_star[ang_fields]

    if args.transform is not None:
        r = np.array(json.loads(args.transform))
        star = transform_star(star, r, inplace=True)

    if args.recenter:
        star["rlnCoordinateX"] = star["rlnCoordinateX"] - star["rlnOriginX"]
        star["rlnCoordinateY"] = star["rlnCoordinateY"] - star["rlnOriginY"]
        star["rlnOriginX"] = 0
        star["rlnOriginY"] = 0

    if args.copy_paths is not None:
        path_star = parse_star(args.copy_paths, keep_index=False)
        star["rlnImageName"] = path_star["rlnImageName"]

    if args.pick:
        fields = ["rlnCoordinateX", "rlnCoordinateY", "rlnAnglePsi", "rlnClassNumber", "rlnAutopickFigureOfMerit", "rlnMicrographName"]
        containing_fields = [f for q in fields for f in star.columns if q in f]
        containing_fields = list(set(star.columns) - set(containing_fields))
        star.drop(containing_fields, axis=1, inplace=True, errors="ignore")

    if args.subsample is not None and args.suffix != "":
        if args.subsample < 1:
            print("Specific integer sample size")
            return 1
        nsamplings = args.bootstrap if args.bootstrap is not None else star.shape[0]/np.int(args.subsample)
        inds = np.random.choice(star.shape[0], size=(nsamplings, np.int(args.subsample)), replace=args.bootstrap is not None)
        for i, ind in enumerate(inds):
            write_star(os.path.join(args.output, os.path.basename(args.input)[:-5] + args.suffix +  "_%d" % (i+1)), star.iloc[ind])

    if args.split_micrographs:
        gb = star.groupby("rlnMicrographName")
        for g in gb:
            g[1].drop("rlnMicrographName", axis=1, inplace=True, errors="ignore")
            write_star(os.path.join(args.output, os.path.basename(g[0])[:-4]) + args.suffix, g[1])
        return 0

    if args.auxout is not None and otherstar is not None:
        write_star(args.auxout, otherstar)
        
    if args.output is not None:
        write_star(args.output, star)
    return 0


def calculate_apix(star):
    return 10000.0 * star.iloc[0]['rlnDetectorPixelSize'] / star.iloc[0]['rlnMagnification']


def select_classes(star, classes):
    clsfields = [f for f in star.columns if "ClassNumber" in f]
    if len(clsfields) == 0:
        raise RuntimeError("No class labels found")
    ind = star[clsfields[0]].isin(classes)
    if not np.any(ind):
        raise RuntimeError("Specified classes have no members")
    return star.loc[ind]


def recenter_row(row):
    remx, offsetx = modf(row["rlnOriginX"])
    remy, offsety = modf(row["rlnOriginY"])
    offsetx = row["rlnCoordinateX"] - offsetx
    offsety = row["rlnCoordinateY"] - offsety
    return pd.Series({"rlnCoordinateX": offsetx, "rlnCoordinateY": offsety,
                "rlnOriginX": remx, "rlnOriginY": remy})


def recenter(star, inplace=False):
    if inplace:
        newstar = star
    else:
        newstar = star.copy()
    newvals = star.apply(recenter_row, axis=1)
    newstar[COORDS + ORIGINS] = newvals[COORDS + ORIGINS]
    return newstar


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
    if not starfile.endswith(".star"):
        starfile += ".star"
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


def transform_star(star, r, t=None, inplace=False):
    """
    Transform particle angles and origins according to a rotation
    matrix (in radians) and an optional translation vector.
    The translation may also be given as the 4th column of a 3x4 matrix.
    """
    assert(r.shape[0] == 3)
    if r.shape[1] == 4 and t is None:
        t = r[:,-1]
        r = r[:,:3]
    else:
        assert(r.shape == (3,3))

    psi, theta, phi = rot2euler(r)

    if inplace:
        newstar = star
    else:
        newstar = star.copy()

    newstar["rlnAngleRot"] += np.rad2deg(phi)
    newstar["rlnAngleTilt"] += np.rad2deg(theta)
    newstar["rlnAnglePsi"] += np.rad2deg(psi)

    if t is not None:
        assert(len(t) == 3)
        tt = r.dot(t)
        newstar["rlnOriginX"] += tt[0]
        newstar["rlnOriginY"] += tt[1]

    return newstar


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--auxout", help="Auxilliary output .star file with deselected particles",
                        type=str)
    parser.add_argument("--bootstrap", help="Sample with replacement when creating multiple outputs",
                        type=int, default=None)
    parser.add_argument("--class", help="Keep this class in output, may be passed multiple times",
                        action="append", type=int, dest="cls")
    parser.add_argument("--copy-angles", help="Source for particle Euler angles (must align exactly with input .star file)",
                        type=str)
    parser.add_argument("--copy-paths", help="Source for particle paths (must align exactly with input .star file)",
                        type=str)
    parser.add_argument("--drop-angles", help="Drop tilt, psi and rot angles from output",
                        action="store_true")
    parser.add_argument("--drop-containing",
                        help="Drop fields containing string from output, may be passed multiple times",
                        action="append")
    parser.add_argument("--info", help="Print information about initial file",
                        action="store_true")
    parser.add_argument("--invert", help="Invert field match conditions",
                        action="store_true")
    parser.add_argument("--offset-group", help="Add fixed offset to group number",
                        type=int)
    parser.add_argument("--pick", help="Only keep fields output by Gautomatch",
                        action="store_true")
    parser.add_argument("--recenter", help="Subtract origin from coordinates",
                        action="store_true")
#    parser.add_argument("--seed", help="Seed for random number generators",
#                        type=int)
    parser.add_argument("--split-micrographs", help="Write separate output file for each micrograph",
                        action="store_true")
    parser.add_argument("--subsample", help="Randomly subsample remaining particles",
                        type=float, metavar="N") 
    parser.add_argument("--subsample-micrographs", help="Randomly subsample micrographs",
                        type=float)
    parser.add_argument("--suffix", help="Suffix for multiple output files",
                        type=str, default="")
    parser.add_argument("--transform", help="Apply rotation matrix or 3x4 rotation plus translation matrix to particles (Numpy format)",
                        type=str)
    parser.add_argument("input", help="Input .star file")
    parser.add_argument("output", help="Output .star file",
                        default=None, nargs="?")
    sys.exit(main(parser.parse_args()))

