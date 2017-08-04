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
from collections import Counter
import numpy as np
import pandas as pd
import json
from glob import glob
from math import modf
from util import rot2euler

MICROGRAPH_NAME = "rlnMicrographName"
IMAGE_NAME = "rlnImageName"
COORDS = ["rlnCoordinateX", "rlnCoordinateY"]
ORIGINS = ["rlnOriginX", "rlnOriginY"]
ANGLES = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
CTF_PARAMS = ["rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle", "rlnSphericalAberration", "rlnCtfBfactor",
              "rlnCtfScaleFactor", "rlnPhaseShift", "rlnAmplitudeContrast", "rlnCtfMaxResolution",
              "rlnCtfFigureOfMerit"]
MICROSCOPE_PARMS = ["rlnVoltage", "rlnMagnification", "rlnDetectorPixelSize"]
MICROGRAPH_COORDS = [MICROGRAPH_NAME] + COORDS
PICK_PARAMS = MICROGRAPH_COORDS + ["rlnAnglePsi", "rlnClassNumber", "rlnAutopickFigureOfMerit"]


def main(args):
    if args.info:
        args.input.append(args.output)

    star = pd.concat((parse_star(inp, keep_index=False) for inp in args.input), join="inner")

    otherstar = None

    if args.cls is not None:
        star = select_classes(star, args.cls)

    if args.info:
        if is_particle_star(star) and "rlnClassNumber" in star.columns:
            c = star["rlnClassNumber"].value_counts()
            print("%s particles in %d classes" % ("{:,}".format(star.shape[0]), len(c)))
            print("Class distribution:  " + ",    ".join(['%s (%.2f %%)' % ("{:,}".format(i), 100.*i/c.sum()) for i in c]))
        elif is_particle_star(star):
            print("%s particles" % "{:,}".format(star.shape[0]))
        if "rlnMicrographName" in star.columns:
            mgraphcnt = star["rlnMicrographName"].value_counts()
            print("%s micrographs, %s +/- %s particles per micrograph" %
                    ("{:,}".format(len(mgraphcnt)), "{:,.3f}".format(np.mean(mgraphcnt)), "{:,.3f}".format(np.std(mgraphcnt))))
        try:
            print("%f A/px (%sX magnification)" % (calculate_apix(star), "{:,.0f}".format(star["rlnMagnification"][0])))
        except KeyError:
            pass
        return 0

    if args.drop_angles:
        star.drop(ANGLES, axis=1, inplace=True, errors="ignore")

    if args.drop_containing is not None:
        containing_fields = [f for q in args.drop_containing for f in star.columns if q in f]
        if args.invert:
            containing_fields = star.columns.difference(containing_fields)
        star.drop(containing_fields, axis=1, inplace=True, errors="ignore")

    if args.offset_group is not None:
        star["rlnGroupNumber"] += args.offset_group

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
        star = smart_merge(star, angle_star, fields=ANGLES)

    if args.transform is not None:
        r = np.array(json.loads(args.transform))
        star = transform_star(star, r, inplace=True)

    if args.recenter:
        star = recenter(star, inplace=True)

    if args.zero_origins:
        star = zero_origins(star, inplace=True)

    if args.copy_paths is not None:
        path_star = parse_star(args.copy_paths, keep_index=False)
        star[IMAGE_NAME] = path_star[IMAGE_NAME]

    if args.copy_ctf is not None:
        ctf_star = pd.concat((parse_star(inp, keep_index=False) for inp in glob(args.copy_ctf)), join="inner")
        star = smart_merge(star, ctf_star, CTF_PARAMS)

    if args.copy_micrograph_coordinates is not None:
        coord_star = pd.concat(
            (parse_star(inp, keep_index=False) for inp in glob(args.copy_micrograph_coordinates)), join="inner")
        star = smart_merge(star, coord_star, fields=MICROGRAPH_COORDS)

    if args.pick:
        star.drop(star.columns.difference(PICK_PARAMS), axis=1, inplace=True, errors="ignore")

    if args.subsample is not None and args.suffix != "":
        if args.subsample < 1:
            print("Specific integer sample size")
            return 1
        nsamplings = args.bootstrap if args.bootstrap is not None else star.shape[0] / np.int(args.subsample)
        inds = np.random.choice(star.shape[0], size=(nsamplings, np.int(args.subsample)),
                                replace=args.bootstrap is not None)
        for i, ind in enumerate(inds):
            write_star(os.path.join(args.output, os.path.basename(args.input[0])[:-5] + args.suffix + "_%d" % (i + 1)),
                       star.iloc[ind])

    if args.split_micrographs:
        stars = split_micrographs(star)
        for mg in stars:
            write_star(os.path.join(args.output, os.path.basename(mg)[:-4]) + args.suffix, stars[mg])
        return 0

    if args.auxout is not None and otherstar is not None:
        write_star(args.auxout, otherstar)

    if args.output is not None:
        write_star(args.output, star)
    return 0


def smart_merge(s1, s2, fields):
    key = merge_key(s1, s2)
    s2 = s2.set_index(key, drop=False)
    s1 = s1.merge(s2[s2.columns.intersection(fields)], left_on=key, right_index=True, suffixes=["_x", ""])
    x = [c for c in s1.columns if "_x" in c]
    if len(x) > 0:
        y = [c.split("_")[0] for c in s1.columns if c in x]
        s1[y] = s1[y].fillna(s1[x])
        s1 = s1.drop(x, axis=1)
    return s1.reset_index(drop=True)


def merge_key(s1, s2):
    inter = s1.columns.intersection(s2.columns)
    if inter.empty:
        return None
    if IMAGE_NAME in inter:
        c = Counter(s1[IMAGE_NAME])
        shared = sum(c[i] for i in set(s2[IMAGE_NAME]))
        if shared > s1.shape[0] * 0.5:
            return IMAGE_NAME
    mgraph_coords = inter.intersection(MICROGRAPH_COORDS)
    if MICROGRAPH_NAME in mgraph_coords:
        c = Counter(s1[MICROGRAPH_NAME])
        shared = sum(c[i] for i in set(s2[MICROGRAPH_NAME]))
        can_merge_mgraph_name = MICROGRAPH_NAME in mgraph_coords and shared > s1.shape[0] * 0.5
        if can_merge_mgraph_name and not mgraph_coords.intersection(COORDS).empty:
            return MICROGRAPH_COORDS
        elif can_merge_mgraph_name:
            return MICROGRAPH_NAME
    return None


def is_particle_star(star):
    return not star.columns.intersection([IMAGE_NAME] + COORDS).empty


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


def split_micrographs(star):
    gb = star.groupby("rlnMicrographName")
    stars = {}
    for g in gb:
        g[1].drop("rlnMicrographName", axis=1, inplace=True, errors="ignore")
        stars[g[0]] = g[1]
    return stars


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


def zero_origins(star, inplace=False):
    if inplace:
        newstar = star
    else:
        newstar = star.copy()
    newstar["rlnCoordinateX"] = newstar["rlnCoordinateX"] - newstar["rlnOriginX"]
    newstar["rlnCoordinateY"] = newstar["rlnCoordinateY"] - newstar["rlnOriginY"]
    newstar["rlnOriginX"] = 0
    newstar["rlnOriginY"] = 0
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
    assert (r.shape[0] == 3)
    if r.shape[1] == 4 and t is None:
        t = r[:, -1]
        r = r[:, :3]
    else:
        assert (r.shape == (3, 3))

    psi, theta, phi = rot2euler(r)

    if inplace:
        newstar = star
    else:
        newstar = star.copy()

    newstar["rlnAngleRot"] += np.rad2deg(phi)
    newstar["rlnAngleTilt"] += np.rad2deg(theta)
    newstar["rlnAnglePsi"] += np.rad2deg(psi)

    if t is not None:
        assert (len(t) == 3)
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
    parser.add_argument("--copy-angles",
                        help="Source for particle Euler angles (must align exactly with input .star file)",
                        type=str)
    parser.add_argument("--copy-ctf", help="Source for CTF parameters (file or quoted glob)")
    parser.add_argument("--copy-micrograph-coordinates", help="Source for micrograph paths and particle coordinates (file or quoted glob)",
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
    parser.add_argument("--recenter", help="Subtract origin from coordinates, leaving subpixel information in origin",
                        action="store_true")
    parser.add_argument("--zero-origins", help="Subtract origin from coordinates and set origin to zero",
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
    parser.add_argument("--transform",
                        help="Apply rotation matrix or 3x4 rotation plus translation matrix to particles (Numpy format)",
                        type=str)
    parser.add_argument("input", help="Input .star file(s) or unquoted glob", nargs="*")
    parser.add_argument("output", help="Output .star file")
    sys.exit(main(parser.parse_args()))
