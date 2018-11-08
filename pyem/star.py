#!/usr/bin/env python2.7
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
from pyem.algo import query_connected
from pyem.util import rot2euler
from pyem.util import euler2rot


class Relion:
    MICROGRAPH_NAME = "rlnMicrographName"
    IMAGE_NAME = "rlnImageName"
    IMAGE_ORIGINAL_NAME = "rlnImageOriginalName"
    COORDX = "rlnCoordinateX"
    COORDY = "rlnCoordinateY"
    ORIGINX = "rlnOriginX"
    ORIGINY = "rlnOriginY"
    ANGLEROT = "rlnAngleRot"
    ANGLETILT = "rlnAngleTilt"
    ANGLEPSI = "rlnAnglePsi"
    CLASS = "rlnClassNumber"
    DEFOCUSU = "rlnDefocusU"
    DEFOCUSV = "rlnDefocusV"
    DEFOCUS = [DEFOCUSU, DEFOCUSV]
    DEFOCUSANGLE = "rlnDefocusAngle"
    CS = "rlnSphericalAberration"
    PHASESHIFT = "rlnPhaseShift"
    AC = "rlnAmplitudeContrast"
    VOLTAGE = "rlnVoltage"
    MAGNIFICATION = "rlnMagnification"
    DETECTORPIXELSIZE = "rlnDetectorPixelSize"
    GROUPNUMBER = "rlnGroupNumber"
    RANDOMSUBSET = "rlnRandomSubset"
    COORDS = [COORDX, COORDY]
    ORIGINS = [ORIGINX, ORIGINY]
    ANGLES = [ANGLEROT, ANGLETILT, ANGLEPSI]
    CTF_PARAMS = [DEFOCUSU, DEFOCUSV, DEFOCUSANGLE, CS, PHASESHIFT, AC,
                  "rlnCtfScaleFactor", "rlnCtfBfactor", "rlnCtfMaxResolution",
                  "rlnCtfFigureOfMerit"]
    MICROSCOPE_PARAMS = [VOLTAGE, MAGNIFICATION, DETECTORPIXELSIZE]
    MICROGRAPH_COORDS = [MICROGRAPH_NAME] + COORDS
    PICK_PARAMS = MICROGRAPH_COORDS + [ANGLEPSI, CLASS, "rlnAutopickFigureOfMerit"]


class UCSF:
    IMAGE_PATH = "ucsfImagePath"
    IMAGE_BASENAME = "ucsfImageBasename"
    IMAGE_INDEX = "ucsfImageIndex"
    IMAGE_ORIGINAL_PATH = "ucsfImageOriginalPath"
    IMAGE_ORIGINAL_BASENAME = "ucsfImageOriginalBasename"
    IMAGE_ORIGINAL_INDEX = "ucsfImageOriginalIndex"


def main(args):
    if args.info:
        args.input.append(args.output)

    df = pd.concat((parse_star(inp, augment=args.augment) for inp in args.input), join="inner")
    
    dfaux = None

    if args.cls is not None:
        df = select_classes(df, args.cls)

    if args.info:
        if is_particle_star(df) and Relion.CLASS in df.columns:
            c = df[Relion.CLASS].value_counts()
            print("%s particles in %d classes" % ("{:,}".format(df.shape[0]), len(c)))
            print("    ".join(['%d: %s (%.2f %%)' % (i, "{:,}".format(s), 100.*s/c.sum())
                for i,s in c.sort_index().iteritems()]))
        elif is_particle_star(df):
            print("%s particles" % "{:,}".format(df.shape[0]))
        if Relion.MICROGRAPH_NAME in df.columns:
            mgraphcnt = df[Relion.MICROGRAPH_NAME].value_counts()
            print("%s micrographs, %s +/- %s particles per micrograph" %
                    ("{:,}".format(len(mgraphcnt)), "{:,.3f}".format(np.mean(mgraphcnt)), "{:,.3f}".format(np.std(mgraphcnt))))
        try:
            print("%f A/px (%sX magnification)" % (calculate_apix(df), "{:,.0f}".format(df[Relion.MAGNIFICATION][0])))
        except KeyError:
            pass
        return 0

    if args.drop_angles:
        df.drop(Relion.ANGLES, axis=1, inplace=True, errors="ignore")

    if args.drop_containing is not None:
        containing_fields = [f for q in args.drop_containing for f in df.columns if q in f]
        if args.invert:
            containing_fields = df.columns.difference(containing_fields)
        df.drop(containing_fields, axis=1, inplace=True, errors="ignore")

    if args.offset_group is not None:
        df[Relion.GROUPNUMBER] += args.offset_group

    if args.subsample_micrographs is not None:
        if args.bootstrap is not None:
            print("Only particle sampling allows bootstrapping")
            return 1
        mgraphs = df[Relion.MICROGRAPH_NAME].unique()
        if args.subsample_micrographs < 1:
            args.subsample_micrographs = np.int(max(np.round(args.subsample_micrographs * len(mgraphs)), 1))
        else:
            args.subsample_micrographs = np.int(args.subsample_micrographs)
        ind = np.random.choice(len(mgraphs), size=args.subsample_micrographs, replace=False)
        mask = df[Relion.MICROGRAPH_NAME].isin(mgraphs[ind])
        if args.auxout is not None:
            dfaux = df.loc[~mask]
        df = df.loc[mask]

    if args.subsample is not None and args.suffix == "":
        if args.subsample < 1:
            args.subsample = np.int(max(np.round(args.subsample * df.shape[0]), 1))
        else:
            args.subsample = np.int(args.subsample)
        ind = np.random.choice(df.shape[0], size=args.subsample, replace=False)
        mask = df.index.isin(ind)
        if args.auxout is not None:
            dfaux = df.loc[~mask]
        df = df.loc[mask]

    if args.copy_angles is not None:
        angle_star = parse_star(args.copy_angles, augment=args.augment)
        df = smart_merge(df, angle_star, fields=Relion.ANGLES)

    if args.transform is not None:
        if args.transform.count(",") == 2:
            r = euler2rot(*np.deg2rad([np.double(s) for s in args.transform.split(",")]))
        else:
            r = np.array(json.loads(args.transform))
        df = transform_star(df, r, inplace=True)

    if args.invert_hand:
        df[Relion.ANGLEROT] = -df[Relion.ANGLEROT]
        df[Relion.ANGLETILT] = 180 - df[Relion.ANGLETILT]

    if args.copy_paths is not None:
        path_star = parse_star(args.copy_paths)
        df[Relion.IMAGE_NAME] = path_star[Relion.IMAGE_NAME]

    if args.copy_ctf is not None:
        ctf_star = pd.concat((parse_star(inp, augment=args.augment) for inp in glob(args.copy_ctf)), join="inner")
        df = smart_merge(df, ctf_star, Relion.CTF_PARAMS)

    if args.copy_micrograph_coordinates is not None:
        coord_star = pd.concat(
            (parse_star(inp, augment=args.augment) for inp in glob(args.copy_micrograph_coordinates)), join="inner")
        df = smart_merge(df, coord_star, fields=Relion.MICROGRAPH_COORDS)

    if args.scale is not None:
        scale_coordinates(df, args.scale, inplace=True)
        scale_origins(df, args.scale, inplace=True)
        scale_magnification(df, args.scale, inplace=True)

    if args.scale_particles is not None:
        scale_origins(df, args.scale, inplace=True)
        scale_magnification(df, args.scale, inplace=True)

    if args.scale_coordinates is not None:
        scale_coordinates(df, args.scale_coordinates, inplace=True)

    if args.scale_origins is not None:
        scale_origins(df, args.scale_origins, inplace=True)

    if args.scale_magnification is not None:
        scale_magnification(df, args.scale_magnfication, inplace=True)

    if args.recenter:
        df = recenter(df, inplace=True)

    if args.zero_origins:
        df = zero_origins(df, inplace=True)

    if args.pick:
        df.drop(df.columns.difference(Relion.PICK_PARAMS), axis=1, inplace=True, errors="ignore")

    if args.subsample is not None and args.suffix != "":
        if args.subsample < 1:
            print("Specific integer sample size")
            return 1
        nsamplings = args.bootstrap if args.bootstrap is not None else df.shape[0] / np.int(args.subsample)
        inds = np.random.choice(df.shape[0], size=(nsamplings, np.int(args.subsample)),
                                replace=args.bootstrap is not None)
        for i, ind in enumerate(inds):
            write_star(os.path.join(args.output, os.path.basename(args.input[0])[:-5] + args.suffix + "_%d" % (i + 1)),
                       df.iloc[ind])

    if args.to_micrographs:
        gb = df.groupby(Relion.MICROGRAPH_NAME)
        mu = gb.mean()
        df = mu[[c for c in Relion.CTF_PARAMS + Relion.MICROSCOPE_PARAMS + [Relion.MICROGRAPH_NAME] if c in mu]].reset_index()

    if args.micrograph_range:
        df.set_index(Relion.MICROGRAPH_NAME, inplace=True)
        m, n = [int(tok) for tok in args.micrograph_range.split(",")]
        mg = df.index.unique().sort_values()
        outside = list(range(0,m)) + list(range(n,len(mg)))
        dfaux = df.loc[mg[outside]].reset_index()
        df = df.loc[mg[m:n]].reset_index()

    if args.micrograph_path is not None:
        df = replace_micrograph_path(df, args.micrograph_path, inplace=True)

    if args.min_separation is not None:
        gb = df.groupby(Relion.MICROGRAPH_NAME)
        dupes = []
        for n, g in gb:
            nb = query_connected(g[Relion.COORDS], args.min_separation / calculate_apix(df))
            dupes.extend(g.index[~np.isnan(nb)])
        dfaux = df.loc[dupes]
        df.drop(dupes, inplace=True)

    if args.merge_source is not None:
        if args.merge_fields is not None:
            if "," in args.merge_fields:
                args.merge_fields = args.merge_fields.split(",")
        else:
            print("Merge fields must be specified using --merge-fields")
            return 1
        if args.merge_key is not None:
            if "," in args.merge_key:
                args.merge_key = args.merge_key.split(",")
        merge_star = parse_star(args.merge_source, augment=args.augment)
        df = smart_merge(df, merge_star, fields=args.merge_fields, key=args.merge_key)

    if args.split_micrographs:
        dfs = split_micrographs(df)
        for mg in dfs:
            write_star(os.path.join(args.output, os.path.basename(mg)[:-4]) + args.suffix, dfs[mg])
        return 0

    if args.auxout is not None and dfaux is not None:
        write_star(args.auxout, dfaux, simplify=args.augment)

    if args.output is not None:
        write_star(args.output, df, simplify=args.augment)
    return 0


def smart_merge(s1, s2, fields, key=None):
    if key is None:
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
    if not inter.size:
        return None
    if Relion.IMAGE_NAME in inter:
        c = Counter(s1[Relion.IMAGE_NAME])
        shared = sum(c[i] for i in set(s2[Relion.IMAGE_NAME]))
        if shared > s1.shape[0] * 0.5:
            return Relion.IMAGE_NAME
        if UCSF.IMAGE_BASENAME in inter:
            c = Counter(s1[UCSF.IMAGE_BASENAME])
            shared = sum(c[i] for i in set(s2[UCSF.IMAGE_BASENAME]))
            if shared > s1.shape[0] * 0.5:
                return [UCSF.IMAGE_BASENAME, UCSF.IMAGE_INDEX]
    mgraph_coords = inter.intersection(Relion.MICROGRAPH_COORDS)
    if Relion.MICROGRAPH_NAME in mgraph_coords:
        c = Counter(s1[Relion.MICROGRAPH_NAME])
        shared = sum(c[i] for i in set(s2[Relion.MICROGRAPH_NAME]))
        can_merge_mgraph_name = Relion.MICROGRAPH_NAME in mgraph_coords and shared > s1.shape[0] * 0.5
        if can_merge_mgraph_name and mgraph_coords.intersection(Relion.COORDS).size:
            return Relion.MICROGRAPH_COORDS
        elif can_merge_mgraph_name:
            return Relion.MICROGRAPH_NAME
    return None


def is_particle_star(df):
    return df.columns.intersection([Relion.IMAGE_NAME] + Relion.COORDS).size


def calculate_apix(df):
    try:
        if df.ndim == 2:
            return 10000.0 * df.iloc[0][Relion.DETECTORPIXELSIZE] / df.iloc[0][Relion.MAGNIFICATION]
        elif df.ndim == 1:
            return 10000.0 * df[Relion.DETECTORPIXELSIZE] / df[Relion.MAGNIFICATION]
        else:
            raise ValueError
    except KeyError:
        return None


def select_classes(df, classes):
    clsfields = [f for f in df.columns if Relion.CLASS in f]
    if len(clsfields) == 0:
        raise RuntimeError("No class labels found")
    ind = df[clsfields[0]].isin(classes)
    if not np.any(ind):
        raise RuntimeError("Specified classes have no members")
    return df.loc[ind]


def split_micrographs(df):
    gb = df.groupby(Relion.MICROGRAPH_NAME)
    dfs = {}
    for g in gb:
        g[1].drop(Relion.MICROGRAPH_NAME, axis=1, inplace=True, errors="ignore")
        dfs[g[0]] = g[1]
    return dfs


def replace_micrograph_path(df, path, inplace=False):
    df = df if inplace else df.copy()
    df[Relion.MICROGRAPH_NAME] = df[Relion.MICROGRAPH_NAME].apply(
        lambda x: os.path.join(path, os.path.basename(x)))
    return df


def set_original_fields(df, inplace=False):
    df = df if inplace else df.copy()
    if Relion.IMAGE_NAME in df:
        df[Relion.IMAGE_ORIGINAL_NAME] = df[Relion.IMAGE_NAME]
    if UCSF.IMAGE_INDEX in df:
        df[UCSF.IMAGE_ORIGINAL_INDEX] = df[UCSF.IMAGE_INDEX]
    if UCSF.IMAGE_PATH in df:
        df[UCSF.IMAGE_ORIGINAL_PATH] = df[UCSF.IMAGE_PATH]
    return df


def all_same_class(df, inplace=False):
    vc = df[Relion.IMAGE_NAME].value_counts()
    n = vc.max()
    si = df.set_index([Relion.IMAGE_NAME, Relion.CLASS], inplace=inplace)
    vci = si.index.value_counts()
    si = si.loc[vci[vci == n].index].reset_index(inplace=inplace)
    return si


def recenter(df, inplace=False):
    df = df if inplace else df.copy()
    intoff = np.round(df[Relion.ORIGINS]).values
    diffxy = df[Relion.ORIGINS] - intoff
    df[Relion.COORDS] = df[Relion.COORDS] - intoff
    df[Relion.ORIGINS] = diffxy
    return df


def recenter_modf(df, inplace=False):
    df = df if inplace else df.copy()
    remxy, offsetxy = np.vectorize(modf)(df[Relion.ORIGINS])
    df[Relion.ORIGINS] = remxy
    df[Relion.COORDS] = df[Relion.COORDS] - offsetxy
    return df


def zero_origins(df, inplace=False):
    df = df if inplace else df.copy()
    df[Relion.COORDX] = df[Relion.COORDX] - df[Relion.ORIGINX]
    df[Relion.COORDY] = df[Relion.COORDY] - df[Relion.ORIGINY]
    df[Relion.ORIGINX] = 0
    df[Relion.ORIGINY] = 0
    return df


def scale_coordinates(df, factor, inplace=False):
    df = df if inplace else df.copy()
    df[Relion.COORDS] = df[Relion.COORDS] * factor
    return df


def scale_origins(df, factor, inplace=False):
    df = df if inplace else df.copy()
    df[Relion.ORIGINS] = df[Relion.ORIGINS] * factor
    return df


def scale_magnification(df, factor, inplace=False):
    df = df if inplace else df.copy()
    df[Relion.MAGNIFICATION] = df[Relion.MAGNIFICATION] * factor
    return df


def parse_star(starfile, keep_index=False, augment=False):
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
    df = pd.read_table(starfile, skiprows=ln, delimiter='\s+', header=None)
    df.columns = headers
    if augment:
        augment_star_ucsf(df, inplace=True)
    return df


def write_star(starfile, df, reindex=True, simplify=True):
    if not starfile.endswith(".star"):
        starfile += ".star"
    if simplify and len([c for c in df.columns if "ucsf" in c or "eman" in c]) > 0:
        df = simplify_star_ucsf(df)
    indexed = re.search("#\d+$", df.columns[0]) is not None  # Check first column for '#N' index.
    if reindex and not indexed:  # No index present, append consecutive indices to sorted headers.
        order = np.argsort(df.columns)
        names = [df.columns[idx] + " #%d" % (i + 1) for i, idx in enumerate(order)]
    elif reindex and indexed:  # Replace existing indices with consecutive indices after sorting headers.
        names = [c.split("#")[0].rstrip()for c in df.columns]
        order = np.argsort(names)
        names = [df.columns[idx] + " #%d" % (i + 1) for i, idx in enumerate(order)]
    else:
        order = np.arange(df.shape[1])
        names = df.columns
    with open(starfile, 'w') as f:
        f.write('\n')
        f.write("data_images" + '\n')
        f.write('\n')
        f.write("loop_" + '\n')
        for name in names:
            line = name + " \n"
            line = line if line.startswith('_') else '_' + line
            f.write(line)
    df[df.columns[order]].to_csv(starfile, mode='a', sep=' ', header=False, index=False)


def transform_star(df, r, t=None, inplace=False, rots=None, invert=False, rotate=True, adjust_defocus=False):
    """
    Transform particle angles and origins according to a rotation
    matrix (in radians) and an optional translation vector.
    The translation may also be given as the 4th column of a 3x4 matrix,
    or as a scalar distance to be applied along the axis of rotation.
    """
    assert (r.shape[0] == 3)
    if r.shape[1] == 4 and t is None:
        t = r[:, -1]
        r = r[:, :3]
    assert (r.shape == (3, 3))
    assert t is None or np.isscalar(t) or len(t) == 3

    if inplace:
        newstar = df
    else:
        newstar = df.copy()

    if rots is None:
        rots = [euler2rot(*np.deg2rad(row[1])) for row in df[Relion.ANGLES].iterrows()]

    if invert:
        r = r.T

    newrots = [ptcl.dot(r) for ptcl in rots]
    if rotate:
        angles = [np.rad2deg(rot2euler(q)) for q in newrots]
        newstar[Relion.ANGLES] = angles

    if t is not None and np.linalg.norm(t) > 0:
        if np.isscalar(t):
            if invert:
                tt = -np.vstack([np.squeeze(q[:, 2]) * t for q in rots])
            else:
                tt = np.vstack([np.squeeze(q[:, 2]) * t for q in newrots])
        else:
            if invert:
                tt = -np.vstack([q.dot(t) for q in rots])
            else:
                tt = np.vstack([q.dot(t) for q in newrots])
        newshifts = df[Relion.ORIGINS] + tt[:, :-1]
        newstar[Relion.ORIGINS] = newshifts
        if adjust_defocus:
            newstar[Relion.DEFOCUSU] += tt[:, -1] * calculate_apix(df)
            newstar[Relion.DEFOCUSV] += tt[:, -1] * calculate_apix(df)
            newstar[Relion.DEFOCUSANGLE] = np.rad2deg(np.arctan2(newstar[Relion.DEFOCUSV], newstar[Relion.DEFOCUSV]))

    return newstar


def augment_star_ucsf(df, inplace=True):
    df = df if inplace else df.copy()
    df.reset_index(inplace=True)
    if Relion.IMAGE_NAME in df:
        df[UCSF.IMAGE_INDEX], df[UCSF.IMAGE_PATH] = \
            df[Relion.IMAGE_NAME].str.split("@").str
        df[UCSF.IMAGE_INDEX] = pd.to_numeric(df[UCSF.IMAGE_INDEX]) - 1
        df[UCSF.IMAGE_BASENAME] = df[UCSF.IMAGE_PATH].apply(os.path.basename)

        if Relion.IMAGE_ORIGINAL_NAME not in df:
            df[Relion.IMAGE_ORIGINAL_NAME] = df[Relion.IMAGE_NAME]

    if Relion.IMAGE_ORIGINAL_NAME in df:
        df[UCSF.IMAGE_ORIGINAL_INDEX], df[UCSF.IMAGE_ORIGINAL_PATH] = \
            df[Relion.IMAGE_ORIGINAL_NAME].str.split("@").str
        df[UCSF.IMAGE_ORIGINAL_INDEX] = pd.to_numeric(df[UCSF.IMAGE_ORIGINAL_INDEX]) - 1
        df[UCSF.IMAGE_ORIGINAL_BASENAME] = df[UCSF.IMAGE_ORIGINAL_PATH].apply(os.path.basename)
    return df


def simplify_star_ucsf(df, inplace=True):
    df = df if inplace else df.copy()
    if UCSF.IMAGE_ORIGINAL_INDEX in df and UCSF.IMAGE_ORIGINAL_PATH in df:
        df[Relion.IMAGE_ORIGINAL_NAME] = df[UCSF.IMAGE_ORIGINAL_INDEX].map(
            lambda x: "%.6d" % (x + 1)).str.cat(df[UCSF.IMAGE_ORIGINAL_PATH],
                                                sep="@")
    if UCSF.IMAGE_INDEX in df and UCSF.IMAGE_PATH in df:
        df[Relion.IMAGE_NAME] = df[UCSF.IMAGE_INDEX].map(
            lambda x: "%.6d" % (x + 1)).str.cat(df[UCSF.IMAGE_PATH], sep="@")
    df.drop([c for c in df.columns if "ucsf" in c or "eman" in c],
            axis=1, inplace=True)
    if "index" in df.columns:
        df.set_index("index", inplace=True)
        df.sort_index(inplace=True, kind="mergesort")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--auxout", help="Auxilliary output .star file with deselected particles",
                        type=str)
    parser.add_argument("--augment", help="Always augment inputs and simplify outputs",
                        action="store_true")
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
    parser.add_argument("--merge-source", help="Source .star for merge")
    parser.add_argument("--merge-fields", help="Field(s) to merge", metavar="f1,f2...fN", type=str)
    parser.add_argument("--merge-key", help="Override merge key detection with explicit key field(s)",
                        metavar="f1,f2...fN", type=str)
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
    parser.add_argument("--min-separation", help="Minimum distance in Angstroms between particle coordinates", type=float)
    parser.add_argument("--scale", help="Factor to rescale particle coordinates, origins, and magnification",
                        type=float)
    parser.add_argument("--scale-particles",
                        help="Factor to rescale particle origins and magnification (rebin refined particles)",
                        type=float)
    parser.add_argument("--scale-coordinates", help="Factor to rescale particle coordinates",
                        type=float)
    parser.add_argument("--scale-origins", help="Factor to rescale particle origins",
                        type=float)
    parser.add_argument("--scale-magnification", help="Factor to rescale magnification (pixel size)",
                        type=float)
    parser.add_argument("--split-micrographs", help="Write separate output file for each micrograph",
                        action="store_true")
    parser.add_argument("--micrograph-range", help="Write micrographs with alphanumeric sort index [m, n) to output file",
                        metavar="m,n")
    parser.add_argument("--subsample", help="Randomly subsample remaining particles",
                        type=float, metavar="N")
    parser.add_argument("--subsample-micrographs", help="Randomly subsample micrographs",
                        type=float)
    parser.add_argument("--suffix", help="Suffix for multiple output files",
                        type=str, default="")
    parser.add_argument("--to-micrographs", help="Convert particles STAR to micrographs STAR",
                        action="store_true")
    parser.add_argument("--micrograph-path", help="Replacement path for micrographs")
    parser.add_argument("--transform",
                        help="Apply rotation matrix or 3x4 rotation plus translation matrix to particles (Numpy format)",
                        type=str)
    parser.add_argument("--invert-hand", help="Alter Euler angles to invert handedness of reconstruction",
                        action="store_true")
    parser.add_argument("input", help="Input .star file(s) or unquoted glob", nargs="*")
    parser.add_argument("output", help="Output .star file")
    sys.exit(main(parser.parse_args()))
