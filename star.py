#!/usr/bin/env python2.7
# Copyright (C) 2018 Daniel Asarnow
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
from six import iteritems
import glob
import json
import numpy as np
import os.path
import pandas as pd
import sys
from pyem import algo
from pyem import star


def main(args):
    if args.info:
        args.input.append(args.output)

    df = pd.concat((star.parse_star(inp, augment=args.augment) for inp in args.input), join="inner")

    dfaux = None

    if args.cls is not None:
        df = star.select_classes(df, args.cls)

    if args.info:
        if star.is_particle_star(df) and star.Relion.CLASS in df.columns:
            c = df[star.Relion.CLASS].value_counts()
            print("%s particles in %d classes" % ("{:,}".format(df.shape[0]), len(c)))
            print("    ".join(['%d: %s (%.2f %%)' % (i, "{:,}".format(s), 100. * s / c.sum())
                               for i, s in iteritems(c.sort_index())]))
        elif star.is_particle_star(df):
            print("%s particles" % "{:,}".format(df.shape[0]))
        if star.Relion.MICROGRAPH_NAME in df.columns:
            mgraphcnt = df[star.Relion.MICROGRAPH_NAME].value_counts()
            print("%s micrographs, %s +/- %s particles per micrograph" %
                  ("{:,}".format(len(mgraphcnt)), "{:,.3f}".format(np.mean(mgraphcnt)),
                   "{:,.3f}".format(np.std(mgraphcnt))))
        try:
            print("%f A/px (%sX magnification)" % (star.calculate_apix(df), "{:,.0f}".format(df[star.Relion.MAGNIFICATION][0])))
        except KeyError:
            pass
        if len(df.columns.intersection(star.Relion.ORIGINS3D)) > 0:
            print("Largest shift is %f pixels" %
                  np.max(np.abs(df[df.columns.intersection(star.Relion.ORIGINS3D)].values)))
        return 0

    if args.drop_angles:
        df.drop(star.Relion.ANGLES, axis=1, inplace=True, errors="ignore")

    if args.drop_containing is not None:
        containing_fields = [f for q in args.drop_containing for f in df.columns if q in f]
        if args.invert:
            containing_fields = df.columns.difference(containing_fields)
        df.drop(containing_fields, axis=1, inplace=True, errors="ignore")

    if args.offset_group is not None:
        df[star.Relion.GROUPNUMBER] += args.offset_group

    if args.subsample_micrographs is not None:
        if args.bootstrap is not None:
            print("Only particle sampling allows bootstrapping")
            return 1
        mgraphs = df[star.Relion.MICROGRAPH_NAME].unique()
        if args.subsample_micrographs < 1:
            args.subsample_micrographs = np.int(max(np.round(args.subsample_micrographs * len(mgraphs)), 1))
        else:
            args.subsample_micrographs = np.int(args.subsample_micrographs)
        ind = np.random.choice(len(mgraphs), size=args.subsample_micrographs, replace=False)
        mask = df[star.Relion.MICROGRAPH_NAME].isin(mgraphs[ind])
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
        angle_star = star.parse_star(args.copy_angles, augment=args.augment)
        df = star.smart_merge(df, angle_star, fields=star.Relion.ANGLES)

    if args.copy_alignments is not None:
        align_star = star.parse_star(args.copy_alignments, augment=args.augment)
        df = star.smart_merge(df, align_star, fields=star.Relion.ALIGNMENTS)

    if args.copy_reconstruct_images is not None:
        recon_star = star.parse_star(args.copy_reconstruct_images, augment=args.augment)
        df[star.Relion.RECONSTRUCT_IMAGE_NAME] = recon_star[star.Relion.IMAGE_NAME]

    if args.transform is not None:
        if args.transform.count(",") == 2:
            r = star.euler2rot(*np.deg2rad([np.double(s) for s in args.transform.split(",")]))
        else:
            r = np.array(json.loads(args.transform))
        df = star.transform_star(df, r, inplace=True)

    if args.invert_hand:
        df[star.Relion.ANGLEROT] = -df[star.Relion.ANGLEROT]
        df[star.Relion.ANGLETILT] = 180 - df[star.Relion.ANGLETILT]

    if args.copy_paths is not None:
        path_star = star.parse_star(args.copy_paths)
        df[star.Relion.IMAGE_NAME] = path_star[star.Relion.IMAGE_NAME]

    if args.copy_ctf is not None:
        ctf_star = pd.concat((star.parse_star(inp, augment=args.augment) for inp in glob.glob(args.copy_ctf)), join="inner")
        df = star.smart_merge(df, ctf_star, star.Relion.CTF_PARAMS)

    if args.copy_micrograph_coordinates is not None:
        coord_star = pd.concat(
            (star.parse_star(inp, augment=args.augment) for inp in glob.glob(args.copy_micrograph_coordinates)), join="inner")
        df = star.smart_merge(df, coord_star, fields=star.Relion.MICROGRAPH_COORDS)

    if args.scale is not None:
        star.scale_coordinates(df, args.scale, inplace=True)
        star.scale_origins(df, args.scale, inplace=True)
        star.scale_magnification(df, args.scale, inplace=True)

    if args.scale_particles is not None:
        star.scale_origins(df, args.scale_particles, inplace=True)
        star.scale_magnification(df, args.scale_particles, inplace=True)

    if args.scale_coordinates is not None:
        star.scale_coordinates(df, args.scale_coordinates, inplace=True)

    if args.scale_origins is not None:
        star.scale_origins(df, args.scale_origins, inplace=True)

    if args.scale_magnification is not None:
        star.scale_magnification(df, args.scale_magnfication, inplace=True)

    if args.recenter:
        df = star.recenter(df, inplace=True)

    if args.zero_origins:
        df = star.zero_origins(df, inplace=True)

    if args.pick:
        df.drop(df.columns.difference(star.Relion.PICK_PARAMS), axis=1, inplace=True, errors="ignore")

    if args.subsample is not None and args.suffix != "":
        if args.subsample < 1:
            print("Specific integer sample size")
            return 1
        nsamplings = args.bootstrap if args.bootstrap is not None else df.shape[0] / np.int(args.subsample)
        inds = np.random.choice(df.shape[0], size=(nsamplings, np.int(args.subsample)),
                                replace=args.bootstrap is not None)
        for i, ind in enumerate(inds):
            star.write_star(os.path.join(args.output, os.path.basename(args.input[0])[:-5] + args.suffix + "_%d" % (i + 1)),
                       df.iloc[ind])

    if args.to_micrographs:
        gb = df.groupby(star.Relion.MICROGRAPH_NAME)
        mu = gb.mean()
        df = mu[[c for c in star.Relion.CTF_PARAMS + star.Relion.MICROSCOPE_PARAMS + [star.Relion.MICROGRAPH_NAME] if
                 c in mu]].reset_index()

    if args.micrograph_range:
        df.set_index(star.Relion.MICROGRAPH_NAME, inplace=True)
        m, n = [int(tok) for tok in args.micrograph_range.split(",")]
        mg = df.index.unique().sort_values()
        outside = list(range(0, m)) + list(range(n, len(mg)))
        dfaux = df.loc[mg[outside]].reset_index()
        df = df.loc[mg[m:n]].reset_index()

    if args.micrograph_path is not None:
        df = star.replace_micrograph_path(df, args.micrograph_path, inplace=True)

    if args.min_separation is not None:
        gb = df.groupby(star.Relion.MICROGRAPH_NAME)
        dupes = []
        for n, g in gb:
            nb = algo.query_connected(g[star.Relion.COORDS], args.min_separation / star.calculate_apix(df))
            dupes.extend(g.index[~np.isnan(nb)])
        dfaux = df.loc[dupes]
        df.drop(dupes, inplace=True)

    if args.merge_source is not None:
        if args.merge_fields is not None:
            if "," in args.merge_fields:
                args.merge_fields = args.merge_fields.split(",")
            else:
                args.merge_fields = [args.merge_fields]
        else:
            print("Merge fields must be specified using --merge-fields")
            return 1
        if args.merge_key is not None:
            if "," in args.merge_key:
                args.merge_key = args.merge_key.split(",")
        merge_star = star.parse_star(args.merge_source, augment=args.augment)
        df = star.smart_merge(df, merge_star, fields=args.merge_fields, key=args.merge_key)

    if args.split_micrographs:
        dfs = star.split_micrographs(df)
        for mg in dfs:
            star.write_star(os.path.join(args.output, os.path.basename(mg)[:-4]) + args.suffix, dfs[mg])
        return 0

    if args.auxout is not None and dfaux is not None:
        star.write_star(args.auxout, dfaux, simplify=args.augment_output)

    if args.output is not None:
        star.write_star(args.output, df, simplify=args.augment_output)
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--auxout", help="Auxilliary output .star file with deselected particles",
                        type=str)
    parser.add_argument("--augment", help="Always augment inputs",
                        action="store_true")
    parser.add_argument("--augment-output", help="Write augmented .star files with non-standard fields", action="store_false")
    parser.add_argument("--bootstrap", help="Sample with replacement when creating multiple outputs",
                        type=int, default=None)
    parser.add_argument("--class", help="Keep this class in output, may be passed multiple times",
                        action="append", type=int, dest="cls")
    parser.add_argument("--copy-angles",
                        help="Source for particle Euler angles (must align exactly with input .star file)",
                        type=str)
    parser.add_argument("--copy-alignments", help="Source for alignment parameters (angles and shifts)")
    parser.add_argument("--copy-ctf", help="Source for CTF parameters (file or quoted glob)")
    parser.add_argument("--copy-micrograph-coordinates", help="Source for micrograph paths and particle coordinates (file or quoted glob)",
                        type=str)
    parser.add_argument("--copy-paths", help="Source for particle paths (must align exactly with input .star file)",
                        type=str)
    parser.add_argument("--copy-reconstruct-images", help="Source for rlnReconstructImage (must align exactly with input .star file)")
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
