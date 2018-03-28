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
import argparse
import sys
import json
import numpy as np
import pandas as pd
from glob import glob
from pyem import star
from pyem.util import expmap
from pyem.util import rot2euler


general = {u'uid': None,
           u'split': "rlnRandomSubset",
           u'ctf_params.akv': "rlnVoltage",
           u'ctf_params.angast_deg': "rlnDefocusAngle",
           u'ctf_params.angast_rad': None,
           u'ctf_params.cs': "rlnSphericalAberration",
           u'ctf_params.detector_psize': "rlnDetectorPixelSize",
           u'ctf_params.df1': "rlnDefocusU",
           u'ctf_params.df2': "rlnDefocusV",
           u'ctf_params.mag': "rlnMagnification",
           u'ctf_params.phase_shift': "rlnPhaseShift",
           u'ctf_params.psize': None,
           u'ctf_params.wgh': "rlnAmplitudeContrast",
           u'data_input_relpath': "rlnImageName",
           u'data_input_idx': None}

model = {u'alignments.model.U': None,
         u'alignments.model.dr': None,
         u'alignments.model.dt': None,
         u'alignments.model.ess_R': None,
         u'alignments.model.ess_S': None,
         u'alignments.model.phiC': None,
         u'alignments.model.r.0': "rlnAngleRot",
         u'alignments.model.r.1': "rlnAngleTilt",
         u'alignments.model.r.2': "rlnAnglePsi",
         u'alignments.model.t.0': "rlnOriginX",
         u'alignments.model.t.1': "rlnOriginY"}


def main(args):
    meta = parse_metadata(args.input)  # Read cryosparc metadata file.
    meta["data_input_idx"] = ["%.6d" % (i + 1) for i in meta["data_input_idx"]]  # Reformat particle idx for Relion.

    if "data_input_relpath" not in meta.columns:
        if args.data_path is None:
            print("Data path missing, use --data-path to specify particle stack path")
            return 1
        meta["data_input_relpath"] = args.data_path

    meta["data_input_relpath"] = meta["data_input_idx"].str.cat(
        meta["data_input_relpath"], sep="@")  # Construct rlnImageName field.
    # Take care of trivial mappings.
    rlnheaders = [general[h] for h in meta.columns if h in general and general[h] is not None]
    df = meta[[h for h in meta.columns if h in general and general[h] is not None]].copy()
    df.columns = rlnheaders

    if "rlnRandomSubset" in df.columns:
        df["rlnRandomSubset"] = df["rlnRandomSubset"].apply(lambda x: ord(x) - 64)

    if "rlnPhaseShift" in df.columns:
        df["rlnPhaseShift"] = np.rad2deg(df["rlnPhaseShift"])

    # Class assignments and other model parameters.
    phic = meta[[h for h in meta.columns if "phiC" in h]]  # Posterior probability over class assignments.
    if len(phic.columns) > 0:  # Check class assignments exist in input.
        # phic.columns = [int(h[21]) for h in meta.columns if "phiC" in h]
        phic.columns = range(len(phic.columns))
        cls = phic.idxmax(axis=1)
        for p in model:
            if model[p] is not None:
                pspec = p.split("model")[1]
                param = meta[[h for h in meta.columns if pspec in h]]
                if len(param.columns) > 0:
                    param.columns = phic.columns
                    df[model[p]] = param.lookup(param.index, cls)
        df["rlnClassNumber"] = cls + 1  # Add one for Relion indexing.
    else:
        for p in model:
            if model[p] is not None and p in meta.columns:
                df[model[p]] = meta[p]
        df["rlnClassNumber"] = 1

    if args.cls is not None:
        df = star.select_classes(df, args.cls)

    # Convert axis-angle representation to Euler angles (degrees).
    if df.columns.intersection(star.Relion.ANGLES).size == len(star.Relion.ANGLES):
        df[star.Relion.ANGLES] = np.rad2deg(
            df[star.Relion.ANGLES].apply(lambda x: rot2euler(expmap(x)),
                                         axis=1, raw=True, broadcast=True))

    if args.minphic is not None:
        mask = np.all(phic < args.minphic, axis=1)
        if args.keep_bad:
            df.loc[mask, "rlnClassNumber"] = 0
        else:
            df.drop(df[mask].index, inplace=True)

    if args.copy_micrograph_coordinates is not None:
        coord_star = pd.concat(
            (star.parse_star(inp, keep_index=False) for inp in
             glob(args.copy_micrograph_coordinates)), join="inner")
        df = star.smart_merge(df, coord_star, fields=star.Relion.MICROGRAPH_COORDS)

    if args.transform is not None:
        r = np.array(json.loads(args.transform))
        df = star.transform_star(df, r, inplace=True)

    # Write Relion .star file with correct headers.
    star.write_star(args.output, df, reindex=True)
    return 0


def parse_metadata(csvfile):
    with open(csvfile, 'rU') as f:
        lines = enumerate(f)
        idx = -1
        headers = None
        for i, l in lines:
            if l.startswith("_header"):
                _, headers = next(lines)
                headers = headers.rstrip().split(",")
            if l.startswith("_dtypes"):
                _i, dtypes = next(lines)
                dtypes = dtypes.rstrip().split(",")
                idx = _i + 1
                break
    meta = pd.read_csv(csvfile, skiprows=idx, header=None, skip_blank_lines=True)
    if headers is None:
        return None
    meta.columns = headers
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input Cryosparc metadata .csv file")
    parser.add_argument("output", help="Output .star file")
    parser.add_argument("--class", help="Keep this class in output, may be passed multiple times",
                        action="append", type=int, dest="cls")
    parser.add_argument("--minphic", help="Minimum posterior probability for class assignment", type=float)
    parser.add_argument("--keep-bad", help="Keep low-confidence particles and assign them to class 0 (incompatible with Relion)", action="store_true")
    parser.add_argument("--data-path", help="Path to single particle stack", type=str)
    parser.add_argument("--copy-micrograph-coordinates",
                        help="Source for micrograph paths and particle coordinates (file or quoted glob)",
                        type=str)
    parser.add_argument("--transform",
                        help="Apply rotation matrix or 3x4 rotation plus translation matrix to particles (Numpy format)",
                        type=str)
    sys.exit(main(parser.parse_args()))
