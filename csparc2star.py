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
from pyem import metadata
from pyem import star


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
    if args.input.endswith(".cs"):
        df = metadata.parse_cryosparc_2_cs(args.input, args.minphic)
    else:
        meta = metadata.parse_cryosparc_065_csv(args.input)  # Read cryosparc metadata file.
        df = metadata.cryosparc_065_csv2star(meta, args.minphic)

    if args.cls is not None:
        df = star.select_classes(df, args.cls)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input Cryosparc metadata .csv (v0.6.5) or .cs (v2+) file")
    parser.add_argument("output", help="Output .star file")
    parser.add_argument("--class", help="Keep this class in output, may be passed multiple times",
                        action="append", type=int, dest="cls")
    parser.add_argument("--minphic", help="Minimum posterior probability for class assignment", type=float, default=0)
    parser.add_argument("--data-path", help="Path to single particle stack", type=str)
    parser.add_argument("--copy-micrograph-coordinates",
                        help="Source for micrograph paths and particle coordinates (file or quoted glob)",
                        type=str)
    parser.add_argument("--transform",
                        help="Apply rotation matrix or 3x4 rotation plus translation matrix to particles (Numpy format)",
                        type=str)
    sys.exit(main(parser.parse_args()))
