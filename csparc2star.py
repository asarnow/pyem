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
import argparse
import json
import logging
import sys
import numpy as np
import pandas as pd
from glob import glob
from pyem import metadata
from pyem import star


def main(args):
    log = logging.getLogger('root')
    hdlr = logging.StreamHandler(sys.stdout)
    log.addHandler(hdlr)
    log.setLevel(logging.getLevelName(args.loglevel.upper()))

    if args.input[0].endswith(".cs"):
        log.debug("Detected CryoSPARC 2+ .cs file")
        cs = np.load(args.input[0])
        try:
            df = metadata.parse_cryosparc_2_cs(cs, passthroughs=args.input[1:], minphic=args.minphic, boxsize=args.boxsize, swapxy=args.swapxy)
        except (KeyError, ValueError) as e:
            log.error(e, exc_info=True)
            log.error("Required fields could not be mapped. Are you using the right input file(s)?")
            return 1
    else:
        log.debug("Detected CryoSPARC 0.6.5 .csv file")
        if len(args.input) > 1:
            log.error("Only one file at a time supported for CryoSPARC 0.6.5 .csv format")
            return 1
        meta = metadata.parse_cryosparc_065_csv(args.input[0])  # Read cryosparc metadata file.
        df = metadata.cryosparc_065_csv2star(meta, args.minphic)

    if args.cls is not None:
        df = star.select_classes(df, args.cls)

    if args.copy_micrograph_coordinates is not None:
        df = star.augment_star_ucsf(df, inplace=True)
        coord_star = pd.concat(
            (star.parse_star(inp, keep_index=False, augment=True) for inp in
             glob(args.copy_micrograph_coordinates)), join="inner")
        key = star.merge_key(df, coord_star)
        log.debug("Coordinates merge key: %s" % key)
        if args.cached or key == star.Relion.IMAGE_NAME:
            fields = star.Relion.MICROGRAPH_COORDS
        else:
            fields = star.Relion.MICROGRAPH_COORDS + [star.UCSF.IMAGE_INDEX, star.UCSF.IMAGE_PATH]
        df = star.smart_merge(df, coord_star, fields=fields, key=key)
        star.simplify_star_ucsf(df)

    if args.micrograph_path is not None:
        df = star.replace_micrograph_path(df, args.micrograph_path, inplace=True)

    if args.transform is not None:
        r = np.array(json.loads(args.transform))
        df = star.transform_star(df, r, inplace=True)

    df = star.check_defaults(df, inplace=True)

    if args.relion2:
        df = star.remove_new_relion31(df, inplace=True)
        star.write_star(args.output, df, resort_records=True, optics=False)
    else:
        df = star.remove_deprecated_relion2(df, inplace=True)
        star.write_star(args.output, df, resort_records=True, optics=True)

    log.info("Output fields: %s" % ", ".join(df.columns))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Cryosparc metadata .csv (v0.6.5) or .cs (v2+) files", nargs="*")
    parser.add_argument("output", help="Output .star file")
    parser.add_argument("--boxsize", help="Cryosparc refinement box size (if different from particles)", type=float)
    # parser.add_argument("--passthrough", "-p", help="List file required for some Cryosparc 2+ job types")
    parser.add_argument("--class", help="Keep this class in output, may be passed multiple times",
                        action="append", type=int, dest="cls")
    parser.add_argument("--minphic", help="Minimum posterior probability for class assignment", type=float, default=0)
    parser.add_argument("--stack-path", help="Path to single particle stack", type=str)
    parser.add_argument("--micrograph-path", help="Replacement path for micrographs")
    parser.add_argument("--copy-micrograph-coordinates",
                        help="Source for micrograph paths and particle coordinates (file or quoted glob)",
                        type=str)
    parser.add_argument("--swapxy", help="Swap X and Y axes when converting particle coordinates", action="store_true")
    parser.add_argument("--cached", help="Keep paths from the Cryosparc 2+ cache when merging coordinates",
                        action="store_true")
    parser.add_argument("--transform",
                        help="Apply rotation matrix or 3x4 rotation plus translation matrix to particles (Numpy format)",
                        type=str)
    parser.add_argument("--relion2", "-r2", help="Relion 2 compatible outputs", action="store_true")
    parser.add_argument("--loglevel", "-l", type=str, default="WARNING", help="Logging level and debug output")
    sys.exit(main(parser.parse_args()))
