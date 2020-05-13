#!/usr/bin/env python
# Copyright (C) 2017-2018 Daniel Asarnow
# University of California, San Francisco
#
# Efficiently combine image stacks.
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
import numpy as np
import os.path
import pandas as pd
import sys
from pyem import metadata
from pyem import mrc
from pyem import star


def main(args):
    log = logging.getLogger('root')
    hdlr = logging.StreamHandler(sys.stdout)
    log.addHandler(hdlr)
    log.setLevel(logging.getLevelName(args.loglevel.upper()))
    # apix = args.apix = hdr["xlen"] / hdr["nx"]

    for fn in args.input:
        if not (fn.endswith(".star") or fn.endswith(".mrcs") or
                fn.endswith(".mrc") or fn.endswith(".par")):
            log.error("Only .star, .mrc, .mrcs, and .par files supported")
            return 1

    first_ptcl = 0
    dfs = []
    with mrc.ZSliceWriter(args.output) as writer:
        for fn in args.input:
            if fn.endswith(".star"):
                df = star.parse_star(fn, augment=True)
                if args.cls is not None:
                    df = star.select_classes(df, args.cls)
                star.set_original_fields(df, inplace=True)
                if args.resort:
                    df = df.sort_values([star.UCSF.IMAGE_ORIGINAL_PATH,
                                         star.UCSF.IMAGE_ORIGINAL_INDEX])
                for idx, row in df.iterrows():
                    if args.stack_path is not None:
                        input_stack_path = os.path.join(args.stack_path, row[star.UCSF.IMAGE_ORIGINAL_PATH])
                    else:
                        input_stack_path = row[star.UCSF.IMAGE_ORIGINAL_PATH]
                    with mrc.ZSliceReader(input_stack_path) as reader:
                        i = row[star.UCSF.IMAGE_ORIGINAL_INDEX]
                        writer.write(reader.read(i))
            elif fn.endswith(".par"):
                if args.stack_path is None:
                    log.error(".par file input requires --stack-path")
                    return 1
                df = metadata.par2star(metadata.parse_fx_par(fn), data_path=args.stack_path)
                # star.set_original_fields(df, inplace=True)  # Redundant.
                star.augment_star_ucsf(df)
            elif fn.endswith(".csv"):
                return 1
            elif fn.endswith(".cs"):
                return 1
            else:
                if fn.endswith(".mrcs"):
                    with mrc.ZSliceReader(fn) as reader:
                        for img in reader:
                            writer.write(img)
                        df = pd.DataFrame(
                            {star.UCSF.IMAGE_ORIGINAL_INDEX: np.arange(reader.nz)})
                    df[star.UCSF.IMAGE_ORIGINAL_PATH] = fn
                else:
                    print("Unrecognized input file type")
                    return 1
            if args.star is not None:
                df[star.UCSF.IMAGE_INDEX] = np.arange(first_ptcl,
                                                      first_ptcl + df.shape[0])
                if args.abs_path:
                    df[star.UCSF.IMAGE_PATH] = writer.path
                else:
                    df[star.UCSF.IMAGE_PATH] = os.path.relpath(writer.path, os.path.dirname(args.star))
                df["index"] = df[star.UCSF.IMAGE_INDEX]
                star.simplify_star_ucsf(df)
                dfs.append(df)
            first_ptcl += df.shape[0]

    if args.star is not None:
        df = pd.concat(dfs, join="inner")
        # df = pd.concat(dfs)
        # df = df.dropna(df, axis=1, how="any")
        if not args.relion2:  # Relion 3.1 style output.
            df = star.remove_deprecated_relion2(df, inplace=True)
            star.write_star(args.star, df, resort_records=False, optics=True)
        else:
            df = star.remove_new_relion31(df, inplace=True)
            star.write_star(args.star, df, resort_records=False, optics=False)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                        help="Input image(s), stack(s) and/or .star file(s)",
                        nargs="*")
    parser.add_argument("output", help="Output stack")
    parser.add_argument("--abs-path", "-a", help="Don't solve relative path between star and stack", action="store_true")
    parser.add_argument("--star", "-s", help="Optional composite .star output file")
    parser.add_argument("--stack-path", help="(PAR file only) Particle stack for input file")
    parser.add_argument("--class", "-c", help="Keep this class in output, may be passed multiple times",
                        action="append", type=int, dest="cls")
    parser.add_argument("--relion2", "-r2", action="store_true")
    parser.add_argument("--loglevel", "-l", type=str, default="WARNING",
                        help="Logging level and debug output")
    parser.add_argument("--resort", help="Natural sort the particle image names", action="store_true")
    sys.exit(main(parser.parse_args()))
