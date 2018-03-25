#!/usr/bin/env python2.7
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
import pandas as pd
import sys
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
                fn.endswith(".mrc")):
            log.error("Only .star, .mrc, and .mrcs files supported")
            return 1

    first_ptcl = 0
    dfs = []
    with mrc.ZSliceWriter(args.output) as writer:
        for fn in args.input:
            if fn.endswith(".star"):
                df = star.parse_star(fn, keep_index=False)
                star.augment_star_ucsf(df)
                df = df.sort_values([star.UCSF.IMAGE_ORIGINAL_PATH,
                                     star.UCSF.IMAGE_ORIGINAL_INDEX])
                gb = df.groupby(star.UCSF.IMAGE_ORIGINAL_PATH)
                for name, g in gb:
                    with mrc.ZSliceReader(name) as reader:
                        for i in g[star.UCSF.IMAGE_ORIGINAL_INDEX].values:
                            writer.write(reader.read(i))
            else:
                with mrc.ZSliceReader(fn) as reader:
                    for img in reader:
                        writer.write(img)
                    df = pd.DataFrame(
                        {star.UCSF.IMAGE_ORIGINAL_INDEX: np.arange(reader.nz)})
                df[star.UCSF.IMAGE_ORIGINAL_PATH] = fn

            if args.star is not None:
                df[star.UCSF.IMAGE_INDEX] = np.arange(first_ptcl,
                                                      first_ptcl + df.shape[0])
                df[star.UCSF.IMAGE_PATH] = writer.path
                df["index"] = df[star.UCSF.IMAGE_INDEX]
                star.simplify_star_ucsf(df)
                dfs.append(df)
            first_ptcl += df.shape[0]

            if args.star is not None:
                df = pd.concat(dfs, axis=1, join="inner")
                star.write_star(args.star, df)

    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                        help="Input image(s), stack(s) and/or .star file(s)",
                        nargs="*")
    parser.add_argument("output", help="Output stack")
    parser.add_argument("--star", help="Optional composite .star output file")
    parser.add_argument("--loglevel", "-l", type=str, default="WARNING",
                        help="Logging level and debug output")
    sys.exit(main(parser.parse_args()))
