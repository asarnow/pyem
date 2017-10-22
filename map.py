#!/usr/bin/env python2.7
# Copyright (C) 2017 Daniel Asarnow
# University of California, San Francisco
#
# Simple map modification utility.
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
import numpy as np
import sys
from pyem.mrc import read
from pyem.mrc import write


def main(args):
    data, hdr = read(args.input, inc_header=True)
    if args.normalize is not None:
        if args.normalize:
            ref, refhdr = read(args.input, inc_header=True)
            sigma = np.std(ref)
        else:
            sigma = np.std(data)

        mu = np.mean(data)
        final = (data - mu) / std

    if args.apix is None:
        args.apix = hdr["nx"] / hdr["xlen"]

    write(args.output, final, psz=args.apix)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input volume MRC file")
    parser.add_argument("output", help="Output mask MRC file")
    parser.add_argument("--apix", "--angpix", "-a", help="Pixel size in Angstroms")
    parser.add_argument("--normalize", "-n", help="Convert map densities to Z-scores",
                        nargs="?", const=False, metavar="reference")
    sys.exit(main(parser.parse_args()))

