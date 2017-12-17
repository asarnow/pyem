#!/usr/bin/env python2.7
# Copyright (C) 2017 Daniel Asarnow
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
from __future__ import print_function
import os
import os.path
import sys
from pyem.mrc import append
from pyem.mrc import read
from pyem.mrc import write
from pyem.star import parse_star


def main(args):
    if os.path.exists(args.output):
        os.remove(args.output)
    first = True
    for fn in args.input:
        if fn.endswith(".star"):
            star = parse_star(fn, keep_index=False)
            for p in star["rlnImageName"]:
                stack = p.split("@")[1]
                idx = int(p.split("@")[0]) - 1
                try:
                    img = read(stack, idx)
                    if first:
                        write(args.output, img)
                        first = False
                    else:
                        append(args.output, img)
                except Exception:
                    print("Error at %s" % p)
        else:
            data, hdr = read(fn, inc_header=True)
            apix = args.apix = hdr["xlen"] / hdr["nx"]
            if first:
                write(args.output, data, psz=apix)
                first = False
            else:
                append(args.output, data)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input particle image(s), stack(s) and/or .star file(s)", nargs="*")
    parser.add_argument("output", help="Output stack")
    sys.exit(main(parser.parse_args()))

