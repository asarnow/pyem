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
from builtins import range
import glob
import sys
from EMAN2 import EMData
from EMAN2 import EMUtil
from pyem.star import parse_star


def main(args):
    for fn in args.input:
        if fn.endswith(".star"):
            star = parse_star(fn, keep_index=False)
            for p in star["rlnImageName"]:
                stack = p.split("@")[1]
                idx = int(p.split("@")[0]) - 1
                img = EMData(stack, idx)
                img.append_image(args.output)
        else:
            n = EMUtil.get_image_count(fn)
            for i in range(n):
                img = EMData(fn, i)
                img.append_image(args.output)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input particle image(s), stack(s) and/or .star file(s)", nargs="*")
    parser.add_argument("output", help="Output stack")
    sys.exit(main(parser.parse_args()))

