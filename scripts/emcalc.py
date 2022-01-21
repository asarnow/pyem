#!/usr/bin/env python
# Copyright (C) 2020 Daniel Asarnow
# University of California, San Francisco
#
# Command line calculator for EM data using numexpr.
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
import numexpr as ne
import numpy as np
import sys
from pyem.mrc import read
from pyem.mrc import write
from pyem import vop
from string import ascii_lowercase


def main(args):
    log = logging.getLogger(__name__)
    hdlr = logging.StreamHandler(sys.stdout)
    log.addHandler(hdlr)
    log.setLevel(logging.getLevelName(args.loglevel.upper()))
    
    data = {}
    hdr = {}
    for i,inp in enumerate(args.input[1:]):
        d, h = read(inp, inc_header=True)
        if args.normalize:
            d = vop.normalize(d)
        data[ascii_lowercase[i]] = d
        hdr[ascii_lowercase[i]] = h
    if args.eval:
        final = eval(args.input[0], globals(), data)
    else:
        final = ne.evaluate(args.input[0], local_dict=data)

    if args.apix is None:
        args.apix = hdr[ascii_lowercase[0]]['xlen'] / hdr[ascii_lowercase[0]]['nx']
    
    write(args.output, final.astype(np.single), psz=args.apix)
    return 0

def _main_():
    import argparse
    parser = argparse.ArgumentParser(description="Use equals sign when passing arguments with negative numbers.")
    parser.add_argument("input", help="Input volume (MRC file)", nargs="*")
    parser.add_argument("output", help="Output volume (MRC file)")
    parser.add_argument("--apix", help="Output pixel size")
    parser.add_argument("--normalize", "-n", help="Normalize all input maps", action="store_true")
    parser.add_argument("--eval", "-e", help="Use eval builtin instead of numexpr.evaluate", action="store_true")
    parser.add_argument("--loglevel", "-l", type=str, default="WARNING", help="Logging level and debug output")
    sys.exit(main(parser.parse_args()))

if __name__ == "__main__":
    _main_()

