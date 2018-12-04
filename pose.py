#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# Copyright (C) 2018 Daniel Asarnow
# University of California, San Francisco
#
# Program for single-particle pose analysis in electron microscopy.
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
import os
import sys
from pyem import mrc
from pyem import plot
from pyem import star
from pyem import util
from pyem import vop


def main(args):
    log = logging.getLogger('root')
    hdlr = logging.StreamHandler(sys.stdout)
    log.addHandler(hdlr)
    log.setLevel(logging.getLevelName(args.loglevel.upper()))

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    dfs = [star.parse_star(inp, keep_index=False) for inp in args.input]
    size_err = np.array(args.input)[np.where(~np.equal([df.shape[0] for df in dfs[1:]], dfs[0].shape[0]))]
    if len(size_err) > 0:
        log.error("All files must have same number of particles. Offending files:\n%s" % ", ".join(size_err))
        return 1

    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="*")
    parser.add_argument("output")
    parser.add_argument("--sample", type=int)
    parser.add_argument("--threads", "-j", type=int)
    sys.exit(main(parser.parse_args()))
