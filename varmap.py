#!/usr/bin/env python2.7
# Copyright (C) 2017 Daniel Asarnow
# University of California, San Francisco
#
# Simple program for computing mean and variance of maps.
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
import numpy as np
import sys
from pyem import mrc


def main(args):
    x = mrc.read(args.input[0])
    sigma = np.zeros(x.shape)
    mu = x.copy()
    for i, f in enumerate(args.input[1:]):
        x = mrc.read(f)
        olddif = x - mu
        mu += (x - mu) / (i + 1)
        sigma += olddif * (x - mu)
    sigma_sq = np.power(sigma, 2)
    mrc.write(args.output, sigma_sq)
    if args.mean is not None:
        mrc.write(args.mean, mu)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input map path(s)", nargs="*")
    parser.add_argument("output", help="Variance map output path")
    parser.add_argument("--mean", help="Mean map output path")
    sys.exit(main(parser.parse_args()))
