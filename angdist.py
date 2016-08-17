#!/usr/bin/env python
# Copyright (C) 2016 Daniel Asarnow
# University of California, San Francisco
#
# Program for plotting angular distribution.
# See help text and README file for more information.
#
# Program for projection subtraction in electron microscopy.
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
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from star import parse_star


def main(args):
    star = parse_star(args.input)

    if args.cmap not in plt.colormaps():
        print("Colormap " + args.cmap + " is not available")
        print("Use one of: " + ", ".join(plt.colormaps()))
        return 1

    xfields = [f for f in star.columns if "Tilt" in f]
    if len(xfields) == 0:
        print("No tilt angle found")
        return 1
    if args.psi:
        yfields = [f for f in star.columns if "Psi" in f]
        if len(yfields) == 0:
            print("No psi angle found")
            return 1
    else:
        yfields = [f for f in star.columns if "Rot" in f]
        if len(yfields) == 0:
            print("No rot angle found")
            return 1

    if args.cls is not None:
        clsfields = [f for f in star.columns if "ClassNumber" in f]
        if len(clsfields) == 0:
            print("No class labels found")
            return 1
        if args.cls > 0:
            ind = star[clsfields[0]] == args.cls
            if not np.any(ind):
                print("Specified class has no members")
                return 1
            xdata = star.loc[ind][xfields[0]]
            ydata = star.loc[ind][yfields[0]]
        else:
            raise NotImplementedError("Class ranges are not yet supported")
    else:
        xdata = star[xfields[0]]
        ydata = star[yfields[0]]

    if args.subplot is not None:
        raise NotImplementedError("Sublots are not yet supported")

    h, x, y = np.histogram2d(xdata, ydata, bins=args.samples, normed=True)
    xc = (x[:-1] + x[1:]) / 2
    yc = (y[:-1] + y[1:]) / 2
    coords = np.array([(xi, yi) for xi in xc for yi in yc])
    theta = np.deg2rad(coords[:, 0])
    r = coords[:, 1]
    area = h.flat / np.max(h) * args.scale
    colors = h.flat

    plt.figure(figsize=(args.figsize, args.figsize), dpi=args.dpi)
    plt.subplot(111, polar=True)
    c = plt.scatter(theta, r, c=colors, s=area, cmap=args.cmap)
    c.set_alpha(args.alpha)
    if args.full_circle:
        c = plt.scatter(theta + np.pi, r, c=colors, s=area, cmap=args.cmap)
        c.set_alpha(args.alpha)

    # plt.xlim((0, 180))

    if args.rmax is None:
        if np.max(r) < 50:
            args.rmax = 50
        else:
            args.rmax = 180

    plt.ylim((0, args.rmax))

    plt.savefig(args.output, format=args.format, bbox_inches="tight", dpi="figure", transparent=args.transparent)

    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", help="Scatter plot alpha value",
                        type=float, default=0.75)
    parser.add_argument("--cmap", help="Colormap for matplotlib",
                        default="magma")
    parser.add_argument("--class", help="Breakdown angular distribution by class",
                        type=int, dest="cls", nargs="?", const=0)
    parser.add_argument("--dpi", help="DPI of output",
                        type=int, default=300)
    parser.add_argument("--figsize", help="Figure size for matplotlib",
                        type=int, default=10)
    parser.add_argument("--format", help="Output image format",
                        default="png", choices=["png", "pdf", "ps", "eps", "svg"])
    parser.add_argument("--full-circle", help="Extend domain from [0, pi] to [0, 2*pi]",
                        action="store_true")
    parser.add_argument("--psi", help="Plot tilt and psi instead of tilt and rot",
                        action="store_true")
    parser.add_argument("--rmax", help="Upper limit of radial axis (probably ~45 or 180)")
    parser.add_argument("--samples", help="Number of angular samples in [0, pi] (e.g. 36 for 5 deg. steps)",
                        type=int, default=36)
    parser.add_argument("--scale", help="Size of largest scatter point",
                        type=float, default=20)
    parser.add_argument("--subplot", help="Draw multiple plots as subplots of a single figure")
    parser.add_argument("--transparent", help="Use transparent background in output figure",
                        action="store_true")
    parser.add_argument("input", help="Input .star file")
    parser.add_argument("output", help="Output image file")

    sns.set()

    sys.exit(main(parser.parse_args()))
