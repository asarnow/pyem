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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.projections.polar import PolarTransform
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist import angle_helper
from mpl_toolkits.axisartist import floating_axes
from pyem.star import parse_star


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
    xfield = xfields[0]

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
    yfield = yfields[0]

    if args.cls is not None:
        clsfields = [f for f in star.columns if "ClassNumber" in f]
        if len(clsfields) == 0:
            print("No class labels found")
            return 1
        clsfield = clsfields[0]
        if args.cls > 0:
            ind = star[clsfield] == args.cls
            if not np.any(ind):
                print("Specified class has no members")
                return 1
            data = star.loc[ind][[xfield, yfield]]
        else:
            classes = np.unique(star[clsfield])
            ind = (star[clsfields[0]] == cls for cls in classes)
            data = [star.loc[i][[xfield, yfield]] for i in ind]
            for d, cls in zip(data, classes):
                h, theta, r = compute_histogram(d, args.samples)
                fig, ax, aux_ax = make_figure(h, theta, r, rmax=args.rmax, figsize=args.figsize, dpi=args.dpi,
                                              scale=args.scale, cmap=args.cmap, alpha=args.alpha)
                if args.psi:
                    ax.axis["left"].label.set_text("Psi Angle")
                else:
                    ax.axis["left"].label.set_text("Rotation Angle")

                fig.savefig(args.output + "_class%d." % cls + args.format, format=args.format, bbox_inches="tight",
                            dpi="figure",
                            transparent=args.transparent)
                plt.close(fig)
            return 0
    else:
        data = star[[xfield, yfield]]

    if args.subplot is not None:
        raise NotImplementedError("Subplots are not yet supported")

    h, theta, r = compute_histogram(data, args.samples)
    fig, ax, aux_ax = make_figure(h, theta, r, rmax=args.rmax, figsize=args.figsize, dpi=args.dpi, scale=args.scale,
                                  cmap=args.cmap, alpha=args.alpha)
    if args.psi:
        ax.axis["left"].label.set_text("Psi Angle")
    else:
        ax.axis["left"].label.set_text("Rotation Angle")
    fig.savefig(args.output, format=args.format, bbox_inches="tight", dpi="figure", transparent=args.transparent)
    return 0


def compute_histogram(data, bins=36):
    h, x, y = np.histogram2d(data[data.columns[0]], data[data.columns[1]], bins=bins, normed=True)
    xc = (x[:-1] + x[1:]) / 2
    yc = (y[:-1] + y[1:]) / 2
    coords = np.array([(xi, yi) for xi in xc for yi in yc])
    theta = coords[:, 0]
    r = coords[:, 1]
    return h.flatten(), theta, r


def make_figure(h, theta, r, rmax=None, figsize=10, dpi=300, scale=500, cmap="magma", alpha=0.75):
    area = h / np.max(h) * scale
    colors = h

    if rmax is None:
        if np.max(r) <= 45:
            rmax = 45
        else:
            rmax = 180

    fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
    ax, aux_ax = setup_axes(fig, 111, rmax)

    # if args.title is not None:
    #     # ax.axis["top"].title.set_text(args.title)
    #     ax.set_title(args.title)
    # elif args.cls is not None and args.cls > 0:
    #     # ax.axis["top"].title.set_text("Angular Distribution within Class %d" % args.cls)
    #     ax.set_title("Angular Distribution within Class %d" % args.cls)
    # else:
    #     # ax.axis["top"].title.set_text("Angular Distribution")
    #     ax.set_title("Angular Distribution")

    c = aux_ax.scatter(theta, r, c=colors, s=area, cmap=cmap, zorder=3)
    c.set_alpha(alpha)
    return fig, ax, aux_ax


def setup_axes(fig, rect, rmax):
    tr_rotate = Affine2D().translate(0, 0)
    tr_scale = Affine2D().scale(np.pi / 180, 1)
    tr = tr_rotate + tr_scale + PolarTransform()
    grid_locator1 = angle_helper.LocatorDMS(12)
    grid_locator2 = angle_helper.LocatorDMS(3)
    tick_formatter1 = angle_helper.FormatterDMS()
    tick_formatter2 = angle_helper.FormatterDMS()
    ra0, ra1 = 0, 180
    cz0, cz1 = 0, rmax
    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(ra0, ra1, cz0, cz1),
        grid_locator1=grid_locator1,
        grid_locator2=grid_locator2,
        tick_formatter1=tick_formatter1,
        tick_formatter2=tick_formatter2)
    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax1)
    ax1.axis["left"].set_axis_direction("bottom")
    ax1.axis["right"].set_axis_direction("top")
    ax1.axis["bottom"].set_visible(False)
    ax1.axis["top"].set_axis_direction("bottom")
    ax1.axis["top"].toggle(ticklabels=True, label=True)
    ax1.axis["top"].major_ticklabels.set_axis_direction("top")
    ax1.axis["top"].label.set_axis_direction("top")
    ax1.axis["top"].label.set_text("Tilt Angle")
    aux_ax = ax1.get_aux_axes(tr)
    aux_ax.patch = ax1.patch
    ax1.patch.zorder = 0.9
    return ax1, aux_ax


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", help="Scatter plot alpha value",
                        type=float, default=0.75)
    parser.add_argument("--cmap", help="Colormap for matplotlib",
                        default="magma")
    parser.add_argument("--class", help="Breakdown angular distribution by class",
                        type=int, dest="cls")
    parser.add_argument("--dpi", help="DPI of output",
                        type=int, default=300)
    parser.add_argument("--figsize", help="Figure size for matplotlib",
                        type=int, default=10)
    parser.add_argument("--format", help="Output image format",
                        default="png", choices=["png", "pdf", "ps", "eps", "svg"])
    parser.add_argument("--psi", help="Plot tilt and psi instead of tilt and rot",
                        action="store_true")
    parser.add_argument("--rmax", help="Upper limit of radial axis (probably ~45 or 180)",
                        type=int)
    parser.add_argument("--samples", help="Number of angular samples in [0, pi] (e.g. 36 for 5 deg. steps)",
                        type=int, default=36)
    parser.add_argument("--scale", help="Size of largest scatter point",
                        type=float, default=20)
    parser.add_argument("--subplot", help="Draw multiple plots as subplots of a single figure")
    parser.add_argument("--title", help="Custom figure title")
    parser.add_argument("--transparent", help="Use transparent background in output figure",
                        action="store_true")
    parser.add_argument("input", help="Input .star file")
    parser.add_argument("output", help="Output image file")

    sns.set()

    sys.exit(main(parser.parse_args()))
