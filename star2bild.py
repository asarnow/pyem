#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# Copyright (C) 2018 Daniel Asarnow
# University of California, San Francisco
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
from healpy import pix2ang
from pyem import geom
from pyem import star
from pyem import util
from scipy.spatial import cKDTree


def main(args):
    log = logging.getLogger('root')
    hdlr = logging.StreamHandler(sys.stdout)
    log.addHandler(hdlr)
    log.setLevel(logging.getLevelName(args.loglevel.upper()))
    if args.boxsize is None:
        log.error("Please specify box size")
        return 1
    df = star.parse_star(args.input, keep_index=False)
    if args.cls is not None:
        df = star.select_classes(df, args.cls)
    if args.apix is None:
        args.apix = star.calculate_apix(df)
    if args.sym is not None:
        args.sym = util.relion_symmetry_group(args.sym)
        df[star.Relion.ANGLEPSI] = 0
        rots = geom.e2r_vec(np.deg2rad(df[star.Relion.ANGLES].values))
        dfs = [star.transform_star(df, op, rots=rots) for op in args.sym]
        dfi = pd.concat(dfs, axis=0, keys=[0, 1, 2, 3])
        newrots = np.array([geom.e2r_vec(np.deg2rad(x[star.Relion.ANGLES].values)) for x in dfs])
        mag = np.array([geom.phi5(r) for r in newrots.reshape(-1, 3, 3)]).reshape(4, -1)
        idx = np.argmin(mag, axis=0)
        midx = [(i, a) for a, i in enumerate(idx)]
        df = dfi.loc[midx]
    nside = 2**args.healpix_order
    angular_sampling = np.sqrt(3 / np.pi) * 60 / nside
    theta, phi = pix2ang(nside, np.arange(12 * nside ** 2))
    phi = np.pi - phi
    hp = np.column_stack((np.sin(theta) * np.cos(phi),
                          np.sin(theta) * np.sin(phi),
                          np.cos(theta)))
    kdtree = cKDTree(hp)
    st = np.sin(np.deg2rad(df[star.Relion.ANGLETILT]))
    ct = np.cos(np.deg2rad(df[star.Relion.ANGLETILT]))
    sp = np.sin(np.deg2rad(df[star.Relion.ANGLEROT]))
    cp = np.cos(np.deg2rad(df[star.Relion.ANGLEROT]))
    ptcls = np.column_stack((st * cp, st * sp, ct))
    _, idx = kdtree.query(ptcls)
    cnts = np.bincount(idx, minlength=theta.size)
    frac = cnts / np.max(cnts).astype(np.float64)
    mu = np.mean(frac)
    sigma = np.std(frac)
    color_scale = (frac - mu) / sigma
    color_scale[color_scale > 5] = 5
    color_scale[color_scale < -1] = -1
    color_scale /= 6
    color_scale += 1 / 6.
    r = args.boxsize * args.apix / 2
    rp = np.reshape(r + r * frac * args.height_scale, (-1, 1))
    base1 = hp * r
    base2 = hp * rp
    base1 = base1[:, [0, 1, 2]] + np.array([r]*3)
    base2 = base2[:, [0, 1, 2]] + np.array([r]*3)
    height = np.squeeze(np.abs(rp - r))
    idx = np.where(height >= 0.01)[0]
    width = args.width_scale * np.pi * r * angular_sampling / 360
    bild = np.hstack((base1, base2, np.ones((base1.shape[0], 1)) * width))
    fmt_color = ".color %f 0 %f\n"
    fmt_cyl = ".cylinder %f %f %f %f %f %f %f\n"
    with open(args.output, "w") as f:
        for i in idx:
            f.write(fmt_color % (color_scale[i], 1 - color_scale[i]))
            f.write(fmt_cyl % tuple(bild[i]))
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input .star file")
    parser.add_argument("output", help="Output .bild file")
    parser.add_argument("--healpix-order",
                        help="Healpix order (Relion convention)", type=int,
                        default=4)
    parser.add_argument("--apix",
                        help="Angstroms per pixel (calculate from STAR by default)",
                        type=float)
    parser.add_argument("--boxsize", help="Box size in pixels", type=int)
    parser.add_argument("--height-scale", type=float, default=0.3)
    parser.add_argument("--width-scale", type=float, default=0.5)
    parser.add_argument("--loglevel", help="Log level", default="WARNING")
    parser.add_argument("--class",
                        help="Only use the specified class, may be passed multiple times",
                        type=int, action="append", dest="cls")
    parser.add_argument("--sym", help="Symmetry group to impose on distribution (Relion conventions)")
    sys.exit(main(parser.parse_args()))
