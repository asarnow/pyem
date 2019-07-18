#!/usr/bin/env python2.7
# Copyright (C) 2017 Daniel Asarnow
# University of California, San Francisco
#
# Generate subparticles for "local reconstruction" methods.
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
import json
import numpy as np
import os
import os.path
import sys
from pyem import geom
from pyem import star
from pyem import util


def main(args):
    log = logging.getLogger(__name__)
    hdlr = logging.StreamHandler(sys.stdout)
    log.addHandler(hdlr)
    log.setLevel(logging.getLevelName(args.loglevel.upper()))

    if args.target is None and args.sym is None and args.transform is None and args.euler is None:
        log.error("At least a target, transformation matrix, Euler angles, or a symmetry group must be provided")
        return 1
    elif args.target is not None and args.boxsize is None and args.origin is None:
        log.error("An origin must be provided via --boxsize or --origin")
        return 1

    if args.apix is None:
        df = star.parse_star(args.input, nrows=1)
        args.apix = star.calculate_apix(df)
        if args.apix is None:
            log.warn("Could not compute pixel size, default is 1.0 Angstroms per pixel")
            args.apix = 1.0
            df[star.Relion.MAGNIFICATION] = 10000
            df[star.Relion.DETECTORPIXELSIZE] = 1.0

    if args.target is not None:
        try:
            args.target = np.array([np.double(tok) for tok in args.target.split(",")])
        except:
            log.error("Target must be comma-separated list of x,y,z coordinates")
            return 1

    if args.euler is not None:
        try:
            args.euler = np.deg2rad(np.array([np.double(tok) for tok in args.euler.split(",")]))
            args.transform = np.zeros((3, 4))
            args.transform[:, :3] = geom.euler2rot(*args.euler)
            if args.target is not None:
                args.transform[:, -1] = args.target
        except:
            log.error("Euler angles must be comma-separated list of rotation, tilt, skew in degrees")
            return 1

    if args.transform is not None and not hasattr(args.transform, "dtype"):
        if args.target is not None:
            log.warn("--target supersedes --transform")
        try:
            args.transform = np.array(json.loads(args.transform))
        except:
            log.error("Transformation matrix must be in JSON/Numpy format")
            return 1

    if args.origin is not None:
        if args.boxsize is not None:
            log.warn("--origin supersedes --boxsize")
        try:
            args.origin = np.array([np.double(tok) for tok in args.origin.split(",")])
            args.origin /= args.apix
        except:
            log.error("Origin must be comma-separated list of x,y,z coordinates")
            return 1
    elif args.boxsize is not None:
        args.origin = np.ones(3) * args.boxsize / 2
    
    if args.sym is not None:
        args.sym = util.relion_symmetry_group(args.sym)

    df = star.parse_star(args.input)

    if star.calculate_apix(df) != args.apix:
        log.warn("Using specified pixel size of %f instead of calculated size %f" %
                 (args.apix, star.calculate_apix(df)))

    if args.cls is not None:
        df = star.select_classes(df, args.cls)

    if args.target is not None:
        args.target /= args.apix
        c = args.target - args.origin
        c = np.where(np.abs(c) < 1, 0, c)  # Ignore very small coordinates.
        d = np.linalg.norm(c)
        ax = c / d
        r = geom.euler2rot(*np.array([np.arctan2(ax[1], ax[0]), np.arccos(ax[2]), np.deg2rad(args.psi)]))
        d = -d
    elif args.transform is not None:
        r = args.transform[:, :3]
        if args.transform.shape[1] == 4:
            d = args.transform[:, -1] / args.apix
            d = r.dot(args.origin) + d - args.origin
        else:
            d = 0
    elif args.sym is not None:
        r = np.identity(3)
        d = -args.displacement / args.apix
    else:
        log.error("At least a target or symmetry group must be provided via --target or --sym")
        return 1

    log.debug("Final rotation: %s" % str(r).replace("\n", "\n" + " " * 16))
    ops = [op.dot(r.T) for op in args.sym] if args.sym is not None else [r.T]
    log.debug("Final translation: %s (%f px)" % (str(d), np.linalg.norm(d)))
    dfs = list(subparticle_expansion(df, ops, d, rotate=args.shift_only, invert=args.invert, adjust_defocus=args.adjust_defocus))
 
    if args.recenter:
        for s in dfs:
            star.recenter(s, inplace=True)
    
    if args.suffix is None and not args.skip_join:
        if len(dfs) > 1:
            df = util.interleave(dfs)
        else:
            df = dfs[0]
        star.write_star(args.output, df)
    else:
        for i, s in enumerate(dfs):
            star.write_star(os.path.join(args.output, args.suffix + "_%d" % i), s)
    return 0


def subparticle_expansion(s, ops=None, dists=0, rots=None, rotate=True, invert=False, adjust_defocus=False):
    log = logging.getLogger(__name__)
    if ops is None:
        ops = [np.eye(3)]
    if rots is None:
        rots = geom.e2r_vec(np.deg2rad(s[star.Relion.ANGLES].values))
    dists = np.atleast_2d(dists)
    if len(dists) == 1:
        dists = np.repeat(dists, len(ops), axis=0)
    for i in range(len(ops)):
        log.debug("Yielding expansion %d" % i)
        log.debug("Rotation: %s" % str(ops[i]).replace("\n", "\n" + " " * 10))
        log.debug("Translation: %s (%f px)" % (str(dists[i]), np.linalg.norm(dists[i])))
        yield star.transform_star(s, ops[i], dists[i], rots=rots, rotate=rotate, invert=invert, adjust_defocus=adjust_defocus)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="STAR file with source particles")
    parser.add_argument("output", help="Output file path (and prefix for output files)")
    parser.add_argument("--apix", "--angpix", help="Angstroms per pixel (calculate from STAR by default)", type=float)
    parser.add_argument("--boxsize", help="Particle box size in pixels (used to define origin only)", type=int)
    parser.add_argument("--class", help="Keep this class in output, may be passed multiple times",
                        action="append", type=int, dest="cls")
    parser.add_argument("--displacement", help="Distance of new origin from symmetrix axis in Angstroms",
                        type=float, default=0)
    parser.add_argument("--origin", help="Origin coordinates in Angstroms", metavar="x,y,z")
    parser.add_argument("--target", help="Target coordinates in Angstroms", metavar="x,y,z")
    parser.add_argument("--invert", help="Invert the transformation", action="store_true")
    parser.add_argument("--target-invert", action="store_true", dest="invert", help=argparse.SUPPRESS)
    parser.add_argument("--psi", help="Additional in-plane rotation of target in degrees", type=float, default=0)
    parser.add_argument("--euler", help="Euler angles (ZYZ intrinsic) to rotate particles", metavar="rot,tilt,psi")
    parser.add_argument("--transform", help="Transformation matrix (3x3 or 3x4) in Numpy format")
    parser.add_argument("--recenter", help="Recenter subparticle coordinates by subtracting X and Y shifts (e.g. for "
                                           "extracting outside Relion)", action="store_true")
    parser.add_argument("--adjust-defocus", help="Add Z component of shifts to defocus", action="store_true")
    parser.add_argument("--shift-only", help="Keep original view axis after target transformation", action="store_false")
    parser.add_argument("--loglevel", "-l", type=str, default="WARNING", help="Logging level and debug output")
    parser.add_argument("--skip-join", help="Force multiple output files even if no suffix provided",
                        action="store_true", default=False)
    parser.add_argument("--suffix", help="Suffix for multiple output files")
    parser.add_argument("--sym", help="Symmetry group for whole-particle expansion or symmetry-derived subparticles ("
                                      "Relion conventions)")

    sys.exit(main(parser.parse_args()))

