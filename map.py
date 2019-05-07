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
import json
import logging
import numpy as np
import sys
from pyem.mrc import read
from pyem.mrc import write
from pyem.util import euler2rot
from pyem.util import rot2euler
from pyem.util import vec2rot
from pyem import vop
from scipy.ndimage import affine_transform
from scipy.ndimage import shift
from scipy.ndimage import zoom


def main(args):
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    hdlr = logging.StreamHandler(sys.stdout)
    if args.quiet:
        hdlr.setLevel(logging.ERROR)
    elif args.verbose:
        hdlr.setLevel(logging.INFO)
    else:
        hdlr.setLevel(logging.WARN)
    log.addHandler(hdlr)

    data, hdr = read(args.input, inc_header=True)
    final = None
    box = np.array([hdr[a] for a in ["nx", "ny", "nz"]])
    center = box // 2

    if args.fft:
        if args.final_mask is not None:
            final_mask = read(args.final_mask)
            data *= final_mask
        data_ft = vop.vol_ft(data.T, pfac=args.pfac, threads=args.threads)
        np.save(args.output, data_ft)
        return 0

    if args.transpose is not None:
        try:
            tax = [np.int64(a) for a in args.transpose.split(",")]
            data = np.transpose(data, axes=tax)
        except:
            log.error("Transpose axes must be comma-separated list of three integers")
            return 1

    if args.normalize:
        if args.reference is not None:
            ref, refhdr = read(args.reference, inc_header=True)
            final, mu, sigma = vop.normalize(data, ref=ref, return_stats=True)
        else:
            final, mu, sigma = vop.normalize(data, return_stats=True)
        final = (data - mu) / sigma
        if args.verbose:
            log.info("Mean: %f, Standard deviation: %f" % (mu, sigma))

    if args.apix is None:
        args.apix = hdr["xlen"] / hdr["nx"]
        log.info("Using computed pixel size of %f Angstroms" % args.apix)

    if args.target and args.matrix:
        log.warn("Target pose transformation will be applied after explicit matrix")
    if args.euler is not None and (args.target is not None or args.matrix is not None):
        log.warn("Euler transformation will be applied after target pose transformation")
    if args.translate is not None and (args.euler is not None or args.target is not None or args.matrix is not None):
        log.warn("Translation will be applied after other transformations")

    if args.origin is not None:
        try:
            args.origin = np.array([np.double(tok) for tok in args.origin.split(",")]) / args.apix
            assert np.all(args.origin < box)
        except:
            log.error("Origin must be comma-separated list of x,y,z coordinates and lie within the box")
            return 1
    else:
        args.origin = center
        log.info("Origin set to box center, %s" % (args.origin * args.apix))

    if not (args.target is None and args.euler is None and args.matrix is None and args.boxsize is None) \
            and vop.ismask(data) and args.spline_order != 0:
        log.warn("Input looks like a mask, --spline-order 0 (nearest neighbor) is recommended")

    if args.matrix is not None:
        try:
            r = np.array(json.loads(args.matrix))
        except:
            log.error("Matrix format is incorrect")
            return 1
        data = vop.resample_volume(data, r=r, t=None, ori=None, order=args.spline_order)

    if args.target is not None:
        try:
            args.target = np.array([np.double(tok) for tok in args.target.split(",")]) / args.apix
        except:
            log.error("Standard pose target must be comma-separated list of x,y,z coordinates")
            return 1
        args.target -= args.origin
        args.target = np.where(np.abs(args.target) < 1, 0, args.target)
        ori = None if args.origin is center else args.origin - center
        r = vec2rot(args.target)
        t = np.linalg.norm(args.target)
        log.info("Euler angles are %s deg and shift is %f px" % (np.rad2deg(rot2euler(r)), t))
        data = vop.resample_volume(data, r=r, t=args.target, ori=ori, order=args.spline_order, invert=args.target_invert)

    if args.euler is not None:
        try:
            args.euler = np.deg2rad(np.array([np.double(tok) for tok in args.euler.split(",")]))
        except:
            log.error("Eulers must be comma-separated list of phi,theta,psi angles")
            return 1
        r = euler2rot(*args.euler)
        offset = args.origin - 0.5
        offset = offset - r.T.dot(offset)
        data = affine_transform(data, r.T, offset=offset, order=args.spline_order)

    if args.translate is not None:
        try:
            args.translate = np.array([np.double(tok) for tok in args.translate.split(",")]) / args.apix
        except:
            log.error("Translation vector must be comma-separated list of x,y,z coordinates")
            return 1
        args.translate -= args.origin
        data = shift(data, -args.translate, order=args.spline_order)

    if args.boxsize is not None:
        args.boxsize = np.double(args.boxsize)
        data = zoom(data, args.boxsize / box, order=args.spline_order)
        args.apix = args.apix * box[0] / args.boxsize

    if final is None:
        final = data

    if args.final_mask is not None:
        final_mask = read(args.final_mask)
        final *= final_mask

    write(args.output, final, psz=args.apix)
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Use equals sign when passing arguments with negative numbers.")
    parser.add_argument("input", help="Input volume (MRC file)")
    parser.add_argument("output", help="Output volume (MRC file)")
    parser.add_argument("--apix", "--angpix", "-a", help="Pixel size in Angstroms", type=float)
    parser.add_argument("--mask", help="Final mask to apply after any operations", dest="final_mask")
    parser.add_argument("--transpose", help="Swap volume axes order", metavar="a1,a2,a3")
    parser.add_argument("--normalize", "-n", help="Convert map densities to Z-scores", action="store_true")
    parser.add_argument("--reference", "-r", help="Normalization reference volume (MRC file)")
    parser.add_argument("--fft", help="Cache padded 3D FFT for projections.", action="store_true")
    parser.add_argument("--threads", help="Thread count for FFTW", type=int, default=1)
    parser.add_argument("--pfac", help="Padding factor for 3D FFT", type=int, default=2)
    parser.add_argument("--origin", help="Origin coordinates in Angstroms (volume center by default)", metavar="x,y,z")
    parser.add_argument("--target", help="Target pose (view axis and origin) coordinates in Angstroms", metavar="x,y,z")
    parser.add_argument("--target-invert", help="Undo target pose transformation", action="store_true")
    parser.add_argument("--euler", help="Euler angles in degrees (Relion conventions)", metavar="phi,theta,psi")
    parser.add_argument("--translate", help="Translation coordinates in Angstroms", metavar="x,y,z")
    parser.add_argument("--matrix",
                        help="Transformation matrix (3x3 or 3x4 with translation in Angstroms) in Numpy/json format")
    parser.add_argument("--boxsize", help="Set the output box dimensions, accounting for pixel size", type=int)
    parser.add_argument("--spline-order",
                        help="Order of spline interpolation (0 for nearest, 1 for trilinear, default is cubic)",
                        type=int, default=3, choices=np.arange(6))
    parser.add_argument("--quiet", "-q", help="Print errors only", action="store_true")
    parser.add_argument("--verbose", "-v", help="Print info messages", action="store_true")
    sys.exit(main(parser.parse_args()))
