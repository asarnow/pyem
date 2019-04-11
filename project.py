#!/usr/bin/env python2.7
# Copyright (C) 2016 Daniel Asarnow, Eugene Palovcak
# University of California, San Francisco
#
# Program for projecting density maps in electron microscopy.
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
import logging
import numpy as np
import sys
from multiprocessing import cpu_count
from pyem import ctf
from pyem import mrc
from pyem import star
from pyem import util
from pyem import vop
from numpy.fft import fftshift
from pyfftw.builders import irfft2


def main(args):
    log = logging.getLogger('root')
    hdlr = logging.StreamHandler(sys.stdout)
    log.addHandler(hdlr)
    log.setLevel(logging.getLevelName(args.loglevel.upper()))
    df = star.parse_star(args.input, keep_index=False)
    star.augment_star_ucsf(df)
    maxshift = np.round(np.max(np.abs(df[star.Relion.ORIGINS].values)))

    if args.map is not None:
        if args.map.endswith(".npy"):
            log.info("Reading precomputed 3D FFT of volume")
            f3d = np.load(args.map)
            log.info("Finished reading 3D FFT of volume")
            if args.size is None:
                args.size = (f3d.shape[0] - 3) // args.pfac
        else:
            vol = mrc.read(args.map, inc_header=False, compat="relion")
            if args.mask is not None:
                mask = mrc.read(args.mask, inc_header=False, compat="relion")
                vol *= mask
            if args.size is None:
                args.size = vol.shape[0]
            if args.crop is not None and args.size // 2 < maxshift + args.crop // 2:
                log.error("Some shifts are too large to crop (maximum crop is %d)" % (args.size - 2 * maxshift))
                return 1
            log.info("Preparing 3D FFT of volume")
            f3d = vop.vol_ft(vol, pfac=args.pfac, threads=args.threads)
            log.info("Finished 3D FFT of volume")
    else:
        log.error("Please supply a map")
        return 1

    sz = (f3d.shape[0] - 3) // args.pfac
    apix = star.calculate_apix(df) * np.double(args.size) / sz
    sx, sy = np.meshgrid(np.fft.rfftfreq(sz), np.fft.fftfreq(sz))
    s = np.sqrt(sx ** 2 + sy ** 2)
    a = np.arctan2(sy, sx)
    log.info("Projection size is %d, unpadded volume size is %d" % (args.size, sz))
    log.info("Effective pixel size is %f A/px" % apix)

    if args.subtract and args.size != sz:
        log.error("Volume and projections must be same size when subtracting")
        return 1

    if args.crop is not None and args.size // 2 < maxshift + args.crop // 2:
        log.error("Some shifts are too large to crop (maximum crop is %d)" % (args.size - 2 * maxshift))
        return 1

    ift = None

    with mrc.ZSliceWriter(args.output, psz=apix) as zsw:
        for i, p in df.iterrows():
            f2d = project(f3d, p, s, sx, sy, a, pfac=args.pfac, apply_ctf=args.ctf, size=args.size, flip_phase=args.flip)
            if ift is None:
                ift = irfft2(f2d.copy(),
                             threads=args.threads,
                             planner_effort="FFTW_ESTIMATE",
                             auto_align_input=True,
                             auto_contiguous=True)
            proj = fftshift(ift(f2d.copy(), np.zeros(ift.output_shape, dtype=ift.output_dtype)))
            log.debug("%f +/- %f" % (np.mean(proj), np.std(proj)))
            if args.subtract:
                with mrc.ZSliceReader(p["ucsfImagePath"]) as zsr:
                    img = zsr.read(p["ucsfImageIndex"])
                log.debug("%f +/- %f" % (np.mean(img), np.std(img)))
                proj = img - proj
            if args.crop is not None:
                orihalf = args.size // 2
                newhalf = args.crop // 2
                x = orihalf - np.int(np.round(p[star.Relion.ORIGINX]))
                y = orihalf - np.int(np.round(p[star.Relion.ORIGINY]))
                proj = proj[y-newhalf:y+newhalf, x-newhalf:x+newhalf]
            zsw.write(proj)
            log.debug("%d@%s: %d/%d" % (p["ucsfImageIndex"], p["ucsfImagePath"], i + 1, df.shape[0]))

    if args.star is not None:
        log.info("Writing output .star file")
        if args.crop is not None:
            df = star.recenter(df, inplace=True)
        if args.subtract:
            df[star.UCSF.IMAGE_ORIGINAL_PATH] = df[star.UCSF.IMAGE_PATH]
            df[star.UCSF.IMAGE_ORIGINAL_INDEX] = df[star.UCSF.IMAGE_INDEX]
        df[star.UCSF.IMAGE_PATH] = args.output
        df[star.UCSF.IMAGE_INDEX] = np.arange(df.shape[0])
        star.simplify_star_ucsf(df)
        star.write_star(args.star, df)
    return 0


def project(f3d, p, s, sx, sy, a, pfac=2, apply_ctf=False, size=None, flip_phase=False):
    orient = util.euler2rot(np.deg2rad(p[star.Relion.ANGLEROT]),
                            np.deg2rad(p[star.Relion.ANGLETILT]),
                            np.deg2rad(p[star.Relion.ANGLEPSI]))
    pshift = np.exp(-2 * np.pi * 1j * (-p[star.Relion.ORIGINX] * sx +
                                       -p[star.Relion.ORIGINY] * sy))
    f2d = vop.interpolate_slice_numba(f3d, orient, pfac=pfac, size=size)
    f2d *= pshift
    if apply_ctf or flip_phase:
        apix = star.calculate_apix(p) * np.double(size) / (f3d.shape[0] // pfac - 1)
        c = ctf.eval_ctf(s / apix, a,
                         p[star.Relion.DEFOCUSU], p[star.Relion.DEFOCUSV],
                         p[star.Relion.DEFOCUSANGLE],
                         p[star.Relion.PHASESHIFT], p[star.Relion.VOLTAGE],
                         p[star.Relion.AC], p[star.Relion.CS], bf=0,
                         lp=2 * apix)
        if flip_phase:
            c = np.sign(c)
        f2d *= c
    return f2d


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="STAR file with particle metadata")
    parser.add_argument("output", help="Output particle stack")
    parser.add_argument("--map", help="Map used to calculate projections")
    parser.add_argument("--mask", help="Mask to apply to map before projection")
    parser.add_argument("--ctf", help="Apply CTF to projections", action="store_true")
    parser.add_argument("--flip", help="Only flip phases when applying CTF to projections", action="store_true")
    parser.add_argument("--pfac", help="Zero padding factor for 3D FFT (default: %(default)d)", type=int, default=2)
    parser.add_argument("--size", help="Size of projections (before subtraction)", type=int)
    parser.add_argument("--crop", help="Size to crop recentered output images (after subtraction)", type=int)
    parser.add_argument("--star", help="Output STAR file with projection metadata")
    parser.add_argument("--subtract", help="Subtract projection from experimental images", action="store_true")
    parser.add_argument("--threads", "-j", help="Number of threads for FFTs (default: CPU count = %(default)d)",
                        metavar="N", type=int, default=cpu_count())
    parser.add_argument("--loglevel", "-l", help="Logging level and debug output", metavar="LEVEL", type=str,
                        default="WARNING")
    sys.exit(main(parser.parse_args()))
