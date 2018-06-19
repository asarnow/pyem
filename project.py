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
    if args.map is not None:
        vol = mrc.read(args.map, inc_header=False, compat="relion")
        if args.mask is not None:
            mask = mrc.read(args.mask, inc_header=False, compat="relion")
            vol *= mask
    else:
        print("Please supply a map")
        return 1

    pfac = 2
    f3d = vop.vol_ft(vol, pfac=pfac, threads=cpu_count(), normfft=pfac**3 * vol.shape[0])
    sz = f3d.shape[0] // 2 - 1
    sx, sy = np.meshgrid(np.fft.rfftfreq(sz), np.fft.fftfreq(sz))
    s = np.sqrt(sx ** 2 + sy ** 2)
    a = np.arctan2(sy, sx)

    ift = None

    with mrc.ZSliceWriter(args.output) as zsw:
        for i, p in df.iterrows():
            f2d = project(f3d, p, s, sx, sy, a, apply_ctf=args.ctf)
            if ift is None:
                ift = irfft2(f2d.copy(),
                             threads=cpu_count(),
                             planner_effort="FFTW_ESTIMATE",
                             auto_align_input=True,
                             auto_contiguous=True)
            proj = fftshift(ift(f2d.copy(), np.zeros(vol.shape[:-1], dtype=vol.dtype), normalise_idft=False, ortho=False))
            log.debug("%f +/- %f" % (np.mean(proj), np.std(proj)))
            if args.subtract:
                with mrc.ZSliceReader(p["ucsfImagePath"]) as zsr:
                    img = zsr.read(p["ucsfImageIndex"])
                log.debug("%f +/- %f" % (np.mean(img), np.std(img)))
                proj = img - proj
            zsw.write(proj)
            log.info("Wrote %d@%s: %d/%d" % (p["ucsfImageIndex"], p["ucsfImagePath"], i, df.shape[0]))

    if args.star is not None:
        df["ucsfImagePath"] = args.output
        df["ucsfImageIndex"] = np.arange(df.shape[0])
        star.simplify_star_ucsf(df)
        star.write_star(args.star, df)
    return 0


def project(f3d, p, s, sx, sy, a, apply_ctf=False):
    orient = util.euler2rot(np.deg2rad(p[star.Relion.ANGLEROT]),
                            np.deg2rad(p[star.Relion.ANGLETILT]),
                            np.deg2rad(p[star.Relion.ANGLEPSI]))
    pshift = np.exp(-2 * np.pi * 1j * (-p[star.Relion.ORIGINX] * sx +
                                       -p[star.Relion.ORIGINY] * sy))
    f2d = vop.interpolate_slice_numba(f3d, orient)
    f2d *= pshift
    if apply_ctf:
        apix = star.calculate_apix(p)
        c = ctf.eval_ctf(s / apix, a,
                         p[star.Relion.DEFOCUSU], p[star.Relion.DEFOCUSV],
                         p[star.Relion.DEFOCUSANGLE],
                         p[star.Relion.PHASESHIFT], p[star.Relion.VOLTAGE],
                         p[star.Relion.AC], p[star.Relion.CS], bf=0,
                         lp=2 * apix)
        f2d *= c
    return f2d


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="STAR file with particle metadata")
    parser.add_argument("output", help="Output particle stack")
    parser.add_argument("--map", help="Map used to calculate projections")
    parser.add_argument("--mask",
                        help="Mask to apply to map before projection")
    parser.add_argument("--ctf", help="Apply CTF to projections",
                        action="store_true")
    parser.add_argument("--star",
                        help="Output STAR file with projection metadata")
    parser.add_argument("--subtract",
                        help="Subtract projection from experimental images",
                        action="store_true")
    parser.add_argument("--loglevel", "-l", type=str, default="WARNING", help="Logging level and debug output")
    sys.exit(main(parser.parse_args()))
