#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2018 Daniel Asarnow
# University of California, San Francisco
#
# Program for sorting particles in electron microscopy.
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
import pyfftw
import sys
from numpy.fft import fftshift
from numpy.fft import fftfreq
from numpy.fft import rfftfreq
from pyem import ctf
from pyem import mrc
from pyem import star
from pyem import util
from pyem import vop
from pyfftw.interfaces.numpy_fft import rfft2
from pyfftw.interfaces.numpy_fft import irfft2


def main(args):
    pyfftw.interfaces.cache.enable()

    refmap = mrc.read(args.key, compat="relion")
    df = star.parse_star(args.input, keep_index=False)
    star.augment_star_ucsf(df)
    refmap_ft = vop.vol_ft(refmap, threads=args.threads)

    apix = star.calculate_apix(df)
    sz = refmap_ft.shape[0] // 2 - 1
    sx, sy = np.meshgrid(rfftfreq(sz), fftfreq(sz))
    s = np.sqrt(sx ** 2 + sy ** 2)
    r = s * sz
    r = np.round(r).astype(np.int64)
    r[r > sz // 2] = sz // 2 + 1
    a = np.arctan2(sy, sx)

    def1 = df["rlnDefocusU"].values
    def2 = df["rlnDefocusV"].values
    angast = df["rlnDefocusAngle"].values
    phase = df["rlnPhaseShift"].values
    kv = df["rlnVoltage"].values
    ac = df["rlnAmplitudeContrast"].values
    cs = df["rlnSphericalAberration"].values
    xshift = df["rlnOriginX"].values
    yshift = df["rlnOriginY"].values

    score = np.zeros(df.shape[0])

    # TODO parallelize
    for i, row in df.iterrows():
        xcor = particle_xcorr(row, refmap_ft)

    if args.top is None:
        args.top = df.shape[0]

    top = df.iloc[np.argsort(score)][:args.top]
    star.simplify_star_ucsf(top)
    star.write_star(args.output, top)
    return 0


def particle_xcorr(ptcl, refmap_ft):
    r = util.euler2rot(*np.deg2rad(ptcl[star.Relion.ANGLES]))
    proj = vop.interpolate_slice_numba(refmap_ft, r)
    c = ctf.eval_ctf(s / apix, a, def1[i], def2[i], angast[i], phase[i],
                     kv[i], ac[i], cs[i], bf=0, lp=2 * apix)
    pshift = np.exp(-2 * np.pi * 1j * (-xshift[i] * sx + -yshift * sy))
    proj_ctf = proj * pshift * c

    with mrc.ZSliceReader(ptcl[star.Relion.IMAGE_NAME]) as f:
        exp_image_fft = rfft2(fftshift(f.read(i)))

    xcor_fft = exp_image_fft * proj_ctf
    xcor = fftshift(irfft2(xcor_fft))
    return xcor


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--key")
    parser.add_argument("--highpass")
    parser.add_argument("--lowpass")
    parser.add_argument("--top", help="Write top N particles",
                        metavar="N", type=int)
    parser.add_argument("--threads", "-j", help="Number of parallel threads",
                        type=int)
    sys.exit(main(parser.parse_args()))
