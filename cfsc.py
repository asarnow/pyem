#!/usr/bin/env python
# Copyright (C) 2018 Daniel Asarnow
# University of California, San Francisco
#
# Rapidly computes conical Fourier shell correlations (3DFSC).
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
import numpy as np
import pyfftw
import pyfftw.interfaces.numpy_fft as fft
import sys
import time
from healpy import pix2vec
from pyem import mrc
from pyem.algo import bincorr
from scipy.spatial import cKDTree


def main(args):
    log = logging.getLogger('root')
    hdlr = logging.StreamHandler(sys.stdout)
    log.addHandler(hdlr)
    log.setLevel(logging.getLevelName(args.loglevel.upper()))
    pyfftw.interfaces.cache.enable()
    vol1 = mrc.read(args.volume1, inc_header=False, compat="relion")
    vol2 = mrc.read(args.volume2, inc_header=False, compat="relion")
    if args.mask is not None:
        mask = mrc.read(args.mask, inc_header=False, compat="relion")
        vol1 *= mask
        vol2 *= mask
    f3d1 = fft.rfftn(vol1, threads=args.threads)
    f3d2 = fft.rfftn(vol2, threads=args.threads)
    nside = 2**args.healpix_order
    x, y, z = pix2vec(nside, np.arange(12 * nside ** 2))
    xhalf = x >= 0
    hp = np.column_stack([x[xhalf], y[xhalf], z[xhalf]])
    t0 = time.time()
    fcor = calc_dfsc(f3d1, f3d2, hp, np.deg2rad(args.arc))
    log.info("Computed CFSC in %0.2f s" % (time.time() - t0))
    fsc = calc_fsc(f3d1, f3d2)
    t0 = time.time()
    log.info("Computed GFSC in %0.2f s" % (time.time() - t0))
    freqs = np.fft.rfftfreq(f3d1.shape[0])
    np.save(args.output, np.row_stack([freqs, fsc, fcor]))
    return 0


def calc_dfsc(f3d1, f3d2, vecs, arc):
    log = logging.getLogger('root')
    n = f3d1.shape[0]
    sz, sy, sx = np.meshgrid(np.fft.fftfreq(n),
                             np.fft.fftfreq(n),
                             np.fft.rfftfreq(n), indexing="ij")
    s = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
    r = s * n
    r[r > n // 2] = n // 2 + 1
    r = np.round(r).astype(np.int64)
    nr = np.max(r) + 1
    grid = np.column_stack([sx.reshape(-1), sy.reshape(-1), sz.reshape(-1)])
    grid = grid / np.linalg.norm(grid, axis=1).reshape(-1, 1)
    t0 = time.time()
    kdtree = cKDTree(grid[1:], balanced_tree=False)
    log.info("Constructed kD-tree in %0.2f s" % (time.time() - t0))
    maxdist = 2 * np.sin(arc / 2)
    fcor = np.zeros((len(vecs), nr - 1))
    t0 = time.time()
    for i, vec in enumerate(vecs):
        idx = kdtree.query_ball_point(vec, maxdist)
        idx = np.asarray(idx) + 1
        fcor[i] = np.abs(bincorr(
            f3d1.flat[idx], f3d2.flat[idx], r.flat[idx], minlength=nr)[:-1])
    log.info("Evaluated %d cones in %0.2f s" % (len(vecs), time.time() - t0))
    return np.row_stack(fcor)


def calc_fsc(f3d1, f3d2):
    n = f3d1.shape[0]
    sz, sy, sx = np.meshgrid(np.fft.fftfreq(n),
                             np.fft.fftfreq(n),
                             np.fft.rfftfreq(n), indexing="ij")
    s = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
    r = s * n
    r = np.round(r).astype(np.int64)
    r[r > n // 2] = n // 2 + 1
    nr = np.max(r) + 1
    return np.abs(bincorr(f3d1, f3d2, r, minlength=nr)[:-1])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("volume1")
    parser.add_argument("volume2")
    parser.add_argument("output")
    parser.add_argument("--arc", help="Cone width in degrees", type=float, default=5)
    parser.add_argument("--healpix-order", help="Healpix order", type=int, default=2)
    parser.add_argument("--mask", "-m", help="Mask for FSC calculation")
    parser.add_argument("--threads", "-j", help="Number of threads for FFTW", type=int, default=1)
    parser.add_argument("--loglevel", "-l", type=str, default="WARNING", help="Logging level and debug output")
    sys.exit(main(parser.parse_args()))
