#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# Copyright (C) 2018 Daniel Asarnow
# University of California, San Francisco
#
# Program for single-particle pose analysis in electron microscopy.
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
import os
import os.path
import sys
import time
from pyem import geom
from pyem import mrc
from pyem import star
from pyem import util


def main(args):
    log = logging.getLogger('root')
    hdlr = logging.StreamHandler(sys.stdout)
    log.addHandler(hdlr)
    log.setLevel(logging.getLevelName(args.loglevel.upper()))

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["NUMBA_NUM_THREADS"] = str(args.threads)

    outdir = os.path.dirname(args.output)
    outbase = os.path.basename(args.output)

    dfs = [star.parse_star(inp, keep_index=False) for inp in args.input]
    size_err = np.array(args.input)[np.where(~np.equal([df.shape[0] for df in dfs[1:]], dfs[0].shape[0]))]
    if len(size_err) > 0:
        log.error("All files must have same number of particles. Offending files:\n%s" % ", ".join(size_err))
        return 1

    dfo = dfs[0]
    dfn = dfs[1]

    oq = geom.e2q_vec(np.deg2rad(dfo[star.Relion.ANGLES].values))
    nq = geom.e2q_vec(np.deg2rad(dfn[star.Relion.ANGLES].values))
    oqu = geom.normq(oq)
    nqu = geom.normq(nq)
    resq = geom.qtimes(geom.qconj(oqu), nqu)
    mu = geom.meanq(resq)
    resqu = geom.normq(resq, mu)

    si_mult = np.random.choice(resqu.shape[0]/args.multimer, args.sample/args.multimer, replace=False)
    si = np.array([si_mult[i] * args.multimer + k for i in range(si_mult.shape[0]) for k in range(args.multimer)])
    not_si = np.setdiff1d(np.arange(resqu.shape[0], dtype=np.int), si)

    samp = resqu[si, :].copy()

    t = time.time()
    d = geom.pdistq(samp, np.zeros((samp.shape[0], samp.shape[0]), dtype=np.double))
    log.info("Sample pairwise distances calculated in %0.3f s" % (time.time() - t))

    g = geom.double_center(d, inplace=False)

    t = time.time()
    vals, vecs = np.linalg.eigh(g)
    log.info("Sample Gram matrix decomposed in %0.3f s" % (time.time() - t))

    np.save(args.output + "_evals.npy", vals)
    np.save(args.output + "_evecs.npy", vecs)

    x = vecs[:, [-1, -2, -3]].dot(np.diag(np.sqrt(vals[[-1, -2, -3]])))

    np.save(args.output + "_xtrain.npy", x)

    test = resqu[not_si].copy()

    t = time.time()
    ga = geom.cdistq(test, samp, np.zeros((test.shape[0], samp.shape[0]), dtype=np.single))
    log.info("Test pairwise distances calculated in %0.3f s" % (time.time() - t))

    ga = geom.double_center(ga, reference=d, inplace=True)

    xa = ga.dot(x) / vals[[-1, -2, -3]].reshape(1, 3)

    np.save(args.output + "_xtest.npy", xa)

    vol, hdr = mrc.read(args.volume, inc_header=True)
    psz = hdr["xlen"] / hdr["nx"]
    for pc in range(2):
        keyq = geom.findkeyq(test, xa, nkey=10, pc_cyl_ptile=args.outlier_radius, pc_ptile=args.outlier_length, pc=pc)
        keyq_exp = geom.qslerp_mult_balanced(keyq, 10)
        volbase = os.path.basename(args.volume).rstrip(".mrc") + "_kpc%d" % pc + "_%.4d.mrc"
        util.write_q_series(vol, keyq_exp, os.path.join(outdir, volbase), psz=psz, order=args.spline_order)

    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("original")
    parser.add_argument("new")
    parser.add_argument("output")
    parser.add_argument("--sample-size", "-s", type=int)
    parser.add_argument("--multimer", "-m", type=int)
    parser.add_argument("--volume", "-v")
    parser.add_argument("--spline-order", "-ord", default=3)
    parser.add_argument("--outlier-radius", "-or", default=90.)
    parser.add_argument("--outlier-length", "-ol", default=25.)
    parser.add_argument("--threads", "-j", type=int)
    sys.exit(main(parser.parse_args()))
