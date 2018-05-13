#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2018 Daniel Asarnow, Eugene Palovcak
# University of California, San Francisco
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
import logging
import numba
import numpy as np
import os.path
import pandas as pd
# import pyfftw
import Queue
import sys
import threading
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
from numpy.fft import fftshift
from pyem import mrc
from pyem.algo import bincorr_nb
from pyem.ctf import eval_ctf
from pyem.star import calculate_apix
from pyem.star import parse_star
from pyem.star import write_star
from pyem.util.convert_numba import euler2rot
from pyem.vop import interpolate_slice_numba
from pyem.vop import vol_ft
# from numpy.fft import rfft2
# from numpy.fft import irfft2
# from pyfftw.interfaces.numpy_fft import rfft2
# from pyfftw.interfaces.numpy_fft import irfft2
from pyfftw.builders import rfft2
from pyfftw.builders import irfft2


def main(args):
    """
    Projection subtraction program entry point.
    :param args: Command-line arguments parsed by ArgumentParser.parse_args()
    :return: Exit status
    """
    log = logging.getLogger('root')
    hdlr = logging.StreamHandler(sys.stdout)
    log.addHandler(hdlr)
    log.setLevel(logging.getLevelName(args.loglevel.upper()))

    log.debug("Reading particle .star file")
    df = parse_star(args.input, keep_index=False)
    df.reset_index(inplace=True)
    df["rlnImageOriginalName"] = df["rlnImageName"]
    df["ucsfOriginalParticleIndex"], df["ucsfOriginalImagePath"] = \
        df["rlnImageOriginalName"].str.split("@").str
    df["ucsfOriginalParticleIndex"] = pd.to_numeric(
        df["ucsfOriginalParticleIndex"])
    df.sort_values("rlnImageOriginalName", inplace=True, kind="mergesort")
    gb = df.groupby("ucsfOriginalImagePath")
    df["ucsfParticleIndex"] = gb.cumcount() + 1
    df["ucsfImagePath"] = df["ucsfOriginalImagePath"].map(
        lambda x: os.path.join(
            args.dest,
            args.prefix +
            os.path.basename(x).replace(".mrcs", args.suffix + ".mrcs")))
    df["rlnImageName"] = df["ucsfParticleIndex"].map(
        lambda x: "%.6d" % x).str.cat(df["ucsfImagePath"], sep="@")
    log.debug("Read particle .star file")

    if args.submap_ft is None:
        submap = mrc.read(args.submap, inc_header=False, compat="relion")
        submap_ft = vol_ft(submap, threads=min(args.threads, cpu_count()))
    else:
        log.debug("Loading %s" % args.submap_ft)
        submap_ft = np.load(args.submap_ft)
        log.debug("Loaded %s" % args.submap_ft)

    sz = submap_ft.shape[0] // 2 - 1
    sx, sy = np.meshgrid(np.fft.rfftfreq(sz), np.fft.fftfreq(sz))
    s = np.sqrt(sx ** 2 + sy ** 2)
    r = s * sz
    r = np.round(r).astype(np.int64)
    r[r > sz // 2] = sz // 2 + 1
    nr = np.max(r) + 1
    a = np.arctan2(sy, sx)

    if args.refmap is not None:
        coefs_method = 1
        if args.refmap_ft is None:
            refmap = mrc.read(args.refmap, inc_header=False, compat="relion")
            refmap_ft = vol_ft(refmap, threads=min(args.threads, cpu_count()))
        else:
            log.debug("Loading %s" % args.refmap_ft)
            refmap_ft = np.load(args.refmap_ft)
            log.debug("Loaded %s" % args.refmap_ft)
    else:
        coefs_method = 0
        refmap_ft = np.empty(submap_ft.shape, dtype=submap_ft.dtype)
    apix = calculate_apix(df)

    log.debug("Constructing particle metadata references")
    # npart = df.shape[0]
    idx = df["ucsfOriginalParticleIndex"].values
    stack = df["ucsfOriginalImagePath"].values.astype(np.str, copy=False)
    def1 = df["rlnDefocusU"].values
    def2 = df["rlnDefocusV"].values
    angast = df["rlnDefocusAngle"].values
    phase = df["rlnPhaseShift"].values
    kv = df["rlnVoltage"].values
    ac = df["rlnAmplitudeContrast"].values
    cs = df["rlnSphericalAberration"].values
    az = df["rlnAngleRot"].values
    el = df["rlnAngleTilt"].values
    sk = df["rlnAnglePsi"].values
    xshift = df["rlnOriginX"].values
    yshift = df["rlnOriginY"].values
    new_idx = df["ucsfParticleIndex"].values
    new_stack = df["ucsfImagePath"].values.astype(np.str, copy=False)

    log.debug("Grouping particles by output stack")
    gb = df.groupby("ucsfImagePath")

    iothreads = threading.BoundedSemaphore(args.io_thread_pairs)
    qsize = args.io_queue_length
    fftthreads = args.fft_threads
    # pyfftw.interfaces.cache.enable()

    log.debug("Instantiating worker pool")
    pool = Pool(processes=args.threads)
    threads = []

    try:
        for fname, particles in gb.indices.iteritems():
            log.debug("Instantiating queue")
            queue = Queue.Queue(maxsize=qsize)
            log.debug("Create producer for %s" % fname)
            prod = threading.Thread(
                target=producer,
                args=(pool, queue, submap_ft, refmap_ft, fname, particles, idx,
                      stack, sx, sy, s, a, apix, def1, def2, angast, phase, kv,
                      ac, cs, az, el, sk, xshift, yshift, new_idx, new_stack,
                      coefs_method, r, nr, fftthreads))
            log.debug("Create consumer for %s" % fname)
            cons = threading.Thread(
                target=consumer,
                args=(queue, fname, apix, fftthreads, iothreads))
            threads.append((prod, cons))
            iothreads.acquire()
            log.debug("iotheads at %d" % iothreads._Semaphore__value)
            log.debug("Start consumer for %s" % fname)
            cons.start()
            log.debug("Start producer for %s" % fname)
            prod.start()
    except KeyboardInterrupt:
        log.debug("Main thread wants out!")

    for pair in threads:
        for thread in pair:
            try:
                thread.join()
            except RuntimeError as e:
                log.debug(e)

    pool.close()
    pool.join()
    pool.terminate()

    df.drop([c for c in df.columns if "ucsf" in c or "eman" in c],
            axis=1, inplace=True)

    df.set_index("index", inplace=True)
    df.sort_index(inplace=True, kind="mergesort")

    write_star(args.output, df, reindex=True)

    return 0


def subtract_outer(*args, **kwargs):
    ft = rfft2(fftshift(args[0]), threads=kwargs["fftthreads"],
               planner_effort="FFTW_ESTIMATE",
               overwrite_input=False,
               auto_align_input=True,
               auto_contiguous=True)
    p1 = ft()
    p1s = subtract(p1, *args[1:])
    ift = irfft2(p1s, threads=kwargs["fftthreads"],
                 planner_effort="FFTW_ESTIMATE",
                 auto_align_input=True,
                 auto_contiguous=True)
    new_image = fftshift(ift())
    return new_image


@numba.jit(cache=False, nopython=True, nogil=True)
def subtract(p1, submap_ft, refmap_ft,
             sx, sy, s, a, apix, def1, def2, angast, phase, kv, ac, cs,
             az, el, sk, xshift, yshift, coefs_method, r, nr):
    c = eval_ctf(s / apix, a, def1, def2, angast, phase, kv, ac, cs, bf=0,
                 lp=2 * apix)
    orient = euler2rot(np.deg2rad(az), np.deg2rad(el), np.deg2rad(sk))
    pshift = np.exp(-2 * np.pi * 1j * (-xshift * sx + -yshift * sy))
    p2 = interpolate_slice_numba(submap_ft, orient)
    p2 *= pshift
    if coefs_method < 1:
        p1s = p1 - p2 * c
    elif coefs_method == 1:
        p3 = interpolate_slice_numba(refmap_ft, orient)
        p3 *= pshift
        frc = np.abs(bincorr_nb(p1, p3 * c, r, nr))
        coefs = np.take(frc, r)
        p1s = p1 - p2 * c * coefs
    return p1s


def producer(pool, queue, submap_ft, refmap_ft, fname, particles, idx, stack,
             sx, sy, s, a, apix, def1, def2, angast, phase, kv, ac, cs,
             az, el, sk, xshift, yshift,
             new_idx, new_stack, coefs_method, r, nr, fftthreads=1):
    log = logging.getLogger('root')
    log.debug("Producing %s" % fname)
    zreader = mrc.ZSliceReader(stack[particles[0]])
    for i in particles:
        log.debug("Produce %d@%s" % (idx[i], stack[i]))
        # p1r = mrc.read_imgs(stack[i], idx[i] - 1, compat="relion")
        p1r = zreader.read(idx[i] - 1)
        log.debug("Apply")
        ri = pool.apply_async(
            subtract_outer,
            (p1r, submap_ft, refmap_ft,
             sx, sy, s, a, apix,
             def1[i], def2[i], angast[i],
             phase[i], kv[i], ac[i], cs[i],
             az[i], el[i], sk[i], xshift[i], yshift[i],
             coefs_method, r, nr), {"fftthreads": fftthreads})
        log.debug("Put")
        queue.put((new_idx[i], ri), block=True)
        log.debug("Queue for %s is size %d" % (stack[i], queue.qsize()))
    zreader.close()
    # Either the poison-pill-put blocks, we have multiple queues and
    # consumers, or the consumer knows maps results to multiple files.
    log.debug("Put poison pill")
    queue.put((-1, None), block=False)


def consumer(queue, stack, apix=1.0, fftthreads=1, iothreads=None):
    log = logging.getLogger('root')
    with mrc.ZSliceWriter(stack, psz=apix) as zwriter:
        while True:
            log.debug("Get")
            i, ri = queue.get(block=True)
            log.debug("Got %d, queue for %s is size %d" %
                      (i, stack, queue.qsize()))
            if i == -1:
                break
            new_image = ri.get()
            log.debug("Result for %d was shape (%d,%d)" %
                      (i, new_image.shape[0], new_image.shape[1]))
            zwriter.write(new_image)
            queue.task_done()
            log.debug("Wrote %d to %d@%s" % (i, zwriter.i, stack))
    if iothreads is not None:
        iothreads.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(version="projection_subtraction.py 2.0a")
    parser.add_argument("input", type=str,
                        help="STAR file with original particles")
    parser.add_argument("output", type=str,
                        help="STAR file with subtracted particles)")
    parser.add_argument("--dest", "-d", type=str, help="Destination directory for subtracted particle stacks")
    parser.add_argument("--refmap", "-r", type=str, help="Map used to calculate reference projections")
    parser.add_argument("--submap", "-s", type=str, help="Map used to calculate subtracted projections")
    parser.add_argument("--refmap_ft", type=str, help="Fourier transform used to calculate reference projections (.npy)")
    parser.add_argument("--submap_ft", type=str, help="Fourier transform used to calculate subtracted projections (.npy)")
    parser.add_argument("--threads", "-j", type=int, default=None, help="Number of simultaneous threads")
    parser.add_argument("--io-thread-pairs", type=int, default=1)
    parser.add_argument("--io-queue-length", type=int, default=1000)
    parser.add_argument("--fft-threads", type=int, default=1)
    parser.add_argument("--loglevel", "-l", type=str, default="WARNING", help="Logging level and debug output")
    parser.add_argument("--low-cutoff", "-L", type=float, default=0.0, help="Low cutoff frequency (Å)")
    parser.add_argument("--high-cutoff", "-H", type=float, default=0.5, help="High cutoff frequency (Å)")
    parser.add_argument("--prefix", type=str, help="Additional prefix for particle stacks", default="")
    parser.add_argument("--suffix", type=str, help="Additional suffix for particle stacks", default="")

    sys.exit(main(parser.parse_args()))
