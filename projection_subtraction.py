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
try:
    import Queue
except ImportError:
    import queue
import sys
import threading
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
from numpy.fft import fftshift
from pyem import mrc
from pyem import star
from pyem import algo
from pyem import ctf
from pyem import vop
from pyem.geom.convert_numba import euler2rot
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

    if args.dest is None and args.suffix == "":
        args.dest = ""
        args.suffix = "_subtracted"

    log.info("Reading particle .star file")
    df = star.parse_star(args.input, keep_index=False)
    star.augment_star_ucsf(df)
    if not args.original:
        df[star.UCSF.IMAGE_ORIGINAL_PATH] = df[star.UCSF.IMAGE_PATH]
        df[star.UCSF.IMAGE_ORIGINAL_INDEX] = df[star.UCSF.IMAGE_INDEX]
    df.sort_values(star.UCSF.IMAGE_ORIGINAL_PATH, inplace=True, kind="mergesort")
    gb = df.groupby(star.UCSF.IMAGE_ORIGINAL_PATH)
    df[star.UCSF.IMAGE_INDEX] = gb.cumcount()
    df[star.UCSF.IMAGE_PATH] = df[star.UCSF.IMAGE_ORIGINAL_PATH].map(
        lambda x: os.path.join(
            args.dest,
            args.prefix +
            os.path.basename(x).replace(".mrcs", args.suffix + ".mrcs")))

    if args.submap_ft is None:
        log.info("Reading volume")
        submap = mrc.read(args.submap, inc_header=False, compat="relion")
        if args.submask is not None:
            log.info("Masking volume")
            submask = mrc.read(args.submask, inc_header=False, compat="relion")
            submap *= submask
        log.info("Preparing 3D FFT of volume")
        submap_ft = vop.vol_ft(submap, pfac=args.pfac, threads=min(args.threads, cpu_count()))
        log.info("Finished 3D FFT of volume")
    else:
        log.info("Loading 3D FFT from %s" % args.submap_ft)
        submap_ft = np.load(args.submap_ft)
        log.info("Loaded 3D FFT from %s" % args.submap_ft)

    sz = (submap_ft.shape[0] - 3) // args.pfac

    maxshift = np.round(np.max(np.abs(df[star.Relion.ORIGINS].values)))
    if args.crop is not None and sz < 2 * maxshift + args.crop:
        log.error("Some shifts are too large to crop (maximum crop is %d)" % (sz - 2 * maxshift))
        return 1

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
            refmap_ft = vop.vol_ft(refmap, pfac=args.pfac, threads=min(args.threads, cpu_count()))
        else:
            log.info("Loading 3D FFT from %s" % args.refmap_ft)
            refmap_ft = np.load(args.refmap_ft)
            log.info("Loaded 3D FFT from %s" % args.refmap_ft)
    else:
        coefs_method = 0
        refmap_ft = np.empty(submap_ft.shape, dtype=submap_ft.dtype)

    apix = star.calculate_apix(df)
    log.info("Computed pixel size is %f A" % apix)

    log.debug("Grouping particles by output stack")
    gb = df.groupby(star.UCSF.IMAGE_PATH)

    iothreads = threading.BoundedSemaphore(args.io_thread_pairs)
    qsize = args.io_queue_length
    fftthreads = args.fft_threads

    def init():
        global tls
        tls = threading.local()

    log.info("Instantiating thread pool with %d workers" % args.threads)
    pool = Pool(processes=args.threads, initializer=init)
    threads = []
    
    log.info("Performing projection subtraction")

    try:
        for fname, particles in gb:
            log.debug("Instantiating queue")
            queue = Queue.Queue(maxsize=qsize)
            log.debug("Create producer for %s" % fname)
            prod = threading.Thread(
                target=producer,
                args=(pool, queue, submap_ft, refmap_ft, fname, particles,
                      sx, sy, s, a, apix, coefs_method, r, nr, fftthreads, args.crop, args.pfac))
            log.debug("Create consumer for %s" % fname)
            cons = threading.Thread(
                target=consumer,
                args=(queue, fname, apix, iothreads))
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

    log.info("Finished projection subtraction")

    log.info("Writing output .star file")
    if args.crop is not None:
        df = star.recenter(df, inplace=True)
    star.simplify_star_ucsf(df)
    star.write_star(args.output, df, reindex=True)

    return 0


def subtract_outer(p1r, ptcl, submap_ft, refmap_ft, sx, sy, s, a, apix, coefs_method, r, nr, **kwargs):
    log = logging.getLogger('root')
    log.debug("%d@%s Exp %f +/- %f" % (ptcl[star.UCSF.IMAGE_ORIGINAL_INDEX], ptcl[star.UCSF.IMAGE_ORIGINAL_PATH], np.mean(p1r), np.std(p1r)))
    ft = getattr(tls, 'ft', None)
    if ft is None:
        ft = rfft2(fftshift(p1r.copy()), threads=kwargs["fftthreads"],
                   planner_effort="FFTW_ESTIMATE",
                   overwrite_input=False,
                   auto_align_input=True,
                   auto_contiguous=True)
        tls.ft = ft
    if coefs_method >= 1:
        p1 = ft(p1r.copy(), np.zeros(ft.output_shape, dtype=ft.output_dtype)).copy()
    else:
        p1 = np.empty(ft.output_shape, ft.output_dtype)

    p1s = subtract(p1, submap_ft, refmap_ft, sx, sy, s, a, apix,
                   ptcl[star.Relion.DEFOCUSU], ptcl[star.Relion.DEFOCUSV], ptcl[star.Relion.DEFOCUSANGLE],
                   ptcl[star.Relion.PHASESHIFT], ptcl[star.Relion.VOLTAGE], ptcl[star.Relion.AC], ptcl[star.Relion.CS],
                   ptcl[star.Relion.ANGLEROT], ptcl[star.Relion.ANGLETILT], ptcl[star.Relion.ANGLEPSI],
                   ptcl[star.Relion.ORIGINX], ptcl[star.Relion.ORIGINY], coefs_method, r, nr, kwargs["pfac"])

    ift = getattr(tls, 'ift', None)
    if ift is None:
        ift = irfft2(p1s.copy(), threads=kwargs["fftthreads"],
                     planner_effort="FFTW_ESTIMATE",
                     auto_align_input=True,
                     auto_contiguous=True)
        tls.ift = ift
    p1sr = fftshift(ift(p1s.copy(), np.zeros(ift.output_shape, dtype=ift.output_dtype)).copy())
    log.debug("%d@%s Exp %f +/- %f, Sub %f +/- %f" % (ptcl[star.UCSF.IMAGE_ORIGINAL_INDEX], ptcl[star.UCSF.IMAGE_ORIGINAL_PATH], np.mean(p1r), np.std(p1r), np.mean(p1sr), np.std(p1sr)))
    new_image = p1r - p1sr
    if kwargs["crop"] is not None:
        orihalf = new_image.shape[0] // 2
        newhalf = kwargs["crop"] // 2
        x = orihalf - np.int(np.round(ptcl[star.Relion.ORIGINX]))
        y = orihalf - np.int(np.round(ptcl[star.Relion.ORIGINY]))
        new_image = new_image[y - newhalf:y + newhalf, x - newhalf:x + newhalf]
    return new_image


@numba.jit(cache=False, nopython=True, nogil=True)
def subtract(p1, submap_ft, refmap_ft,
             sx, sy, s, a, apix, def1, def2, angast, phase, kv, ac, cs,
             az, el, sk, xshift, yshift, coefs_method, r, nr, pfac):
    c = ctf.eval_ctf(s / apix, a, def1, def2, angast, phase, kv, ac, cs, bf=0, lp=2 * apix)
    orient = euler2rot(np.deg2rad(az), np.deg2rad(el), np.deg2rad(sk))
    pshift = np.exp(-2 * np.pi * 1j * (-xshift * sx + -yshift * sy))
    p2 = vop.interpolate_slice_numba(submap_ft, orient, pfac=pfac)
    p2 *= pshift
    if coefs_method < 1:
        # p1s = p1 - p2 * c
        p1s = p2 * c
    elif coefs_method == 1:
        p3 = vop.interpolate_slice_numba(refmap_ft, orient, pfac=pfac)
        p3 *= pshift
        frc = np.abs(algo.bincorr_nb(p1, p3 * c, r, nr))
        coefs = np.take(frc, r)
        # p1s = p1 - p2 * c * coefs
        p1s = p2 * c * coefs
    return p1s


def producer(pool, queue, submap_ft, refmap_ft, fname, particles,
             sx, sy, s, a, apix, coefs_method, r, nr, fftthreads=1, crop=None, pfac=2):
    log = logging.getLogger('root')
    log.debug("Producing %s" % fname)
    zreader = mrc.ZSliceReader(particles[star.UCSF.IMAGE_ORIGINAL_PATH].iloc[0])
    for i, ptcl in particles.iterrows():
        log.debug("Produce %d@%s" % (ptcl[star.UCSF.IMAGE_ORIGINAL_INDEX], ptcl[star.UCSF.IMAGE_ORIGINAL_PATH]))
        # p1r = mrc.read_imgs(stack[i], idx[i] - 1, compat="relion")
        p1r = zreader.read(ptcl[star.UCSF.IMAGE_ORIGINAL_INDEX])
        log.debug("Apply")
        ri = pool.apply_async(
            subtract_outer,
            (p1r, ptcl, submap_ft, refmap_ft, sx, sy, s, a, apix, coefs_method, r, nr),
            {"fftthreads": fftthreads, "crop": crop, "pfac": pfac})
        log.debug("Put")
        queue.put((ptcl[star.UCSF.IMAGE_INDEX], ri), block=True)
        log.debug("Queue for %s is size %d" % (ptcl[star.UCSF.IMAGE_ORIGINAL_PATH], queue.qsize()))
    zreader.close()
    log.debug("Put poison pill")
    queue.put((-1, None), block=True)


def consumer(queue, stack, apix=1.0, iothreads=None):
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

    parser = argparse.ArgumentParser(version="projection_subtraction.py 2.1b")
    parser.add_argument("input", type=str,
                        help="STAR file with original particles")
    parser.add_argument("output", type=str,
                        help="STAR file with subtracted particles)")
    parser.add_argument("--dest", "-d", type=str, help="Destination directory for subtracted particle stacks")
    parser.add_argument("--refmap", "-r", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--submap", "-s", type=str, help="Map used to calculate subtracted projections")
    parser.add_argument("--refmap_ft", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--submap_ft", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--submask", type=str, help="Mask to apply to submap before subtracting")
    parser.add_argument("--original", help="Read original particle images instead of current", action="store_true")
    parser.add_argument("--threads", "-j", type=int, default=None, help="Number of simultaneous threads")
    parser.add_argument("--io-thread-pairs", type=int, default=1)
    parser.add_argument("--io-queue-length", type=int, default=1000)
    parser.add_argument("--fft-threads", type=int, default=1)
    parser.add_argument("--pfac", help="Padding factor for 3D FFT", type=int, default=2)
    parser.add_argument("--loglevel", "-l", type=str, default="WARNING", help="Logging level and debug output")
    parser.add_argument("--low-cutoff", "-L", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--high-cutoff", "-H", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--crop", help="Size to crop recentered output images", type=int)
    parser.add_argument("--prefix", type=str, help="Additional prefix for particle stacks", default="")
    parser.add_argument("--suffix", type=str, help="Additional suffix for particle stacks", default="")

    sys.exit(main(parser.parse_args()))
