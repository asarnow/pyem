#!/usr/bin/env python
# Copyright (C) 2022 Daniel Asarnow
# University of Washington
#
# Quickly display particle picks on a micrograph to verify coordinates.
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
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pyfftw.interfaces.numpy_fft as fft
import scipy.ndimage as ndi
import seaborn as sns
import sys
from pyem import ctf
from pyem import mrc
from pyem import star


def main(args):
    log = logging.getLogger('root')
    hdlr = logging.StreamHandler(sys.stdout)
    log.addHandler(hdlr)
    log.setLevel(logging.getLevelName(args.loglevel.upper()))
    sns.set()
    if args.fast:
        df = star.parse_star(args.input, nrows=10000)
    else:
        df = star.parse_star(args.input)
    gb = df.groupby(star.UCSF.MICROGRAPH_BASENAME)
    if args.mic is None:
        log.info("Searching for median micrograph")
        args.mic = gb.size().argsort().index[gb.ngroups // 2 + args.offset_mics]  # Micrograph w/ median particle count.
        group = gb.get_group(args.mic)
        log.info("Median micrograph has %d particles" % group.shape[0])
        mic_path = group.iloc[0][star.Relion.MICROGRAPH_NAME]
    elif np.char.isnumeric(args.mic):
        log.info("Using micrograph %d" % int(args.mic))
        group = gb.nth[int(args.mic)]
        args.mic = group.index[0]
        mic_path = group.iloc[0][star.Relion.MICROGRAPH_NAME]
    else:
        mic_path = args.mic
        args.mic = os.path.basename(args.mic)
        group = gb.get_group(args.mic)
    x = group[star.Relion.COORDX]
    y = group[star.Relion.COORDY]
    log.info("Selected %s for display" % mic_path)
    im, hdr = mrc.read(mic_path, compat="mrc2014", inc_header=True)
    im_min = np.min(im)
    im = (im - im_min) / np.max(np.abs(im - im_min))
    I = fft.rfft2(im)
    if args.phase_flip:
        log.info("Phase flipping micrograph")
        group_avg = group.mean(numeric_only=True)
        apix = hdr['xlen'] / hdr['nx']
        sx, sy = np.meshgrid(np.fft.rfftfreq(im.shape[1]), np.fft.fftfreq(im.shape[0]))
        s = np.sqrt(sx ** 2 + sy ** 2)
        a = np.arctan2(sy, sx)
        c = ctf.eval_ctf(s / apix, a,
             group_avg[star.Relion.DEFOCUSU], group_avg[star.Relion.DEFOCUSV],
             group_avg[star.Relion.DEFOCUSANGLE],
             group_avg[star.Relion.PHASESHIFT], group_avg[star.Relion.VOLTAGE],
             group_avg[star.Relion.AC], group_avg[star.Relion.CS], bf=0,
             lp=2 * apix)
        c = np.sign(c)
        I *= c
    if args.filt:
        GH = ndi.fourier_gaussian(I, sigma=1000, n=im.shape[0])
        gH = np.real(fft.irfft2(GH))
        GL = ndi.fourier_gaussian(I, sigma=10, n=im.shape[0])
        gL = np.real(fft.irfft2(GL))
        g = gL / gH
        p2, p98 = np.percentile(g, [4, 98])
        g = np.clip(g, p2, p98)
    else:
        g = np.real(fft.irfft2(I))
    g = g[:, ::-1].T
    if args.invertx:
        log.info("Inverting X coordinates")
        x = im.shape[0] - x
    if args.inverty:
        log.info("Inverting Y coordinates")
        y = im.shape[1] - y
    if args.swapxy:
        x, y = y, x
    f, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(g, cmap="gray")
    ax.grid(None)
    if args.disp:
        circles = [plt.Circle(coord, 100, color=[0, 1, 0], fill=False, linewidth=0.5, alpha=1) for coord in zip(x, y)]
        for c in circles:
            ax.add_patch(c)
    if args.output is None:
        plt.show()
    else:
        log.info("Saving figure to %s" % args.output)
        f.savefig(args.output, dpi=300, bbox_inches="tight")
    return 0


def _main_():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="STAR file with particle coordinates and valid relative micrograph paths")
    parser.add_argument("output", help="Output image (show pyplot window if omitted)", nargs="?")
    parser.add_argument("--micrograph", "-m", dest="mic",
                        help="Path, basename, or numeric index of specific micrograph for display")
    parser.add_argument("--offset-micrographs", "-o", type=int, default=0, dest="offset_mics",
                        help="Display micrograph with N positions more or fewer particles than median"
                             "(e.g. if the automatically selected micrograph is not good)")
    parser.add_argument("--fast", "-f", action="store_true", help="Only read the first few thousand particles")
    parser.add_argument("--invertx", "-x", action="store_true", help="Subtract coordinate from micrograph size in X")
    parser.add_argument("--inverty", "-y", action="store_true", help="Subtract coordinate from micrograph size in Y")
    parser.add_argument("--swapxy", "-s", action="store_true",
                        help="Swap X & Y (NOT THE SAME as --swapxy in csparc2star.py)")
    parser.add_argument("--phase-flip", "-p", action="store_true", help="Flip CTF phases in micrograph before display")
    parser.add_argument("--nodisp", "-nd", dest="disp", help="Don't display particles, micrograph only",
                        action="store_false")
    parser.add_argument("--nofilt", "-nf", dest="filt", help="Skip flat-fielding, etc", action="store_false")
    parser.add_argument("--loglevel", "-l", help="Set log verbosity", default="warning")
    sys.exit(main(parser.parse_args()))


if __name__ == "__main__":
    _main_()

