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
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pyfftw.interfaces.numpy_fft as fft
import scipy.ndimage as ndi
import seaborn as sns
import skimage as ski
import sys
from pyem import mrc
from pyem import star


def main(args):
    sns.set()
    if args.fast:
        df = star.parse_star(args.input, nrows=10000)
    else:
        df = star.parse_star(args.input)
    df.set_index(star.UCSF.MICROGRAPH_BASENAME, inplace=True)
    if args.mic is None:
        vc = df.index.value_counts()
        args.mic = vc.index[vc.shape[0] // 2 + args.offset_mics]  # Micrograph with median particle count.
        mic_path = df.loc[args.mic].iloc[0][star.Relion.MICROGRAPH_NAME]
    elif np.char.isnumeric(args.mic):
        args.mic = df.index[args.mic]
        mic_path = df.loc[args.mic].iloc[0][star.Relion.MICROGRAPH_NAME]
    else:
        mic_path = args.mic
        args.mic = os.path.basename(args.mic)
    im = mrc.read(mic_path, compat="mrc2014")
    im_min = np.min(im)
    im = (im - im_min) / np.max(np.abs(im - im_min))
    I = fft.fft2(im)
    GH = ndi.fourier_gaussian(I, sigma=1000)
    gH = np.real(fft.ifft2(GH))
    GL = ndi.fourier_gaussian(I, sigma=10)
    gL = np.real(fft.ifft2(GL))
    g = gL / gH
    p2, p98 = np.percentile(g, [4, 98])
    g = ski.exposure.rescale_intensity(g, in_range=(p2, p98))
    x = df.loc[args.mic][star.Relion.COORDX]
    y = df.loc[args.mic][star.Relion.COORDY]
    g = ndi.rotate(g, 90)
    if args.invertx:
        x = im.shape[0] - x
    if args.inverty:
        y = im.shape[1] - y
    if args.swapxy:
        x, y = y, x
    f, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(g, cmap="gray")
    ax.grid(None)
    circles = [plt.Circle(coord, 100, color=[0, 1, 0], fill=False, linewidth=0.5, alpha=1) for coord in zip(x, y)]
    for c in circles:
        ax.add_patch(c)
    if args.output is None:
        plt.show()
    else:
        f.savefig(args.output, dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
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
    sys.exit(main(parser.parse_args()))
