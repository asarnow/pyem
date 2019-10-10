#!/usr/bin/env python
# Copyright (C) 2017 Daniel Asarnow
# University of California, San Francisco
#
# Generate soft masks for use in classification and refinement.
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
import numpy as np
import sys
from pyem.mrc import read
from pyem.mrc import write
from pyem.vop import binary_sphere
from pyem.vop import binary_dilate
from pyem.vop import binarize_volume
from scipy.interpolate import interp1d
from scipy.ndimage import binary_closing
from scipy.ndimage import distance_transform_edt


def main(args):
    if args.threshold is None:
        print("Please provide a binarization threshold")
        return 1
    data, hdr = read(args.input, inc_header=True)
    mask = binarize_volume(data, args.threshold, minvol=args.minvol, fill=args.fill)
    if args.base_map is not None:
        base_map = read(args.base_map, inc_header=False)
        base_mask = binarize_volume(base_map, args.threshold, minvol=args.minvol, fill=args.fill)
        total_width = args.extend + args.edge_width
        excl_mask = binary_dilate(mask, total_width, strel=args.relion)
        base_mask = binary_dilate(base_mask, args.extend, strel=args.relion)
        base_mask = base_mask &~ excl_mask
        if args.overlap > 0:
            incl_mask = binary_dilate(base_mask, args.overlap, strel=args.relion) & excl_mask
            base_mask = base_mask | incl_mask
        mask = base_mask
    elif args.extend > 0:
        mask = binary_dilate(mask, args.extend, strel=args.relion)
    if args.close:
        se = binary_sphere(args.extend, False)
        mask = binary_closing(mask, structure=se, iterations=1)
    final = mask.astype(np.single)
    if args.edge_width != 0:
        dt = distance_transform_edt(~mask)  # Compute *outward* distance transform of mask.
        idx = (dt <= args.edge_width) & (dt > 0)  # Identify edge points by distance from mask.
        x = np.arange(1, args.edge_width + 1)  # Domain of the edge profile.
        if "sin" in args.edge_profile:
            y = np.sin(np.linspace(np.pi/2, 0, args.edge_width + 1))  # Range of the edge profile.
        f = interp1d(x, y[1:])
        final[idx] = f(dt[idx])  # Insert edge heights interpolated at distance transform values.
    write(args.output, final, psz=hdr["xlen"] / hdr["nx"])
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="\n".join([
            "The mask is generated according to the following procedure:",
            "  1. Threshold map",
            "  2. Optionally delete small segments",
            "  3. Optionally fill holes",
            "  4. Extend initial mask",
            "  5. Optional morphological closing",
            "  6. Add soft edge"]))
    parser.add_argument("input", help="Input volume MRC file")
    parser.add_argument("output", help="Output mask MRC file")
    parser.add_argument("--threshold", "-t", help="Threshold for initial mask",
                        type=float)
    parser.add_argument("--extend", "-e", help="Structuring element size for dilating initial mask",
                        type=int, default=0)
    parser.add_argument("--edge-width", "-w", help="Width for soft edge",
                        type=int, default=0)
    parser.add_argument("--edge-profile", "-p", help="Soft edge profile type",
                        choices=["sinusoid"],
                        default="sinusoid")
    parser.add_argument("--fill", "-f", help="Flood fill initial mask",
                        action="store_true")
    parser.add_argument("--minvol", "-m", help="Minimum volume for mask segments (pass -1 for largest segment only)", type=int, default=0)
    parser.add_argument("--close", "-c", help="Perform morphological closing", action="store_true")
    parser.add_argument("--relion", help="Mimics relion_mask_create output (slower)", action="store_true")
    parser.add_argument("--base-map", "-b", help="Create and write a matched mask instead of regular output (see project wiki)")
    parser.add_argument("--overlap", "-o", help="Overlap width for matched mask (default: %(default)d)", type=int, default=0)
    sys.exit(main(parser.parse_args()))

