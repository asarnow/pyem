#!/usr/bin/env python2.7
# Copyright (C) 2017 Daniel Asarnow
# University of California, San Francisco
#
# Generate subparticles for "local reconstruction" methods.
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
import argparse
import glob
import numpy as np
import os
import os.path
import pandas as pd
import sys
import xml.etree.cElementTree as etree
from pyem.star import calculate_apix
from pyem.star import parse_star
from pyem.star import write_star
from pyem.star import transform_star
from pyem.star import select_classes
from pyem.star import recenter
from pyem.star import ANGLES
from pyem.star import COORDS
from pyem.star import ORIGINS
from pyem.util import euler2rot
from pyem.util import rot2euler
from pyem.util import expmap
from pyem.util import relion_symmetry_group


def main(args):
    if args.markers is None and args.sym is None:
        print("Markers and/or a symmetry group must be provided")
        return 1

    star = parse_star(args.input, keep_index=False)

    if args.apix is None:
        args.apix = calculate_apix(star)

    if args.cls is not None:
        star = select_classes(star, args.cls)

    if args.sym is not None:
        args.sym = relion_symmetry_group(args.sym)

    if args.markers is not None:
        cmmfiles = glob.glob(args.markers)
        markers = []
        for cmmfile in cmmfiles:
            cmms = parse_cmm(cmmfile) / args.apix
            markers.append(cmms[1:] - cmms[0])
        markers = np.vstack(markers)
        stars = []
        rots = [euler2rot(*np.deg2rad(r[1])) for r in star[ANGLES].iterrows()]
        shifts = star[ORIGINS].copy()
        for cm in markers:
            cm_ax = cm / np.linalg.norm(cm)
            cmr = euler2rot(*np.array([np.arctan2(cm_ax[1], cm_ax[0]), np.arccos(cm_ax[2]), 0.]))
            angles = [np.rad2deg(rot2euler(r.dot(cmr.T))) for r in rots]
            star[ANGLES] = angles
            newshifts = shifts + np.array([r.dot(cm)[:-1] for r in rots])
            star[ORIGINS] = newshifts
            star = recenter(star, inplace=True)
            stars.append(star.copy())
    else:
        stars = symmetry_expansion(star, args.sym)
    
    if args.suffix is None and not args.skip_join:
        bigstar = pd.concat(stars)
        write_star(args.output, bigstar)
    else:
        for i, star in enumerate(stars):
            write_star(os.path.join(args.output, args.suffix + "_%d" % i), star)
    return 0


def parse_cmm(cmmfile):
    tree = etree.parse(cmmfile)
    cmms = np.array([[np.double(cm.get(ax)) for ax in ['x', 'y', 'z']] for cm in tree.findall("marker")])
    return cmms


def symmetry_expansion(s, ops=[np.eye(3)]):
    for op in ops:
        yield transform_star(s, op)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="STAR file with source particles")
    parser.add_argument("output", help="Output file path (and prefix for output files)")
    parser.add_argument("--apix", "--angpix", help="Angstroms per pixel (calculate from STAR by default)")
    parser.add_argument("--class", help="Keep this class in output, may be passed multiple times",
                        action="append", type=int, dest="cls")
    parser.add_argument("--markers", help="Marker file from Chimera, or *quoted* file glob")
    parser.add_argument("--skip-join", help="Force multiple output files even if no suffix provided",
                        action="store_true", default=False)
    parser.add_argument("--suffix", help="Suffix for multiple output files")
    parser.add_argument("--sym", help="Symmetry group for symmetry expansion or symmetry-derived subparticles (Relion conventions)")

    sys.exit(main(parser.parse_args()))

