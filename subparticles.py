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
import glob
import logging
import numpy as np
import os
import os.path
import sys
import xml.etree.cElementTree as etree
from pyem.star import calculate_apix
from pyem.star import parse_star
from pyem.star import write_star
from pyem.star import transform_star
from pyem.star import select_classes
from pyem.star import recenter
from pyem.star import Relion
from pyem.util import euler2rot
from pyem.util import relion_symmetry_group
from pyem.util import interleave


def main(args):
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    hdlr = logging.StreamHandler(sys.stdout)
    if args.quiet:
        hdlr.setLevel(logging.WARNING)
    else:
        hdlr.setLevel(logging.INFO)
    log.addHandler(hdlr)

    if args.markers is None and args.target is None and args.sym is None:
        log.error("A marker or symmetry group must be provided via --target, --markers, or --sym")
        return 1
    elif args.sym is None and args.markers is None and args.boxsize is None and args.origin is None:
        log.error("An origin must be provided via --boxsize, --origin, or --markers")
        return 1
    elif args.sym is not None and args.markers is None and args.target is None and \
            (args.boxsize is not None or args.origin is not None):
        log.warn("Symmetry expansion alone will ignore --target or --origin")

    if args.target is not None:
        try:
            args.target = np.array([np.double(tok) for tok in args.target.split(",")])
        except:
            log.error("Target must be comma-separated list of x,y,z coordinates")
            return 1

    if args.origin is not None:
        if args.boxsize is not None:
            log.warn("--origin supersedes --boxsize")
        try:
            args.origin = np.array([np.double(tok) for tok in args.origin.split(",")])
        except:
            log.error("Origin must be comma-separated list of x,y,z coordinates")
            return 1

    if args.marker_sym is not None:
        args.marker_sym = relion_symmetry_group(args.marker_sym)

    star = parse_star(args.input, keep_index=False)

    if args.apix is None:
        args.apix = calculate_apix(star)
        if args.apix is None:
            log.warn("Could not compute pixel size, default is 1.0 Angstroms per pixel")
            args.apix = 1.0

    if args.cls is not None:
        star = select_classes(star, args.cls)

    cmms = []

    if args.markers is not None:
        cmmfiles = glob.glob(args.markers)
        for cmmfile in cmmfiles:
            for cmm in parse_cmm(cmmfile):
                cmms.append(cmm / args.apix)
    
    if args.target is not None:
        cmms.append(args.target / args.apix)

    stars = []

    if len(cmms) > 0:
        if args.origin is not None:
            args.origin /= args.apix
        elif args.boxsize is not None:
            args.origin = np.ones(3) * args.boxsize / 2
        else:
            log.warn("Using first marker as origin")
            if len(cmms) == 1:
                log.error("Using first marker as origin, expected at least two markers")
                return 1
            args.origin = cmms[0]
            cmms = cmms[1:]

        markers = [cmm - args.origin for cmm in cmms]
        markers = [np.where(np.abs(m) < 1, 0, m) for m in markers]

        if args.marker_sym is not None and len(markers) == 1:
            markers = [op.dot(markers[0]) for op in args.marker_sym]
        elif args.marker_sym is not None:
            log.error("Exactly one marker is required for symmetry-derived subparticles")
            return 1
        
        rots = [euler2rot(*np.deg2rad(r[1])) for r in star[Relion.ANGLES].iterrows()]
        #origins = star[ORIGINS].copy()
        for m in markers:
            d = np.linalg.norm(m)
            ax = m / d
            op = euler2rot(*np.array([np.arctan2(ax[1], ax[0]), np.arccos(ax[2]), np.deg2rad(args.psi)]))
            stars.append(transform_star(star, op.T, -d, rots=rots))
        
    if args.sym is not None:
        args.sym = relion_symmetry_group(args.sym)
        if len(stars) > 0:
            stars = [se for s in stars for se in subparticle_expansion(s, args.sym, -args.displacement / args.apix)]
        else:
            stars = list(subparticle_expansion(star, args.sym, -args.displacement / args.apix))
 
    if args.recenter:
        for s in stars:
            recenter(s, inplace=True)
    
    if args.suffix is None and not args.skip_join:
        if len(stars) > 1:
            star = interleave(stars)
        else:
            star = stars[0]
        write_star(args.output, star)
    else:
        for i, star in enumerate(stars):
            write_star(os.path.join(args.output, args.suffix + "_%d" % i), star)
    return 0


def parse_cmm(cmmfile):
    tree = etree.parse(cmmfile)
    cmms = [np.array([np.double(cm.get(ax)) for ax in ['x', 'y', 'z']]) for cm in tree.findall("marker")]
    return cmms


def subparticle_expansion(s, ops=[np.eye(3)], dists=None, rots=None):
    if rots is None:
        rots = [euler2rot(*np.deg2rad(r[1])) for r in s[Relion.ANGLES].iterrows()]
    if dists is not None:
        if np.isscalar(dists):
            dists = [dists] * len(ops)
        for i in range(len(ops)):
            yield transform_star(s, ops[i], dists[i], rots=rots)
    else:
        for op in ops:
            yield transform_star(s, op, rots=rots)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="STAR file with source particles")
    parser.add_argument("output", help="Output file path (and prefix for output files)")
    parser.add_argument("--apix", "--angpix", help="Angstroms per pixel (calculate from STAR by default)", type=float)
    parser.add_argument("--boxsize", help="Particle box size in pixels (used to define origin only)", type=int)
    parser.add_argument("--class", help="Keep this class in output, may be passed multiple times",
                        action="append", type=int, dest="cls")
    parser.add_argument("--displacement", help="Distance of new origin from symmetrix axis in Angstroms",
                        type=float, default=0)
    parser.add_argument("--origin", help="Origin coordinates in Angstroms", metavar="x,y,z")
    parser.add_argument("--target", help="Target coordinates in Angstroms", metavar="x,y,z")
    parser.add_argument("--psi", help="Additional in-plane rotation of target in degrees", type=float, default=0)
    parser.add_argument("--markers", help="Marker file from Chimera, or *quoted* file glob")
    parser.add_argument("--marker-sym", help="Symmetry group for symmetry-derived subparticles (Relion conventions)")
    parser.add_argument("--recenter", help="Recenter subparticle coordinates by subtracting X and Y shifts (e.g. for "
                                           "extracting outside Relion)", action="store_true")
    parser.add_argument("--quiet", help="Don't print info messages", action="store_true")
    parser.add_argument("--skip-join", help="Force multiple output files even if no suffix provided",
                        action="store_true", default=False)
    parser.add_argument("--skip-origins", help="Skip update of particle origins", action="store_true")
    parser.add_argument("--suffix", help="Suffix for multiple output files")
    parser.add_argument("--sym", help="Symmetry group for whole-particle expansion or symmetry-derived subparticles ("
                                      "Relion conventions)")

    sys.exit(main(parser.parse_args()))

