#!/usr/bin/env python2.7
# Copyright (C) 2015 Eugene Palovcak, Daniel Asarnow
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
import os.path
import pandas as pd
import sys
from EMAN2 import EMData
from EMAN2 import EMNumPy as emn
from EMAN2 import Transform
from EMAN2 import Vec3f
from pathos.multiprocessing import ProcessPool as Pool
from pyem.mrc import append
from pyem.mrc import read_imgs
from pyem.mrc import write
from pyem.star import parse_star
from pyem.star import write_star
from sparx import filt_ctf
from sparx import generate_ctf


def main(args):
    """
    Projection subtraction program entry point.
    :param args: Command-line arguments parsed by ArgumentParser.parse_args()
    :return: Exit status
    """

    log = logging.getLogger(__name__)
    log.setLevel(logging.getLevelName(args.loglevel.upper()))

    # rchop = lambda x, y: x if not x.endswith(y) or len(y) == 0 else x[:-len(y)]
    # args.output = rchop(args.output, ".star")
    # args.suffix = rchop(args.suffix, ".mrc")
    # args.suffix = rchop(args.suffix, ".mrcs")

    star = parse_star(args.input, keep_index=False)

    sub_dens = EMData(args.submap)

    if args.wholemap is not None:
        dens = EMData(args.wholemap)
    elif args.low_cutoff is not None or args.high_cutoff is not None:
        log.error("Reference map is required for FRC normalization.")
        return 1

    if args.recenter:  # Compute difference vector between new and old mass centers.
        if args.wholemap is None:
            log.error("Reference map required for automatic recentering.")
            return 1

        new_dens = dens - sub_dens
        # Note the sign of the shift in coordinate frame is opposite the shift in the CoM.
        recenter = Vec3f(*dens.phase_cog()[:3]) - Vec3f(*new_dens.phase_cog()[:3])
        # TODO shift(dens, recenter)
        # TODO shift(sub_dens, recenter)
        # TODO transform_star(star, recenter)

    star.reset_index(inplace=True)
    star["rlnOriginalImageName"] = star["rlnImageName"]
    star["ucsfOriginalParticleIndex"], star["ucsfOriginalImagePath"] = star["rlnOriginalImageName"].str.split("@").str
    star["ucsfOriginalParticleIndex"] = pd.to_numeric(star["ucsfOriginalParticleIndex"])
    star.sort_values("rlnOriginalImageName", inplace=True, kind="mergesort")
    gb = star.groupby("ucsfOriginalImagePath")
    star["ucsfParticleIndex"] = gb.cumcount()
    star["ucsfImagePath"] = star["ucsfOriginalImagePath"].map(lambda x: os.path.join(args.suffix, os.path.basename(x)))
    star["rlnImageName"] = star["ucsfParticleIndex"].map(lambda x: "%.6d" % x).str.cat(star["ucsfImagePath"], sep="@")

    # Extra columns for EMAN2 CTF.
    star["emanDefocusAngle"] = 90.0 - star['rlnDefocusAngle']
    star["emanDefocusDifference"] = (star['rlnDefocusU'] - star['rlnDefocusV']) / 10000.0
    star["emanDefocus"] = (star['rlnDefocusU'] + star['rlnDefocusV']) / 20000.0
    star["emanPixelSize"] = star['rlnDetectorPixelSize'] / star['rlnMagnification'] * 10000.0
    star["emanBFactor"] = 0.

    # subtract(star, sub_dens, dens, low_cutoff=args.low_cutoff, high_cutoff=args.high_cutoff)

    gb = star.groupby("ucsfOriginalImagePath")

    subf = lambda x: subtract(x[1], sub_dens, dens, args.low_cutoff, args.high_cutoff)

    pool = Pool(nodes=args.nproc)

    results = pool.imap(subf, [(name, group) for name, group in gb])
    codes = list(results)

    pool.close()
    pool.join()
    pool.terminate()

    star.drop([c for c in star.columns if "ucsf" in c or "eman" in c], axis=1, inplace=True)

    star.set_index("index", inplace=True)
    star.sort_index(inplace=True, kind="mergesort")

    write_star(args.output, star, reindex=True)

    return 0


def subtract(s, sub_dens, dens=None, low_cutoff=None, high_cutoff=None):
    for i, row in s.iterrows():
        img = read_imgs(row["ucsfOriginalImagePath"], row["ucsfOriginalParticleIndex"] - 1, 1)
        ptcl = emn.numpy2em(img)
        ctfproj = make_proj(dens, row)
        ctfproj_sub = make_proj(sub_dens, row)
        if dens is not None:
            ptcl_sub = ptcl.process("math.sub.optimal",
                                    {"ref": ctfproj, "actual": ctfproj_sub, "low_cutoff_frequency": low_cutoff,
                                     "high_cutoff_frequency": high_cutoff})
        else:
            ptcl_sub = ptcl - ctfproj_sub
        fname = row["ucsfImagePath"]
        subimg = emn.em2numpy(ptcl_sub)
        if row["ucsfParticleIndex"] == 0:
            write(fname, subimg, psz=row["emanPixelSize"])
        else:
            append(fname, subimg)
    return 0


def make_proj(dens, row):
    """
    Project and CTF filter density according to particle metadata.
    :param dens: EMData density
    :param row: Metadata row for particle
    :return: CTF-filtered projection
    """
    t = Transform()
    t.set_rotation(
        {'psi': row["rlnAnglePsi"], 'phi': row["rlnAngleRot"], 'theta': row["rlnAngleTilt"], 'type': 'spider'})
    t.set_trans(-row["rlnOriginX"] * row["emanPixelSize"], -row["rlnOriginY"] * row["emanPixelSize"])
    proj = dens.project("standard", t)
    ctf_params = row[["emanDefocus", "rlnSphericalAberration", "rlnVoltage", "emanPixelSize", "emanBFactor",
                     "rlnAmplitudeContrast", "emanDefocusDifference", "emanDefocusAngle"]]
    ctf = generate_ctf(ctf_params)
    ctf_proj = filt_ctf(proj, ctf)
    return ctf_proj


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(version="projection_subtraction.py 1.9")
    parser.add_argument("--input", type=str, help="RELION .star file listing input particle image stack(s)")
    parser.add_argument("--wholemap", type=str, help="Map used to calculate projections for normalization")
    parser.add_argument("--submap", type=str, help="Map used to calculate subtracted projections")
    parser.add_argument("--output", type=str, help="RELION .star file for listing output particle image stack(s)")
    parser.add_argument("--nproc", type=int, default=None, help="Number of parallel processes")
    parser.add_argument("--maxchunk", type=int, default=1000, help="Maximum task chunk size")
    parser.add_argument("--loglevel", type=str, default="WARNING", help="Logging level and debug output")
    parser.add_argument("--recenter", action="store_true", default=False,
                        help="Shift particle origin to new center of mass")
    parser.add_argument("--low-cutoff", type=float, default=0.0, help="Low cutoff frequency in FRC normalization")
    parser.add_argument("--high-cutoff", type=float, default=0.7071, help="High cutoff frequency in FRC normalization")
    parser.add_argument("suffix", type=str, help="Relative path and suffix for output image stack(s)")

    sys.exit(main(parser.parse_args()))
