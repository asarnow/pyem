#!/usr/bin/env python
# Copyright (C) 2022 Daniel Asarnow
# University of Washington
#
# Builds a corrected_micrographs.star file from individual motion
# correction .star files, such as those from MotionCor2 -OutStar 1.
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
import glob
import logging
import os.path
import pandas as pd
import sys
from pyem import star


def main(args):
    log = logging.getLogger('root')
    hdlr = logging.StreamHandler(sys.stdout)
    log.addHandler(hdlr)
    log.setLevel(logging.getLevelName(args.loglevel.upper()))
    mic_stars = glob.glob(os.path.join(args.input, "*.star"))

    if args.nodw:
        mics = [m[:-5] + ".mrc" for m in mic_stars]
    else:
        mics = [m[:-5] + "_DW.mrc" for m in mic_stars]
    for m in mics:
        if not os.path.exists(m):
            log.warning("%s does not exist" % m)

    df = pd.DataFrame({star.Relion.MICROGRAPH_NAME: mics,
                       star.Relion.MICROGRAPHMETADATA: mic_stars})

    if args.set_optics is None:
        df[star.Relion.OPTICSGROUP] = 1
    elif args.set_optics.isnumeric():
        df[star.Relion.OPTICSGROUP] = int(args.set_optics)
    else:
        tok = args.set_optics.split(",")
        df = star.set_optics_groups(df, sep=tok[0], idx=int(tok[1]), inplace=True)
        df.dropna(axis=0, how="any", inplace=True)

    mic_star = star.parse_star_tables(mic_stars[0])

    if args.apix is None:
        if star.Relion.MICROGRAPHORIGINALPIXELSIZE in mic_star['data_general']:
            args.apix = float(mic_star['data_general'][star.Relion.MICROGRAPHORIGINALPIXELSIZE])
        if "rlnOriginalPixelSize" in mic_star['data_general']:
            args.apix = float(mic_star["data_general"]["rlnOriginalPixelSize"])

    if args.bin is None and star.Relion.MICROGRAPHBINNING in mic_star['data_general']:
        args.bin = float(mic_star['data_general'][star.Relion.MICROGRAPHBINNING])
    elif args.bin is None:
        args.bin = 1.0

    df[star.Relion.MICROGRAPHORIGINALPIXELSIZE] = args.apix
    df[star.Relion.MICROGRAPHPIXELSIZE] = args.apix * args.bin
    df[star.Relion.MICROGRAPHBINNING] = args.bin

    if args.kv is None and star.Relion.VOLTAGE in mic_star['data_general']:
        args.kv = mic_star['data_general'][star.Relion.VOLTAGE]
    if args.cs is None and star.Relion.CS in mic_star['data_general']:
        args.cs = mic_star['data_general'][star.Relion.CS]
    if args.ac is None and star.Relion.AC in mic_star['data_general']:
        args.ac = mic_star['data_general'][star.Relion.AC]

    df[star.Relion.VOLTAGE] = args.kv
    df[star.Relion.CS] = args.cs
    df[star.Relion.AC] = args.ac

    if args.mtf is not None:
        df[star.Relion.MTFFILENAME] = args.mtf

    star.write_star(args.output, df)

    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", default=".", help="Motion correction output directory with individual .star files")
    parser.add_argument("output", default="corrected_micrographs.star", help="Path for output micrographs .star file")
    parser.add_argument("--nodw", action="store_true", help="Don't add _DW to micrograph names")
    parser.add_argument("--apix", "--angpix", type=float, help="Pixel size in Angstroms")
    parser.add_argument("--bin", "-b", type=float, default=None, help="Binning factor during motion correction")
    parser.add_argument("--ac", "-ac", type=float, default=None, help="Amplitude contrast")
    parser.add_argument("--cs", "-cs", type=float, default=None, help="Spherical abberation")
    parser.add_argument("--kv", "-kv", type=float, default=None, help="Acceleration voltage (kV)")
    parser.add_argument("--mtf", "-m", help="Path of detector MTF STAR file")
    parser.add_argument("--set-optics", type=str,
                        help="Determine optics groups from micrograph basename using a separator and index (e.g. _,4)")
    parser.add_argument("--loglevel", default="WARNING")
    sys.exit(main(parser.parse_args()))
