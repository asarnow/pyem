#!/usr/bin/env python
# Copyright (C) 2016 Daniel Asarnow
# University of California, San Francisco
#
# Simple program for parsing and altering Relion .star files.
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
import argparse
import sys
import pandas as pd
from pyem.star import write_star


convert = {u'uid': None,
           u'ctf_params.akv': "_rlnVoltage",
           u'ctf_params.angast_deg': "_rlnDefocusAngle",
           u'ctf_params.angast_rad': None,
           u'ctf_params.cs': "_rlnSphericalAberration",
           u'ctf_params.detector_psize': "_rlnDetectorPixelSize",
           u'ctf_params.df1': "_rlnDefocusU",
           u'ctf_params.df2': "_rlnDefocusV",
           u'ctf_params.mag': "_rlnMagnification",
           u'ctf_params.psize': None,
           u'ctf_params.wgh': "_rlnAmplitudeContrast",
           u'data_input_relpath': "_rlnImageName",
           u'data_input_idx': None,
           u'alignments.model.U': None,
           u'alignments.model.dr': None,
           u'alignments.model.dt': None,
           u'alignments.model.ess_R': None,
           u'alignments.model.ess_S': None,
           u'alignments.model.phiC': None,
           u'alignments.model.r.0': "_rlnAngleRot",
           u'alignments.model.r.1': "_rlnAngleTilt",
           u'alignments.model.r.2': "_rlnAnglePsi",
           u'alignments.model.t.0': "_rlnCoordinateX",
           u'alignments.model.t.1': "_rlnCoordinateY"}


def main(args):
    meta = parse_metadata(args.input)
    meta["data_input_idx"] = ["%.6d" % i for i in meta["data_input_idx"]]
    meta["data_input_relpath"] = meta["data_input_idx"].str.cat(meta["data_input_relpath"], sep="@")
    rlnheaders = [convert[h] for h in meta.columns if convert[h] is not None]
    star = meta[[h for h in meta.columns if convert[h] is not None]]
    star.columns = rlnheaders
    write_star(args.output, star, reindex=True)
    return 0


def parse_metadata(csvfile):
    with open(csvfile, 'rU') as f:
        lines = enumerate(f)
        idx = -1
        headers = None
        for i, l in lines:
            if l.startswith("_header"):
                _, headers = next(lines)
                headers = headers.rstrip().split(",")
            if l.startswith("_dtypes"):
                _i, dtypes = next(lines)
                dtypes = dtypes.rstrip().split(",")
                idx = _i + 1
                break
    meta = pd.read_csv(csvfile, skiprows=idx, header=None)
    if headers is None:
        return None
    meta.columns = headers
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input Cryosparc metadata .csv file")
    parser.add_argument("output", help="Output .star file")
    sys.exit(main(parser.parse_args()))
