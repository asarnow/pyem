# Copyright (C) 2017 Daniel Asarnow
# University of California, San Francisco
#
# Handles metadata from cryoSPARC v0.65 and earlier.
# See README file for more information.
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
import numpy as np
import pandas as pd
from pyem import geom
from pyem import star


def parse_cryosparc_065_csv(csvfile):
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
    meta = pd.read_csv(csvfile, skiprows=idx, header=None, skip_blank_lines=True)
    if headers is None:
        return None
    meta.columns = headers
    return meta


def cryosparc_065_csv2star(meta, minphic=0):
    general = {u'uid': None,
               u'split': "rlnRandomSubset",
               u'ctf_params.akv': "rlnVoltage",
               u'ctf_params.angast_deg': "rlnDefocusAngle",
               u'ctf_params.angast_rad': None,
               u'ctf_params.cs': "rlnSphericalAberration",
               u'ctf_params.detector_psize': "rlnDetectorPixelSize",
               u'ctf_params.df1': "rlnDefocusU",
               u'ctf_params.df2': "rlnDefocusV",
               u'ctf_params.mag': "rlnMagnification",
               u'ctf_params.phase_shift': "rlnPhaseShift",
               u'ctf_params.psize': None,
               u'ctf_params.wgh': "rlnAmplitudeContrast",
               u'data_input_relpath': "rlnImageName",
               u'data_input_idx': None}
    model = {u'alignments.model.U': None,
             u'alignments.model.dr': None,
             u'alignments.model.dt': None,
             u'alignments.model.ess_R': None,
             u'alignments.model.ess_S': None,
             u'alignments.model.phiC': None,
             u'alignments.model.r.0': "rlnAngleRot",
             u'alignments.model.r.1': "rlnAngleTilt",
             u'alignments.model.r.2': "rlnAnglePsi",
             u'alignments.model.t.0': "rlnOriginX",
             u'alignments.model.t.1': "rlnOriginY"}
    meta["data_input_idx"] = ["%.6d" % (i + 1) for i in meta[
        "data_input_idx"]]  # Reformat particle idx for Relion.
    if "data_input_relpath" not in meta.columns:
        meta["data_input_relpath"] = ""
    meta["data_input_relpath"] = meta["data_input_idx"].str.cat(
        meta["data_input_relpath"], sep="@")  # Construct rlnImageName field.
    # Take care of trivial mappings.
    rlnheaders = [general[h] for h in meta.columns if
                  h in general and general[h] is not None]
    df = meta[[h for h in meta.columns if
               h in general and general[h] is not None]].copy()
    df.columns = rlnheaders
    df = star.augment_star_ucsf(df, inplace=True)
    if "rlnRandomSubset" in df.columns:
        df["rlnRandomSubset"] = df["rlnRandomSubset"].apply(
            lambda x: ord(x) - 64)
    if "rlnPhaseShift" in df.columns:
        df["rlnPhaseShift"] = np.rad2deg(df["rlnPhaseShift"])
    # Class assignments and other model parameters.
    phic = meta[[h for h in meta.columns if
                 "phiC" in h]]  # Posterior probability over class assignments.
    if len(phic.columns) > 0:  # Check class assignments exist in input.
        # phic.columns = [int(h[21]) for h in meta.columns if "phiC" in h]
        phic.columns = range(len(phic.columns))
        cls = phic.idxmax(axis=1)
        for p in model:
            if model[p] is not None:
                pspec = p.split("model")[1]
                param = meta[[h for h in meta.columns if pspec in h]]
                if len(param.columns) > 0:
                    param.columns = phic.columns
                    df[model[p]] = param.lookup(param.index, cls)
        df["rlnClassNumber"] = cls + 1  # Add one for Relion indexing.
    else:
        for p in model:
            if model[p] is not None and p in meta.columns:
                df[model[p]] = meta[p]
        df["rlnClassNumber"] = 1
    if df.columns.intersection(star.Relion.ANGLES).size == len(star.Relion.ANGLES):
        df[star.Relion.ANGLES] = np.rad2deg(geom.rot2euler(geom.expmap(df[star.Relion.ANGLES].values)))
    if phic is not None and minphic > 0:
        mask = np.all(phic < minphic, axis=1)
        df.drop(df[mask].index, inplace=True)
    return df
