#!/usr/bin/env python2.7
# Copyright (C) 2017 Daniel Asarnow
# University of California, San Francisco
#
# I/O routines in pyem.
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
from . import star
from . import util


def parse_par(fn):
    head_data = {"Input particle images": None,
           "Beam energy (keV)": None,
           "Spherical aberration (mm)": None,
           "Amplitude contrast": None,
           "Pixel size of images (A)": None}
    ln = 1
    skip = 0
    with open(fn, 'rU') as f:
        lastheader = False
        firstblock = True
        for l in f:
            if l.startswith("C") and firstblock:
                if "PSI" in l or "DF1" in l:
                    lastheader = True
                    headers = l.rstrip().split()
                if ":" in l:
                    tok = l.split(":")
                    tok[1] = tok[1].lstrip().rstrip()
                    tok[0] = tok[0].lstrip("C ")
                    if tok[0] in head_data:
                        try:
                            head_data[tok[0]] = float(tok[1])
                        except ValueError:
                            head_data[tok[0]] = tok[1]
                if lastheader:
                    skip = ln
                    firstblock = False
            elif l.startswith("C"):
                break
            else:
                headers = ["C", "PHI", "THETA", "PSI", "SHX", "SHY",
                        "MAG", "FILM", "DF1", "DF2", "ANGAST",
                        "OCC", "LogP", "SIGMA", "SCORE", "CHANGE"]
                break
                
            ln += 1

    if skip == 0:
        n = None
    else:
        n = ln - skip - 1
    par = pd.read_table(fn, skiprows=skip, nrows=n, delimiter="\s+", header=None, comment="C")
    par.columns = headers
    for k in head_data:
        if head_data[k] is not None:
            par[k] = head_data[k]
    return par


def par2star(par, v9=True):
    general = {"PHI": None,
            "THETA": None,
            "PSI": None,
            "SHX": None,
            "SHY": None,
            "MAG": "rlnMagnification",
            "FILM": "rlnGroupNumber",
            "DF1": "rlnDefocusU",
            "DF2": "rlnDefocusV",
            "ANGAST": "rlnDefocusAngle",
            "Beam energy (keV)": "rlnVoltage",
            "Spherical aberration (mm)": "rlnSphericalAberration",
            "Amplitude contrast": "rlnAmplitudeContrast",
            "C": None,
            "Pixel size of images (A)": None
            }
    rlnheaders = [general[h] for h in par.columns if h in general and general[h] is not None]
    star = par[[h for h in par.columns if h in general and general[h] is not None]].copy()
    star.columns = rlnheaders
    star["rlnImageName"] = ["%.6d" % (i + 1) for i in par["C"]]  # Reformat particle idx for Relion.
    star["rlnImageName"] = star["rlnImageName"].str.cat(par["Input particle images"], sep="@")
    star["rlnDetectorPixelSize"] = par["Pixel size of images (A)"] * par["MAG"] / 10000.0
    star["rlnOriginX"] = par["SHX"] / par["Pixel size of images (A)"]
    star["rlnOriginY"] = par["SHY"] / par["Pixel size of images (A)"]
    star["rlnAngleRot"] = -par["PSI"]
    star["rlnAngleTilt"] = -par["THETA"]
    star["rlnAnglePsi"] = -par["PHI"]
    return star


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
        df[star.Relion.ANGLES] = np.rad2deg(
            df[star.Relion.ANGLES].apply(lambda x: util.rot2euler(util.expmap(x)),
                                         axis=1, raw=True, broadcast=True))
    if phic is not None and minphic > 0:
        mask = np.all(phic < minphic, axis=1)
        df.drop(df[mask].index, inplace=True)
    return df


def parse_cryosparc_2_cs(csfile, passthrough=None, minphic=0):
    cs = csfile if type(csfile) is np.ndarray else np.load(csfile)
    if passthrough is not None:
        pt = passthrough if type(passthrough) is np.ndarray else np.load(passthrough)
        cs = util.join_struct_arrays([cs, pt[[n for n in pt.dtype.names if n != 'uid']]])
    general = {u'uid': None,
               u'ctf/accel_kv': "rlnVoltage",
               u'blob/psize_A': "rlnDetectorPixelSize",
               u'ctf/ac': "rlnAmplitudeContrast",
               u'ctf/cs_mm': "rlnSphericalAberration",
               u'ctf/df1_A': "rlnDefocusU",
               u'ctf/df2_A': "rlnDefocusV",
               u'ctf/df_angle_rad': "rlnDefocusAngle",
               u'ctf/phase_shift_rad': "rlnPhaseShift",
               u'ctf/cross_corr_ctffind4': "rlnCtfFigureOfMerit",
               u'ctf/ctf_fit_to_A': "rlnCtfMaxResolution",
               u'blob/path': "ucsfImagePath",
               u'blob/idx': "ucsfImageIndex"}
    model = {u'split': "rlnRandomSubset",
             u'shift': star.Relion.ORIGINS,
             u'pose': star.Relion.ANGLES,
             u'error': None,
             u'error_min': None,
             u'resid_pow': None,
             u'slice_pow': None,
             u'image_pow': None,
             u'cross_cor': None,
             u'alpha': None,
             u'weight': None,
             u'pose_ess': None,
             u'shift_ess': None,
             u'class_posterior': None,
             u'class': "rlnClassNumber",
             u'class_ess': None}
    names = [k for k in general if general[k] is not None and k in cs.dtype.names]
    df = pd.DataFrame.from_records(cs[names])
    df.columns = [general[k] for k in names]
    df.reset_index(inplace=True)
    df["rlnDefocusAngle"] = np.rad2deg(df["rlnDefocusAngle"])
    df["rlnPhaseShift"] = np.rad2deg(df["rlnPhaseShift"])
    df["rlnMagnification"] = 10000.0
    star.simplify_star_ucsf(df)

    phic = [n for n in cs.dtype.names if "class_posterior" in n]
    if len(phic) > 1:
        cls = np.argmax([cs[p] for p in phic], axis=0)
        for k in model:
            if model[k] is not None:
                names = [n for n in cs.dtype.names if n.endswith(k)]
                df[model[k]] = pd.DataFrame(np.array(
                        [cs[names[c]][i] for i, c in enumerate(cls)]))
    else:
        if "alignments2D" in phic[0]:
            model["pose"] = star.Relion.ANGLEPSI
        for k in model:
            if model[k] is not None:
                name = phic[0].replace("class_posterior", k)
                df[model[k]] = pd.DataFrame(cs[name])

    if "rlnRandomSubset" in df.columns:
        df["rlnRandomSubset"] += 1
    if "rlnClassNumber" in df.columns:
        df["rlnClassNumber"] += 1

    if df.columns.intersection(star.Relion.ANGLES).size == len(star.Relion.ANGLES):
        df[star.Relion.ANGLES] = np.rad2deg(
                df[star.Relion.ANGLES].apply(
                    lambda x: util.rot2euler(util.expmap(x)),
                    axis=1, raw=True, broadcast=True))
    else:
        df[star.Relion.ANGLEPSI] = np.rad2deg(df[star.Relion.ANGLEPSI])

    return df

