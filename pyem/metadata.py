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
import logging
import numpy as np
import pandas as pd
import sys
from . import star
from . import geom
from . import util


def parse_f9_par(fn):
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


def parse_fx_par(fn):
    df = pd.read_csv(fn, delimiter="\s+", skipfooter=2, engine="python")
    return df


def write_f9_par(fn, df):
    formatters = {"C": lambda x: "%7d" % x,
                  "PSI": lambda x: "%7.2f" % x,
                  "THETA": lambda x: "%7.2f" % x,
                  "PHI": lambda x: "%7.2f" % x,
                  "SHX": lambda x: "%9.2f" % x,
                  "SHY": lambda x: "%9.2f" % x,
                  "MAG": lambda x: "%7.0f" % x,
                  "INCLUDE": lambda x: "%5d" % x,
                  "DF1": lambda x: "%8.1f" % x,
                  "DF2": lambda x: "%8.1f" % x,
                  "ANGAST": lambda x: "%7.2f" % x,
                  "PSHIFT": lambda x: "%7.2f" % x,
                  "OCC": lambda x: "%7.2f" % x,
                  "LogP": lambda x: "%9d" % x,
                  "SIGMA": lambda x: "%10.4f" % x,
                  "SCORE": lambda x: "%7.2f" % x,
                  "CHANGE": lambda x: "%7.2f" % x}
    with open(fn, 'w') as f:
        f.write(df.to_string(formatters=formatters, index=False))


def write_fx_par(fn, df):
    formatters = {"C": lambda x: "%7d" % x,
                  "PSI": lambda x: "%7.2f" % x,
                  "THETA": lambda x: "%7.2f" % x,
                  "PHI": lambda x: "%7.2f" % x,
                  "SHX": lambda x: "%9.2f" % x,
                  "SHY": lambda x: "%9.2f" % x,
                  "MAG": lambda x: "%7.0f" % x,
                  "INCLUDE": lambda x: "%5d" % x,
                  "DF1": lambda x: "%8.1f" % x,
                  "DF2": lambda x: "%8.1f" % x,
                  "ANGAST": lambda x: "%7.2f" % x,
                  "PSHIFT": lambda x: "%7.2f" % x,
                  "OCC": lambda x: "%7.2f" % x,
                  "LogP": lambda x: "%9d" % x,
                  "SIGMA": lambda x: "%10.4f" % x,
                  "SCORE": lambda x: "%7.2f" % x,
                  "CHANGE": lambda x: "%7.2f" % x}
    with open(fn, 'w') as f:
        f.write("C           PSI   THETA     PHI       SHX       SHY     "
                "MAG  INCLUDE   DF1      DF2  ANGAST  PSHIFT     "
                "OCC      LogP      SIGMA   SCORE  CHANGE\n")
        df.to_string(buf=f, formatters=formatters, index=False, header=None)
        f.write("\nC Blank line\n")


def par2star(par, data_path, apix=1.0, cs=2.0, ac=0.07, kv=300, invert_eulers=True):
    general = {"PHI": None,
            "THETA": None,
            "PSI": None,
            "SHX": None,
            "SHY": None,
            "MAG": None,
            "FILM": star.Relion.GROUPNUMBER,
            "DF1": star.Relion.DEFOCUSU,
            "DF2": star.Relion.DEFOCUSV,
            "ANGAST": star.Relion.DEFOCUSANGLE,
            "C": None,
            "CLASS": star.Relion.CLASS
            }
    rlnheaders = [general[h] for h in par.columns if h in general and general[h] is not None]
    df = par[[h for h in par.columns if h in general and general[h] is not None]].copy()
    df.columns = rlnheaders
    df[star.UCSF.IMAGE_INDEX] = par["C"] - 1
    df[star.UCSF.IMAGE_PATH] = data_path
    df[star.Relion.IMAGEPIXELSIZE] = apix
    df[star.Relion.CS] = cs
    df[star.Relion.AC] = ac
    df[star.Relion.VOLTAGE] = kv
    df[star.Relion.ORIGINXANGST] = -par["SHX"]
    df[star.Relion.ORIGINYANGST] = -par["SHY"]
    if invert_eulers:
        df[star.Relion.ANGLEROT] = -par["PSI"]
        df[star.Relion.ANGLETILT] = -par["THETA"]
        df[star.Relion.ANGLEPSI] = -par["PHI"]
    else:
        df[star.Relion.ANGLES] = par[["PHI", "THETA", "PSI"]]
    return df


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


def cryosparc_2_cs_particle_locations(cs, df=None, swapxy=False, invertx=False, inverty=True):
    log = logging.getLogger('root')
    if df is None:
        df = pd.DataFrame()
    if u'location/center_x_frac' in cs.dtype.names:
        log.debug("Converting normalized particle coordinates to absolute")
        df[star.Relion.COORDX] = cs[u'location/center_x_frac']
        df[star.Relion.COORDY] = cs[u'location/center_y_frac']
        # df[star.Relion.MICROGRAPH_NAME] = cs[u'location/micrograph_path']
        if invertx:
            df[star.Relion.COORDX] = 1 - df[star.Relion.COORDX]
        if inverty:
            df[star.Relion.COORDY] = 1 - df[star.Relion.COORDY]
        if swapxy:
            df[star.Relion.COORDS] = np.round(df[star.Relion.COORDS] *
                                              cs['location/micrograph_shape'][:, ::-1]).astype(np.int)
        else:
            df[star.Relion.COORDS] = np.round(df[star.Relion.COORDS] * cs['location/micrograph_shape']).astype(np.int)
        log.info("Converted particle coordinates from normalized to absolute")
    return df


def cryosparc_2_cs_ctf_parameters(cs, df=None):
    log = logging.getLogger('root')
    if df is None:
        df = pd.DataFrame()
    if 'ctf/tilt_A' in cs.dtype.names:
        log.debug("Recovering beam tilt")
        df[star.Relion.BEAMTILTX] = cs['ctf/tilt_A'][:, 0]
        df[star.Relion.BEAMTILTY] = cs['ctf/tilt_A'][:, 1]
    if 'ctf/shift_A' in cs.dtype.names:
        pass
    if 'ctf/trefoil_A' in cs.dtype.names:
        pass
        # df[star.Relion.ODDZERNIKE] = cs['ctf/trefoil_A']
    if 'ctf/tetrafoil_A' in cs.dtype.names:
        pass
        # df[star.Relion.EVENZERNIKE] = cs['ctf/tetra_A']
    if 'ctf/anisomag' in cs.dtype.names:
        df[star.Relion.MAGMAT00] = cs['ctf/anisomag'][:, 0]
        df[star.Relion.MAGMAT01] = cs['ctf/anisomag'][:, 1]
        df[star.Relion.MAGMAT10] = cs['ctf/anisomag'][:, 2]
        df[star.Relion.MAGMAT11] = cs['ctf/anisomag'][:, 3]
    return df


def cryosparc_2_cs_model_parameters(cs, df=None, minphic=0):
    model = {u'split': star.Relion.RANDOMSUBSET,
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
             u'class': star.Relion.CLASS,
             u'class_ess': None}
    log = logging.getLogger('root')
    if df is None:
        df = pd.DataFrame()
    phic_names = [n for n in cs.dtype.names if "class_posterior" in n]
    if len(phic_names) > 1:
        log.info("Collecting particle parameters from most likely classes")
        phic = np.array([cs[p] for p in phic_names])
        cls = np.argmax(phic, axis=0)
        cls_prob = np.choose(cls, phic)
        for k in model:
            if model[k] is not None:
                names = [n for n in cs.dtype.names if n.endswith(k)]
                df[model[k]] = pd.DataFrame(np.array(
                    [cs[names[c]][i] for i, c in enumerate(cls)]))
        if minphic > 0:
            df.drop(df.loc[cls_prob < minphic].index, inplace=True)
    elif len(phic_names) == 1:
        log.info("Assigning parameters 2D classes or single 3D class")
        if "alignments2D" in phic_names[0]:
            log.info("Assigning skew angle from 2D classification")
            model["pose"] = star.Relion.ANGLEPSI
        for k in model:
            if model[k] is not None:
                name = phic_names[0].replace("class_posterior", k)
                df[model[k]] = pd.DataFrame(cs[name])
    else:
        log.info("Classification parameters not found")
    return df


def cryosparc_2_cs_array_parameters(cs, df=None):
    log = logging.getLogger('root')
    if df is None:
        df = pd.DataFrame()
    if "blob/shape" in cs.dtype.names:
        log.info("Copying image size")
        df[star.Relion.IMAGESIZE] = cs["blob/shape"][:, 0]
    return df


def parse_cryosparc_2_cs(csfile, passthroughs=None, minphic=0, boxsize=None,
                         swapxy=False, invertx=False, inverty=False):
    micrograph = {u'uid': star.UCSF.UID,
                  u'micrograph_blob/path': star.Relion.MICROGRAPH_NAME,
                  u'micrograph_blob/psize_A': star.Relion.DETECTORPIXELSIZE,
                  u'mscope_params/accel_kv': None,
                  u'mscope_params/cs_mm': None,
                  u'ctf/accel_kv': star.Relion.VOLTAGE,
                  u'ctf/amp_contrast': star.Relion.AC,
                  u'ctf/cs_mm': star.Relion.CS,
                  u'ctf/df1_A': star.Relion.DEFOCUSU,
                  u'ctf/df2_A': star.Relion.DEFOCUSV,
                  u'ctf/df_angle_rad': star.Relion.DEFOCUSANGLE,
                  u'ctf/phase_shift_rad': star.Relion.PHASESHIFT,
                  u'ctf/cross_corr_ctffind4': star.Relion.CTFFIGUREOFMERIT,
                  u'ctf/ctf_fit_to_A': star.Relion.CTFMAXRESOLUTION}
    general = {u'uid': star.UCSF.UID,
               u'ctf/accel_kv': star.Relion.VOLTAGE,
               u'blob/psize_A': star.Relion.DETECTORPIXELSIZE,
               u'ctf/ac': star.Relion.AC,
               u'ctf/amp_contrast': star.Relion.AC,
               u'ctf/cs_mm': star.Relion.CS,
               u'ctf/df1_A': star.Relion.DEFOCUSU,
               u'ctf/df2_A': star.Relion.DEFOCUSV,
               u'ctf/df_angle_rad': star.Relion.DEFOCUSANGLE,
               u'ctf/phase_shift_rad': star.Relion.PHASESHIFT,
               u'ctf/cross_corr_ctffind4': star.Relion.CTFFIGUREOFMERIT,
               u'ctf/ctf_fit_to_A': star.Relion.CTFMAXRESOLUTION,
               u'ctf/bfactor': star.Relion.CTFBFACTOR,
               u'ctf/exp_group_id': star.Relion.OPTICSGROUP,
               u'blob/path': star.UCSF.IMAGE_PATH,
               u'blob/idx': star.UCSF.IMAGE_INDEX,
               u'location/center_x_frac': None,
               u'location/center_y_frac': None,
               u'location/micrograph_path': star.Relion.MICROGRAPH_NAME,
               u'location/micrograph_shape': None}
    log = logging.getLogger('root')
    log.debug("Reading primary file")
    cs = csfile if type(csfile) is np.ndarray else np.load(csfile)
    df = util.dataframe_from_records_mapped(cs, general)
    df = cryosparc_2_cs_particle_locations(cs, df, swapxy=swapxy, invertx=invertx, inverty=inverty)
    df = cryosparc_2_cs_model_parameters(cs, df, minphic=minphic)
    df = cryosparc_2_cs_array_parameters(cs, df)
    if passthroughs is not None:
        for passthrough in passthroughs:
            if type(passthrough) is np.ndarray:
                log.info("Passing np.ndarray at %s" % str(id(passthrough)))
                pt = passthrough
            else:
                log.info("Reading auxiliary file %s" % passthrough)
                pt = np.load(passthrough)
            names = [n for n in pt.dtype.names if n != 'uid' and n not in cs.dtype.names]
            if len(names) > 0:
                ptdf = util.dataframe_from_records_mapped(pt, {**general, **micrograph})
                ptdf = cryosparc_2_cs_particle_locations(pt, ptdf, swapxy=swapxy, invertx=invertx, inverty=inverty)
                # ptdf = cryosparc_2_cs_model_parameters(pt, ptdf, minphic=minphic)
                ptdf = cryosparc_2_cs_array_parameters(pt, ptdf)
                key = star.UCSF.UID
                log.info("Trying to merge: %s" % ", ".join(names))
                fields = [c for c in ptdf.columns if c not in df.columns]
                log.info("Merging: %s" % ", ".join(fields))
                df = star.smart_merge(df, ptdf, fields=fields, key=key)
            else:
                log.info("This file contains no new information and will be ignored")

    if sys.version_info >= (3, 0):
        if star.Relion.MICROGRAPH_NAME in df:
            df[star.Relion.MICROGRAPH_NAME] = df[star.Relion.MICROGRAPH_NAME].apply(lambda x: x.decode('UTF-8'))
        if star.UCSF.IMAGE_PATH in df:
            df[star.UCSF.IMAGE_PATH] = df[star.UCSF.IMAGE_PATH].apply(lambda x: x.decode('UTF-8'))

    df[star.Relion.MAGNIFICATION] = 10000.0

    log.info("Directly copied fields: %s" % ", ".join(df.columns))

    if star.Relion.DEFOCUSANGLE in df:
        log.debug("Converting DEFOCUSANGLE from degrees to radians")
        df[star.Relion.DEFOCUSANGLE] = np.rad2deg(df[star.Relion.DEFOCUSANGLE])
    elif star.Relion.DEFOCUSV in df and star.Relion.DEFOCUSU in df:
        log.warn("Defocus angles not found")
    else:
        log.warn("Defocus values not found")

    if star.Relion.PHASESHIFT in df:
        log.debug("Converting PHASESHIFT from degrees to radians")
        df[star.Relion.PHASESHIFT] = np.rad2deg(df[star.Relion.PHASESHIFT])

    if star.Relion.ORIGINX in df.columns and boxsize is not None:
        df[star.Relion.ORIGINS] *= cs["blob/shape"][0] / boxsize

    if star.Relion.RANDOMSUBSET in df.columns:
        log.debug("Changing RANDOMSUBSET to 1-based index")
        df[star.Relion.RANDOMSUBSET] += 1

    if star.Relion.CLASS in df.columns:
        log.debug("Changing CLASS to 1-based index")
        df[star.Relion.CLASS] += 1

    if star.Relion.OPTICSGROUP in df.columns:
        log.debug("Changing OPTICSGROUP to 1-based index")
        df[star.Relion.OPTICSGROUP] += 1

    if df.columns.intersection(star.Relion.ANGLES).size == len(star.Relion.ANGLES):
        log.debug("Converting Rodrigues coordinates to Euler angles")
        df[star.Relion.ANGLES] = np.rad2deg(geom.rot2euler(geom.expmap(df[star.Relion.ANGLES].values)))
        log.info("Converted Rodrigues coordinates to Euler angles")
    elif star.Relion.ANGLEPSI in df:
        log.debug("Converting ANGLEPSI from degrees to radians")
        df[star.Relion.ANGLEPSI] = np.rad2deg(df[star.Relion.ANGLEPSI])
    elif star.is_particle_star(df):
        log.warn("Angular alignment parameters not found")
    return df

