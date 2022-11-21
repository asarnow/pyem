# Copyright (C) 2017-2022 Daniel Asarnow
# University of California, San Francisco
#
# Handles metadata from cryoSPARC 2.0 and later.
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
import sys
import numpy as np
import os.path
import pandas as pd
from pyem import geom
from pyem import star
from pyem import util


micrograph = {u'uid': star.UCSF.UID,
              u'micrograph_blob/path': star.Relion.MICROGRAPH_NAME,
              u'micrograph_blob/psize_A': star.Relion.MICROGRAPHPIXELSIZE,
              u'mscope_params/accel_kv': star.Relion.VOLTAGE,
              u'mscope_params/cs_mm': star.Relion.CS,
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
           u'blob/psize_A': star.Relion.IMAGEPIXELSIZE,
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
           u'location/micrograph_shape': None,
           u'filament/filament_uid': star.Relion.HELICALTUBEID,
           u'filament/filament_pose': None}
movie = {u'movie_blob/path': star.Relion.MICROGRAPHMOVIE_NAME,
         u'movie_blob/psize_A': star.Relion.MICROGRAPHORIGINALPIXELSIZE,
         u'gain_ref_blob/path': star.Relion.MICROGRAPHGAIN_NAME,
         u'rigid_motion/frame_start': star.Relion.MICROGRAPHSTARTFRAME,
         u'rigid_motion/frame_end': star.Relion.MICROGRAPHENDFRAME,
         u'mscope_params/total_dose_e_per_A2': star.Relion.MICROGRAPHDOSERATE}


def cryosparc_2_cs_particle_locations(cs, df=None, swapxy=True, invertx=False, inverty=True):
    log = logging.getLogger('root')
    if df is None:
        df = pd.DataFrame()
    if u'location/center_x_frac' in cs.dtype.names:
        log.debug("Converting normalized particle coordinates to absolute")
        df[star.Relion.COORDX] = cs[u'location/center_x_frac']
        df[star.Relion.COORDY] = cs[u'location/center_y_frac']
        # df[star.Relion.MICROGRAPH_NAME] = cs[u'location/micrograph_path']
        if invertx:
            # Might rarely be needed, if your K3 images are "tall" in SerialEM.
            df[star.Relion.COORDX] = 1 - df[star.Relion.COORDX]
        if inverty:
            # cryoSPARC coordinates have origin in "bottom left" so inverting Y is default for Relion correctness
            # (and therefore also for Import Particles). However, cryoSPARC Patch Motion flips images physically
            # vs. SerialEM, Motioncor2 doesn't, so "inverting twice" (not inverting) is required if switching.
            df[star.Relion.COORDY] = 1 - df[star.Relion.COORDY]
        if swapxy:
            # In cryoSPARC, fast axis is long axis of K3, 'location/micrograph_shape' is [y, x].
            # In Relion and numpy (e.g. pyem.mrc), the fast axis is the short axis of K3, shape is (x, y).
            # cryoSPARC import particles correctly imports *Relion convention* coordinates, which we also want.
            # Default behavior is now to always swap.
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
        log.debug("Recovering beam tilt and converting to mrad")
        df[star.Relion.BEAMTILTX] = np.arcsin(cs['ctf/tilt_A'][:, 0] / cs['ctf/cs_mm'] * 1e-7) * 1e3
        df[star.Relion.BEAMTILTY] = np.arcsin(cs['ctf/tilt_A'][:, 1] / cs['ctf/cs_mm'] * 1e-7) * 1e3
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
    if u'alignments3D/class_posterior' in cs.dtype.names:
        log.info("Assigning pose from single 3D refinement")
        for k in model:
            if model[k] is not None:
                name = u'alignments3D/' + k
                df[model[k]] = pd.DataFrame(cs[name])
    elif len(phic_names) > 1:
        log.info("Assigning pose from most likely 3D classes")
        phic = np.array([cs[p] for p in phic_names if u'alignments2D' not in p])
        cls = np.argmax(phic, axis=0)
        cls_prob = phic[cls, range(cls.shape[0])]
        for k in model:
            if model[k] is not None:
                names = [n for n in cs.dtype.names if n.endswith(k)]
                df[model[k]] = pd.DataFrame(np.array(
                    [cs[names[c]][i] for i, c in enumerate(cls)]))
        if minphic > 0:
            df.drop(df.loc[cls_prob < minphic].index, inplace=True)
    elif u'alignments2D/class_posterior' in cs.dtype.names:
        log.info("Assigning pose from 2D classes")
        log.info("Assigning skew angle from 2D classification")
        model["pose"] = star.Relion.ANGLEPSI
        for k in model:
            if model[k] is not None:
                name = "alignments2D/" + k
                df[model[k]] = pd.DataFrame(cs[name])
    else:
        log.info("Particle poses not found")
    return df


def cryosparc_2_cs_array_parameters(cs, df=None):
    log = logging.getLogger('root')
    if df is None:
        df = pd.DataFrame()
    if "blob/shape" in cs.dtype.names:
        log.info("Copying image size")
        df[star.Relion.IMAGESIZE] = cs["blob/shape"][:, 0]
    elif "movie_blob/shape" in cs.dtype.names:
        log.info("Copying movie size")
        df[star.Relion.IMAGESIZEX] = cs["movie_blob/shape"][:, 1]
        df[star.Relion.IMAGESIZEY] = cs["movie_blob/shape"][:, 0]
        df[star.Relion.IMAGESIZEZ] = cs["movie_blob/shape"][:, 2]
    elif "micrograph_blob/shape" in cs.dtype.names:
        log.info("Copying micrograph size")
        df[star.Relion.IMAGESIZEX] = cs["micrograph_blob/shape"][:, 1]
        df[star.Relion.IMAGESIZEY] = cs["micrograph_blob/shape"][:, 0]
    return df


def cryosparc_2_cs_filament_parameters(cs, df=None):
    log = logging.getLogger('root')
    if df is None:
        df = pd.DataFrame()
    if 'filament/filament_pose' in cs.dtype.names:
        log.info('Copying filament pose')
        df[star.Relion.ANGLEPSI] = -cs['filament/filament_pose'] + np.pi/2
    return df


def cryosparc_2_cs_motion_parameters(cs, trajdir="."):
    log = logging.getLogger('root')
    log.info("Creating movie data_general tables")
    data_general = util.dataframe_from_records_mapped(cs, {**movie, **micrograph, **general})
    data_general = cryosparc_2_cs_array_parameters(cs, data_general)
    data_general[star.Relion.MOTIONMODELVERSION] = 0
    data_general[star.Relion.MICROGRAPHBINNING] = \
        data_general[star.Relion.MICROGRAPHPIXELSIZE] / data_general[star.Relion.MICROGRAPHORIGINALPIXELSIZE]
    data_general[star.Relion.MICROGRAPHDOSERATE] /= data_general[star.Relion.IMAGESIZEZ]
    data_general[star.Relion.MICROGRAPHPREEXPOSURE] = \
        data_general[star.Relion.MICROGRAPHDOSERATE] * data_general[star.Relion.MICROGRAPHSTARTFRAME]
    data_general[star.Relion.MICROGRAPHSTARTFRAME] += 1
    data_general[star.Relion.MICROGRAPHMOVIE_NAME] = data_general[star.Relion.MICROGRAPHMOVIE_NAME].apply(
        lambda x: x.decode('UTF-8'))
    log.info("Reading movie trajectory files")
    for i in range(cs.shape[0]):
        trajfile = cs['rigid_motion/path'][i].decode('UTF-8')
        trajfile = os.path.join(trajdir, trajfile)
        traj = np.load(trajfile).reshape((-1, 2))
        log.debug("%s: %d-%d, (%d x %d)" %
                  (trajfile, cs['rigid_motion/frame_start'][i], cs['rigid_motion/frame_end'][i],
                  traj.shape[0], traj.shape[1]))
        d = {star.Relion.MICROGRAPHFRAMENUMBER: np.arange(cs['rigid_motion/frame_start'][i] + 1,
                                                          cs['rigid_motion/frame_end'][i] + 1),
             star.Relion.MICROGRAPHSHIFTX: traj[:, 1],
             star.Relion.MICROGRAPHSHIFTY: traj[:, 0]}
        try:
            data_shift = pd.DataFrame(d)
            mic = {star.Relion.GENERALDATA: data_general.iloc[i], star.Relion.GLOBALSHIFTDATA: data_shift}
        except ValueError:
            log.debug("Couldn't convert %s, skipping" % trajfile)
            continue
        yield mic


def parse_cryosparc_2_cs(csfile, passthroughs=None, minphic=0, boxsize=None,
                         swapxy=False, invertx=False, inverty=False):

    log = logging.getLogger('root')
    log.debug("Reading primary file")
    cs = csfile if type(csfile) is np.ndarray else np.load(csfile)
    df = util.dataframe_from_records_mapped(cs, general)
    df = cryosparc_2_cs_particle_locations(cs, df, swapxy=swapxy, invertx=invertx, inverty=inverty)
    df = cryosparc_2_cs_model_parameters(cs, df, minphic=minphic)
    df = cryosparc_2_cs_array_parameters(cs, df)
    df = cryosparc_2_cs_filament_parameters(cs, df)
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
                ptdf = cryosparc_2_cs_filament_parameters(pt, ptdf)
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

    # df[star.Relion.MAGNIFICATION] = 10000.0

    log.info("Directly copied fields: %s" % ", ".join(df.columns))

    if star.Relion.DEFOCUSANGLE in df:
        log.debug("Converting DEFOCUSANGLE from degrees to radians")
        df[star.Relion.DEFOCUSANGLE] = np.rad2deg(df[star.Relion.DEFOCUSANGLE])
    elif star.Relion.DEFOCUSV in df and star.Relion.DEFOCUSU in df:
        log.warning("Defocus angles not found")
    else:
        log.warning("Defocus values not found")

    if star.Relion.PHASESHIFT in df:
        log.debug("Converting PHASESHIFT from degrees to radians")
        df[star.Relion.PHASESHIFT] = np.rad2deg(df[star.Relion.PHASESHIFT])

    if star.Relion.ORIGINX in df.columns and boxsize is not None:
        df[star.Relion.ORIGINS] *= cs["blob/shape"][0] / boxsize

    if star.Relion.RANDOMSUBSET in df.columns:
        log.debug("Changing RANDOMSUBSET to 1-based index")
        if df[star.Relion.RANDOMSUBSET].value_counts().size == 1:
            df.drop(star.Relion.RANDOMSUBSET, axis=1, inplace=True)
        else:
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
        log.warning("Angular alignment parameters not found")
    return df
