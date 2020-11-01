# Copyright (C) 2016 Daniel Asarnow
# University of California, San Francisco
#
# Library for parsing and altering Relion .star files.
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
import re
import os.path
from collections import Counter
import numpy as np
import pandas as pd
import sys
from math import modf
from pyem.geom import e2r_vec
from pyem.geom import rot2euler
from pyem.util import natsort_values


class Relion:
    # Relion 2+ fields.
    MICROGRAPH_NAME = "rlnMicrographName"
    MICROGRAPH_NAME_NODW = "rlnMicrographNameNoDW"
    IMAGE_NAME = "rlnImageName"
    IMAGE_ORIGINAL_NAME = "rlnImageOriginalName"
    RECONSTRUCT_IMAGE_NAME = "rlnReconstructImageName"
    COORDX = "rlnCoordinateX"
    COORDY = "rlnCoordinateY"
    ORIGINX = "rlnOriginX"
    ORIGINY = "rlnOriginY"
    ORIGINZ = "rlnOriginZ"
    ANGLEROT = "rlnAngleRot"
    ANGLETILT = "rlnAngleTilt"
    ANGLEPSI = "rlnAnglePsi"
    CLASS = "rlnClassNumber"
    DEFOCUSU = "rlnDefocusU"
    DEFOCUSV = "rlnDefocusV"
    DEFOCUS = [DEFOCUSU, DEFOCUSV]
    DEFOCUSANGLE = "rlnDefocusAngle"
    CS = "rlnSphericalAberration"
    PHASESHIFT = "rlnPhaseShift"
    AC = "rlnAmplitudeContrast"
    VOLTAGE = "rlnVoltage"
    MAGNIFICATION = "rlnMagnification"
    DETECTORPIXELSIZE = "rlnDetectorPixelSize"
    BEAMTILTX = "rlnBeamTiltX"
    BEAMTILTY = "rlnBeamTiltY"
    BEAMTILTCLASS = "rlnBeamTiltClass"
    CTFSCALEFACTOR = "rlnCtfScalefactor"
    CTFBFACTOR = "rlnCtfBfactor"
    CTFMAXRESOLUTION = "rlnCtfMaxResolution"
    CTFFIGUREOFMERIT = "rlnCtfFigureOfMerit"
    GROUPNUMBER = "rlnGroupNumber"
    RANDOMSUBSET = "rlnRandomSubset"
    AUTOPICKFIGUREOFMERIT = "rlnAutopickFigureOfMerit"

    # Relion 3 fields.
    OPTICSGROUP = "rlnOpticsGroup"
    OPTICSGROUPNAME = "rlnOpticsGroupName"
    ODDZERNIKE = "rlnOddZernike"
    EVENZERNIKE = "rlnEvenZernike"
    MAGMAT00 = "rlnMagMat00"
    MAGMAT01 = "rlnMagMat01"
    MAGMAT10 = "rlnMagMat10"
    MAGMAT11 = "rlnMagMat11"
    IMAGEPIXELSIZE = "rlnImagePixelSize"
    IMAGESIZE = "rlnImageSize"
    IMAGEDIMENSION = "rlnImageDimensionality"
    ORIGINXANGST = "rlnOriginXAngst"
    ORIGINYANGST = "rlnOriginYAngst"
    ORIGINZANGST = "rlnOriginZAngst"
    MICROGRAPHPIXELSIZE = "rlnMicrographPixelSize"
    MICROGRAPHORIGINALPIXELSIZE = "rlnMicrographOriginalPixelSize"
    MTFFILENAME = "rlnMtfFileName"

    # Field lists.
    COORDS = [COORDX, COORDY]
    ORIGINS = [ORIGINX, ORIGINY]
    ORIGINS3D = [ORIGINX, ORIGINY, ORIGINZ]
    ORIGINSANGST = [ORIGINXANGST, ORIGINYANGST]
    ORIGINSANGST3D = [ORIGINXANGST, ORIGINYANGST, ORIGINZANGST]
    ANGLES = [ANGLEROT, ANGLETILT, ANGLEPSI]
    ALIGNMENTS = ANGLES + ORIGINS3D + ORIGINSANGST3D
    CTF_PARAMS = [DEFOCUSU, DEFOCUSV, DEFOCUSANGLE, CS, PHASESHIFT, AC,
                  BEAMTILTX, BEAMTILTY, BEAMTILTCLASS, CTFSCALEFACTOR, CTFBFACTOR,
                  CTFMAXRESOLUTION, CTFFIGUREOFMERIT]
    MICROSCOPE_PARAMS = [VOLTAGE, MAGNIFICATION, DETECTORPIXELSIZE]
    MICROGRAPH_COORDS = [MICROGRAPH_NAME] + COORDS
    PICK_PARAMS = MICROGRAPH_COORDS + [ANGLEPSI, CLASS, AUTOPICKFIGUREOFMERIT]

    FIELD_ORDER = [IMAGE_NAME, IMAGE_ORIGINAL_NAME, MICROGRAPH_NAME, MICROGRAPH_NAME_NODW] + \
                   COORDS + ALIGNMENTS + MICROSCOPE_PARAMS + CTF_PARAMS + \
                  [CLASS + GROUPNUMBER + RANDOMSUBSET + OPTICSGROUP]

    RELION2 = ORIGINS3D + [MAGNIFICATION, DETECTORPIXELSIZE]

    RELION30 = [BEAMTILTCLASS]

    RELION31 = ORIGINSANGST3D + [BEAMTILTX, BEAMTILTY, OPTICSGROUP, OPTICSGROUPNAME,
                ODDZERNIKE, EVENZERNIKE, MAGMAT00, MAGMAT01, MAGMAT10, MAGMAT11,
                IMAGEPIXELSIZE, IMAGESIZE, IMAGEDIMENSION]

    OPTICSGROUPTABLE = [AC, CS, VOLTAGE, BEAMTILTX, BEAMTILTY, OPTICSGROUPNAME, ODDZERNIKE, EVENZERNIKE,
                        MAGMAT00, MAGMAT01, MAGMAT10, MAGMAT11, IMAGEPIXELSIZE, IMAGESIZE, IMAGEDIMENSION]

    # Data tables.
    OPTICDATA = "data_optics"
    MICROGRAPHDATA = "data_micrographs"
    PARTICLEDATA = "data_particles"
    IMAGEDATA = "data_images"

    # Data type specification.
    DATATYPES = {OPTICSGROUP: np.int}


class UCSF:
    IMAGE_PATH = "ucsfImagePath"
    IMAGE_BASENAME = "ucsfImageBasename"
    IMAGE_INDEX = "ucsfImageIndex"
    IMAGE_ORIGINAL_PATH = "ucsfImageOriginalPath"
    IMAGE_ORIGINAL_BASENAME = "ucsfImageOriginalBasename"
    IMAGE_ORIGINAL_INDEX = "ucsfImageOriginalIndex"
    MICROGRAPH_BASENAME = "ucsfMicrographBasename"
    UID = "ucsfUid"
    PARTICLE_UID = "ucsfParticleUid"
    MICROGRAPH_UID = "ucsfMicrographUid"


def smart_merge(s1, s2, fields, key=None, left_key=None):
    if key is None:
        key = merge_key(s1, s2)
    if left_key is None:
        left_key = key
    s2 = s2.set_index(key, drop=False)
    s1 = s1.merge(s2[s2.columns.intersection(fields)], left_on=left_key, right_index=True, suffixes=["", "_y"])
    y = [c for c in s1.columns if "_y" in c]  # Columns duplicated in the merge source.
    if len(y) > 0:
        x = [c.split("_")[0] for c in s1.columns if c in y]  # Corresponding original columns.
        for xi, yi in zip(x, y):
            if xi in fields:  # Use values from merge source, default to original values.
                s1[xi] = s1[yi].fillna(s1[xi])
            else:  # Use original values, default to merge source values.
                s1[xi] = s1[xi].fillna(s1[yi])
        s1 = s1.drop(y, axis=1)
    return s1.reset_index(drop=True)


def merge_key(s1, s2, threshold=0.5):
    inter = s1.columns.intersection(s2.columns)
    if not inter.size:
        return None
    if Relion.IMAGE_NAME in inter:
        c = Counter(s1[Relion.IMAGE_NAME])
        shared = sum(c[i] for i in set(s2[Relion.IMAGE_NAME]))
        if shared >= s1.shape[0] * threshold:
            return Relion.IMAGE_NAME
    if UCSF.IMAGE_BASENAME in inter:
        c = Counter(s1[UCSF.IMAGE_BASENAME])
        shared = sum(c[i] for i in set(s2[UCSF.IMAGE_BASENAME]))
        if shared >= s1.shape[0] * threshold:
            return [UCSF.IMAGE_BASENAME, UCSF.IMAGE_INDEX]
    mgraph_coords = inter.intersection(Relion.MICROGRAPH_COORDS)
    if Relion.MICROGRAPH_NAME in mgraph_coords:
        c = Counter(s1[Relion.MICROGRAPH_NAME])
        shared = sum(c[i] for i in set(s2[Relion.MICROGRAPH_NAME]))
        can_merge_mgraph_name = Relion.MICROGRAPH_NAME in mgraph_coords and shared >= s1.shape[0] * threshold
        if can_merge_mgraph_name and mgraph_coords.intersection(Relion.COORDS).size:
            return Relion.MICROGRAPH_COORDS
        elif can_merge_mgraph_name:
            return Relion.MICROGRAPH_NAME
    if UCSF.MICROGRAPH_BASENAME in inter:
        c = Counter(s1[UCSF.MICROGRAPH_BASENAME])
        shared = sum(c[i] for i in set(s2[UCSF.MICROGRAPH_BASENAME]))
        if shared >= s1.shape[0] * threshold:
            return UCSF.MICROGRAPH_BASENAME
    return None


def is_particle_star(df):
    return df.columns.intersection([Relion.IMAGE_NAME] + Relion.COORDS).size


def calculate_apix(df):
    try:
        if df.ndim == 2:
            if Relion.IMAGEPIXELSIZE in df:
                return df.iloc[0][Relion.IMAGEPIXELSIZE]
            if Relion.MICROGRAPHPIXELSIZE in df:
                return df.iloc[0][Relion.MICROGRAPHPIXELSIZE]
            return 10000.0 * df.iloc[0][Relion.DETECTORPIXELSIZE] / df.iloc[0][Relion.MAGNIFICATION]
        elif df.ndim == 1:
            if Relion.IMAGEPIXELSIZE in df:
                return df[Relion.IMAGEPIXELSIZE]
            if Relion.MICROGRAPHPIXELSIZE in df:
                return df[Relion.MICROGRAPHPIXELSIZE]
            return 10000.0 * df[Relion.DETECTORPIXELSIZE] / df[Relion.MAGNIFICATION]
        else:
            raise ValueError
    except KeyError:
        return None


def select_classes(df, classes):
    clsfields = [f for f in df.columns if Relion.CLASS in f]
    if len(clsfields) == 0:
        raise RuntimeError("No class labels found")
    ind = df[clsfields[0]].isin(classes)
    if not np.any(ind):
        raise RuntimeError("Specified classes have no members")
    return df.loc[ind]


def to_micrographs(df):
    gb = df.groupby(Relion.MICROGRAPH_NAME)
    mu = gb.mean()
    df = mu[[c for c in Relion.CTF_PARAMS + Relion.MICROSCOPE_PARAMS +
             [Relion.MICROGRAPH_NAME, Relion.OPTICSGROUP] if c in mu]].reset_index()
    if Relion.OPTICSGROUP in df:
        df = df.astypes(Relion.DATATYPES)
    return df


def split_micrographs(df):
    gb = df.groupby(Relion.MICROGRAPH_NAME)
    dfs = {}
    for g in gb:
        g[1].drop(Relion.MICROGRAPH_NAME, axis=1, inplace=True, errors="ignore")
        dfs[g[0]] = g[1]
    return dfs


def replace_micrograph_path(df, path, inplace=False):
    df = df if inplace else df.copy()
    df[Relion.MICROGRAPH_NAME] = df[Relion.MICROGRAPH_NAME].apply(
        lambda x: os.path.join(path, os.path.basename(x)))
    return df


def set_original_fields(df, inplace=False):
    df = df if inplace else df.copy()
    if Relion.IMAGE_NAME in df:
        df[Relion.IMAGE_ORIGINAL_NAME] = df[Relion.IMAGE_NAME]
    if UCSF.IMAGE_INDEX in df:
        df[UCSF.IMAGE_ORIGINAL_INDEX] = df[UCSF.IMAGE_INDEX]
    if UCSF.IMAGE_PATH in df:
        df[UCSF.IMAGE_ORIGINAL_PATH] = df[UCSF.IMAGE_PATH]
    return df


def all_same_class(df, inplace=False):
    vc = df[Relion.IMAGE_NAME].value_counts()
    n = vc.max()
    si = df.set_index([Relion.IMAGE_NAME, Relion.CLASS], inplace=inplace)
    vci = si.index.value_counts()
    si = si.loc[vci[vci == n].index].reset_index(inplace=inplace)
    return si


def recenter(df, inplace=False):
    df = df if inplace else df.copy()
    intoff = np.round(df[Relion.ORIGINS]).values
    diffxy = df[Relion.ORIGINS] - intoff
    df[Relion.COORDS] = df[Relion.COORDS] - intoff
    df[Relion.ORIGINS] = diffxy
    return df


def recenter_modf(df, inplace=False):
    df = df if inplace else df.copy()
    remxy, offsetxy = np.vectorize(modf)(df[Relion.ORIGINS])
    df[Relion.ORIGINS] = remxy
    df[Relion.COORDS] = df[Relion.COORDS] - offsetxy
    return df


def zero_origins(df, inplace=False):
    df = df if inplace else df.copy()
    df[Relion.COORDX] = df[Relion.COORDX] - df[Relion.ORIGINX]
    df[Relion.COORDY] = df[Relion.COORDY] - df[Relion.ORIGINY]
    df[Relion.ORIGINX] = 0
    df[Relion.ORIGINY] = 0
    return df


def scale_coordinates(df, factor, inplace=False):
    df = df if inplace else df.copy()
    df[Relion.COORDS] = df[Relion.COORDS] * factor
    return df


def scale_origins(df, factor, inplace=False):
    df = df if inplace else df.copy()
    df[Relion.ORIGINS] = df[Relion.ORIGINS] * factor
    return df


def scale_magnification(df, factor, inplace=False):
    df = df if inplace else df.copy()
    df[Relion.MAGNIFICATION] = df[Relion.MAGNIFICATION] * factor
    return df


def scale_apix(df, factor, inplace=False):
    df = df if inplace else df.copy()
    df[Relion.IMAGEPIXELSIZE] = df[Relion.IMAGEPIXELSIZE] * factor
    return df


def invert_hand(df, inplace=False):
    df = df if inplace else df.copy()
    df[Relion.ANGLEROT] = -df[Relion.ANGLEROT]
    df[Relion.ANGLETILT] = 180 - df[Relion.ANGLETILT]
    return df


def set_optics_groups(df, sep="_", idx=4, inplace=False):
    df = df if inplace else df.copy()
    df[Relion.OPTICSGROUPNAME] = df[UCSF.MICROGRAPH_BASENAME].str.split(sep, expand=True).loc[:, idx]
    df[Relion.OPTICSGROUP] = pd.Categorical(df[Relion.OPTICSGROUPNAME]).codes + 1
    return df


def parse_star_table(starfile, offset=0, nrows=None, keep_index=False):
    headers = []
    foundheader = False
    ln = 0
    with open(starfile, 'r') as f:
        f.seek(offset)
        for l in f:
            if l.lstrip().startswith("_"):
                foundheader = True
                lastheader = True
                if keep_index:
                    head = l.strip()
                else:
                    head = l.split('#')[0].strip().lstrip('_')
                headers.append(head)
            else:
                lastheader = False
            if foundheader and not lastheader:
                break
            ln += 1
        f.seek(offset)
        df = pd.read_csv(f, delimiter='\s+', header=None, skiprows=ln, nrows=nrows)
    df.columns = headers
    return df


def star_table_offsets(starfile):
    tables = {}
    with open(starfile) as f:
        l = f.readline()  # Current line
        ln = 0  # Current line number.
        offset = 0  # Char offset of current table.
        cnt = 0  # Number of tables.
        in_table = False  # True if file cursor is inside a table.
        in_loop = False
        blank_terminates = False
        while l:
            if l.lstrip().startswith("data"):
                table_name = l.strip()
                if in_table:
                    tables[table_name] = (offset, lineno, ln - 1, ln - data_line - 1)
                in_table = True
                in_loop = False
                blank_terminates = False
                offset = f.tell()  # Record byte offset of table.
                lineno = ln  # Record start line of table.
                cnt += 1  # Increment table count.
            if l.lstrip().startswith("loop"):
                in_loop = True
            elif in_loop and not l.startswith("_"):
                in_loop = False
                blank_terminates = True
                data_line = ln
            if blank_terminates and in_table and l.isspace():  # Allow blankline to terminate table.
                in_table = False
                tables[table_name] = (offset, lineno, ln - 1, ln - data_line)
            l = f.readline()  # Read next line.
            ln += 1  # Increment line number.
        if in_table and table_name not in tables:
            tables[table_name] = (offset, lineno, ln, ln - data_line)
        return tables


def parse_star(starfile, keep_index=False, augment=True, nrows=sys.maxsize):
    tables = star_table_offsets(starfile)
    dfs = {t: parse_star_table(starfile, offset=tables[t][0], nrows=min(tables[t][3], nrows), keep_index=keep_index)
           for t in tables}
    if Relion.OPTICDATA in dfs:
        if Relion.PARTICLEDATA in dfs:
            data_table = Relion.PARTICLEDATA
        elif Relion.MICROGRAPHDATA in dfs:
            data_table = Relion.MICROGRAPHDATA
        elif Relion.IMAGEDATA in dfs:
            data_table = Relion.IMAGEDATA
        else:
            data_table = None
        if data_table is not None:
            df = pd.merge(dfs[Relion.OPTICDATA], dfs[data_table], on=Relion.OPTICSGROUP)
        else:
            df = dfs[Relion.OPTICDATA]
    else:
        df = dfs[next(iter(dfs))]
    df = check_defaults(df, inplace=True)
    if augment:
        augment_star_ucsf(df, inplace=True)
    return df


def parse_star_tables(starfile, keep_index=False, nrows=sys.maxsize):
    tables = star_table_offsets(starfile)
    dfs = {t: parse_star_table(starfile, offset=tables[t][0], nrows=min(tables[t][3], nrows), keep_index=keep_index)
           for t in tables}
    return dfs


def write_star_table(starfile, df, table="data_", resort_fields=True, mode='w'):
    indexed = re.search("#\d+$", df.columns[0]) is not None  # Check first column for '#N' index.
    if not indexed:
        if resort_fields:
            df = sort_fields(df, inplace=True)
        names = [idx + " #%d" % (i + 1) for i, idx in enumerate(df.columns)]
    else:
        names = df.columns
    with open(starfile, mode) as f:
        f.write('\n')
        f.write(table + '\n')
        f.write('\n')
        f.write("loop_" + '\n')
        for name in names:
            line = name + " \n"
            line = line if line.startswith('_') else '_' + line
            f.write(line)
    df.to_csv(starfile, mode='a', sep=' ', header=False, index=False, float_format='%.6f')


def write_star_tables(starfile, dfs, resort_fields=True):
    for i, t in enumerate(dfs):
        mode = 'w' if i == 0 else 'a+'
        write_star_table(starfile, dfs[t], table=t, resort_fields=resort_fields, mode=mode)


def write_star(starfile, df, resort_fields=True, resort_records=False, simplify=True, optics=True):
    if not starfile.endswith(".star"):
        starfile += ".star"
    if resort_records:
        df = sort_records(df, inplace=True)
    if simplify and len([c for c in df.columns if "ucsf" in c or "eman" in c]) > 0:
        df = simplify_star_ucsf(df)
    
    if optics:
        if Relion.OPTICSGROUP not in df:
            df[Relion.OPTICSGROUP] = 1
        gb = df.groupby(Relion.OPTICSGROUP)
        df_optics = gb[df.columns.intersection(Relion.OPTICSGROUPTABLE)].first().reset_index(drop=False)
        df = df.drop(columns=Relion.OPTICSGROUPTABLE, errors="ignore")
        data_table = Relion.PARTICLEDATA if is_particle_star(df) else Relion.MICROGRAPHDATA
        dfs = {Relion.OPTICDATA: df_optics, data_table: df}
        write_star_tables(starfile, dfs, resort_fields=resort_fields)
    else:
        write_star_table(starfile, df, table=Relion.IMAGEDATA, resort_fields=resort_fields)


def transform_star(df, r, t=None, inplace=False, rots=None, invert=False, rotate=True, adjust_defocus=False):
    """
    Transform particle angles and origins according to a rotation
    matrix (in radians) and an optional translation vector.
    The translation may also be given as the 4th column of a 3x4 matrix,
    or as a scalar distance to be applied along the axis of rotation.
    """
    assert (r.shape[0] == 3)
    if r.shape[1] == 4 and t is None:
        t = r[:, -1]
        r = r[:, :3]
    assert (r.shape == (3, 3))
    assert t is None or np.array(t).size == 1 or len(t) == 3

    if inplace:
        newstar = df
    else:
        newstar = df.copy()

    if rots is None:
        rots = e2r_vec(np.deg2rad(df[Relion.ANGLES].values))

    if invert:
        r = r.T

    newrots = np.dot(rots, r)  # Works with 3D array and list of 2D arrays.
    if rotate:
        angles = np.rad2deg(rot2euler(newrots))
        newstar[Relion.ANGLES] = angles

    if t is not None and np.linalg.norm(t) > 0:
        if np.array(t).size == 1:
            if invert:
                tt = -(t * rots)[:, :, 2]  # Works with 3D array and list of 2D arrays.
            else:
                tt = newrots[:, :, 2] * t
        else:
            if invert:
                tt = -np.dot(rots, t)
            else:
                tt = np.dot(newrots, t)
        if Relion.ORIGINX in newstar:
            newstar[Relion.ORIGINX] += tt[:, 0]
        if Relion.ORIGINY in newstar:
            newstar[Relion.ORIGINY] += tt[:, 1]
        if Relion.ORIGINZ in newstar:
            newstar[Relion.ORIGINZ] += tt[:, 2]
        if adjust_defocus:
            newstar[Relion.DEFOCUSU] += tt[:, -1] * calculate_apix(df)
            newstar[Relion.DEFOCUSV] += tt[:, -1] * calculate_apix(df)
            newstar[Relion.DEFOCUSANGLE] = np.rad2deg(np.arctan2(newstar[Relion.DEFOCUSV], newstar[Relion.DEFOCUSV]))

    return newstar


def augment_star_ucsf(df, inplace=True):
    df = df if inplace else df.copy()
    df.reset_index(inplace=True)
    if Relion.IMAGE_NAME in df:
        df[UCSF.IMAGE_INDEX], df[UCSF.IMAGE_PATH] = \
            df[Relion.IMAGE_NAME].str.split("@").str
        df[UCSF.IMAGE_INDEX] = pd.to_numeric(df[UCSF.IMAGE_INDEX]) - 1

        if Relion.IMAGE_ORIGINAL_NAME not in df:
            df[Relion.IMAGE_ORIGINAL_NAME] = df[Relion.IMAGE_NAME]

    if Relion.IMAGE_ORIGINAL_NAME in df:
        df[UCSF.IMAGE_ORIGINAL_INDEX], df[UCSF.IMAGE_ORIGINAL_PATH] = \
            df[Relion.IMAGE_ORIGINAL_NAME].str.split("@").str
        df[UCSF.IMAGE_ORIGINAL_INDEX] = pd.to_numeric(df[UCSF.IMAGE_ORIGINAL_INDEX]) - 1

    if UCSF.IMAGE_PATH in df:
        df[UCSF.IMAGE_BASENAME] = df[UCSF.IMAGE_PATH].apply(os.path.basename)

    if UCSF.IMAGE_ORIGINAL_PATH in df:
        df[UCSF.IMAGE_ORIGINAL_BASENAME] = df[UCSF.IMAGE_ORIGINAL_PATH].apply(os.path.basename)

    if Relion.MICROGRAPH_NAME in df:
        df[UCSF.MICROGRAPH_BASENAME] = df[Relion.MICROGRAPH_NAME].apply(os.path.basename)
    return df


def simplify_star_ucsf(df, resort_index=False, inplace=True, drop=True):
    df = df if inplace else df.copy()
    if UCSF.IMAGE_ORIGINAL_INDEX in df and UCSF.IMAGE_ORIGINAL_PATH in df:
        df[Relion.IMAGE_ORIGINAL_NAME] = df[UCSF.IMAGE_ORIGINAL_INDEX].map(
            lambda x: "%.6d" % (x + 1)).str.cat(df[UCSF.IMAGE_ORIGINAL_PATH],
                                                sep="@")
    if UCSF.IMAGE_INDEX in df and UCSF.IMAGE_PATH in df:
        df[Relion.IMAGE_NAME] = df[UCSF.IMAGE_INDEX].map(
            lambda x: "%.6d" % (x + 1)).str.cat(df[UCSF.IMAGE_PATH], sep="@")
    if drop:
        df.drop([c for c in df.columns if "ucsf" in c or "eman" in c],
                axis=1, inplace=True)
    if resort_index and "index" in df.columns:
        df.set_index("index", inplace=True)
        df.sort_index(inplace=True, kind="mergesort")
    elif drop and "index" in df.columns:
        df.drop("index", axis=1, inplace=True)
    return df


def sort_fields(df, inplace=False):
    df = df if inplace else df.copy()
    columns = [c for c in Relion.FIELD_ORDER if c in df] + \
              [c for c in df.columns if c not in Relion.FIELD_ORDER]
    df = df.reindex(columns=columns, copy=False)
    return df


def sort_records(df, inplace=False):
    df = df if inplace else df.copy()
    if is_particle_star(df):
        if UCSF.IMAGE_INDEX in df:
            # df.sort_values([UCSF.IMAGE_PATH, UCSF.IMAGE_INDEX], inplace=True)
            df = natsort_values(df, df[UCSF.IMAGE_PATH] + "_" + df[UCSF.IMAGE_INDEX].astype(str), inplace=True)
    elif Relion.MICROGRAPH_NAME in df:
        df = natsort_values(df, Relion.MICROGRAPH_NAME, inplace=True)
    return df


def original_field(field):
    tok = re.findall("[A-Z][a-z]+", field)
    tok = tok[0] + "Original" + "".join(tok[1:])
    lead = re.match(r".*?[a-z].*?(?=[A-Z])", field).group()
    field = lead + tok
    return field


def check_defaults(df, inplace=False):
    df = df if inplace else df.copy()
    if Relion.PHASESHIFT not in df:
        df[Relion.PHASESHIFT] = 0

    if Relion.IMAGEPIXELSIZE in df:
        if Relion.DETECTORPIXELSIZE not in df and Relion.MAGNIFICATION not in df:
            df[Relion.DETECTORPIXELSIZE] = df[Relion.IMAGEPIXELSIZE]
            df[Relion.MAGNIFICATION] = 10000
        elif Relion.DETECTORPIXELSIZE in df:
            df[Relion.MAGNIFICATION] = df[Relion.DETECTORPIXELSIZE] / df[Relion.IMAGEPIXELSIZE] * 10000
        elif Relion.MAGNIFICATION in df:
            df[Relion.DETECTORPIXELSIZE] = df[Relion.MAGNIFICATION] * df[Relion.IMAGEPIXELSIZE] / 10000
    elif Relion.DETECTORPIXELSIZE in df and Relion.MAGNIFICATION in df:
        df[Relion.IMAGEPIXELSIZE] = df[Relion.DETECTORPIXELSIZE] * df[Relion.MAGNIFICATION] / 10000

    for it in zip(Relion.ORIGINSANGST3D, Relion.ORIGINS3D):
        if it[0] in df:
            df[it[1]] = df[it[0]] / df[Relion.IMAGEPIXELSIZE]
        elif it[1] in df:
            df[it[0]] = df[it[1]] * df[Relion.IMAGEPIXELSIZE]

    if Relion.ORIGINZANGST in df:
        df[Relion.IMAGEDIMENSION] = 3
    else:
        df[Relion.IMAGEDIMENSION] = 2

    if Relion.OPTICSGROUPNAME in df and Relion.OPTICSGROUP not in df:
        df[Relion.OPTICSGROUP] = df[Relion.OPTICSGROUPNAME].astype('category').cat.codes

    if Relion.BEAMTILTCLASS in df and Relion.OPTICSGROUP not in df:
        df[Relion.OPTICSGROUP] = df[Relion.BEAMTILTCLASS]
    return df


def remove_deprecated_relion2(df, inplace=False):
    df = df if inplace else df.copy()
    df.drop(columns=Relion.RELION2 + Relion.RELION30, inplace=True, errors="ignore")
    return df


def remove_new_relion31(df, inplace=False):
    df = df if inplace else df.copy()
    df.drop(columns=Relion.RELION31, inplace=True, errors="ignore")
    return df


def compatible(df, version=None, inplace=False, relion2=False):
    df = df if inplace else df.copy()
    if version is None:
        version = 30 if relion2 else 31
    if version < 10:
        version = int(10 * version)
    if version < 30:
        df.drop(columns=Relion.RELION30 + Relion.RELION31, inplace=True, errors="ignore")
    if version == 30:
        df.drop(columns=Relion.RELION31, inplace=True, errors="ignore")
    if version >= 31:
        df.drop(columns=Relion.RELION2 + Relion.RELION30, inplace=True, errors="ignore")
    return df


def revert_original(df, inplace=False):
    df = df if inplace else df.copy()
    if Relion.IMAGE_ORIGINAL_NAME in df and Relion.IMAGE_NAME in df:
        df.rename(columns={Relion.IMAGE_NAME: Relion.IMAGE_ORIGINAL_NAME,
                       Relion.IMAGE_ORIGINAL_NAME: Relion.IMAGE_NAME}, inplace=True)
    elif Relion.IMAGE_ORIGINAL_NAME in df:
        df[Relion.IMAGE_NAME] = df[Relion.IMAGE_ORIGINAL_NAME]

    if UCSF.IMAGE_ORIGINAL_INDEX in df and UCSF.IMAGE_ORIGINAL_PATH in df \
            and UCSF.IMAGE_INDEX in df and UCSF.IMAGE_PATH in df:
        df.rename(columns={UCSF.IMAGE_INDEX: UCSF.IMAGE_ORIGINAL_INDEX,
                       UCSF.IMAGE_ORIGINAL_INDEX: UCSF.IMAGE_INDEX,
                       UCSF.IMAGE_PATH: UCSF.IMAGE_ORIGINAL_PATH,
                       UCSF.IMAGE_ORIGINAL_PATH: UCSF.IMAGE_PATH}, inplace=True)
        if UCSF.IMAGE_ORIGINAL_BASENAME in df and UCSF.IMAGE_BASENAME in df:
            df.rename(columns={UCSF.IMAGE_BASENAME: UCSF.IMAGE_ORIGINAL_BASENAME,
                               UCSF.IMAGE_ORIGINAL_BASENAME: UCSF.IMAGE_BASENAME}, inplace=True)
    elif UCSF.IMAGE_ORIGINAL_INDEX in df and UCSF.IMAGE_ORIGINAL_PATH in df:
        df[UCSF.IMAGE_INDEX] = df[UCSF.IMAGE_ORIGINAL_INDEX]
        df[UCSF.IMAGE_PATH] = df[UCSF.IMAGE_ORIGINAL_PATH]
        if UCSF.IMAGE_ORIGINAL_BASENAME in df:
            df[UCSF.IMAGE_BASENAME] = df[UCSF.IMAGE_ORIGINAL_BASENAME]
    return df
