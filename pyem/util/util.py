# Copyright (C) 2017-2018 Daniel Asarnow
# University of California, San Francisco
#
# Library of miscellaneous utility functions.
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
from __future__ import absolute_import
import bisect
import numpy as np
import pandas as pd
import subprocess
from distutils.spawn import find_executable as which
from .. import geom
from .. import mrc
from .. import vop


def cent2edge(bins):
    """Convert bin centers to bin edges"""
    return np.r_[-np.Inf, 0.5 * (bins[:-1] + bins[1:]), np.Inf]


def relion_symmetry_group(sym):
    relion = which("relion_refine")
    if relion is None:
        raise RuntimeError(
            "Need relion_refine on PATH to obtain symmetry operators")
    stdout = subprocess.check_output(
        ("%s --sym %s --i /dev/null --o /dev/null --print_symmetry_ops" %
         (relion, sym)).split())
    lines = stdout.split("\n")[2:-1]
    return [np.array(
        [[np.double(val) for val in l.split()] for l in lines[i:i + 3]])
        for i in range(1, len(lines), 4)]


def aligndf(df1, df2, fields=None):
    ww = df1.set_index(fields)
    dd = df2.set_index(fields)
    i1 = set(tuple(f) for f in df1[fields].values)
    i2 = set(tuple(f) for f in df2[fields].values)
    iboth = list(i1.intersection(i2))
    df1a = ww.loc[iboth].copy()
    df2a = dd.loc[iboth].copy()
    return df1a, df2a


def interleave(dfs, drop=True):
    return pd.concat(dfs).sort_index(kind="mergesort").reset_index(drop=drop)


def join_struct_arrays(arrays):
    sizes = np.array([a.itemsize for a in arrays])
    offsets = np.r_[0, sizes.cumsum()]
    n = len(arrays[0])
    joint = np.empty((n, offsets[-1]), dtype=np.uint8)
    for a, size, offset in zip(arrays, sizes, offsets):
        joint[:, offset:offset + size] = a.view(np.uint8).reshape(n, size)
    dtype = sum((a.dtype.descr for a in arrays), [])
    return joint.ravel().view(dtype)


def dataframe_from_records_mapped(rec, field_dict):
    names = [str(k) for k in field_dict if field_dict[k] is not None and k in rec.dtype.names]
    df = pd.DataFrame.from_records(rec[names])
    df.columns = [field_dict[k] for k in names]
    return df


def nearest_good_box_size(n):
    b = [32, 36, 40, 48, 52, 56, 64, 66, 70, 72, 80, 84, 88, 100, 104, 108,
         112, 120, 128, 130, 132, 140, 144, 150, 160, 162, 168, 176, 180, 182,
         192, 200, 208, 216, 220, 224, 240, 256, 264, 288, 300, 308, 320, 324,
         336, 338, 352, 364, 384, 400, 420, 432, 448, 450, 462, 480, 486, 500,
         504, 512, 520, 528, 546, 560, 576, 588, 600, 640, 648, 650, 660, 672,
         686, 700, 702, 704, 720, 726, 728, 750, 768, 770, 784, 800, 810, 840,
         882, 896, 910, 924, 936, 972, 980, 1008, 1014, 1020, 1024, 1080, 1125,
         1152, 1200, 1215, 1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536,
         1600, 1620, 1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048, 2160,
         2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916,
         3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840,
         3888, 4000, 4050, 4320, 4374, 4500, 4608, 4800, 4860, 5000, 5120,
         5184, 5400, 5625, 5760, 5832, 6000, 6075, 6144, 6250, 6400, 6480,
         6750, 6912, 7200, 7290, 7500, 7680, 7776, 8000, 8100]
    return b[bisect.bisect(b, n) - 1]


def chimera_xform(xform, o=None, apix=1.):
    if o is None:
        o = np.array([0, 0, 0])
    r = xform[:, :3]
    v = xform[:, 3] / apix
    u = r.T.dot(o - v) - o
    return r, u


def chimera_xform2str(r, v):
    transform_string = np.column_stack([r, v]).tolist().__repr__()
    return transform_string


def chimera_xform2target(t0, r, u, o=None, apix=1.):
    if o is None:
        o = np.array([0, 0, 0])
    t1 = (r.dot(t0 / apix - o) + u + o) * apix
    return t1


def write_q_series(vol, qarr, basename, psz=1., order=1):
    for i, q in enumerate(qarr):
        r = geom.quat2rot(q / np.linalg.norm(q))
        decoy = vop.resample_volume(vol, r=r, order=order)
        mrc.write(basename % i, decoy, psz=psz)
