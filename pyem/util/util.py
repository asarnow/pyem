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
import numpy as np
import pandas as pd
import subprocess
from distutils.spawn import find_executable as which


def cent2edge(bins):
    """Convert bin centers to bin edges"""
    return np.r_[-np.Inf, 0.5 * (bins[:-1] + bins[1:]), np.Inf]


def isrotation(r, tol=1e-4):
    """Check for valid rotation matrix"""
    check = np.identity(3, dtype=r.dtype) - np.dot(r.T, r)
    if tol is None:
        return check
    return np.linalg.norm(check) < tol


def relion_symmetry_group(sym):
    relion = which("relion_refine")
    if relion is None:
        raise RuntimeError("Need relion_refine on PATH to obtain symmetry operators")
    stdout = subprocess.check_output(("%s --sym %s --i /dev/null --o /dev/null --print_symmetry_ops" % (relion, sym)).split())
    lines = stdout.split("\n")[2:-1]
    return [np.array([[np.double(val) for val in l.split()] for l in lines[i:i + 3]]) for i in range(1, len(lines), 4)]


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

