#!/usr/bin/env python
# Copyright (C) 2017 Daniel Asarnow
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


def rot2euler(r):
    """Decompose rotation matrix into Euler angles"""
    #    assert(isrotation(r))
    #r = r.T
    #    psi = np.arctan2(r[1,2], r[2,2])
    #    c2 = np.sqrt(np.power(r[0,0], 2) + np.power(r[0,1], 2))
    #    theta = np.arctan2(-r[0,2], c2)
    #    s1 = np.sin(psi)
    #    c1 = np.cos(psi)
    #    phi = np.arctan2(s1*r[2,0] - c1*r[1,0], c1*r[1,1] - s1*r[2,1])

    # Shoemake rotation matrix decomposition algorithm with same conventions as Relion.
    epsilon = np.finfo(np.double).eps
    abs_sb = np.sqrt(r[0, 2] ** 2 + r[1, 2] ** 2)
    if abs_sb > 16 * epsilon:
        gamma = np.arctan2(r[1, 2], -r[0, 2])
        alpha = np.arctan2(r[2, 1], r[2, 0])
        if np.abs(np.sin(gamma)) < epsilon:
            sign_sb = np.sign(-r[0, 2]) / np.cos(gamma)
        else:
            sign_sb = np.sign(r[1, 2]) if np.sin(gamma) > 0 else -np.sign(r[1, 2])
        beta = np.arctan2(sign_sb * abs_sb, r[2, 2])
    else:
        if np.sign(r[2, 2]) > 0:
            alpha = 0
            beta = 0
            gamma = np.arctan2(-r[1, 0], r[0, 0])
        else:
            alpha = 0
            beta = np.pi
            gamma = np.arctan2(r[1, 0], -r[0, 0])
    return alpha, beta, gamma


def euler2rot(alpha, beta, gamma):
    ca = np.cos(alpha)
    cb = np.cos(beta)
    cg = np.cos(gamma)
    sa = np.sin(alpha)
    sb = np.sin(beta)
    sg = np.sin(gamma)
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa
    r = np.array([[cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb],
                  [-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb],
                  [sc, ss, cb]])
    return r


def expmap(e):
    """Convert axis-angle vector into 3D rotation matrix"""
    theta = np.linalg.norm(e)
    if theta < 1e-16:
        return np.identity(3, dtype=e.dtype)
    w = e / theta
    k = np.array([[0, -w[2], w[1]],
                  [w[2], 0, -w[0]],
                  [-w[1], w[0], 0]], dtype=e.dtype)
    r = np.identity(3, dtype=e.dtype) + np.sin(theta) * k + (1 - np.cos(theta)) * np.dot(k, k)
    return r.T


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
