# Copyright (C) 2017-2018 Daniel Asarnow
# University of California, San Francisco
#
# Library for interconversion of rotation representations.
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
import numba
import numpy as np


@numba.jit(nopython=True, nogil=True)
def rot2euler(r):
    """Decompose rotation matrix into Euler angles"""
    #assert(isrotation(r))
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


@numba.jit(nopython=True, nogil=True)
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
    r = np.array(((cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb),
                  (-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb),
                  (sc, ss, cb)))
    return r


@numba.jit(nopython=True, nogil=True)
def vec2rot(v):
    ax = v / np.linalg.norm(v)
    return euler2rot(*np.array([np.arctan2(ax[1], ax[0]), np.arccos(ax[2]), 0.]))


@numba.jit(nopython=True, nogil=True)
def quat2aa(q):
    n = np.sqrt(np.sum(q[1:] ** 2))
    ax = q[1:] / n
    theta = 2 * np.arctan2(n, q[0])  # Or 2 * np.arccos(q[0])
    return theta * ax


@numba.jit(nopython=True, nogil=True)
def quat2rot(q):
    aa = q[0] ** 2
    bb = q[1] ** 2
    cc = q[2] ** 2
    dd = q[3] ** 2
    ab = q[0] * q[1]
    ac = q[0] * q[2]
    ad = q[0] * q[3]
    bc = q[1] * q[2]
    bd = q[1] * q[3]
    cd = q[2] * q[3]
    r = np.array([[aa + bb - cc - dd, 2*bc - 2*ad,       2*bd + 2*ac],
                  [2*bc + 2*ad,       aa - bb + cc - dd, 2*cd - 2*ab],
                  [2*bd - 2*ac,       2*cd + 2*ab,       aa - bb - cc + dd]], dtype=q.dtype)
    return r


@numba.jit(nopython=True, nogil=True)
def euler2quat(alpha, beta, gamma):
    q = np.array([np.cos((alpha + gamma) / 2) * np.cos(beta / 2),
                  np.cos((alpha - gamma) / 2) * np.sin(beta / 2),
                  np.sin((alpha - gamma) / 2) * np.sin(beta / 2),
                  np.sin((alpha + gamma) / 2) * np.cos(beta / 2)])
    return q


@numba.jit(nopython=True, nogil=True)
def quat2euler(q):
    aa = q[0] ** 2
    bb = q[1] ** 2
    cc = q[2] ** 2
    dd = q[3] ** 2
    ab = q[0] * q[1]
    ac = q[0] * q[2]
    bd = q[1] * q[3]
    cd = q[2] * q[3]

    alpha = np.arctan2(bd + ac, ab - cd)

    if alpha < 0:
        alpha += 2 * np.pi

    beta = np.arccos(aa - bb - cc + dd)

    gamma = np.arctan2(bd - ac, ab + cd)

    if gamma < 0:
        gamma += 2 * np.pi

    return alpha, beta, gamma


@numba.jit(nopython=True, nogil=True)
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
