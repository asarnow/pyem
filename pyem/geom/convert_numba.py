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
def rot2euler(r, out=None):
    """Decompose rotation matrix into Euler angles"""
    # assert(isrotation(r))
    # Shoemake rotation matrix decomposition algorithm with same conventions as Relion.
    # epsilon = np.finfo(np.double).eps
    epsilon = 1e-16
    r = r.reshape(-1, 9).reshape(-1, 3, 3)
    if out is None:
        out = np.zeros((len(r), 3), dtype=r.dtype)
    for i in range(len(r)):
        abs_sb = np.sqrt(r[i, 0, 2] ** 2 + r[i, 1, 2] ** 2)
        if abs_sb > 16 * epsilon:
            gamma = np.arctan2(r[i, 1, 2], -r[i, 0, 2])
            alpha = np.arctan2(r[i, 2, 1], r[i, 2, 0])
            if np.abs(np.sin(gamma)) < epsilon:
                sign_sb = np.sign(-r[i, 0, 2]) / np.cos(gamma)
            else:
                sign_sb = np.sign(r[i, 1, 2]) if np.sin(gamma) > 0 else -np.sign(r[i, 1, 2])
            beta = np.arctan2(sign_sb * abs_sb, r[i, 2, 2])
        else:
            if np.sign(r[i, 2, 2]) > 0:
                alpha = 0
                beta = 0
                gamma = np.arctan2(-r[i, 1, 0], r[i, 0, 0])
            else:
                alpha = 0
                beta = np.pi
                gamma = np.arctan2(r[i, 1, 0], -r[i, 0, 0])
        out[i, 0] = alpha
        out[i, 1] = beta
        out[i, 2] = gamma
    return out


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
    theta = 2 * np.arctan2(np.abs(n), q[0])  # Or 2 * np.arccos(q[0])
    if np.abs(theta) > 1e-12:
        ax = q[1:] / np.sin(0.5 * theta)
    else:
        ax = np.zeros(3, dtype=q.dtype)
    return theta * ax


@numba.jit(nopython=True, nogil=True)
def aa2quat(ax):
    theta = np.linalg.norm(ax)
    if theta == 0:
        return np.array([1, 0, 0, 0], dtype=ax.dtype)
    q = np.zeros(4, dtype=ax.dtype)
    q[0] = np.cos(0.5 * theta)
    q[1:] = np.sin(0.5 * theta)
    q[1:] *= ax
    q[1:] /= theta
    if q[0] < 0:
        q[:] = -q[:]
    return q


@numba.jit(nopython=True, nogil=True)
def quat2rot(q):
    # aa = q[0] ** 2
    bb = 2 * q[1] ** 2
    cc = 2 * q[2] ** 2
    dd = 2 * q[3] ** 2
    ab = 2 * q[0] * q[1]
    ac = 2 * q[0] * q[2]
    ad = 2 * q[0] * q[3]
    bc = 2 * q[1] * q[2]
    bd = 2 * q[1] * q[3]
    cd = 2 * q[2] * q[3]
    # These formulas are equivalent forms taken from Wikipedia, but are transposed.
    # r = np.array([[aa + bb - cc - dd, 2*bc - 2*ad,       2*bd + 2*ac],
    #               [2*bc + 2*ad,       aa - bb + cc - dd, 2*cd - 2*ab],
    #               [2*bd - 2*ac,       2*cd + 2*ab,       aa - bb - cc + dd]], dtype=q.dtype)
    # r = np.array([[1 - 2 * (cc + dd), 2 * (bc - ad), 2 * (bd + ac)],
    #               [2 * (bc + ad), 1 - 2 * (bb + dd), 2 * (cd - ab)],
    #               [2 * (bd - ac), 2 * (cd + ab), 1 - 2 * (bb + cc)]], dtype=q.dtype)
    # Tentatively correct formula (assuming multiplication by 2):
    r = np.array([[1 - cc - dd, bc + ad, bd - ac],
                  [bc - ad, 1 - bb - dd, cd + ab],
                  [bd + ac, cd - ab, 1 - bb - cc]], dtype=q.dtype)
    # This formula is incorrect after fixing euler2quat for ZYZ.
    # r = np.array([[1 - 2 * (bb + dd), 2 * (ad - bc), -2 * (cd + ab)],
    #               [-2 * (bc + ad), 1 - 2 * (cc + dd), 2 * (bd - ac)],
    #               [2 * (ab - cd), 2 * (bd + ac), 1 - 2 * (bb + cc)]], dtype=q.dtype)
    # This assumes the multiplications by 2 are done first:
    # r = np.array([[1 - (bb + dd), (ad - bc), -(cd + ab)],
    #               [-(bc + ad), 1 - (cc + dd), (bd - ac)],
    #               [(ab - cd), (bd + ac), 1 - (bb + cc)]], dtype=q.dtype)
    return r


@numba.jit(nopython=True, nogil=True)
def rot2quat(r):
    q = np.zeros(4, dtype=r.dtype)
    tr = np.trace(r)
    if tr > 0:
        q[0] = np.sqrt(tr + 1) / 2
        sinv = 1 / (q[0] * 4)
        q[1] = sinv * (r[1, 2] - r[2, 1])
        q[2] = sinv * (r[2, 0] - r[0, 2])
        q[3] = sinv * (r[0, 1] - r[1, 0])
    else:
        mi = np.argmax(np.diag(r))
        if mi == 0:
            q[1] = np.sqrt(r[0, 0] - r[1, 1] - r[2, 2] + 1) / 2
            sinv = 1 / (q[1] * 4)
            q[0] = sinv * (r[1, 2] - r[2, 1])
            q[2] = sinv * (r[0, 1] + r[1, 0])
            q[3] = sinv * (r[0, 2] + r[2, 0])
        elif mi == 1:
            q[2] = np.sqrt(r[1, 1] - r[2, 2] - r[0, 0] + 1) / 2
            sinv = 1 / (q[2] * 4)
            q[0] = sinv * (r[2, 0] - r[0, 2])
            q[1] = sinv * (r[0, 1] + r[1, 0])
            q[3] = sinv * (r[1, 2] + r[2, 1])
        else:
            q[3] = np.sqrt(r[2, 2] - r[0, 0] - r[1, 1] + 1) / 2
            sinv = 1 / (q[3] * 4)
            q[0] = sinv * (r[0, 1] - r[1, 0])
            q[1] = sinv * (r[0, 2] + r[2, 0])
            q[2] = sinv * (r[1, 2] + r[2, 1])
    if q[0] < 0:
        q[:] = -q[:]
    return q


@numba.jit(nopython=True, nogil=True)
def euler2quat(alpha, beta, gamma):
    q = np.array([np.cos((alpha + gamma) / 2) * np.cos(beta / 2),
                  np.sin((gamma - alpha) / 2) * np.sin(beta / 2),
                  np.cos((alpha - gamma) / 2) * np.sin(beta / 2),
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

    # alpha = np.arctan2(bd + ac, ab - cd)
    alpha = np.arctan2(np.abs(ab - cd), bd + ac)

    # if alpha < 0:
    #     alpha += 2 * np.pi

    beta = np.arccos(aa - bb - cc + dd)

    # gamma = np.arctan2(bd - ac, ab + cd)
    gamma = 2 * np.arctan2(np.abs(bd - ac), ab + cd)

    # if gamma < 0:
    #     gamma += 2 * np.pi

    return alpha, beta, gamma


@numba.jit(nopython=True, nogil=True)
def expmap(e, out=None):
    """Convert axis-angle vector into 3D rotation matrix"""
    # theta = np.linalg.norm(e)
    # if theta < 1e-16:
    #     return np.identity(3, e.dtype)
    # w = e / theta
    # k = np.array([[0, w[2], -w[1]],
    #               [-w[2], 0, w[0]],
    #               [w[1], -w[0], 0]], dtype=e.dtype)
    # r = np.identity(3, e.dtype) + np.sin(theta) * k + (np.ones(1, dtype=e.dtype) - np.cos(theta)) * np.dot(k, k)
    e = np.atleast_2d(e)
    if out is None:
        out = np.zeros((len(e), 3, 3), dtype=e.dtype)
    for i in range(len(e)):
        theta = np.linalg.norm(e[i, ...])
        if theta < 1e-16:
            out[i, 0, 0] = out[i, 1, 1] = out[i, 2, 2] = 1
        w = e[i, ...] / theta
        s = np.sin(theta)
        c = 1 - np.cos(theta)
        w0 = s * w[0]
        w1 = s * w[1]
        w2 = s * w[2]
        w00 = -w[0] ** 2
        w11 = -w[1] ** 2
        w22 = -w[2] ** 2
        w01 = c * w[0] * w[1]
        w02 = c * w[0] * w[2]
        w12 = c * w[1] * w[2]
        out[i, 0, 0] = 1 + c * (w22 + w11)
        out[i, 0, 1] = w2 + w01
        out[i, 0, 2] = -w1 + w02
        out[i, 1, 0] = -w2 + w01
        out[i, 1, 1] = 1 + c * (w22 + w00)
        out[i, 1, 2] = w0 + w12
        out[i, 2, 0] = w1 + w02
        out[i, 2, 1] = -w0 + w12
        out[i, 2, 2] = 1 + c * (w11 + w00)
    return out


@numba.jit(nopython=True, nogil=True)
def aa2rot(e):
    return expmap(e)


@numba.jit(nopython=True, nogil=True)
def logmap(r):
    angle = np.arccos((np.trace(r) - 1) * 0.5)
    ax = np.zeros(3, dtype=r.dtype)
    if angle < 1e-12:
        return ax
    if np.abs(angle - np.pi) < 1e-12:
        ax[:3] = np.sqrt(0.5 * (np.diag(r) + 1.0))
    else:
        maginv = 0.5 / np.sin(angle)
        ax[0] = r[1, 2] - r[2, 1]
        ax[1] = r[2, 0] - r[0, 2]
        ax[2] = r[0, 1] - r[1, 0]
        ax *= maginv
        ax *= angle
    return ax


@numba.jit(nopython=True, nogil=True)
def rot2aa(r):
    return logmap(r)


def parallel_convert_func(f):
    @numba.jit(nopython=True, parallel=True)
    def g(arr, out):
        for i in numba.prange(len(arr)):
            out[i] = f(arr[i])
        return out
    return g


@numba.jit(nopython=True, nogil=True, fastmath=True)
def e2r_vec(eu, out=None):
    eu = np.atleast_2d(eu)
    if out is None:
        out = np.zeros((len(eu), 3, 3))
    for i in range(len(eu)):
        ca = np.cos(eu[i, 0])
        cb = np.cos(eu[i, 1])
        cg = np.cos(eu[i, 2])
        sa = np.sin(eu[i, 0])
        sb = np.sin(eu[i, 1])
        sg = np.sin(eu[i, 2])
        cc = cb * ca
        cs = cb * sa
        sc = sb * ca
        ss = sb * sa
        out[i, 0, :] = cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb
        out[i, 1, :] = -sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb
        out[i, 2, :] = sc, ss, cb
    return out


@numba.jit(nopython=True, nogil=True, fastmath=True)
def e2q_vec(eu, out=None):
    eu = np.atleast_2d(eu)
    if out is None:
        out = np.zeros((len(eu), 4))
    for i in range(len(eu)):
        out[i, 0] = np.cos((eu[i, 0] + eu[i, 2]) / 2) * np.cos(eu[i, 1] / 2)
        out[i, 1] = np.cos((eu[i, 0] - eu[i, 2]) / 2) * np.sin(eu[i, 1] / 2)
        out[i, 2] = np.sin((eu[i, 0] - eu[i, 2]) / 2) * np.sin(eu[i, 1] / 2)
        out[i, 3] = np.sin((eu[i, 0] + eu[i, 2]) / 2) * np.cos(eu[i, 1] / 2)
    return out
