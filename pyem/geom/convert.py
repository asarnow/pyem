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
import numpy as np


def rot2euler(r):
    """Decompose rotation matrix into Euler angles"""
    # assert(isrotation(r))
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


def vec2rot(v):
    ax = v / np.linalg.norm(v)
    return euler2rot(*np.array([np.arctan2(ax[1], ax[0]), np.arccos(ax[2]), 0.]))


def quat2aa(q):
    n = np.linalg.norm(q[1:])
    ax = q[1:] / n if n > 0 else np.zeros(3, dtype=q.dtype)
    theta = 2 * np.arctan2(n, q[0])  # Or 2 * np.arccos(q[0])
    return theta * ax


def aa2quat(ax, theta=None):
    if theta is None:
        theta = np.linalg.norm(ax)
        if theta != 0:
            ax = ax / theta
    q = np.zeros(4, dtype=ax.dtype)
    q[0] = np.cos(theta / 2)
    q[1:] = ax * np.sin(theta / 2)
    return q


def quat2rot(q):
    n = np.sum(q**2)
    s = 0 if n == 0 else 2 / n
    wx = s * q[0] * q[1]
    wy = s * q[0] * q[2]
    wz = s * q[0] * q[3]
    xx = s * q[1] * q[1]
    xy = s * q[1] * q[2]
    xz = s * q[1] * q[3]
    yy = s * q[2] * q[2]
    yz = s * q[2] * q[3]
    zz = s * q[3] * q[3]
    r = np.array([[1 - (yy + zz), xy + wz,       xz - wy],
                  [xy - wz,       1 - (xx + zz), yz + wx],
                  [xz + wy,       yz - wx,       1 - (xx + yy)]], dtype=q.dtype)
    return r


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
    return q


def euler2quat(alpha, beta, gamma):
    ha, hb, hg = alpha / 2, beta / 2, gamma / 2
    ha_p_hg = ha + hg
    hg_m_ha = hg - ha
    q = np.array([np.cos(ha_p_hg) * np.cos(hb),
                  np.sin(hg_m_ha) * np.sin(hb),
                  np.cos(hg_m_ha) * np.sin(hb),
                  np.sin(ha_p_hg) * np.cos(hb)])
    return q


def quat2euler(q):
    ha1 = np.arctan2(q[1], q[2])
    ha2 = np.arctan2(q[3], q[0])
    alpha = ha2 - ha1  # np.arctan2(r21/r20)
    beta = 2 * np.arccos(np.sqrt(q[0]**2 + q[3]**2))  # np.arccos*r33
    gamma = ha1 + ha2  # np.arctan2(r12/-r02)
    return alpha, beta, gamma


def expmap(e):
    """Convert axis-angle vector into 3D rotation matrix"""
    theta = np.linalg.norm(e)
    if theta < 1e-16:
        return np.identity(3, dtype=e.dtype)
    w = e / theta
    k = np.array([[0, w[2], -w[1]],
                  [-w[2], 0, w[0]],
                  [w[1], -w[0], 0]], dtype=e.dtype)
    r = np.identity(3, dtype=e.dtype) + np.sin(theta) * k + (1 - np.cos(theta)) * np.dot(k, k)
    return r
