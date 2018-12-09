# Copyright (C) 2018 Daniel Asarnow
# University of California, San Francisco
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


def _qconj(q, p):
    p[0] = q[0]
    p[1] = -q[1]
    p[2] = -q[2]
    p[3] = -q[3]
    return p


qconj = np.vectorize(_qconj, signature="(m),(m)->(m)")


def _qtimes(q1, q2, q3):
    q3[0] = q1[0] * q2[0] - (q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3])
    q3[1] = q1[2] * q2[3] - q1[3] * q2[2] + q1[0] * q2[1] + q2[0] * q1[1]
    q3[2] = q1[3] * q2[1] - q1[1] * q2[3] + q1[0] * q2[2] + q2[0] * q1[2]
    q3[3] = q1[1] * q2[2] - q1[2] * q2[1] + q1[0] * q2[3] + q2[0] * q1[3]
    return q3


qtimes = np.vectorize(_qtimes, signature="(m),(m)->(m)")


def _qsqrt(q, p):
    p[0] = q[0] + 1
    p[1:] = q[1:]
    p[:] = p[:] / np.sqrt(2 * (1 + q[0]))
    return p


qsqrt = np.vectorize(_qsqrt, signature="(m),(m)->(m)")


def qslerp(q1, q2, t):
    cos_half_theta = np.dot(q1, q2)
    if cos_half_theta >= 1.0:
        return q1.copy()
    half_theta = np.arccos(cos_half_theta)
    sin_half_theta = np.sqrt(1 - cos_half_theta * cos_half_theta)
    if np.abs(sin_half_theta) < 1E-12:
        return (q1 + q2) / 2
    a = np.sin((1 - t) * half_theta) / sin_half_theta
    b = np.sin(t * half_theta) / sin_half_theta
    return q1 * a + q2 * b


def normq(q, mu=None):
    q = (q.T/np.linalg.norm(q, axis=1)).T
    if mu is not None:
        ang = np.dot(q, mu) < 0
        q[ang] = -q[ang]
    return q


def meanq(q, w=None):
    if w is None:
        return np.linalg.eigh(np.einsum('ij,ik->...jk', q, q))[1][:, -1]
    else:
        return np.linalg.eigh(np.einsum('ij,ik,i->...jk', q, q, w))[1][:, -1]


def distq(q1, q2):
    return 2 * np.arccos(np.abs(np.dot(q1, q2)))


def pdistq(q1, q2=None):
    if q2 is None:
        q2 = q1
    d = np.abs(q1.dot(q2.T))
    # dots[dots > 1.0] = 1.0
    np.clip(d, 0, 1.0, out=d)
    np.arccos(d, out=d)
    d *= 2
    d_over_pi_half = d > np.pi/2
    d[d_over_pi_half] = np.pi - d[d_over_pi_half]
    d += 1e-6
    d **= 2
    d *= -0.5
    return d


def normdq(q, mu=None):
    mag = np.linalg.norm(q.real, axis=1)
    if mu is not None:
        ang = np.dot(q.real, mu) < 0
        q.real[ang] = -q.real[ang]
    return q / mag.reshape(q.shape[0], 1)

