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
import numba
import numpy as np


@numba.jit(nopython=True)
def _qconj(q, p):
    p[0] = q[0]
    p[1] = -q[1]
    p[2] = -q[2]
    p[3] = -q[3]
    return p


@numba.guvectorize(["void(float64[:], float64[:])"],
        "(m)->(m)", nopython=True, cache=True)
def qconj(q, p):
    _qconj(q, p)


@numba.jit(nopython=True)
def _qtimes(q1, q2, q3):
    q3[0] = q1[0] * q2[0] - (q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3])
    q3[1] = q1[2] * q2[3] - q1[3] * q2[2] + q1[0] * q2[1] + q2[0] * q1[1]
    q3[2] = q1[3] * q2[1] - q1[1] * q2[3] + q1[0] * q2[2] + q2[0] * q1[2]
    q3[3] = q1[1] * q2[2] - q1[2] * q2[1] + q1[0] * q2[3] + q2[0] * q1[3]
    return q3


@numba.guvectorize(["void(float64[:], float64[:], float64[:])"], 
        "(m),(m)->(m)", nopython=True, cache=True)
def qtimes(q1, q2, q3):
    _qtimes(q1, q2, q3)


@numba.jit(cache=True, nopython=True)
def qslerp(q1, q2, t):
    cos_half_theta = np.dot(q1, q2)
    if cos_half_theta >= 1.0:
        return q1.copy()
    if cos_half_theta < 0:
        cos_half_theta = -cos_half_theta
        q1 = qconj(q1)
    half_theta = np.arccos(cos_half_theta)
    sin_half_theta = np.sqrt(1 - cos_half_theta * cos_half_theta)
    if np.abs(sin_half_theta) < 1E-12:
        return (q1 + q2) / 2
    a = np.sin((1 - t) * half_theta)
    b = np.sin(t * half_theta)
    return (q1 * a + q2 * b) / sin_half_theta

