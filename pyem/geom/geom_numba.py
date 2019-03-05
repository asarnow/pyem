# Copyright (C) 2019 Daniel Asarnow
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
import numba
import numpy as np


@numba.guvectorize(["void(float64[:], float64[:], float64[:])"],
                   "(m),(m)->(m)", nopython=True, cache=False)
def cross3(u, v, w):
    w[0] = u[1] * v[2] - u[2] * v[1]
    w[1] = u[2] * v[0] - u[0] * v[2]
    w[2] = u[0] * v[1] - u[1] * v[0]


@numba.jit(nopython=True, cache=False, nogil=True)
def cross3_sca(u, v):
    w = np.zeros_like(u)
    w[0] = u[1] * v[2] - u[2] * v[1]
    w[1] = u[2] * v[0] - u[0] * v[2]
    w[2] = u[0] * v[1] - u[1] * v[0]
    return w


@numba.jit(nopython=True, cache=False, nogil=True)
def dualcos(r, d):
    return np.cos(r), -d * np.sin(r)
