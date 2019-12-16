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


@numba.jit(cache=False, nopython=True, nogil=True)
def _bincount_nb(bins, data, out):
    for i in range(len(bins)):
        out[bins[i]] += data[i]
    return out


@numba.jit(cache=False, nopython=True, nogil=True)
def bincount_nb(bins, data, out=None):
    if out is None:
        out = np.zeros(bins.max(), dtype=data.dtype)
    _bincount_nb(bins, data, out)
    return out


@numba.jit(cache=False, nopython=True, nogil=True)
def bincorr_nb(p1, p2, bins, n=-1):
    bflat = bins.reshape(-1)
    p1flat = p1.reshape(-1)
    p2flat = p2.reshape(-1)
    if n == -1:
        n = np.max(bflat)
    fc = p1flat * np.conj(p2flat)
    fcc = _bincount_nb(bflat, fc, out=np.zeros(n, dtype=np.complex128))
    p1r = _bincount_nb(bflat, np.abs(p1flat)**2, out=np.zeros(n, dtype=np.float64))
    p2r = _bincount_nb(bflat, np.abs(p2flat)**2, out=np.zeros(n, dtype=np.float64))
    mag = np.sqrt(p1r * p2r)
    mag[-1] += 1e-17
    frc = fcc / mag
    return frc
