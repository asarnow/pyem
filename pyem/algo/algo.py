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
from scipy.spatial import cKDTree


def bincorr(p1, p2, bins, minlength=0):
    bflat = bins.reshape(-1)
    p1flat = p1.reshape(-1)
    p2flat = p2.reshape(-1)
    fc = p1flat * np.conj(p2flat)
    fcr = np.bincount(bflat, fc.real, minlength=minlength)
    fcc = np.bincount(bflat, fc.imag, minlength=minlength)
    mag = np.sqrt(
        np.bincount(bflat, np.abs(p1flat) ** 2, minlength=minlength) *
        np.bincount(bflat, np.abs(p2flat) ** 2, minlength=minlength))
    mag[-1] += 1e-17
    frc = (fcr + fcc * 1j) / mag
    return frc


def query_connected(kdt, maxdist):
    if type(kdt) is not cKDTree:
        kdt = cKDTree(kdt)
    nb = np.full(kdt.n, np.nan)
    pairs = kdt.query_pairs(maxdist)
    for p in pairs:
        for idx in p[1:]:
            if np.isnan(nb[p[0]]):
                nb[idx] = p[0]
            else:
                nb[idx] = nb[p[0]]
    return nb

