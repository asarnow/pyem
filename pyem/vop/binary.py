#!/usr/bin/env python2.7
# Copyright (C) 2017-2018 Daniel Asarnow
# University of California, San Francisco
#
# Library functions for volume data.
# See README file for more information.
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
from scipy.ndimage import label
from scipy.ndimage import labeled_comprehension


def binary_sphere(r, le=True):
    rr = np.linspace(-r, r, 2 * r + 1)
    x, y, z = np.meshgrid(rr, rr, rr)
    if le:
        sph = (x ** 2 + y ** 2 + z ** 2) <= r ** 2
    else:
        sph = (x ** 2 + y ** 2 + z ** 2) < r ** 2
    return sph


def binary_volume_opening(vol, minvol):
    lb_vol, num_objs = label(vol)
    lbs = np.arange(1, num_objs + 1)
    v = labeled_comprehension(lb_vol > 0, lb_vol, lbs, np.sum, np.int, 0)
    ix = np.isin(lb_vol, lbs[v >= minvol])
    newvol = np.zeros(vol.shape, dtype=np.bool)
    newvol[ix] = vol[ix]
    return newvol

