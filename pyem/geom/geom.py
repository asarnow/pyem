# Copyright (C) 2018 Daniel Asarnow
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


def double_center(arr, reference=None, inplace=False):
    assert arr.ndim == 2
    if reference is None:
        mu0 = np.mean(arr, axis=0, keepdims=True)
        mu = np.mean(arr)
    else:
        mu0 = np.mean(reference, axis=0, keepdims=True)
        mu = np.mean(reference)
    mu1 = np.mean(arr, axis=1, keepdims=True)
    if inplace:
        arr -= mu0
        arr -= mu1
        arr += mu
    else:
        arr = arr - mu0
        arr -= mu1
        arr += mu
    return arr
