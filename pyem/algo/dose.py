# Copyright (C) 2019 Daniel Asarnow
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
import libtiff
import numpy as np


def welford(st):
    x = st[0]
    mu = x.copy()
    sigma = np.zeros(st.shape)
    mu = st.copy()
    for i, x in enumerate(st[1:]):
        x = st[i]
        olddif = x - mu
        mu += (x - mu) / (i + 1)
        sigma += olddif * (x - mu)
    return mu, sigma
