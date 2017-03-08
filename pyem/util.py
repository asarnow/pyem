#!/usr/bin/env python
# Copyright (C) 2017 Daniel Asarnow
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

def cent2edge(bins):
    """Convert bin centers to bin edges"""
    return np.r_[-np.Inf, 0.5 * (bins[:-1] + bins[1:]), np.Inf]


def isrotation(r, tol=1e-4):
    """Check for valid rotation matrix"""
    check = np.identity(3, dtype=r.dtype) - np.dot(r.T, r)
    if tol is None:
        return check
    return np.linalg.norm(check) < tol


def rot2euler(r):
    """Decompose rotation matrix into Euler angles"""
    assert(isrotation(r))

    if np.abs(r[2,0]) != 1:
        theta = -np.arcsin(r[2,0])
        # theta2 = pi - theta
        psi = np.arctan2(r[2,1] / np.cos(theta), r[2,2] / np.cos(theta));
        # psi2 = np.arctan2(r[2,1] / np.cos(theta2), r[2,2] / np.cos(theta2));

        phi = np.arctan2(r[1,0] / np.cos(theta), r[0,0] / np.cos(theta));
        # phi2 = np.arctan2(r[1,0] / np.cos(theta2), r[0,0] / np.cos(theta2));
    else:
        phi = 0
        if r[2,0] == -1:
            theta = pi/2
            psi = phi + np.arctan2(r[0,1], r[0,2])
        else:
            theta = -pi/2
            psi = -phi + np.arctan2(-r[0,1], -r[0,2])
    return psi, theta, phi

