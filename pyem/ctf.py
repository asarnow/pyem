#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# Copyright (C) 2018 Daniel Asarnow, Eugene Palovcak
# University of California, San Francisco
#
# Simple library for calculating contrast transfer function of electron microscopy.
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


def ctf_freqs(shape, ps=1.0):
    nyq = 1. / (2. * ps)
    x, y = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[0]))
    rho = np.sqrt(x**2 + y**2)
    a = np.arctan2(y, x)
    s = rho * nyq
    return s, a


@numba.jit(cache=True, nopython=True)
def eval_ctf(s, a, def1, def2, angast=0, phase=0, kv=300, ac=0.1, cs=2.0, bf=0, lp=0):
    """
    :param s: Precomputed frequency grid for CTF evaluation.
    :param a: Precomputed frequency grid angles.
    :param def1: 1st prinicipal underfocus distance (Å).
    :param def2: 2nd principal underfocus distance (Å).
    :param angast: Angle of astigmatism (deg) from x-axis to azimuth.
    :param phase: Phase shift (deg).
    :param kv:  Microscope acceleration potential (kV).
    :param ac:  Amplitude contrast in [0, 1.0].
    :param cs:  Spherical aberration (mm).
    :param bf:  B-factor, divided by 4 in exponential, lowpass positive.
    :param lp:  Hard low-pass filter (Å).
    """
    angast = np.deg2rad(angast)
    kv = kv * 1E3
    cs = cs * 1E7
    lamb = 12.2639 / np.sqrt(kv * (1. + kv * 0.978466E-6))
    def_avg = -(def1 + def2) * 0.5
    def_dev = -(def1 - def2) * 0.5
    k1 = np.pi / 2. * 2 * lamb
    k2 = np.pi / 2. * cs * lamb**3
    k3 = np.sqrt(1 - ac**2)
    k4 = bf / 4.  # B-factor, follows RELION convention.
    k5 = np.deg2rad(phase)  # Phase shift.
    if lp != 0:  # Hard low- or high-pass.
        s *= s <= (1. / lp)
    s_2 = s**2
    s_4 = s_2**2
    dZ = def_avg + def_dev * (np.cos(2 * (a - angast)))
    gamma = (k1 * dZ * s_2) + (k2 * s_4) - k5
    ctf = -(k3 * np.sin(gamma) - ac*np.cos(gamma))
    if bf != 0:  # Enforce envelope.
        ctf *= np.exp(-k4 * s_2)
    return ctf

