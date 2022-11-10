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


def ctf_freq(shape, d=1.0, full=True):
    """
    :param shape: Shape tuple.
    :param d: Frequency spacing in inverse Å (1 / pixel size).
    :param full: When false, return only unique Fourier half-space for real data. 
    """
    if full:
        xfrq = np.fft.fftfreq(shape[1])
    else:
        xfrq = np.fft.rfftfreq(shape[1])
    x, y = np.meshgrid(xfrq, np.fft.fftfreq(shape[0]))
    rho = np.sqrt(x**2 + y**2)
    a = np.arctan2(y, x)
    s = rho * d
    return s, a


@numba.jit(cache=True, nopython=True, nogil=True)
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
    :param lp:  Hard low-pass filter (Å), should usually be Nyquist.
    """
    angast = np.deg2rad(angast)
    kv = kv * 1e3
    cs = cs * 1e7
    lamb = 12.2643247 / np.sqrt(kv * (1. + kv * 0.978466e-6))
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


@numba.jit(cache=True, nopython=True, nogil=True)
def eval_ctf_between(n, apix, def1, def2, lores=0, hires=0, angast=0, phase=0, kv=300, ac=0.1, cs=2.0, bf=0, out=None):
    if out is None:
        out = np.zeros((n, n // 2 + 1))
    ctf = out.view()
    ctf.shape = (ctf.size,)
    d = 1. / (apix * n)
    kv = kv * 1e3
    cs = cs * 1e7
    lamb = 12.2643247 / np.sqrt(kv * (1. + kv * 0.978466e-6))
    def_avg = -(def1 + def2) * 0.5  # Sign convention for underfocused imaging.
    def_dev = -(def1 - def2) * 0.5
    k1 = np.pi / 2. * 2 * lamb
    k2 = np.pi / 2. * cs * lamb**3
    k3 = np.sqrt(1 - ac**2)
    k4 = bf / 4.  # B-factor, follows RELION convention.
    k5 = np.deg2rad(phase)  # Phase shift.
    rmin = int(lores * apix * n)
    rmin2 = rmin**2
    rmax = int(min(n // 2 - 1, hires * apix * n))
    rmax2 = rmax**2
    cnt = int(0)
    for i in range(n):
        if i <= rmax:
            yp = i
        elif i >= n - rmax:
            yp = i - n
        else:
            continue
        yp2 = yp ** 2
        for j in range(rmax + 1):
            xp = j
            r = xp ** 2 + yp2
            if r < rmin2 or r > rmax2:
                continue
            s = np.sqrt(xp**2 + yp**2) * d
            a = np.arctan2(yp, xp)
            s2 = s**2
            s4 = s**4
            dz = def_avg + def_dev * (np.cos(2 * (a - angast)))
            gamma = (k1 * dz * s2) + (k2 * s4) - k5
            # ctf[i, j] = -(k3 * np.sin(gamma) - ac * np.cos(gamma))
            ctf[cnt] = -(k3 * np.sin(gamma) - ac * np.cos(gamma))
            if bf != 0:  # Enforce envelope.
                # ctf[i, j] *= np.exp(-k4 * s2)
                ctf[cnt] *= np.exp(-k4 * s2)
            cnt += 1
    return out
