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
from .geom_numba import cross3_sca


@numba.jit(nopython=True, cache=False, nogil=True)
def _qconj(q, p):
    p[0] = q[0]
    p[1] = -q[1]
    p[2] = -q[2]
    p[3] = -q[3]
    return p


@numba.guvectorize(["void(float64[:], float64[:])"],
                   "(m)->(m)", nopython=True, cache=False)
def qconj(q, p):
    _qconj(q, p)


@numba.jit(nopython=True, cache=False, nogil=True)
def _qtimes(q1, q2, q3):
    q3[0] = q1[0] * q2[0] - (q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3])
    q3[1] = q1[2] * q2[3] - q1[3] * q2[2] + q1[0] * q2[1] + q2[0] * q1[1]
    q3[2] = q1[3] * q2[1] - q1[1] * q2[3] + q1[0] * q2[2] + q2[0] * q1[2]
    q3[3] = q1[1] * q2[2] - q1[2] * q2[1] + q1[0] * q2[3] + q2[0] * q1[3]
    return q3


@numba.guvectorize(["void(float64[:], float64[:], float64[:])"],
                   "(m),(m)->(m)", nopython=True, cache=False)
def qtimes(q1, q2, q3):
    _qtimes(q1, q2, q3)


@numba.jit(nopython=True, cache=False, nogil=True)
def _qsqrt(q, p):
    p[0] = q[0] + 1
    p[1:] = q[1:]
    p[:] = p[:] / np.sqrt(2 * (1 + q[0]))
    return p


@numba.guvectorize(["void(float64[:], float64[:])"],
                   "(m)->(m)", nopython=True, cache=False)
def qsqrt(q, p):
    _qsqrt(q, p)


@numba.jit(cache=True, nopython=True, nogil=True)
def qslerp(q1, q2, t, longest=False):
    cos_half_theta = np.dot(q1, q2)
    if cos_half_theta >= 1.0:
        return q1.copy()
    if longest:
        if cos_half_theta > 0:
            cos_half_theta = -cos_half_theta
            q1 = -q1
    elif cos_half_theta < 0:
        cos_half_theta = -cos_half_theta
        q1 = -q1
    half_theta = np.arccos(cos_half_theta)
    sin_half_theta = np.sqrt(1 - cos_half_theta * cos_half_theta)
    if np.abs(sin_half_theta) < 1E-12:
        return (q1 + q2) / 2
    a = np.sin((1 - t) * half_theta)
    b = np.sin(t * half_theta)
    return (q1 * a + q2 * b) / sin_half_theta


@numba.jit(cache=False, nopython=True, parallel=True)
def cdistq(q1, q2, d):
    # if q2 is None:
    #     q2 = q1
    # if out is None:
    # d = np.zeros((len(q1), len(q2)), dtype=q1.dtype)
    # else:
    #     d = out
    #     assert len(d) == len(q1)
    #     assert len(d[0]) == len(q2)
    pi_half = np.pi / 2
    for i in numba.prange(d.shape[0]):
        for j in range(d.shape[1]):
            v = np.abs(np.sum(q1[i] * q2[j]))
            if v > 1.0:
                v = 1.0
            v = np.arccos(v)
            v *= 2
            if v > pi_half:
                v = np.pi - v
            # v += 1e-6
            v **= 2
            v *= -0.5
            d[i, j] = v
    return d


@numba.jit(cache=False, nopython=True, parallel=True)
def pdistq(q1, d):
    pi_half = np.pi / 2
    for i in numba.prange(d.shape[0]):
        d[i, i] = 0.0
        for j in range(i + 1, d.shape[1]):
            v = np.abs(np.sum(q1[i] * q1[j]))
            if v > 1.0:
                v = 1.0
            v = np.arccos(v)
            v *= 2
            if v > pi_half:
                v = np.pi - v
            # v += 1e-6
            v **= 2
            v *= -0.5
            d[i, j] = d[j, i] = v
    return d


@numba.guvectorize(["void(complex128[:], complex128[:], complex128[:])"],
                   "(m),(m)->(m)", nopython=True, cache=False)
def dqtimes(q1, q2, q3):
    # Dual part = dual1 * real2 + real1 * dual2
    _qtimes(q1.imag, q2.real, q3.real)  # Store 1st dual term in real.
    _qtimes(q1.real, q2.imag, q3.imag)  # Store 2nd dual term in imag.
    tmp = q3.imag  # Workaround Numba typing error.
    tmp += q3.real  # Sum terms into imag.
    # Real part = real1 * real2
    _qtimes(q1.real, q2.real, q3.real)  # Overwrite real with real part.
    return


@numba.guvectorize(["void(complex128[:], complex128[:])"],
                   "(m)->(m)", nopython=True, cache=False)
def dqconj(q, p):
    _qconj(q.real, p.real)
    _qconj(q.imag, p.imag)


@numba.jit(nopython=True, cache=False, nogil=True)
def dqtimes_sca(q1, q2):
    q3 = np.zeros((4,), dtype=q1.dtype)
    # Dual part = dual1 * real2 + real1 * dual2
    _qtimes(q1.imag, q2.real, q3.real)  # Store 1st dual term in real.
    _qtimes(q1.real, q2.imag, q3.imag)  # Store 2nd dual term in imag.
    tmp = q3.imag  # Workaround Numba typing error.
    tmp += q3.real  # Sum terms into imag.
    # Real part = real1 * real2
    _qtimes(q1.real, q2.real, q3.real)  # Overwrite real with real part.
    return q3


@numba.jit(nopython=True, cache=False, nogil=True)
def dqconj_sca(q):
    p = np.zeros((4,), dtype=q.dtype)
    _qconj(q.real, p.real)
    _qconj(q.imag, p.imag)
    return p


@numba.jit(nopython=True, cache=False, nogil=True)
def dq2sc(q):
    theta = 2 * np.arccos(q[0].real)
    # nr = 1 / np.linalg.norm(q[1:].real)
    nr = np.sin(theta / 2)
    if nr < 1E-6:
        # nr = 0
        nr = 1e6
    else:
        nr = 1 / nr
    d = -2 * q[0].imag * nr
    l = q[1:].real * nr
    m = (q[1:].imag - l * d * q[0].real * 0.5) * nr
    return theta, d, l, m


@numba.jit(nopython=True, cache=False, nogil=True)
def sc2dq(theta, d, l, m):
    q = np.zeros((4,), dtype=np.complex128)
    theta_half = 0.5 * theta
    sin_theta_half = np.sin(theta_half)
    cos_theta_half = np.cos(theta_half)
    d_half = 0.5 * d
    q.real[0] = cos_theta_half
    q.real[:1] = sin_theta_half * l
    q.imag[0] = - d_half * sin_theta_half
    q.imag[1:] = sin_theta_half * m + d_half * cos_theta_half * l
    return q


# @numba.jit(nopython=True, cache=False, nogil=True)
# def dqpower(q, t):
#     theta, d, l, m = dq2sc(q)
#     theta_half = 0.5 * theta
#     d_half = 0.5 * d
#     da_real = t * np.cos(theta_half)
#     da_dual = t * -d_half * np.sin(theta_half)


@numba.jit(nopython=True, cache=False, parallel=True)
def pdistdq(q, d):
    relq = np.zeros((4,), dtype=np.complex128)
    for i in numba.prange(d.shape[0]):
        # relq = geom.dqtimes(geom.dqconj(q[i, None]), q)
        for j in range(i + 1, d.shape[1]):
            relq[:] = dqtimes_sca(dqconj_sca(q[i]), q[j])
            theta, dax, l, m = dq2sc(relq)
            p0 = cross3_sca(l, m)
            r2 = np.sum(p0 ** 2)
            v = np.sqrt(dax ** 2 + theta ** 2 * r2)
            d[i, j] = d[j, i] = v
    return d


@numba.jit(nopython=True, cache=False, parallel=True)
def cdistdq(q1, q2, d):
    relq = np.zeros((4,), dtype=np.complex128)
    for i in numba.prange(d.shape[0]):
        for j in range(d.shape[1]):
            relq = dqtimes_sca(dqconj_sca(q1[i]), q2[j])
            theta, dax, l, m = dq2sc(relq)
            p0 = cross3_sca(l, m)
            r2 = np.sum(p0 ** 2)
            v = np.sqrt(dax ** 2 + theta ** 2 * r2)
            d[i, j] = v
    return d


@numba.jit(nopython=True, cache=False, nogil=True)
def dqblend(q1, q2, t):
    q3 = (1 - t) * q1 + t * q2
    q3 /= np.linalg.norm(q3)
    return q3
