#!/usr/bin/env python2.7
# Copyright (C) 2017 Daniel Asarnow
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
import numpy.ma as ma
from scipy.ndimage import map_coordinates
from pyfftw.interfaces.numpy_fft import rfftn
from .vop_numba import fill_ft


def ismask(vol):
    """
    Even with a soft edge, a mask will have very few unique values (unless it's already been resampled).
    The 1D slice below treats just the central XY section for speed. Real maps have ~20,000 unique values here.
    """
    return np.unique(vol[vol.shape[2] / 2::vol.shape[2]]).size < 100


def resample_volume(vol, r=None, t=None, ori=None, order=3, compat="mrc2014", indexing="ij", invert=False):
    if r is None and t is None:
        return vol.copy()

    center = np.array(vol.shape) // 2

    x, y, z = np.meshgrid(*[np.arange(-c, c) for c in center], indexing=indexing)
    xyz = np.vstack([x.reshape(-1), y.reshape(-1), z.reshape(-1), np.ones(x.size)])

    if ori is not None:
        xyz -= ori[:, None]

    th = np.eye(4)
    if t is None and r.shape[1] == 4:
        t = np.squeeze(r[:, 3])
    elif t is not None:
        th[:3, 3] = t

    rh = np.eye(4)
    if r is not None:
        rh[:3, :3] = r[:3, :3].T

    if invert:
        th[:3, 3] = -th[:3, 3]
        rh[:3, :3] = rh[:3:, :3].T
        xyz = rh.dot(th.dot(xyz))[:3, :] + center[:, None]
    else:
        xyz = th.dot(rh.dot(xyz))[:3, :] + center[:, None]

    xyz = np.array([arr.reshape(vol.shape) for arr in xyz])

    if "relion" in compat.lower() or "xmipp" in compat.lower():
        xyz = xyz[::-1]

    newvol = map_coordinates(vol, xyz, order=order)
    return newvol


def grid_correct(vol, pfac=2, order=1):
    n = vol.shape[0]
    nhalf = n / 2
    x, y, z = np.meshgrid(*[np.arange(-nhalf, nhalf)] * 3, indexing="xy")
    r = np.sqrt(x**2 + y**2 + z**2, dtype=vol.dtype) / (n * pfac)
    with np.errstate(divide="ignore", invalid="ignore"):
        sinc = np.sin(np.pi * r) / (np.pi * r)  # Results in 1 NaN in the center.
    sinc[nhalf, nhalf, nhalf] = 1.
    if order == 0:
       cordata = vol / sinc
    elif order == 1:
       cordata = vol / sinc**2
    else:
        raise NotImplementedError("Only nearest-neighbor and trilinear grid corrections are available")
    return cordata


def interpolate_slice(f3d, rot, pfac=2, size=None):
    nhalf = f3d.shape[0] / 2
    if size is None:
        phalf = nhalf
    else:
        phalf = size / 2
    qot = rot * pfac  # Scaling!
    px, py, pz = np.meshgrid(np.arange(-phalf, phalf), np.arange(-phalf, phalf), 0)
    pr = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    pcoords = np.vstack([px.reshape(-1), py.reshape(-1), pz.reshape(-1)])
    mcoords = qot.T.dot(pcoords)
    mcoords = mcoords[:, pr.reshape(-1) < nhalf]
    pvals = map_coordinates(np.real(f3d), mcoords, order=1, mode="wrap") + \
             1j * map_coordinates(np.imag(f3d), mcoords, order=1, mode="wrap")
    pslice = np.zeros(pr.shape, dtype=np.complex)
    pslice[pr < nhalf] = pvals
    return pslice


def vol_ft(vol, pfac=2, threads=1, normfft=1):
    """ Returns a centered, Nyquist-limited, zero-padded, interpolation-ready 3D Fourier transform.
    :param vol: Volume to be Fourier transformed.
    :param pfac: Size factor for zero-padding.
    :param threads: Number of threads for pyFFTW.
    :param normfft: Normalization constant for Fourier transform.
    """
    vol = grid_correct(vol, pfac=pfac, order=1)
    padvol = np.pad(vol, (vol.shape[0] * pfac - vol.shape[0]) // 2, "constant")
    ft = rfftn(np.fft.ifftshift(padvol), padvol.shape, threads=threads)
    ftc = np.zeros((ft.shape[0] + 3, ft.shape[1] + 3, ft.shape[2]), dtype=ft.dtype)
    fill_ft(ft, ftc, vol.shape[0], normfft=normfft)
    return ftc


def normalize(vol, ref=None, return_stats=False):
    volm = vol.view(ma.MaskedArray)
    sz = volm.shape[0]
    rng = np.arange(-sz/2, sz)
    x, y, z = np.meshgrid(rng, rng, rng)
    r2 = x**2 + y**2 + z**2
    mask = r2 > sz**2
    volm.mask = mask
    if ref is not None:
        ref = ref.view(ma.MaskedArray)
        ref.mask = mask
        sigma = np.std(ref)
        mu = np.mean(ref)
    else:
        sigma = np.std(volm)
        mu = np.mean(volm)
    if return_stats:
        return (vol - mu) / sigma, mu, sigma
    return (vol - mu) / sigma
