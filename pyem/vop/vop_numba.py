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
import numba
import numpy as np


@numba.jit(cache=True, nopython=True, nogil=True)
def fill_ft(ft, ftc, rmax, normfft=1):
    rmax2 = rmax ** 2
    for k in range(ft.shape[0]):
        kp = k if k < ft.shape[2] else k - ft.shape[0]
        for i in range(ft.shape[1]):
            ip = i if i < ft.shape[2] else i - ft.shape[1]
            for j in range(ft.shape[2]):
                jp = j
                r2 = ip**2 + jp**2 + kp**2
                if r2 <= rmax2:
                    ftc[kp + ftc.shape[0]//2, ip + ftc.shape[1]//2, jp] = ft[k, i, j] * normfft


@numba.jit(cache=False, nopython=True, nogil=True)
def interpolate_slice_numba(f3d, rot, pfac=2, size=None):
    linterp = lambda a, l, h: l + (h - l) * a
    ori = f3d.shape[0] // 2 - 1
    n = (f3d.shape[0] - 3) // pfac
    nhalf = n // 2
    if size is None:
        size = n
    phalf = size // 2
    rmax = min(nhalf, (phalf+1) - 1)
    rmax2 = rmax**2
    qot = rot.T * pfac  # Scaling!
    f2d = np.zeros((np.int64(size), np.int64(phalf + 1)), dtype=np.complex64)

    for i in range(size):
        
        if i <= rmax:
            yp = i
        elif i >= size - rmax:
            yp = i - size
        else:
            continue

        yp2 = yp ** 2

        for j in range(rmax + 1):
            xp = j

            if xp**2 + yp2 > rmax2:
                continue

            x = qot[0,0] * xp + qot[0,1] * yp  # Implicit z = 0.
            y = qot[1,0] * xp + qot[1,1] * yp
            z = qot[2,0] * xp + qot[2,1] * yp

            if x < 0:
                x = -x
                y = -y
                z = -z
                negx = True
            else:
                negx = False

            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            z0 = int(np.floor(z))

            ax = x - x0
            ay = y - y0
            az = z - z0

            y0 += ori + 1
            z0 += ori + 1

            x1 = x0 + 1
            y1 = y0 + 1
            z1 = z0 + 1

            f2d[i, j] = linterp(az,
                    linterp(ay,
                            linterp(ax, f3d[z0,y0,x0], f3d[z0,y0,x1]),
                            linterp(ax, f3d[z0,y1,x0], f3d[z0,y1,x1])),
                    linterp(ay,
                            linterp(ax, f3d[z1,y0,x0], f3d[z1,y0,x1]),
                            linterp(ax, f3d[z1,y1,x0], f3d[z1,y1,x1])))

            if negx:
                f2d[i, j] = np.conj(f2d[i, j])
    return f2d


def accumulate_slice_nb(f3d, f2d, rot, pfac=2):
    pass
