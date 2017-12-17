# Copyright (C) 2016 Eugene Palovcak and Daniel Asarnow
# University of Calfornia, San Francisco
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
import os


def read(fname, inc_header=False, compat="mrc2014"):
    hdr = read_header(fname)
    nx = hdr['nx']
    ny = hdr['ny']
    nz = hdr['nz']
    datatype = hdr['datatype']
    with open(fname) as f:
        f.seek(1024)  # seek to start of data
        if datatype == 1:
            data = np.reshape(np.fromfile(f, dtype='int16', count=nx * ny * nz), (nx, ny, nz), order='F')
        if datatype == 2:
            data = np.reshape(np.fromfile(f, dtype='float32', count=nx * ny * nz), (nx, ny, nz), order='F')

    if "relion" in compat.lower() or "xmipp" in compat.lower():
        data = np.transpose(data)

    if inc_header:
        return data, hdr
    else:
        return data


def write(fname, data, psz=1, origin=None, fast=False):
    """ Writes a MRC file. The header will be blank except for nx,ny,nz,datatype=2 for float32. 
    data should be (nx,ny,nz), and will be written in Fortran order as MRC requires."""
    header = np.zeros(256, dtype=np.int32)  # 1024 byte header
    header_f = header.view(np.float32)
    header[:3] = data.shape  # nx, ny, nz
    header[3] = 2  # mode, 2 = float32 datatype
    header[7:10] = data.shape  # mx, my, mz (grid size)
    header_f[10:13] = [psz * i for i in data.shape]  # xlen, ylen, zlen
    header_f[13:16] = 90.0  # CELLB
    header[16:19] = [1, 2, 3]  # axis order
    if not fast:
        header_f[19:22] = [data.min(), data.max(), data.mean()]  # data stats
    if origin is None:
        header_f[49:52] = [0, 0, 0]
    elif origin is "center":
        header_f[49:52] = [psz * i / 2 for i in data.shape]
    else:
        header_f[49:52] = origin
    header[52] = 542130509  # 'MAP ' chars
    header[53] = 16708
    with open(fname, 'wb') as f:
        header.tofile(f)
        np.require(np.reshape(data, (-1,), order='F'), dtype=np.float32).tofile(f)


def append(fname, data):
    with open(fname, 'r+b') as f:
        nx, ny, nz = np.fromfile(f, dtype=np.int32, count=3)  # First 12 bytes of stack.
        f.seek(36)  # First byte of zlen.
        zlen = np.fromfile(f, dtype=np.float32, count=1)
        if data.shape[0] != nx or data.shape[1] != ny:
            raise Exception
        f.seek(0, os.SEEK_END)
        np.require(np.reshape(data, (-1,), order='F'), dtype=np.float32).tofile(f)
        # Update header after new data is written.
        apix = zlen / nz
        nz += data.shape[2]
        zlen += apix * data.shape[2]
        f.seek(8)
        nz.tofile(f)
        f.seek(36)
        zlen.tofile(f)


def write_imgs(fname, idx, data):
    with open(fname, 'r+b') as f:
        nx, ny, nz = np.fromfile(f, dtype=np.int32, count=3)  # First 12 bytes of stack.
        if data.shape[2] > nz:
            raise Exception
        if data.shape[0] != nx or data.shape[1] != ny:
            raise Exception
        f.seek(1024 + idx * nx * ny * 4)
        np.require(np.reshape(data, (-1,), order='F'), dtype=np.float32).tofile(f)


def read_imgs(fname, idx, num=None):
    hdr = read_header(fname)
    nx = hdr['nx']
    ny = hdr['ny']
    nz = hdr['nz']
    datatype = hdr['datatype']
    assert (idx < nz)
    if num is None:
        num = nz - idx
    assert (idx + num <= nz)
    assert (num > 0)
    datasizes = {1: 2, 2: 4}
    with open(fname) as f:
        f.seek(1024 + idx * datasizes[datatype] * nx * ny)  # Seek to start of img idx.
        if datatype == 1:
            return np.reshape(np.fromfile(f, dtype='int16', count=nx * ny * num), (nx, ny, num), order='F')
        if datatype == 2:
            return np.reshape(np.fromfile(f, dtype='float32', count=nx * ny * num), (nx, ny, num), order='F')


def read_header(fname):
    with open(fname) as f:
        hdr = {}
        header = np.fromfile(f, dtype=np.int32, count=256)
        header_f = header.view(np.float32)
        [hdr['nx'], hdr['ny'], hdr['nz'], hdr['datatype']] = header[:4]
        [hdr['xlen'], hdr['ylen'], hdr['zlen']] = header_f[10:13]
        if hdr['xlen'] == hdr['ylen'] == hdr['zlen'] == 0:
            hdr['xlen'] = hdr['nx']
            hdr['ylen'] = hdr['ny']
            hdr['zlen'] = hdr['nz']
        # print "Nx %d Ny %d Nz %d Type %d" % (nx, ny, nz, datatype)
    return hdr

# C************************************************************************
# C                                   *
# C   HEADER FORMAT                           *
# C   1   NX  number of columns (fastest changing in map) *
# C   2   NY  number of rows                  *
# C   3   NZ  number of sections (slowest changing in map)    *
# C   4   MODE    data type :                 *
# C           0   image : signed 8-bit bytes range -128   *
# C                   to 127              *
# C           1   image : 16-bit halfwords        *
# C           2   image : 32-bit reals            *
# C           3   transform : complex 16-bit integers *
# C           4   transform : complex 32-bit reals    *
# C   5   NXSTART number of first column in map           *
# C   6   NYSTART number of first row in map          *
# C   7   NZSTART number of first section in map          *
# C   8   MX  number of intervals along X         *
# C   9   MY  number of intervals along Y         *
# C   10  MZ  number of intervals along Z         *
# C   11-13   CELLA   cell dimensions in angstroms            *
# C   14-16   CELLB   cell angles in degrees              *
# C   17  MAPC    axis corresp to cols (1,2,3 for X,Y,Z)      *
# C   18  MAPR    axis corresp to rows (1,2,3 for X,Y,Z)      *
# C   19  MAPS    axis corresp to sections (1,2,3 for X,Y,Z)  *
# C   20  DMIN    minimum density value               *
# C   21  DMAX    maximum density value               *
# C   22  DMEAN   mean density value              *
# C   23  ISPG    space group number 0 or 1 (default=0)       *
# C   24  NSYMBT  number of bytes used for symmetry data (0 or 80)*
# C   25-49   EXTRA   extra space used for anything           *
# C   50-52   ORIGIN  origin in X,Y,Z used for transforms     *
# C   53  MAP character string 'MAP ' to identify file type   *
# C   54  MACHST  machine stamp                   *
# C   55  RMS rms deviation of map from mean density      *
# C   56  NLABL   number of labels being used         *
# C   57-256  LABEL(80,10) 10 80-character text labels        *
# C                                   *
# C************************************************************************
