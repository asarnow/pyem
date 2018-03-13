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


def _read_header(f):
    hdr = {}
    header = np.fromfile(f, dtype=np.int32, count=256)
    header_f = header.view(np.float32)
    [hdr['nx'], hdr['ny'], hdr['nz'], hdr['datatype']] = header[:4]
    [hdr['xlen'], hdr['ylen'], hdr['zlen']] = header_f[10:13]
    if hdr['xlen'] == hdr['ylen'] == hdr['zlen'] == 0:
        hdr['xlen'] = hdr['nx']
        hdr['ylen'] = hdr['ny']
        hdr['zlen'] = hdr['nz']
    return hdr


def read_header(fname):
    with open(fname) as f:
        hdr = _read_header(f)
        # print "Nx %d Ny %d Nz %d Type %d" % (nx, ny, nz, datatype)
    return hdr


def read(fname, inc_header=False, compat="mrc2014"):
    if "relion" in compat.lower() or "xmipp" in compat.lower():
        order = "C"
    else:
        order = "F"
    with open(fname) as f:
        hdr = _read_header(f)
        nx = hdr['nx']
        ny = hdr['ny']
        nz = hdr['nz']
        datatype = hdr['datatype']
        f.seek(1024)  # seek to start of data
        if nz == 1:
            shape = (nx, ny)
        else:
            shape = (nx, ny, nz)
        if datatype == 1:
            dtype = 'int16'
        elif datatype == 2:
            dtype = 'float32'
        else:
            raise IOError("Unknown MRC data type")
        data = np.reshape(np.fromfile(f, dtype=dtype, count=nx * ny * nz), shape, order=order)
    if inc_header:
        return data, hdr
    else:
        return data


def write(fname, data, psz=1, origin=None, fast=False):
    """ Writes a MRC file. The header will be blank except for nx,ny,nz,datatype=2 for float32. 
    data should be (nx,ny,nz), and will be written in Fortran order as MRC requires."""
    data = np.atleast_3d(data)
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
    data = np.atleast_3d(data)
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
        nz.astype(np.int32).tofile(f)
        f.seek(36)
        zlen.astype(np.float32).tofile(f)


def write_imgs(fname, idx, data):
    with open(fname, 'r+b') as f:
        nx, ny, nz = np.fromfile(f, dtype=np.int32, count=3)
        if data.shape[2] > nz:
            raise Exception
        if data.shape[0] != nx or data.shape[1] != ny:
            raise Exception
        f.seek(1024 + idx * nx * ny * 4)
        np.require(np.reshape(data, (-1,), order='F'), dtype=np.float32).tofile(f)


def read_imgs(fname, idx, num=1, compat="mrc2014"):
    if "relion" in compat or "xmipp" in compat:
        order = "C"
    else:
        order = "F"
    with open(fname) as f:
        nx, ny, nz, datatype = np.fromfile(f, dtype=np.int32, count=4)
        assert (idx < nz)
        if num < 0:
            num = nz - idx
        assert (idx + num <= nz)
        assert (num != 0)
        if num == 1:
            shape = (nx, ny)
        else:
            shape = (nx, ny, num)
        if datatype == 1:
            dtype = 'int16'
            size = 2
        elif datatype == 2:
            dtype = 'float32'
            size = 4
        else:
            raise IOError("Unknown MRC data type")
        f.seek(1024 + idx * size * nx * ny)
        return np.reshape(np.fromfile(f, dtype=dtype, count=nx * ny * num), shape, order=order)


def read_zslices(fname, compat="mrc2014"):
    if "relion" in compat or "xmipp" in compat:
        order = "C"
    else:
        order = "F"
    with open(fname) as f:
        nx, ny, nz, datatype = np.fromfile(f, dtype=np.int32, count=4)
        shape = (nx, ny)
        if datatype == 1:
            dtype = 'int16'
            size = 2
        elif datatype == 2:
            dtype = 'float32'
            size = 4
        else:
            raise IOError("Unknown MRC data type")
        for idx in range(nz):
            f.seek(1024 + idx * size * nx * ny)
            yield np.reshape(np.fromfile(f, dtype=dtype, count=nx * ny), shape, order=order)


class ZSliceReader:
    def __init__(self, fname, compat="mrc2014"):
        if "relion" in compat or "xmipp" in compat:
            self.order = "C"
        else:
            self.order = "F"
        self.path = fname
        self.f = open(self.path)
        self.nx, self.ny, self.nz, datatype = np.fromfile(self.f, dtype=np.int32, count=4)
        self.shape = (self.nx, self.ny)
        self.size = self.nx * self.ny
        if datatype == 1:
            self.dtype = 'int16'
            self.datasize = 2
        elif datatype == 2:
            self.dtype = 'float32'
            self.datasize = 4
        else:
            raise IOError("Unknown MRC data type")
        self.i = 0

    def read(self, i):
        self.i = i
        if i >= self.nz:
            raise IOError("Index %d out of bounds for stack of size %d" % (i, self.nz))
        self.f.seek(1024 + self.i * self.datasize * self.size)
        return np.reshape(np.fromfile(self.f, dtype=self.dtype, count=self.size),
                          self.shape, order=self.order)

    def close(self):
        self.f.close()


# HEADER FORMAT
# 1      (0,4)      NX  number of columns (fastest changing in map)
# 2      (4,8)      NY  number of rows
# 3      (8,12)     NZ  number of sections (slowest changing in map)
# 4      (12,16)    MODE  data type:
#                       0   image: signed 8-bit bytes range -128 to 127
#                       1   image: 16-bit halfwords
#                       2   image: 32-bit reals
#                       3   transform: complex 16-bit integers
#                       4   transform: complex 32-bit reals
# 5      (16,20)    NXSTART number of first column in map
# 6      (20,24)    NYSTART number of first row in map
# 7      (24,28)    NZSTART number of first section in map
# 8      (28,32)    MX      number of intervals along X
# 9      (32,36)    MY      number of intervals along Y
# 10     (36,40)    MZ      number of intervals along Z
# 11-13  (40,52)    CELLA   cell dimensions in angstroms
# 14-16  (52,64)    CELLB   cell angles in degrees
# 17     (64,68)    MAPC    axis corresp to cols (1,2,3 for X,Y,Z)
# 18     (68,72)    MAPR    axis corresp to rows (1,2,3 for X,Y,Z)
# 19     (72,76)    MAPS    axis corresp to sections (1,2,3 for X,Y,Z)
# 20     (76,80)    DMIN    minimum density value
# 21     (80,84)    DMAX    maximum density value
# 22     (84,88)    DMEAN   mean density value
# 23     (88,92)    ISPG    space group number 0 or 1 (default=0)
# 24     (92,96)    NSYMBT  number of bytes used for symmetry data (0 or 80)
# 25-49  (96,196)   EXTRA   extra space used for anything
# 50-52  (196,208)  ORIGIN  origin in X,Y,Z used for transforms
# 53     (208,212)  MAP     character string 'MAP ' to identify file type
# 54     (212,216)  MACHST  machine stamp
# 55     (216,220)  RMS     rms deviation of map from mean density
# 56     (220,224)  NLABL   number of labels being used
# 57-256 (224,1024) LABEL(80,10)    10 80-character text labels
