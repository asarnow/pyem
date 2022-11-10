# -*- coding: utf-8 -*-
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


MODE = {0: np.dtype(np.int8), 1: np.dtype(np.int16), 2: np.dtype(np.float32),
        6: np.dtype(np.uint16), 12: np.dtype(np.float16),
        np.dtype(np.int8): 0, np.dtype(np.int16): 1, np.dtype(np.float32): 2,
        np.dtype(np.uint16): 6, np.dtype(np.float16): 12}
HEADER_LEN = int(1024)  # Bytes.


def mrc_header(shape, dtype=np.float32, psz=1.0):
    header = np.zeros(HEADER_LEN // np.dtype(np.int32).itemsize, dtype=np.int32)
    header_f = header.view(np.float32)
    header[:3] = shape
    if np.dtype(dtype) not in MODE:
        raise ValueError("Invalid dtype for MRC")
    header[3] = MODE[np.dtype(dtype)]
    header[7:10] = header[:3]  # mx, my, mz (grid size)
    header_f[10:13] = psz * header[:3]  # xlen, ylen, zlen
    header_f[13:16] = 90.0  # CELLB
    header[16:19] = 1, 2, 3  # Axis order.
    header_f[19:22] = 1, 0, -1  # Convention for unreliable  values.
    # header[26] = 1329812045  # "MRCO" chars.
    header[27] = 20140  # Version 2014-0.
    header_f[49:52] = 0, 0, 0  # Default origin.
    header[52] = 542130509  # "MAP " chars.
    header[53] = 17476  # 0x00004444 for little-endian.
    header_f[54] = -1  # Convention for unreliable RMS value.
    return header


def mrc_header_complete(data, psz=1.0, origin=None):
    header = mrc_header(data.shape, data.dtype, psz=psz)
    header_f = header.view(np.float32)
    header_f[19:22] = [data.min(), data.max(), data.mean()]
    header_f[54] = np.sqrt(np.mean(data**2))
    if origin is None:
        header_f[49:52] = (0, 0, 0)
    elif origin == "center":
        header_f[49:52] = psz * header[:3] / 2
    else:
        header_f[49:52] = origin
    return header


def _read_header(f):
    hdr = {}
    pos = f.tell()
    f.seek(0)
    header = np.fromfile(f, dtype=np.int32, count=256)
    f.seek(pos)
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
    return hdr


def read(fname, inc_header=False, compat="mrc2014"):
    if "relion" in compat.lower() or "xmipp" in compat.lower():
        order = "C"
    else:
        order = "F"  # Read x, y, z -> data[i, j, k] == data[i][j][k]
    data = None
    with open(fname) as f:
        hdr = _read_header(f)
        nx = hdr['nx']
        ny = hdr['ny']
        nz = hdr['nz']
        datatype = hdr['datatype']
        f.seek(HEADER_LEN)  # seek to start of data
        if nz == 1:
            shape = (nx, ny)
        else:
            shape = (nx, ny, nz)
        if datatype in MODE:
            dtype = MODE[datatype]
        else:
            raise IOError("Unknown MRC data type %d" % datatype)
        data = np.reshape(np.fromfile(f, dtype=dtype, count=nx * ny * nz), shape, order=order)
    if inc_header:
        return data, hdr
    else:
        return data


def write(fname, data, psz=1, origin=None, fast=False):
    """
    Write a MRC file. Fortran axes order is assumed.
    :param fname: Destination path.
    :param data: Array to write.
    :param psz: Pixel size in Å for MRC header.
    :param origin: Coordinate of origin voxel.
    :param fast: Skip computing density statistics in header. Default is False.
    """
    data = np.atleast_3d(data)
    if fast:
        header = mrc_header(data.shape, dtype=data.dtype, psz=psz)
    else:
        header = mrc_header_complete(data, psz=psz, origin=origin)
    with open(fname, 'wb') as f:
        f.write(header.tobytes())
        f.write(np.require(data, dtype=np.float32).tobytes(order="F"))


def append(fname, data):
    data = np.atleast_3d(data)
    with open(fname, 'r+b') as f:
        nx, ny, nz = np.fromfile(f, dtype=np.int32, count=3)
        f.seek(36)  # First byte of zlen.
        zlen = np.fromfile(f, dtype=np.float32, count=1)
        if data.shape[0] != nx or data.shape[1] != ny:
            raise ValueError("Data has different shape than destination file")
        f.seek(0, os.SEEK_END)
        f.write(np.require(data, dtype=np.float32).tobytes(order="F"))
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
        nx, ny, nz, dtype = np.fromfile(f, dtype=np.int32, count=4)
        if dtype in MODE:
            dtype = MODE[dtype]
        else:
            raise ValueError("Invalid dtype for MRC")
        if not np.can_cast(data.dtype, dtype):
            raise ValueError("Can't cast %s to %s" % (data.dtype, dtype))
        if data.shape[2] > nz:
            raise ValueError("Data has more z-slices than destination file")
        if data.shape[0] != nx or data.shape[1] != ny:
            raise ValueError("Data has different shape than destination file")
        f.seek(HEADER_LEN + idx * nx * ny * dtype.itemsize)
        f.write(np.require(data, dtype=dtype).tobytes(order="F"))


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
        if datatype in MODE:
            dtype = MODE[datatype]
        else:
            raise IOError("Unknown MRC data type %d" % datatype)
        f.seek(HEADER_LEN + idx * dtype.itemsize * nx * ny)
        return np.reshape(np.fromfile(f, dtype=dtype, count=nx * ny * num),
                          shape, order=order)


def read_zslices(fname):
    with ZSliceReader(fname) as zsr:
        for i in range(zsr.nz):
            yield zsr.read(i)


class ZSliceReader:
    def __init__(self, fname):
        self.path = fname
        self.f = open(self.path)
        self.nx, self.ny, self.nz, datatype = np.fromfile(self.f, dtype=np.int32, count=4)
        self.shape = (self.nx, self.ny)
        self.size = self.nx * self.ny
        if datatype in MODE:
            self.dtype = MODE[datatype]
        else:
            raise IOError("Unknown MRC data type %d" % datatype)
        self.i = 0

    def read(self, i):
        self.i = i
        if self.i >= self.nz:
            raise IOError("Index %d out of bounds for stack of size %d" % (i, self.nz))
        self.f.seek(HEADER_LEN + self.i * self.dtype.itemsize * self.size)
        # Populate slice C order so that X is fastest axis i.e. j / columns.
        return np.reshape(
            np.fromfile(self.f, dtype=self.dtype, count=self.size), self.shape)

    def close(self):
        self.f.close()

    def __iter__(self):
        self.i = 0
        return self

    def next(self):
        try:
            item = self.read(self.i)
        except IOError:
            raise StopIteration
        self.i += 1
        return item

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ZSliceWriter:
    def __init__(self, fname, shape=None, dtype=np.float32, psz=1.0, mode="w"):
        self.path = fname
        self.shape = None
        self.size = None
        self.psz = psz
        self.dtype = None
        self.f = None
        self.i = 0
        if shape is not None:
            self.set_shape(shape)
        if dtype is not None:
            self.set_dtype(dtype)
        if "a" in mode:
            hdr = read_header(self.path)
            self.f = open(self.path, 'ab')
            self.psz = hdr["xlen"] / hdr["nx"]
            self.set_shape((hdr["nx"], hdr["ny"]))
            if hdr["datatype"] in MODE:
                self.set_dtype(hdr["datatype"])
            else:
                raise IOError("Unknown MRC data type %d" % hdr['datatype'])
            self.f.seek(0, os.SEEK_END)
        else:
            self.f = open(self.path, 'wb')
            # self.f.seek(HEADER_LEN)  # Results in a sparse file?
            self.f.write(b'\x00' * HEADER_LEN)

    def set_dtype(self, dtype):
        if np.dtype(dtype) not in MODE:
            raise ValueError("Unknown MRC data type %d" % dtype)
        self.dtype = np.dtype(dtype)

    def set_shape(self, shape):
        if len(shape) == 1:
            self.shape = (1, shape[0])
        elif len(shape) == 2:
            self.shape = (shape[0], shape[1])
        elif len(shape) == 3:
            self.shape = (shape[1], shape[2])
        else:
            raise ValueError("Number of dimensions must be 1, 2, or 3")
        self.size = np.prod(self.shape)

    def write(self, arr):
        """
        Write one or more z-slices to ZSliceWriter's underlying File object.
        If ZSliceWriter was constructed without a shape, then the first two
        dimensions of the array define the shape (and slice size).
        Subsequent arrays can be of any shape, as long as their size is an
        integer multiple of the slice size.
        :param arr: A numpy array.
        """
        if self.i == 0:
            if self.shape is None:
                self.set_shape(arr.shape)
            if self.dtype is None:
                self.set_dtype(arr.dtype)
        assert np.can_cast(arr.dtype, self.dtype, casting="same_kind")
        assert arr.size % self.size == 0
        # self.f.seek(HEADER_LEN + self.i * self.dtype.itemsize * arr.size)
        self.f.write(np.require(arr, dtype=self.dtype).tobytes())
        self.i += arr.size / self.size

    def close(self):
        header = mrc_header(shape=(self.shape[1], self.shape[0], self.i),
                            dtype=self.dtype, psz=self.psz)
        self.f.seek(0)
        self.f.write(header.tobytes())
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# HEADER FORMAT
# 0      (0,4)      NX  number of columns (fastest changing in map)
# 1      (4,8)      NY  number of rows
# 2      (8,12)     NZ  number of sections (slowest changing in map)
# 3      (12,16)    MODE  data type:
#                       0   image: signed 8-bit bytes range -128 to 127
#                       1   image: 16-bit halfwords
#                       2   image: 32-bit reals
#                       3   transform: complex 16-bit integers
#                       4   transform: complex 32-bit reals
# 4      (16,20)    NXSTART number of first column in map
# 5      (20,24)    NYSTART number of first row in map
# 6      (24,28)    NZSTART number of first section in map
# 7      (28,32)    MX      number of intervals along X
# 8      (32,36)    MY      number of intervals along Y
# 9      (36,40)    MZ      number of intervals along Z
# 10-13  (40,52)    CELLA   cell dimensions in angstroms
# 13-16  (52,64)    CELLB   cell angles in degrees
# 16     (64,68)    MAPC    axis corresp to cols (1,2,3 for X,Y,Z)
# 17     (68,72)    MAPR    axis corresp to rows (1,2,3 for X,Y,Z)
# 18     (72,76)    MAPS    axis corresp to sections (1,2,3 for X,Y,Z)
# 19     (76,80)    DMIN    minimum density value
# 20     (80,84)    DMAX    maximum density value
# 21     (84,88)    DMEAN   mean density value
# 22     (88,92)    ISPG    space group number, 0 for images or 1 for volumes
# 23     (92,96)    NSYMBT  number of bytes in extended header
# 24-49  (96,196)   EXTRA   extra space used for anything
#           26  (104)   EXTTYP      extended header type("MRCO" for MRC)
#           27  (108)   NVERSION    MRC format version (20140)
# 49-52  (196,208)  ORIGIN  origin in X,Y,Z used for transforms
# 52     (208,212)  MAP     character string 'MAP ' to identify file type
# 53     (212,216)  MACHST  machine stamp
# 54     (216,220)  RMS     rms deviation of map from mean density
# 55     (220,224)  NLABL   number of labels being used
# 56-256 (224,1024) LABEL(80,10)    10 80-character text labels
