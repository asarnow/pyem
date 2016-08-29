#!/usr/bin/env python
# Copyright (C) 2016 Daniel Asarnow
# University of California, San Francisco
#
# Simple library for interacting with particle images.
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
from EMAN2 import EMData, Transform
from sparx import generate_ctf, filt_ctf


def particles(star):
    """
    Generator function using StarFile object to produce particle EMData and MetaData objects.
    :param star: StarFile object
    :return: Tuple holding (particle EMData, particle MetaData)
    """
    npart = len(star['rlnImageName'])
    for i in range(npart):
        meta = MetaData(star, i)
        ptcl = EMData(meta.name, meta.number)
        yield ptcl, meta


def make_proj(dens, meta):
    """
    Project and CTF filter density according to particle metadata.
    :param dens: EMData density
    :param meta: Particle metadata (Euler angles and CTF parameters)
    :return: CTF-filtered projection
    """
    t = Transform()
    t.set_rotation({'psi': meta.psi, 'phi': meta.phi, 'theta': meta.theta, 'type': 'spider'})
    t.set_trans(-meta.x_origin, -meta.y_origin)
    proj = dens.project("standard", t)
    ctf = generate_ctf(meta.ctf_params)
    ctf_proj = filt_ctf(proj, ctf)
    return ctf_proj


class MetaData:
    """
    Class representing particle metadata (from .star file).
    Includes Euler angles, particle origin, and CTF parameters.
    """

    def __init__(self, star, i):
        """
        Instantiate MetaData object for i'th particle in StarFile object.
        :param star: StarFile object
        :param i: Index of desired particle
        """
        self.i = i
        self.number = int(star['rlnImageName'][i].split("@")[0]) - 1
        self.name = star['rlnImageName'][i].split("@")[1]
        self.phi = star['rlnAngleRot'][i]
        self.psi = star['rlnAnglePsi'][i]
        self.theta = star['rlnAngleTilt'][i]
        self.x_origin = star['rlnOriginX'][i]
        self.y_origin = star['rlnOriginY'][i]
        # CTFFIND4 --> sparx CTF conventions (from CTER paper).
        self.defocus = (star['rlnDefocusU'][i] + star['rlnDefocusV'][i]) / 20000.0
        self.dfdiff = (star['rlnDefocusU'][i] - star['rlnDefocusV'][i]) / 10000.0
        self.dfang = 90.0 - star['rlnDefocusAngle'][i]
        self.apix = ((10000.0 * star['rlnDetectorPixelSize'][i]) /
                     float(star['rlnMagnification'][i]))
        self.voltage = star["rlnVoltage"][i]
        self.cs = star["rlnSphericalAberration"][i]
        self.ac = star["rlnAmplitudeContrast"][i] * 100.0
        self.bfactor = 0
        self.ctf_params = [self.defocus, self.cs, self.voltage, self.apix, self.bfactor, self.ac, self.dfdiff,
                           self.dfang]
