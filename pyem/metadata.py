#!/usr/bin/env python2.7
# Copyright (C) 2017 Daniel Asarnow
# University of California, San Francisco
#
# I/O routines in pyem.
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
import pandas as pd


def parse_par(fn):
    head_data = {"Input particle images": None,
           "Beam energy (keV)": None,
           "Spherical aberration (mm)": None,
           "Amplitude contrast": None,
           "Pixel size of images (A)": None}
    ln = 1
    skip = 0
    with open(fn, 'rU') as f:
        lastheader = False
        firstblock = True
        for l in f:
            if l.startswith("C") and firstblock:
                if "PSI" in l or "DF1" in l:
                    lastheader = True
                    headers = l.rstrip().split()
                if ":" in l:
                    tok = l.split(":")
                    tok[1] = tok[1].lstrip().rstrip()
                    tok[0] = tok[0].lstrip("C ")
                    if tok[0] in head_data:
                        try:
                            head_data[tok[0]] = float(tok[1])
                        except ValueError:
                            head_data[tok[0]] = tok[1]
                if lastheader:
                    skip = ln
                    firstblock = False
            elif l.startswith("C"):
                break
            else:
                headers = ["C", "PHI", "THETA", "PSI", "SHX", "SHY",
                        "MAG", "FILM", "DF1", "DF2", "ANGAST",
                        "OCC", "LogP", "SIGMA", "SCORE", "CHANGE"]
                break
                
            ln += 1

    if skip == 0:
        n = None
    else:
        n = ln - skip - 1
    par = pd.read_table(fn, skiprows=skip, nrows=n, delimiter="\s+", header=None, comment="C")
    par.columns = headers
    for k in head_data:
        if head_data[k] is not None:
            par[k] = head_data[k]
    return par


def par2star(par, v9=True):
    general = {"PHI": None,
            "THETA": None,
            "PSI": None,
            "SHX": None,
            "SHY": None,
            "MAG": "rlnMagnification",
            "FILM": "rlnGroupNumber",
            "DF1": "rlnDefocusU",
            "DF2": "rlnDefocusV",
            "ANGAST": "rlnDefocusAngle",
            "Beam energy (keV)": "rlnVoltage",
            "Spherical aberration (mm)": "rlnSphericalAberration",
            "Amplitude contrast": "rlnAmplitudeContrast",
            "C": None,
            "Pixel size of images (A)": None
            }
    rlnheaders = [general[h] for h in par.columns if h in general and general[h] is not None]
    star = par[[h for h in par.columns if h in general and general[h] is not None]].copy()
    star.columns = rlnheaders
    star["rlnImageName"] = ["%.6d" % (i + 1) for i in par["C"]]  # Reformat particle idx for Relion.
    star["rlnImageName"] = star["rlnImageName"].str.cat(par["Input particle images"], sep="@")
    star["rlnDetectorPixelSize"] = par["Pixel size of images (A)"] * par["MAG"] / 10000.0
    star["rlnOriginX"] = par["SHX"] / par["Pixel size of images (A)"]
    star["rlnOriginY"] = par["SHY"] / par["Pixel size of images (A)"]
    star["rlnAngleRot"] = -par["PSI"]
    star["rlnAngleTilt"] = -par["THETA"]
    star["rlnAnglePsi"] = -par["PHI"]
    return star

