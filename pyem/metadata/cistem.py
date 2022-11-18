# Copyright (C) 2017 Daniel Asarnow
# University of California, San Francisco
#
# Handles metadata from Frealign9, FrealignX, and cisTEM.
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
from pyem import star


def parse_f9_par(fn):
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


def parse_fx_par(fn):
    with open(fn, 'r') as f:
        columns = f.readline().split()
        df = pd.read_csv(f, delimiter="\s+", header=None, names=columns, comment="C")
    return df


def write_f9_par(fn, df):
    formatters = {"C": lambda x: "%7d" % x,
                  "PSI": lambda x: "%7.2f" % x,
                  "THETA": lambda x: "%7.2f" % x,
                  "PHI": lambda x: "%7.2f" % x,
                  "SHX": lambda x: "%9.2f" % x,
                  "SHY": lambda x: "%9.2f" % x,
                  "MAG": lambda x: "%7.0f" % x,
                  "INCLUDE": lambda x: "%5d" % x,
                  "DF1": lambda x: "%8.1f" % x,
                  "DF2": lambda x: "%8.1f" % x,
                  "ANGAST": lambda x: "%7.2f" % x,
                  "PSHIFT": lambda x: "%7.2f" % x,
                  "OCC": lambda x: "%7.2f" % x,
                  "LogP": lambda x: "%9d" % x,
                  "SIGMA": lambda x: "%10.4f" % x,
                  "SCORE": lambda x: "%7.2f" % x,
                  "CHANGE": lambda x: "%7.2f" % x}
    with open(fn, 'w') as f:
        f.write(df.to_string(formatters=formatters, index=False))


def write_fx_par(fn, df):
    formatters = {"C": lambda x: "%7d" % x,
                  "PSI": lambda x: "%7.2f" % x,
                  "THETA": lambda x: "%7.2f" % x,
                  "PHI": lambda x: "%7.2f" % x,
                  "SHX": lambda x: "%9.2f" % x,
                  "SHY": lambda x: "%9.2f" % x,
                  "MAG": lambda x: "%7.0f" % x,
                  "INCLUDE": lambda x: "%5d" % x,
                  "DF1": lambda x: "%8.1f" % x,
                  "DF2": lambda x: "%8.1f" % x,
                  "ANGAST": lambda x: "%7.2f" % x,
                  "PSHIFT": lambda x: "%7.2f" % x,
                  "OCC": lambda x: "%7.2f" % x,
                  "LogP": lambda x: "%9d" % x,
                  "SIGMA": lambda x: "%10.4f" % x,
                  "SCORE": lambda x: "%7.2f" % x,
                  "CHANGE": lambda x: "%7.2f" % x}
    with open(fn, 'w') as f:
        f.write("C           PSI   THETA     PHI       SHX       SHY     "
                "MAG  INCLUDE   DF1      DF2  ANGAST  PSHIFT     "
                "OCC      LogP      SIGMA   SCORE  CHANGE\n")
        df.to_string(buf=f, formatters=formatters, index=False, header=None)
        f.write("\nC Blank line\n")


def par2star(par, data_path, apix=1.0, cs=2.0, ac=0.07, kv=300, invert_eulers=True):
    general = {"PHI": None,
            "THETA": None,
            "PSI": None,
            "SHX": None,
            "SHY": None,
            "MAG": None,
            "FILM": star.Relion.GROUPNUMBER,
            "DF1": star.Relion.DEFOCUSU,
            "DF2": star.Relion.DEFOCUSV,
            "ANGAST": star.Relion.DEFOCUSANGLE,
            "C": None,
            "CLASS": star.Relion.CLASS
            }
    rlnheaders = [general[h] for h in par.columns if h in general and general[h] is not None]
    df = par[[h for h in par.columns if h in general and general[h] is not None]].copy()
    df.columns = rlnheaders
    df[star.UCSF.IMAGE_INDEX] = par["C"] - 1
    df[star.UCSF.IMAGE_PATH] = data_path
    df[star.Relion.IMAGEPIXELSIZE] = apix
    df[star.Relion.CS] = cs
    df[star.Relion.AC] = ac
    df[star.Relion.VOLTAGE] = kv
    df[star.Relion.ORIGINXANGST] = -par["SHX"]
    df[star.Relion.ORIGINYANGST] = -par["SHY"]
    if invert_eulers:
        df[star.Relion.ANGLEROT] = -par["PSI"]
        df[star.Relion.ANGLETILT] = -par["THETA"]
        df[star.Relion.ANGLEPSI] = -par["PHI"]
    else:
        df[star.Relion.ANGLES] = par[["PHI", "THETA", "PSI"]]
    return df
