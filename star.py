# Copyright (C) 2016 Daniel Asarnow
# University of California, San Francisco
#
# Simple program for parsing and altering Relion .star files.
# See help text and README file for more information.
#
# Program for projection subtraction in electron microscopy.
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
import sys
import pandas as pd


def main(args):
    star = parse_star(args.input)

    if args.drop_angles:
        ang_fields = [f for f in star.columns if "Tilt" in f or "Psi" in f or "Rot" in f]
        star.drop(ang_fields, axis=1, inplace=True, errors="ignore")

    write_star(args.output, star)
    return 0


def parse_star(starfile):
    headers = []
    foundheader = False
    ln = 0
    with open(starfile, 'rU') as f:
        for l in f:
            if l.startswith('_rln'):
                foundheader = True
                lastheader = True
                headers.append(l.rstrip())
            else:
                lastheader = False
            if foundheader and not lastheader:
                break
            ln += 1
    star = pd.read_table(starfile, skiprows=ln, delimiter='\s+', header=None)
    star.columns = headers
    return star


def write_star(starfile, star):
    return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop-angles", help="Tilt, psi and rot angles will be dropped from output")
    parser.add_argument("input", help="Input .star file")
    parser.add_argument("output", help="Output .star file")
    sys.exit(main(parser.parse_args()))
