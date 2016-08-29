#! /usr/bin/python2.7
# Copyright (C) 2016 Daniel Asarnow, Eugene Palovcak
# University of California, San Francisco
#
# Program for projecting density maps in electron microscopy.
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
from EMAN2 import EMData
from pyem.star import parse_star
from pyem.particle import particles, make_proj


def main(args):
    star = parse_star(args.input, keep_index=False)
    dens = EMData(args.map)
    ptcls = particles(star)
    for p, m in ptcls:
        ctf_proj = make_proj(dens, m)
        ctf_proj.write_image(m.name, m.number)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="RELION .star file listing input particle image stack(s)")
    parser.add_argument("--map", help="Map used to calculate projections")
    parser.add_argument("suffix", help="Relative path and suffix for output image stack(s)",
                        default=None)
    sys.exit(main(parser.parse_args()))
