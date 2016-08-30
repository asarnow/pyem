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
from __future__ import print_function
import sys
import os.path
from EMAN2 import EMData
from pathos.multiprocessing import Pool
from pyem.star import parse_star
from pyem.particle import particles, make_proj


def main(args):
    dens = EMData(args.map)
    star = parse_star(args.input, keep_index=False)
    star[["ImageNumber", "ImageName"]] = star['rlnImageName'].str.split("@", expand=True)
    grouped = star.groupby("ImageName")
    pool = None
    if args.nproc > 1:
        pool = Pool(processes=args.nproc)
        results = pool.imap(lambda x: project_stack(x, dens, args.dest), (group for name, group in grouped))
    else:
        results = (project_stack(group, dens, args.dest) for name, group in grouped)
    i = 0
    t = 0
    for r in results:
        i += 1
        t += r
        sys.stdout.write("\rProjected %d particles in %d stacks" % (t, i))
        sys.stdout.flush()

    if pool is not None:
        pool.close()
        pool.join()

    sys.stdout.write('\n')
    sys.stdout.flush()

    return 0


def project_stack(group, dens, dest):
    ptcls = particles(group)
    i = 0
    for p, m in ptcls:
        ctf_proj = make_proj(dens, m)
        fname = os.path.join(dest, os.path.basename(m.name))
        ctf_proj.write_image(fname, i)
        i += 1
    return i


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="RELION .star file listing input particle image stack(s)")
    parser.add_argument("--map", help="Map used to calculate projections")
    parser.add_argument("--nproc", help="Number of parallel processes",
                        type=int, default=1)
    parser.add_argument("dest", help="Destination directory for output image stack(s)")
    sys.exit(main(parser.parse_args()))
