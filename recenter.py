#! /usr/bin/python2.7
# Copyright (C) 2016 Daniel Asarnow
# University of California, San Francisco
#
# Program implementing several methods for recentering particles.
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
import glob
import logging
import sys
import numpy as np
from star import parse_star, write_star
from EMAN2 import EMData, Vec3f, Transform


def main(args):
    star = parse_star(args.input, keep_index=False)

    if args.class_2d is not None:
        refs = glob.glob(args.class_2d)
    elif args.class_3d is not None:
        refs = glob.glob(args.class_3d)
    else:
        refs = []
    
    shifts = []
    for r in refs:
        if args.class_3d is not None:
            refmap = EMData(r)
            com = Vec3f(*refmap.phase_cog()[:3])
            shifts.append(com)
        else:
            stack = EMData.read_images(r)
            for im in stack:
                com = Vec2f(*im.phase_cog()[:2])
                shifts.append(com)

    if args.class_2d is None and args.class_3d is None:
        for ptcl in star.rows:
            im = EMData.read_image(ptcl)
            com = im.phase_cog()
            ptcl["rlnOriginX"] += com[0]
            ptcl["rlnOriginY"] += com[1]
    else:
        for ptcl in star.rows:
            com = shifts[ptcl["rlnClassNumber"]]
            xshift, yshift = transform_com(com, ptcl)
            ptcl["rlnOriginX"] += xshift
            ptcl["rlnOriginY"] += yshift

    if args.zero_origin:
        star["rlnCoordinateX"] = star["rlnCoordinateX"] - star["rlnOriginX"]
        star["rlnCoordinateY"] = star["rlnCoordinateY"] - star["rlnOriginY"]
        star["rlnOriginX"] = 0
        star["rlnOriginY"] = 0

    write_star(args.output, star, reindex=True)

    return 0


def transform_com(com, ptcl):
    t = Transform()
    if len(com) == 3:
        t.set_rotation({'psi': meta.psi, 'phi': meta.phi, 'theta': meta.theta, 'type': 'spider'})
    else:
        t.set_rotation({'psi': meta.psi})
    shift = t.transform(com)
    # The change in the origin is the projection of the transformed difference vector on the new xy plane.
    return shift[0], shift[1]


def find_cm(im):
    l = np.floor(im.shape[0] / 2)
    x, y = np.meshgrid(np.arange(-l, l, dtype=np.double), np.arange(-l, l, dtype=np.double))
    mu_x = np.average(x, axis=None, weights=im)
    mu_y = np.average(y, axis=None, weights=im)
    return mu_x, mu_y


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-2d", help="2D class images for recentering (pass glob in quotes for multiple files)")
    parser.add_argument("--class-3d", help="3D class images for recentering (pass glob in quotes for multiple files)")
    parser.add_argument("--zero-origin", help="Subtract particle origin from particle coordinates in output")
    parser.add_argument("input", help="Input .star file")
    parser.add_argument("output", help="Output .star file")
    sys.exit(main(parser.parse_args()))
