#! /usr/bin/python2.7
# Copyright (C) 2015 Eugene Palovcak, Daniel Asarnow
# University of California, San Francisco
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
import os.path
import logging
from pathos.multiprocessing import Pool
from EMAN2 import EMANVERSION, EMArgumentParser, EMData, Transform, Vec3f
from EMAN2star import StarFile
from pyem.particle import particles, make_proj, MetaData


def main(options):
    """
    Projection subtraction program entry point.
    :param options: Command-line arguments parsed by ArgumentParser.parse_args()
    :return: Exit status
    """
    rchop = lambda x, y: x if not x.endswith(y) or len(y) == 0 else x[:-len(y)]
    options.output = rchop(options.output, ".star")
    options.suffix = rchop(options.suffix, ".mrc")
    options.suffix = rchop(options.suffix, ".mrcs")

    star = StarFile(options.input)
    npart = len(star['rlnImageName'])
    
    sub_dens = EMData(options.submap)

    if options.wholemap is not None:
        dens = EMData(options.wholemap)
    else:
        print "Reference map is required."
        return 1

    # Write star header for output.star.
    top_header = "\ndata_\n\nloop_\n"
    headings = star.keys()
    output_star = open("{0}.star".format(options.output), 'w')

    output_star.write(top_header)
    for i, heading in enumerate(headings):
        output_star.write("_{0} #{1}\n".format(heading, i + 1))

    if options.recenter:  # Compute difference vector between new and old mass centers.
        if options.wholemap is None:
            print "Reference map required for recentering."
            return 1

        new_dens = dens - sub_dens
        # Note the sign of the shift in coordinate frame is opposite the shift in the CoM.
        recenter = Vec3f(*dens.phase_cog()[:3]) - Vec3f(*new_dens.phase_cog()[:3])
    else:
        recenter = None

    pool = None
    if options.nproc > 1:  # Compute subtraction in parallel.
        pool = Pool(processes=options.nproc)
        results = pool.imap(
            lambda x: subtract(x, dens, sub_dens, recenter=recenter, no_frc=options.no_frc,
                               low_cutoff=options.low_cutoff,
                               high_cutoff=options.high_cutoff), particles(star),
            chunksize=min(npart / options.nproc, options.maxchunk))
    else:  # Use serial generator.
        results = (subtract(x, dens, sub_dens, recenter=recenter, no_frc=options.no_frc, low_cutoff=options.low_cutoff,
                            high_cutoff=options.high_cutoff) for x in particles(star))

    # Write subtraction results to .mrcs and .star files.
    i = 0
    nfile = 1
    starpath = None
    mrcs = None
    mrcs_orig = None
    for r in results:
        if i % options.maxpart == 0:
            mrcsuffix = options.suffix + "_%d" % nfile
            nfile += 1
            starpath = "{0}.mrcs".format(
                os.path.sep.join(os.path.relpath(mrcsuffix, options.output).split(os.path.sep)[1:]))
            mrcs = "{0}.mrcs".format(mrcsuffix)
            mrcs_orig = "{0}_original.mrcs".format(mrcsuffix)
            if os.path.exists(mrcs):
                os.remove(mrcs)
            if os.path.exists(mrcs_orig):
                os.remove(mrcs_orig)

        r.ptcl_norm_sub.append_image(mrcs)

        if options.original:
            r.ptcl.append_image(mrcs_orig)

        if logger.getEffectiveLevel() == logging.DEBUG:  # Write additional debug output.
            ptcl_sub_img = r.ptcl.process("math.sub.optimal",
                                          {"ref": r.ctfproj, "actual": r.ctfproj_sub, "return_subim": True})
            ptcl_lowpass = r.ptcl.process("filter.lowpass.gauss", {"apix": 1.22, "cutoff_freq": 0.05})
            ptcl_sub_lowpass = r.ptcl_norm_sub.process("filter.lowpass.gauss", {"apix": 1.22, "cutoff_freq": 0.05})
            ptcl_sub_img.write_image("poreclass_subimg.mrcs", -1)
            ptcl_lowpass.write_image("poreclass_lowpass.mrcs", -1)
            ptcl_sub_lowpass.write_image("poreclass_sublowpass.mrcs", -1)
            r.ctfproj.write_image("poreclass_ctfproj.mrcs", -1)
            r.ctfproj_sub.write_image("poreclass_ctfprojsub.mrcs", -1)

        assert r.meta.i == i  # Assert particle order is preserved.
        star['rlnImageName'][i] = "{0:06d}@{1}".format(i % options.maxpart + 1, starpath)  # Set new image name.
        r.meta.update(star)  # Update StarFile with altered fields.
        line = '  '.join(str(star[key][i]) for key in headings)
        output_star.write("{0}\n".format(line))
        i += 1

    output_star.close()

    if pool is not None:
        pool.close()
        pool.join()

    return 0


def subtract(particle, dens, sub_dens, recenter=None, no_frc=False, low_cutoff=0.0, high_cutoff=0.7071):
    """
    Perform projection subtraction on one particle image.
    :param particle: Tuple holding original (particle EMData, particle MetaData)
    :param dens: Whole density map
    :param sub_dens: Subtraction density map
    :param recenter: Vector between CoM before and after subtraction, or None to skip recenter operation (default None)
    :param no_frc: Skip FRC normalization (default False)
    :param low_cutoff: Low cutoff frequency in FRC normalization (default 0.0)
    :param high_cutoff: High cutoff frequency in FRC normalization (default 0.7071)
    :return: Result object
    """
    ptcl, meta = particle[0], particle[1]
    ctfproj = make_proj(dens, meta)
    ctfproj_sub = make_proj(sub_dens, meta)

    if no_frc:  # Direct subtraction only.
        ptcl_sub = ptcl - ctfproj_sub
    else:  # Per-particle FRC normalization.
        ptcl_sub = ptcl.process("math.sub.optimal", {"ref": ctfproj, "actual": ctfproj_sub,
                                                     "low_cutoff_frequency": low_cutoff,
                                                     "high_cutoff_frequency": high_cutoff})

    ptcl_norm_sub = ptcl_sub.process("normalize")

    if recenter is not None:
        # Rotate the coordinate frame of the CoM difference vector by the Euler angles.
        t = Transform()
        t.set_rotation({'psi': meta.psi, 'phi': meta.phi, 'theta': meta.theta, 'type': 'spider'})
        shift = t.transform(recenter)
        # The change in the origin is the projection of the transformed difference vector on the new xy plane.
        meta.x_origin += shift[0]
        meta.y_origin += shift[1]

    return Result(ptcl, meta, ctfproj, ctfproj_sub, ptcl_sub, ptcl_norm_sub)


class Result:
    """
    Class representing the metadata, intermediate calculation and final result of a subtraction operation.
    """

    def __init__(self, ptcl, meta, ctfproj, ctfproj_sub, ptcl_sub, ptcl_norm_sub):
        """
        Instantiate Result object.
        :param ptcl: Original particle EMData object
        :param meta: Original particle MetaData object
        :param ctfproj: CTF-filtered projection of whole map
        :param ctfproj_sub: CTF-filtered projection of subtraction map
        :param ptcl_sub: Subtracted particle EMData object
        :param ptcl_norm_sub: Normalized, subtracted particle EMData object
        """
        self.ptcl = ptcl
        self.meta = meta
        self.ctfproj = ctfproj
        self.ctfproj_sub = ctfproj_sub
        self.ptcl_sub = ptcl_sub
        self.ptcl_norm_sub = ptcl_norm_sub


def update(self, star):
    """
    Updates StarFile entry to match MetaData instance.
    Beware: ONLY UPDATES FIELDS MODIFIED ELSEWHERE IN THIS PROGRAM!
    :param self: The MetaData object.
    :param star: StarFile object with matching indices
    :return: None
    """
    star['rlnOriginX'][self.i] = self.x_origin
    star['rlnOriginY'][self.i] = self.y_origin
MetaData.update = update  # Monkey patch the MetaData class.


if __name__ == "__main__":
    usage = "projection_subtraction.py [options] output_suffix"
    parser = EMArgumentParser(usage=usage, version="projection_subtraction.py 1.0a, " + EMANVERSION)
    parser.add_argument("--input", type=str, help="RELION .star file listing input particle image stack(s)")
    parser.add_argument("--wholemap", type=str, help="Map used to calculate projections for normalization")
    parser.add_argument("--submap", type=str, help="Map used to calculate subtracted projections")
    parser.add_argument("--output", type=str, help="RELION .star file for listing output particle image stack(s)")
    parser.add_argument("--nproc", type=int, default=1, help="Number of parallel processes")
    parser.add_argument("--maxchunk", type=int, default=1000, help="Maximum task chunk size")
    parser.add_argument("--maxpart", type=int, default=65000, help="Maximum no. of particles per image stack file")
    parser.add_argument("--loglevel", type=str, default="WARNING", help="Logging level and debug output")
    parser.add_argument("--recenter", action="store_true", default=False,
                        help="Shift particle origin to new center of mass")
    parser.add_argument("--original", action="store_true", default=False,
                        help="Also write original (not subtracted) particles to new image stack(s)")
    parser.add_argument("--low-cutoff", type=float, default=0.0, help="Low cutoff frequency in FRC normalization")
    parser.add_argument("--high-cutoff", type=float, default=0.7071, help="High cutoff frequency in FRC normalization")
    parser.add_argument("--no-frc", action="store_true", default=False,
                        help="Perform direct subtraction without FRC normalization")
    # parser.add_argument("--append", action="store_true", default=False, help="Append")
    parser.add_argument("suffix", type=str, help="Relative path and suffix for output image stack(s)")
    (opts, args) = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.getLevelName(opts.loglevel.upper()))

    sys.exit(main(opts))
