#! /usr/bin/python2.7
# Copyright (C) 2015 Eugene Palovcak, Daniel Asarnow
# University of California, San Francisco

# Projection subtraction program
# Given a (presumably partial) map, a particle stack, and
# the RELION star file that generated the map, for each particle, this program:
#        (1) Computes a projection from the map given the RELION angles
#        (2) CTF corrects the image given the RELION star file
#        (3) Appropriately standardizes the projection image (how?)
#        (4) Subtracts the projected image from the CTF corrected image
#        (5) Saves the projection-subtracted image
#  2015-06-23 --particlestack options is replaced by finding the particle files
#             from various .mrcs stacks. Only requires the .star file now.
#  V3: Output generates subtracted stack and equivalent unsubtracted stack
#  2016-07-12 Supports parallelism, requires pathos library, writes multiple stacks
#             to avoid 16-bit overflow in EMAN2.
#  2016-07-13 Changed argument conventions and help text, fixed relative paths in output.
import sys
import os.path
import logging
from pathos.multiprocessing import Pool
from EMAN2 import EMANVERSION, EMArgumentParser, EMData, Transform, Vec3f
from EMAN2star import StarFile
from sparx import generate_ctf, filt_ctf


def main(options):
    rchop = lambda x, y: x if not x.endswith(y) or len(y) == 0 else x[:-len(y)]
    options.output = rchop(options.output, ".star")
    options.suffix = rchop(options.suffix, ".mrc")
    options.suffix = rchop(options.suffix, ".mrcs")

    star = StarFile(options.input)
    npart = len(star['rlnImageName'])

    dens = EMData(options.wholemap)
    sub_dens = EMData(options.submap)

    # Write star header for output.star.
    top_header = "\ndata_\n\nloop_\n"
    headings = star.keys()
    output_star = open("{0}.star".format(options.output), 'w')

    output_star.write(top_header)
    for i, heading in enumerate(headings):
        output_star.write("_{0} #{1}\n".format(heading, i + 1))

    if options.recenter:  # Compute difference vector between new and old mass centers.
        #TODO dens - sub_dens
        recenter = Vec3f(*sub_dens.phase_cog()[:3]) - Vec3f(*dens.phase_cog()[:3])
    else:
        recenter = None

    # Compute subtraction in parallel or using serial generator.
    pool = None
    if options.nproc > 1:
        pool = Pool(processes=options.nproc)
        results = pool.imap(lambda x: subtract(x, dens, sub_dens, recenter), particles(star),
                            chunksize=min(npart / options.nproc, 1000))
    else:
        results = (subtract(x, dens, sub_dens) for x in particles(star))

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

        # Output for testing.
        if logger.getEffectiveLevel() == logging.DEBUG:
            ptcl_sub_img = r.ptcl.process("math.sub.optimal",
                                          {"ref": r.ctfproj, "actual": r.ctfproj_sub, "return_subim": True})
            ptcl_lowpass = r.ptcl.process("filter.lowpass.gauss", {"apix": 1.22, "cutoff_freq": 0.05})
            ptcl_sub_lowpass = r.ptcl_norm_sub.process("filter.lowpass.gauss", {"apix": 1.22, "cutoff_freq": 0.05})
            ptcl_sub_img.write_image("poreclass_subimg.mrcs", -1)
            ptcl_lowpass.write_image("poreclass_lowpass.mrcs", -1)
            ptcl_sub_lowpass.write_image("poreclass_sublowpass.mrcs", -1)
            r.ctfproj.write_image("poreclass_ctfproj.mrcs", -1)
            r.ctfproj_sub.write_image("poreclass_ctfprojsub.mrcs", -1)
        # Change image name and write output.star
        assert r.meta.i == i
        star['rlnImageName'][i] = "{0:06d}@{1}".format(i % options.maxpart + 1, starpath)
        line = '  '.join(str(star[key][i]) for key in headings)
        output_star.write("{0}\n".format(line))
        i += 1

    output_star.close()

    if pool is not None:
        pool.close()
        pool.join()

    return 0


def particles(star):
    npart = len(star['rlnImageName'])
    for i in range(npart):
        ptcl_n = int(star['rlnImageName'][i].split("@")[0]) - 1
        ptcl_name = star['rlnImageName'][i].split("@")[1]
        ptcl = EMData(ptcl_name, ptcl_n)
        meta = MetaData(star, i)
        yield ptcl, meta


def subtract(particle, dens, sub_dens, recenter=None):
    ptcl, meta = particle[0], particle[1]
    ctfproj = make_proj(dens, meta)
    ctfproj_sub = make_proj(sub_dens, meta)
    ptcl_sub = ptcl.process("math.sub.optimal", {"ref": ctfproj, "actual": ctfproj_sub})
    ptcl_norm_sub = ptcl_sub.process("normalize")
    if recenter is not None:
        t = Transform()
        t.set_rotation({'psi': meta.psi, 'phi': meta.phi, 'theta': meta.theta, 'type': 'spider'})
        shift = t.transform(recenter)
        meta.x_origin += shift[0]
        meta.y_origin += shift[1]
    return Result(ptcl, meta, ctfproj, ctfproj_sub, ptcl_sub, ptcl_norm_sub)


def make_proj(dens, meta):
    t = Transform()
    t.set_rotation({'psi': meta.psi, 'phi': meta.phi, 'theta': meta.theta, 'type': 'spider'})
    t.set_trans(-meta.x_origin, -meta.y_origin)
    proj = dens.project("standard", t)
    ctf = generate_ctf(meta.ctf_params)
    ctf_proj = filt_ctf(proj, ctf)
    return ctf_proj


class Result:
    def __init__(self, ptcl, meta, ctfproj, ctfproj_sub, ptcl_sub, ptcl_norm_sub):
        self.ptcl = ptcl
        self.meta = meta
        self.ctfproj = ctfproj
        self.ctfproj_sub = ctfproj_sub
        self.ptcl_sub = ptcl_sub
        self.ptcl_norm_sub = ptcl_norm_sub


class MetaData:
    def __init__(self, star, i):
        self.i = i
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


if __name__ == "__main__":
    usage = "Not written yet"
    parser = EMArgumentParser(usage=usage, version=EMANVERSION)
    parser.add_argument("--input", type=str, help="RELION .star file listing input particle image stack(s)")
    parser.add_argument("--wholemap", type=str, help="Map used to calculate projections for normalization")
    parser.add_argument("--submap", type=str, help="Map used to calculate subtracted projections")
    parser.add_argument("--output", type=str, help="RELION .star file for listing output particle image stack(s)")
    parser.add_argument("--nproc", type=int, default=1, help="Number of parallel processes")
    parser.add_argument("--maxpart", type=int, default=65000, help="Maximum no. of particles per image stack file")
    parser.add_argument("--loglevel", type=str, default="WARNING", help="Logging level and debug output")
    parser.add_argument("--recenter", action="store_true", default=False,
                        help="Shift particle origin to new center of mass")
    parser.add_argument("--original", action="store_true", default=False,
                        help="Also write original (not subtracted) particles to new image stack(s)")
    # parser.add_argument("--append", action="store_true", default=False, help="Append")
    parser.add_argument("suffix", type=str, help="Relative path and suffix for output image stack(s)")
    (opts, args) = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.getLevelName(opts.loglevel.upper()))

    sys.exit(main(opts))
