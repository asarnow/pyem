#! /usr/bin/python2.7
# Eugene Palovcak - 23 June 2015 - UCSF
# V2 - Eugene Palovcak - 10 September 2015 - UCSF

# Projection subtraction program
# Given a (presumably partial) map, a particle stack, and
# the RELION star file that generated the map, for each particle, this program:
#        (1) Computes a projection from the map given the RELION angles
#        (2) CTF corrects the image given the RELION star file
#        (3) Appropriately standardizes the projection image (how?)
#        (4) Subtracts the projected image from the CTF corrected image
#        (5) Saves the projection-subtracted image
#  V2: --particlestack options is replaced by finding the particle files
#      from various .mrcs stacks. Only requires the .star file now.  
#  V3: Output generates subtracted stack and equivalent unsubtracted stack

import os
import sys
from EMAN2 import EMANVERSION, EMArgumentParser, EMData, Transform
from EMAN2star import StarFile
from sparx import generate_ctf, filt_ctf


def main():
    progname = os.path.basename(sys.argv[0])
    usage = "Not written yet"

    # Initialize EMAN2 Argument Parser
    parser = EMArgumentParser(usage=usage, version=EMANVERSION)
    parser.add_argument("--particlestar", type=str, help="RELION .star file for particle stack")
    parser.add_argument("--wholemap", type=str, help="Map used to calculate projections for normalization")
    parser.add_argument("--submap", type=str, help="Map used to calculate subtracted projections")
    parser.add_argument("--output", type=str, help="Name of output stack/star")
    (options, args) = parser.parse_args()

    star = StarFile(options.particlestar)
    npart = len(star['rlnImageName'])

    dens = EMData(options.wholemap)
    sub_dens = EMData(options.submap)

    # Write star header for output.star...
    top_header = "\ndata_\n\nloop_\n"
    headings = star.keys()
    output_star = open("{0}.star".format(options.output), 'w')

    output_star.write(top_header)
    for i, heading in enumerate(headings):
        output_star.write("_{0} #{1}\n".format(heading, i + 1))

    # Calculate projection subtraction and append output.star
    for N in range(npart):
        if N % 100 == 0:
            print "Projection-subtracted {0} of {1} particles...".format(N, npart)
        ptcl_n = int(star['rlnImageName'][N].split("@")[0]) - 1
        ptcl_name = star['rlnImageName'][N].split("@")[1]
        ptcl = EMData(ptcl_name, ptcl_n)
        meta = MetaData(star, N)
        ptcl_sub, ptcl_norm_sub = subtract(ptcl, meta, dens, sub_dens)
        ptcl_norm_sub.write_image("{0}.mrcs".format(options.output), -1)
        ptcl.write_image("{0}_original.mrcs".format(options.output), -1)
        # Output for testing
        #      ptcl_sub_img = ptcl.process("math.sub.optimal", {"ref":ctfproj,
        #                             "actual":ctfproj_sub, "return_subim":True})
        #      ptcl_lowpass = ptcl.process("filter.lowpass.gauss", {"apix":1.22, "cutoff_freq":0.05})
        #      ptcl_sub_lowpass = ptcl_norm_sub.process("filter.lowpass.gauss", {"apix":1.22, "cutoff_freq":0.05})
        #      ptcl_sub_img = ptcl_sub_img.write_image("poreclass_subimg.mrcs", -1)
        #      ptcl_lowpass.write_image("poreclass_lowpass.mrcs", -1)
        #      ptcl_sub_lowpass.write_image("poreclass_sublowpass.mrcs", -1)
        #      ctfproj.write_image("poreclass_ctfproj.mrcs", -1)
        #      ctfproj_sub.write_image("poreclass_ctfprojsub.mrcs", -1)
        # Change image name and write output.star
        star['rlnImageName'][N] = "{0:06d}@{1}".format(N+1, "{0}.mrcs".format(options.output))
        line = '  '.join(str(star[key][N]) for key in headings)
        output_star.write("{0}\n".format(line))


def subtract(ptcl, meta, dens, sub_dens):
    ctfproj = make_proj(dens, meta)
    ctfproj_sub = make_proj(sub_dens, meta)
    ptcl_sub = ptcl.process("math.sub.optimal", {"ref": ctfproj, "actual": ctfproj_sub})
    ptcl_norm_sub = ptcl_sub.process("normalize")
    return ptcl_sub, ptcl_norm_sub


def make_proj(dens, meta):
    t = Transform()
    t.set_rotation({'psi': meta.psi, 'phi': meta.phi, 'theta': meta.theta, 'type': 'spider'})
    t.set_trans(-meta.x_origin, -meta.y_origin)
    proj = dens.project("standard", t)
    ctf = generate_ctf(meta.ctf_params)
    ctf_proj = filt_ctf(proj, ctf)
    return ctf_proj


class MetaData:
    def __init__(self, star, N):
        self.phi = star['rlnAngleRot'][N]
        self.psi = star['rlnAnglePsi'][N]
        self.theta = star['rlnAngleTilt'][N]
        self.x_origin = star['rlnOriginX'][N]
        self.y_origin = star['rlnOriginY'][N]
        # CTFFIND4 --> sparx CTF conventions (from CTER paper)
        self.defocus = (star['rlnDefocusU'][N] + star['rlnDefocusV'][N]) / 20000.0
        self.dfdiff = (star['rlnDefocusU'][N] - star['rlnDefocusV'][N]) / 10000.0
        self.dfang = 90.0 - star['rlnDefocusAngle'][N]
        self.apix = ((10000.0 * star['rlnDetectorPixelSize'][N]) /
                     float(star['rlnMagnification'][N]))
        self.voltage = star["rlnVoltage"][N]
        self.cs = star["rlnSphericalAberration"][N]
        self.ac = star["rlnAmplitudeContrast"][N] * 100.0
        self.bfactor = 0
        self.ctf_params = [self.defocus, self.cs, self.voltage, self.apix, self.bfactor, self.ac, self.dfdiff,
                           self.dfang]


if __name__ == "__main__":
    main()
