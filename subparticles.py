import os
import os.path
import numpy as np
import pandas as pd
import xml.etree.cElementTree as etree
from math import modf
from pyem.star import parse_star
from pyem.star import write_star
from pyem.star import transform_star
from pyem.star import select_classes
from pyem.util import euler2rot
from pyem.util import rot2euler
from pyem.util import expmap
#from pyem.util import logmap

from EMAN2 import EMANVERSION, EMArgumentParser, EMData, Transform, Vec3f

angles = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
origins = ["rlnOriginX", "rlnOriginY"]
coords = ["rlnCoordinateX", "rlnCoordinateY"]

def recenter_row(row):
        remx, offsetx = modf(row["rlnOriginX"])
        remy, offsety = modf(row["rlnOriginY"])
        offsetx = row["rlnCoordinateX"] - offsetx
        offsety = row["rlnCoordinateY"] - offsety
        return pd.Series({"rlnCoordinateX": offsetx, "rlnCoordinateY": offsety,
                "rlnOriginX": remx, "rlnOriginY": remy})

refine = parse_star("Class3D/3dr1_best2dr1_I4_3.7deg/run_it025_data.star", keep_index=False)
star = select_classes(refine, [4])
apix = 10000.0 * star.iloc[0]['rlnDetectorPixelSize'] / star.iloc[0]['rlnMagnification']
starbk = star.copy()

cmm_dir = "cmm_markers"
cmms = os.listdir(cmm_dir)

rots = [euler2rot(*np.deg2rad(r[1])) for r in star[angles].iterrows()]

shifts = star[origins].copy()

stars = []
for cmm in cmms:
    cmm = cmms[0]
    tree = etree.parse(os.path.join(cmm_dir, cmm))
    cm = np.array([np.double(tree.findall("marker")[1].get(ax)) - 
          np.double(tree.findall("marker")[0].get(ax)) for ax in ['x', 'y', 'z']]) / apix
    cm_ax = cm / np.linalg.norm(cm)
    r2 = euler2rot(*np.array([np.arctan2(cm_ax[1], cm_ax[0]), np.arccos(cm_ax[2]), 0.]))
    newangles = [np.rad2deg(rot2euler(r1.dot(r2.T))) for r1 in rots]
    star[angles] = newangles
    newshifts = shifts + np.array([r1.dot(cm)[:-1] for r1 in rots])
    star[origins] = newshifts
    newcoords = star.apply(recenter_row, axis=1)
    star[coords] = newcoords[coords]
    star[origins] = newcoords[origins]
    stars.append(star.copy())

bigstar = pd.concat(stars)
write_startar("LocalRec/bigstar.star", bigstar)
