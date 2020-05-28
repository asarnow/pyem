# Copyright (C) 2017 Daniel Asarnow
# University of California, San Francisco
#
# Library functions for interesting plots.
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
import matplotlib.pyplot as plt
# import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()
infinity = u'\u221e'
angstrom = u'\xc5'


def plot_fsc_curves(fsc, lgdtext=None, title=None, fname=None):
    # if fname is not None:
    #     mpl.rc("savefig", dpi=300)

    if type(fsc) is not list:
        fsc = [fsc]

    if type(fsc[0]) is str:
        fns = fsc
        fsc = []
        for fn in fns:
            fsc.append(pd.read_table(fn, header=None))
        for f in fsc:
            f.columns = ["freq", "fsc"]
            f["res"] = 1 / f["freq"]

    if lgdtext is None:
        lgdtext = ["Curve %d" % (i + 1) for i in range(len(fsc))]

    if title is None:
        title = "FSC Plot"

    sns.set(font_scale=3)
    fg, ax = plt.subplots(figsize=(20, 10))
    for f in fsc:
        f.plot(x="freq", y="fsc", ax=ax, legend=None, linewidth=4.)
    hl = ax.axhline(y=0.143, color='k', linestyle='-', linewidth=2.)
    hl.set_zorder(1)
    ax.set_xlim((np.min(fsc[0]["freq"]), np.max(fsc[0]["freq"])))
    ax.set_xlabel("Resolution (%s)" % angstrom)
    # ax.set_xticks([0., 1/20., 1/10., 1/6.7, 1/5., 1/4., 1/3.5, 1/3., 1/2.5])
    ax.set_xticks(np.arange(np.min(fsc[0]["freq"]), np.max(fsc[0]["freq"]), 0.05))
    # ax.set_xticklabels([infinity] + [("%.1f" % (1/f)).rstrip('0').rstrip('.') if (1/f) > 7 else "%.1f" % (1/f)
    #                                  for f in [1/20., 1/10., 1/6.7, 1/5., 1/4., 1/3.5, 1/3., 1/2.5]])
    ax.set_xticklabels([infinity] + [("%.1f" % f).rstrip('0').rstrip('.') if f > 7 else "%.1f" % f for f in
                                     1 / np.arange(0.05, np.max(fsc[0]["freq"]), 0.05)])
    ax.set_ylabel("Fourier Correlation")
    ax.set_ylim((np.min([f["fsc"] for f in fsc]), 1.02))
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.text(np.max(fsc[0]["freq"]), 0.155, "0.143", fontsize=36, horizontalalignment="right")
    ax.set_title(title)
    ax.legend(lgdtext, fontsize=24)
    if fname is not None:
        fg.savefig(fname, dpi=300)
        # mpl.rc("savefig", dpi=80)
    return fg, ax


def plot_angle_comparison(df1, df2, lgdtext=None, fname=None, maxrot=90):
    # if fname is not None:
    #     mpl.rc("savefig", dpi=300)

    if lgdtext is None:
        lgdtext = [u"Second (deg)", u"First (deg)"]

    sns.set(font_scale=3)
    f, ax = plt.subplots(1, 3, figsize=(30, 10))
    sns.regplot(df2["rlnAngleRot"], df1["rlnAngleRot"], fit_reg=False, scatter_kws={"s": 16}, ax=ax[0])
    ax[0].set_xlim((-maxrot, maxrot))
    ax[0].set_ylim((-maxrot, maxrot))
    ax[0].set_xticks(np.arange(-maxrot, maxrot+1, 15))
    ax[0].set_yticks(np.arange(-maxrot, maxrot+1, 15))
    ax[0].xaxis.label.set_visible(False)
    ax[0].set_ylabel(lgdtext[0])
    ax[0].set_title(u"$\phi$ ( $Z$ )", y=1.01)

    sns.regplot(df2["rlnAngleTilt"], df1["rlnAngleTilt"], fit_reg=False, scatter_kws={"s": 16}, ax=ax[1])
    ax[1].set_xlim((0, 180))
    ax[1].set_ylim((0, 180))
    ax[1].set_xticks(np.arange(0, 181, 30))
    ax[1].set_yticks(np.arange(0, 181, 30))
    ax[1].xaxis.label.set_visible(False)
    ax[1].yaxis.label.set_visible(False)
    ax[1].set_title(u"$\\theta$ ( $Y'$ )", y=1.01)

    sns.regplot(df2["rlnAnglePsi"], df1["rlnAnglePsi"], fit_reg=False, scatter_kws={"s": 16}, ax=ax[2])
    ax[2].set_xlim((-180, 180))
    ax[2].set_ylim((-180, 180))
    ax[2].set_xticks(np.arange(-180, 181, 45))
    ax[2].set_yticks(np.arange(-180, 181, 45))
    ax[2].xaxis.label.set_visible(False)
    ax[2].yaxis.label.set_visible(False)
    ax[2].set_title(u"$\psi$ ( $Z''$  )", y=1.01)
    f.text(0.5, -0.05, lgdtext[1], ha='center', fontsize=36)
    f.tight_layout(pad=1., w_pad=-1.5, h_pad=0.5)
    if fname is not None:
        f.savefig(fname, dpi=300)
        # mpl.rc("savefig", dpi=80)
    return f, ax
