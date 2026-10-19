# from matplotlib import markers
# from matplotlib.patches import Patch
from zoom_analysis.stars import sfhs
from zoom_analysis.zoom_helpers import decentre_coordinates, check_in_all_sims_vol
from plot_constraints.plot_PQRO25_SMHMR import zed_to_data

# from zoom_analysis.stars.star_reader import yt_read_star_ball, read_part_ball
# from zoom_analysis.stars.leja_quench_fit import sfr_ridge_leja22
from zoom_analysis.halo_maker.read_treebricks import (
    read_brickfile,
    convert_brick_units,
    convert_star_units,
    read_zoom_stars,
)
# from zoom_analysis.trees.tree_reader import read_tree_file_rev as read_tree_fev_sim
# from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import (
    get_gal_assoc_file,
    # get_halo_assoc_file,
    # get_halo_props_snap,
    # find_zoom_tgt_gal,
    # find_zoom_tgt_halo,
    # get_central_gal_for_hid,
    get_gal_props_snap,
)
import pandas as pd

from matplotlib import colors
from cycler import cycler

# from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

# from scipy.spatial import cKDTree

import os
import numpy as np



sim_dirs=[
"/data101/jlewis/sims/NH_for_rascas",
"/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id36755",
"/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
"/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
"/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130",
"/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756",
"/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288",
"/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
"/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
]



main_stars = True

bin_size = 1.0
zbins = np.arange(2.0, 9.0+bin_size, bin_size)
# zbins=np.array([2.9])
# zbin_err = 0.05
zbin_err = 0.5*bin_size
sims_per_zbins = np.zeros((len(zbins)))
# find_smallest z with 2 sims !
smallest_z = np.inf


mgal_bins = np.logspace(6.5, 11, 15)


n_zbins = len(zbins)
n_plots = n_zbins

nrows = int(np.sqrt(n_plots))
ncols = np.ceil(n_plots / nrows).astype(int)
overlap = nrows * ncols - n_plots
if overlap / ncols>1:
    nrows=nrows - overlap//ncols

row_fact = 0
height_ratios = [1 for i in range(nrows)]
# if overlap == 0:
#     nrows += 1
#     row_fact = 0.6
#     height_ratios.append(row_fact)
overlap = nrows * ncols - n_plots


row_aspect = nrows - row_fact
col_aspect = ncols

figsize = (col_aspect * 6, row_aspect * 6)

legend_args = {
    # "handles": lines,
    # "labels": labels,
    "ncol": 1,
    "loc": "lower right",
    "framealpha": 0.0,
    # "labelcolor": plot_colors,
}

subplot_args = {
    "nrows": nrows,
    "ncols": ncols,
    "sharex": True,
    "sharey": True,
    "figsize": figsize,
    "height_ratios": height_ratios,
    "layout": "constrained",
}

mf_fig, mf_ax = plt.subplots(**subplot_args)
mf_ax = np.ravel(mf_ax)

vols = np.zeros(len(sim_dirs))
for isim,sim_dir in enumerate(sim_dirs):
    sim = ramses_sim(
    sim_dir,
    full_snaps=False,
    tar_snaps=False,
    mini_snaps=True,
    full_check=False,
    )
    if not "NH" in sim_dir and not 'NewHorizon' in sim_dir:
        vol, _ = sim.get_volume()
    else:
        vol = 16**3
    vols[isim]=vol

for iz, z in enumerate(zbins):

    zbin_txt = f"z={z:.1f} $\pm$ {zbin_err:.1f}"

    mf_ax[iz].text(
        0.05,
        0.95,
        zbin_txt,
        transform=mf_ax[iz].transAxes,
        fontsize=10,
        verticalalignment="top",
    )

    mf = np.zeros(len(mgal_bins)-1)

    for isim, sim_dir in enumerate(sim_dirs):

        sim = ramses_sim(
            sim_dir,
            full_snaps=False,
            tar_snaps=False,
            mini_snaps=True,
            full_check=False,
        )

        print(sim.name)

        sim_snaps = sim.snap_numbers
        sim_aexps = sim.get_snap_exps()
        sim.init_cosmo()
        sim_times = sim.cosmo_model.age(1.0 / sim_aexps - 1.0).value * 1e3


        snap = sim.get_closest_snap(zed=z)

        assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
        assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

        if not snap in assoc_file_nbs:
            continue

        aexp = sim.get_snap_exps(snap)
        zed_sim = 1.0 / aexp - 1
        if np.abs(zed_sim - z) > zbin_err:
            print(f"no close enough snaps to zbin {z:.1f}")
            continue

        time = sim.cosmo_model.age(1.0 / aexp - 1.0).value * 1e3

        gal_props = get_gal_props_snap(sim_dir, snap, main_stars=True)

        #there a a few low mass star bursts that are not centrals
        central = gal_props["central"] * gal_props["host purity"] > 0.9999

        mstels = gal_props["mass"][central]


        if isim == 0:
            mf = binned_statistic(mstels,mstels,"count",bins=mgal_bins).statistic
            # mf_ax[iz].plot(mgal_bins[:-1],mf/vols[:isim+1].sum()/np.diff(np.log10(mgal_bins)),color='k',label='NewHorizon')
            # mf_ax[iz].plot(mgal_bins[:-1],mf/np.diff(np.log10(mgal_bins)),color='k',label='NewHorizon')
            mf_ax[iz].plot(mgal_bins[:-1],mf,color='k',label='NewHorizon')
            mf = 0
        else:
            mf += binned_statistic(mstels,mstels,"count",bins=mgal_bins).statistic
        if isim == len(sim_dirs)-1:
            # mf_ax[iz].plot(mgal_bins[:-1],mf/vols[:isim+1].sum()/np.diff(np.log10(mgal_bins)),label='NewHorizon + zooms')
            # mf_ax[iz].plot(mgal_bins[:-1],mf/np.diff(np.log10(mgal_bins)),label='NewHorizon + zooms')
            mf_ax[iz].plot(mgal_bins[:-1],mf,label='Zooms')


for iax,ax in enumerate(mf_ax):
    ax.set_xscale('log')
    ax.set_yscale('log')
    if iax < ((nrows * ncols) - overlap):
        ax.set_box_aspect(1)


    if overlap > 0:
        for ax in mf_ax[-overlap:]:
            ax.axis("off")


mf_ax[-1].legend(framealpha=0.0)

mf_fig.supxlabel('Stellar mass, Msun')
# mf_fig.supylabel('Mass function, dex$^{-1}.cMpc^{-3}$')
# mf_fig.supylabel('Mass function, dex$^{-1}$')
mf_fig.supylabel('Galaxy counts')

mf_fig.savefig('figs/mass_function_zooms.png')