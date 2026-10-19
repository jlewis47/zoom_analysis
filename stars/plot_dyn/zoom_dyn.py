# from f90nml import read
from turtle import title
import h5py

from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import get_gal_assoc_file
from zoom_analysis.halo_maker.read_treebricks import get_gal_stars
from zoom_analysis.stars.dynamics import extract_nh_kinematics

from scipy.stats import binned_statistic

# from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

import os
import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# from zoom_analysis.sink_reader import *

# from hagn.association import gid_to_stars
# from hagn.utils import get_hagn_sim, adaptahop_to_code_units
# from hagn.tree_reader import read_tree_rev
# from hagn.catalogues import get_halos_cat, make_super_cat, get_cat_hids

# get colour cycler
# from cycler import cycler

# planck cosmo
from astropy.cosmology import Planck18 as cosmo

# from astropy.cosmology import z_at_value
from astropy import units as u

# from hagn.catalogues import make_super_cat

# fig, ax = sink_histories.setup_bh_plot()

tgt_zed = 3.3
# tgt_zed = 2.3
# tgt_zed = 3
# tgt_zed = 4.5
# tgt_zed = 5

stellar_bins = np.logspace(6, 12, 15)

fpure = 1.0 - 1e-4

HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
delta_t = 100  # Myr

sim_dirs = [
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE_highAGNtherm/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highSFE_lowSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE_lowNsink/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_pdrag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_noAGN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_high_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nrg_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_pdrag",
]

# list of all available pyplot markers:
# https://matplotlib.org/stable/api/markers_api.html
markers = [
    "o",
    "v",
    "^",
    "<",
    ">",
    "s",
    "p",
    "P",
    "*",
    "h",
    "H",
    "X",
    "D",
    "d",
    "|",
    "_",
]

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

for isim, sim_dir in enumerate(sim_dirs):

    sim = ramses_sim(sim_dir, nml="cosmo.nml")

    name = sim.name

    intID = int(sim.name.split("_")[0][2:])

    # hagn_sim = get_hagn_sim()
    # hagn_snap = hagn_sim.get_closest_snap(zed=tgt_zed)

    # # super_cat = make_super_cat(
    # #     hagn_snap, outf="/data101/jlewis/hagn/super_cats"
    # # )  # , overwrite=True)

    # # gal_pties = get_cat_hids(super_cat, [intID])

    # hagn_zed = 1.0 / hagn_sim.aexps[hagn_sim.snap_numbers == hagn_snap][0] - 1

    # mbh, dmbh, t_sink, fmerger, t_merge = sink_histories.get_bh_stuff(
    #     hagn_sim, intID, hagn_zed
    # )

    # l = sink_histories.plot_bh_stuff(
    #     ax,
    #     mbh,
    #     dmbh,
    #     t_sink,
    #     fmerger,
    #     t_merge,
    #     0,
    #     cosmo,
    #     label=name,
    #     lw=1,
    # )

    # c = l[0].get_color()
    # find last assoc_file
    found = False
    decal = -1

    while not found:
        sim_snap = sim.snap_numbers[decal]
        sim_zed = 1.0 / sim.aexps[decal] - 1

        gfile = get_gal_assoc_file(sim_dir, sim_snap)

        found_cond = os.path.isfile(gfile)
        if tgt_zed != None:
            found_cond = found_cond and np.abs(sim_zed - tgt_zed) < 0.1

        if found_cond:
            found = True
        else:
            decal -= 1

    with h5py.File(gfile, "r") as src:
        gids = src["gids"][()]

        gal_dict = {}
        for k in src.keys():
            prop = src[k]
            if len(prop.shape) > 1 and prop.shape[0] == 3:
                gal_dict[k] = prop[:, :]
            else:
                gal_dict[k] = prop[:]

    # print(gal_dict.keys())

    host_purity = gal_dict["host purity"]
    central_cond = gal_dict["central"] == 1
    pure_cond = (host_purity > fpure) * central_cond
    gal_mass = gal_dict["mass"][pure_cond]
    gal_poss = gal_dict["pos"][:, pure_cond]
    r50s = 1 * gal_dict["r50"][pure_cond]
    # rmax = gal_dict["rmax"][pure_cond]
    # halo_mass = gal_dict["host mass"][pure_cond]

    print(f"I found {pure_cond.sum():d} galaxies with purity > {fpure:f},")
    print(f"and {len(host_purity)-pure_cond.sum()} galaxies with purity < {fpure:f}")

    label = sim.name
    if len(sim_dirs) > 1:
        label += f", z={sim_zed:.2f}"

    vrots = []
    sigmas = []
    # gal_mass = []

    for gid, gal_pos, r50 in zip(gids[pure_cond], gal_poss.T, r50s):

        stars = get_gal_stars(sim_snap, gid, sim)

        # only get stars within r50 of gal_pos
        dists = np.linalg.norm(stars["pos"].T - gal_pos[:, np.newaxis], axis=0)
        dist_lim = dists < r50

        vrot, sigma = extract_nh_kinematics(
            stars["mpart"][dist_lim],
            stars["pos"][dist_lim],
            stars["vel"][dist_lim],
            gal_pos,
        )

        vrots.append(np.abs(vrot))
        sigmas.append(sigma)
        # gal_mass.append(stars["mpart"][dist_lim].sum())

    stab = np.asarray(vrots) / np.asarray(sigmas)

    points = ax.scatter(gal_mass, stab, label=label, alpha=0.5, marker=markers[isim])

    # get avg values in bins
    bin_means, bin_edges, binnumber = binned_statistic(
        # gal_mass, stab, bins=stellar_bins, statistic="mean"
        gal_mass,
        stab,
        bins=stellar_bins,
        # statistic=np.nanmedian,
        statistic=np.nanmean,
    )
    # print(bin_means)
    bin_width = bin_edges[1:] - bin_edges[:-1]
    bin_centers = bin_edges[:-1] + bin_width * 0.5

    # plot
    # ax.errorbar(bin_centers, bin_means, xerr=bin_width/2, color=points.get_facecolor(), ls="none",alpha=1.0)
    ax.plot(bin_centers, bin_means, color=points.get_facecolor(), ls="-", alpha=1.0)


ax.tick_params(
    axis="both",
    which="both",
    bottom=True,
    # top=True,
    top=False,
    left=True,
    right=True,
    direction="in",
)

ax.grid()

ax.set_xlabel("Stellar mass [M$_\odot$]")
ax.set_ylabel("Vrot/sigma")

# ax.set_yscale("log")
ax.set_xscale("log")
title_txt = ""
if len(sim_dirs) == 1:
    title_txt += f"z={sim_zed:.2f}"
    # ax.text(0.05, 0.95, f"z={sim_zed:.2f}", transform=ax.transAxes)
# also print purity threshold
# ax.text(0.05, 0.9, f"fpure > {fpure:.3f}", transform=ax.transAxes)
title_txt += f" fpure > {fpure:.3f}"

# y2 = ax.twiny()
# # y2.set_xlim(ax.get_xlim())
# # y2.set_xticks(ax.get_xticks())
# # xlim = ax.get_xlim()
# y2.set_xticklabels(
#     [
#         "%.1f" % z_at_value(sim.cosmo_model.age, time_label * u.Gyr, zmax=np.inf)
#         for time_label in ax.get_xticks()
#     ]
# )
# y2.set_xlabel("redshift")

# outdir = "./bhsmr_plots"
# if not os.path.exists(outdir):
# os.makedirs(outdir)

ax.legend(framealpha=0.0, title=title_txt)
if tgt_zed != None:
    fig.savefig(f"zoom_dyn_z{tgt_zed:.2f}.png")
else:
    fig.savefig("zoom_dyn.png")
