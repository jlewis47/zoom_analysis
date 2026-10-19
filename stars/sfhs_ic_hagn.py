# from f90nml import read
from zoom_analysis.stars import sfhs
from zoom_analysis.stars.star_reader import yt_read_star_ball, read_part_ball
from zoom_analysis.halo_maker.read_treebricks import (
    read_brickfile,
    convert_brick_units,
    convert_star_units,
)
from zoom_analysis.constants import ramses_pc

# from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

import os
import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# from zoom_analysis.sink_reader import *

from hagn.association import gid_to_stars
from hagn.utils import get_hagn_sim, adaptahop_to_code_units
from hagn.tree_reader import read_tree_rev
from hagn.catalogues import get_halos_cat, make_super_cat, get_cat_hids

# get colour cycler
# from cycler import cycler

# planck cosmo
from astropy.cosmology import Planck18 as cosmo

from zoom_analysis.stars.leja_quench_fit import sfr_ridge_leja22

from hagn.catalogues import make_super_cat

from f90_tools.star_reader import read_part_ball_hagn, read_part_ball_NCdust


# sim_ids = [
#     "id2757_995pcnt_mstel/",
#     "id19782_500pcnt_mstel/",pip install setuptools
#     "id13600_005pcnt_mstel/",
# ]
# sdir = "/home/jlewis/sims/proposal_zooms/ExtractZoom/mh1e12/"
hagn_snap = 197  # z=2 in the full res box
HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
delta_t = 100  # Myr

# explr_path = f"/data101/jlewis/sims/dust_fid/lvlmax_20/"
explr_path = f"/home/jlewis/zooming/zoom_starts/dust_fid/lvlmax_20/"
explr_dirs = [
    os.path.join(explr_path, d)
    for d in os.listdir(explr_path)
    if os.path.isdir(os.path.join(explr_path, d))
]

# print(explr_dirs)


def go_down_for_ids(explr_dirs):

    # print(dirs)

    names = []

    for explr_dir in explr_dirs:
        # print("looking in %s" % explr_dir)
        if explr_dir.split("/")[-1].strip().startswith("id"):
            names.append(explr_dir)
        else:
            next_dirs = [
                os.path.join(explr_dir, d)
                for d in os.listdir(explr_dir)
                if os.path.isdir(os.path.join(explr_dir, d))
            ]
            if len(next_dirs) > 0:
                names.extend([d for d in go_down_for_ids(next_dirs)])

    return names
    # return [n.split("/")[-1].strip() for n in names]


sdirs = np.asarray(go_down_for_ids(explr_dirs))
sim_ids = np.asarray([d.split("/")[-1].strip() for d in sdirs])

names = np.asarray([d[2:].split("_")[0] for d in sim_ids])
u_names, args = np.unique(names, return_index=True)
sdirs = sdirs[args]
sim_ids = sim_ids[args]
# print(sim_names)

# sim_ids = ["id74099", "id147479", "id242704"]

super_cat = make_super_cat(
    197, outf="/data101/jlewis/hagn/super_cats"
)  # , overwrite=True)


# setup plot
fig, ax = sfhs.setup_sfh_plot()


yax2 = None

hagn_sim = get_hagn_sim()
hagn_sim.init_cosmo()
l_hagn = hagn_sim.unit_l(hagn_sim.aexp_stt) / (ramses_pc * 1e6) / hagn_sim.aexp_stt
# colors = []

hagn_snaps = hagn_sim.snap_numbers
hagn_aexps = hagn_sim.get_snap_exps(param_save=False)
hagn_times = hagn_sim.cosmo_model.age(1.0 / hagn_aexps - 1.0).value * 1e3

tgt_zed = 1.0 / hagn_aexps[hagn_snaps == hagn_snap] - 1.0
tgt_time = cosmo.age(tgt_zed).value


sfr_min, ssfr_min, sfh_min = np.inf, np.inf, np.inf
sfr_max, ssfr_max, sfh_max = -np.inf, -np.inf, -np.inf

last_hagn_id = -1
isim = 0

zoom_ls = ["--", ":", "-."]

# c = "tab:blue"

for sim_id in sim_ids:

    # get the galaxy in HAGN
    intID = int(sim_id[2:].split("_")[0])

    # print("halo id: ", intID)
    print(sim_id)

    gal_pties = get_cat_hids(super_cat, [intID])

    # hagn_tree_hids, hagn_tree_datas, hagn_tree_aexps = read_tree_rev(
    #     tgt_zed,
    #     gal_pties["hid"],
    #     tree_type="halo",
    #     target_fields=["m", "x", "y", "z", "r"],
    # )
    # hagn_tree_times = (
    #     hagn_sim.cosmo_model.age(1.0 / hagn_tree_aexps - 1.0).value * 1e3
    # )  # Myr#

    # print(last_hagn_id != sim_id)

    if last_hagn_id != sim_id.split("_")[0]:

        zoom_style = 0
        last_hagn_id = sim_id.split("_")[0]

        print(gal_pties["gid"], hagn_snap)

        try:
            stars = gid_to_stars(
                gal_pties["gid"],
                hagn_snap,
                hagn_sim,
                ["mass", "birth_time", "metallicity"],
            )
        except ValueError:
            continue

        print("%.1e" % stars["mass"].sum())

        # print(stars["mass"].min())

        # sfh = np.zeros_like(hagn_tree_times, dtype=float)
        # sfr = np.zeros_like(hagn_tree_times, dtype=float)
        # ssfr = np.zeros_like(hagn_tree_times, dtype=float)

        # follow_hagn_halo(
        #     delta_t,
        #     l_hagn,
        #     sfh,
        #     sfr,
        #     ssfr,
        #     hagn_tree_datas,
        #     hagn_tree_aexps,
        #     hagn_sim,
        #     hagn_snaps,
        #     hagn_times,
        #     hagn_tree_times,
        #     read_part_ball_hagn,
        # )

        sfh, sfr, ssfr, t, cosmo = sfhs.get_sf_stuff(stars, tgt_zed, hagn_sim)

        sfr_min = min(sfr[sfr > 0].min(), sfr_min)
        sfr_max = max(sfr.max(), sfr_max)
        ssfr_min = min(ssfr[ssfr > 0].min(), ssfr_min)
        ssfr_max = max(ssfr.max(), ssfr_max)
        sfh_min = min(sfh[sfh > 0].min(), sfh_min)
        sfh_max = max(sfh.max(), sfh_max)

        # # print(sim_id)
        # # print(gal_pties["masses"], sfh[0])
        # # print(gal_pties["sfrs"], sfr[0])
        # # print(gal_pties["ssfrs"], ssfr[0])

        l = sfhs.plot_sf_stuff(
            ax,
            sfh,
            sfr,
            ssfr,
            t,
            0,
            cosmo,
            label=sim_id.split("_")[0],
            lw=3,
        )

        # l = sfhs.plot_sf_stuff(
        #     ax,
        #     sfh[sfh > 0],
        #     sfr[sfh > 0],
        #     ssfr[sfh > 0],
        #     hagn_tree_times[sfh > 0],
        #     0,
        #     cosmo,
        #     # label=sim_id + " zoom",
        #     # label=sim_id,
        #     # ls=zoom_ls[zoom_style],
        #     # ticks=sim_id == sim_ids[-1],
        #     lw=3,
        #     # ticks=True,
        #     # marker="o",
        # )

        c = l[0].get_color()

        # print(c)

    else:
        zoom_style += 1


ax[0].set_ylim(ssfr_min * 0.5, ssfr_max * 1.5)
ax[1].set_ylim(sfr_min * 0.5, sfr_max * 1.5)
ax[2].set_ylim(sfh_min * 0.5, sfh_max * 1.5)

# add key to existing legend

# ax[-1].plot([], [], "k-", label="HAGN")
# ax[-1].plot([], [], "k--", label="zoom")

ax[-1].legend(framealpha=0.0, ncol=3)
# cur_leg = ax[-1].get_legend()
# ax[-1].legend(
#     cur_leg.legendHandles
#     + [
#         Line2D([0], [0], color="k", linestyle="-"),
#         Line2D([0], [0], color="k", ls="--"),
#     ],
#     sim_ids + ["HAGN", "zoom"],
# )

for a in ax:
    a.tick_params(
        axis="both",
        which="both",
        bottom=True,
        top=True,
        left=True,
        right=True,
        direction="in",
    )

xlim = ax[0].get_xlim()
ax[0].text(xlim[0] + 0.2, 1.1e-5, "quenched", color="k", alpha=0.3)
ylim = ax[0].get_ylim()
ax[0].fill_between(xlim, 1e-5, ylim[0], color="k", alpha=0.3)
ax[0].set_ylim(ylim)
for a in ax:
    ax[0].set_xlim(xlim)

y2 = ax[0].twiny()
y2.set_xlim(xlim)
zlabels = [f"{cosmo.age(zed).value:.1f}" for zed in ax[0].get_xticks()]
# y2.set_xticks(ax[0].get_xticks())
y2.set_xticklabels(zlabels)
y2.set_xlabel("redshift")

# plot leja+22 limits
zspan = np.arange(np.min(np.float32(zlabels)), np.max(np.float32(zlabels)), 0.05)

leja_masses = np.asarray([1e10, 1e11, 1e12])
ssfr_leja_z2_lim = [
    (
        sfr_ridge_leja22(z, leja_masses) / (leja_masses) * 1e6
        if 0.3 < z <= 2.7
        else np.full(len(leja_masses), np.nan)
    )
    for z in zspan
]

zspan_time = cosmo.age(zspan).value

for m, ssfr in zip(leja_masses, np.transpose(ssfr_leja_z2_lim)):
    # ax[0].axhline(ssfr, color="k", ls="--", alpha=0.3)
    finite = np.isfinite(ssfr)
    if finite.sum() < 2:
        continue
    ax[0].plot(zspan_time[finite], ssfr[finite], "k--", alpha=0.33)

    ax[0].text(
        zspan_time[finite][0],
        ssfr[finite][0],
        f"quenched Leja+22, M={m:.1e}",
        color="k",
        alpha=0.33,
        ha="right",
    )


fig.savefig("sfhs_ics_comp.png")
