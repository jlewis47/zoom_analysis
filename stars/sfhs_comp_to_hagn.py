# from f90nml import read
from curses import color_content
from zoom_analysis.stars import sfhs
from zoom_analysis.zoom_helpers import decentre_coordinates
# from zoom_analysis.stars.star_reader import read_part_ball
from zoom_analysis.halo_maker.read_treebricks import (
    # read_brickfile,
    # convert_brick_units,
    # convert_star_units,
    read_zoom_stars,
)
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import find_zoom_tgt_halo


# from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory

from gremlin.read_sim_params import ramses_sim

# from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# from scipy.spatial import cKDTree

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

# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_evenlesscoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892"
sdirs = [
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_leastcoarse",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_leastcoarse",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704",  # _leastcoarse",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756",  # _leastcoarse",
]
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892"
# sdir =
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_evenlesscoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479"
# print(explr_dirs)

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

for sdir in sdirs:

    name = sdir.split("/")[-1].strip()

    # sim_id = name[2:].split("_")[0]

    # sim_ids = ["id74099", "id147479", "id242704"]

    # c = "tab:blue"

    # get the galaxy in HAGN
    intID = int(name[2:].split("_")[0])

    hagn_tree_hids, hagn_tree_datas, hagn_tree_aexps = read_tree_rev(
        tgt_zed,
        [intID],
        tree_type="halo",
        target_fields=["m", "x", "y", "z", "r"],
        sim="hagn",
    )
    # print("halo id: ", intID)
    print(intID)

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

    print(gal_pties["gid"], hagn_snap)

    try:
        stars = gid_to_stars(
            gal_pties["gid"],
            hagn_snap,
            hagn_sim,
            ["mass", "birth_time", "metallicity"],
        )
    except ValueError:
        pass

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
        label=name.split("_")[0],
        lw=1,
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

    # now do the zoom_halo

    sim = ramses_sim(sdir, nml="cosmo.nml")

    # find last output with assoc files
    assoc_files = os.listdir(os.path.join(sim.path, "association"))
    assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

    snap_tgt = assoc_file_nbs.max()

    # snap_tgt = sim.snap_numbers[-1]
    snap_zed = 1.0 / sim.get_snap_exps(snap_tgt) - 1

    hagn_tree_arg = np.argmin(np.abs(hagn_tree_aexps - 1.0 / (snap_zed + 1.0)))
    hagn_mass_sim_snap = hagn_tree_datas["m"][0][hagn_tree_arg]

    # compare positions
    # replace with function for position interpolation
    hagn_pos = np.asarray(
        [
            hagn_tree_datas["x"][0][hagn_tree_arg],
            hagn_tree_datas["y"][0][hagn_tree_arg],
            hagn_tree_datas["z"][0][hagn_tree_arg],
        ]
    )
    hagn_pos += 0.5 * l_hagn * hagn_tree_aexps[hagn_tree_arg]
    hagn_pos /= l_hagn * hagn_tree_aexps[hagn_tree_arg]

    hagn_pos = decentre_coordinates(hagn_pos, sim.path)

    hagn_rad = (
        hagn_tree_datas["r"][0][hagn_tree_arg] / l_hagn / hagn_tree_aexps[hagn_tree_arg]
    )

    tgt_hid, halo_dict, hosted_gals = find_zoom_tgt_halo(
        sim, snap_zed, tgt_mass=hagn_mass_sim_snap, tgt_ctr=hagn_pos, tgt_rad=hagn_rad
    )

    print(halo_dict["pos"])
    print(hagn_pos)

    gid = int(hosted_gals["gids"][np.argmax(hosted_gals["mass"])])

    stars = read_zoom_stars(sim, snap_tgt, gid)

    sim_sfh, sim_sfr, sim_ssfr, sim_t, cosmo = sfhs.get_sf_stuff(stars, snap_zed, sim)

    sfr_min = min(sim_sfr[sim_sfr > 0].min(), sfr_min)
    sfr_max = max(sim_sfr.max(), sfr_max)
    ssfr_min = min(sim_ssfr[sim_ssfr > 0].min(), ssfr_min)
    ssfr_max = max(sim_ssfr.max(), ssfr_max)
    sfh_min = min(sim_sfh[sim_sfh > 0].min(), sfh_min)
    sfh_max = max(sim_sfh.max(), sfh_max)

    if len(sim_sfh) > 1:
        l = sfhs.plot_sf_stuff(
            ax,
            sim_sfh,
            sim_sfr,
            sim_ssfr,
            sim_t,
            0,
            cosmo,
            # label=sim_id.split("_")[0],
            color=c,
            lw=3,
        )
    else:
        ax[2].scatter(sim_t * 1e-3, sim_ssfr, color=c, marker="x", s=100)
        ax[1].scatter(sim_t * 1e-3, sim_sfr, color=c, marker="x", s=100)
        ax[0].scatter(sim_t * 1e-3, sim_sfh, color=c, marker="x", s=100)


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
# y2.set_xticks(ax[0].get_xticks())
y2.set_xticklabels([f"{cosmo.age(zed).value:.1f}" for zed in ax[0].get_xticks()])
y2.set_xlabel("redshift")

fig.savefig(f"sfhs_compHAGN.png")
