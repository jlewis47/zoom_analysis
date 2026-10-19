# from f90nml import read
from curses import color_content
from math import isfinite
from zoom_analysis.stars import sfhs
from zoom_analysis.zoom_helpers import decentre_coordinates
from zoom_analysis.stars.star_reader import yt_read_star_ball, read_part_ball
from zoom_analysis.halo_maker.read_treebricks import (
    read_brickfile,
    convert_brick_units,
    convert_star_units,
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
from hagn.tree_reader import read_tree_rev, interpolate_tree_position
from hagn.catalogues import get_halos_cat, make_super_cat, get_cat_hids, get_cat_gids

# get colour cycler
# from cycler import cycler

# planck cosmo
from astropy.cosmology import Planck18 as cosmo

from hagn.catalogues import make_super_cat, get_cat_hids

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
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id21892_leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id180130_leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id242704",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id242756",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id74099",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_lowerSFE",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099",
]
# sdir =
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892"
# sdir =
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_evenlesscoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479"
# print(explr_dirs)


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

zoom_ls = ["-", "--", ":", "-."]

sfr_max = -np.inf
sfr_min = np.inf

ssfr_max = -np.inf
ssfr_min = np.inf

mstel_max = -np.inf
mstel_min = np.inf

for sdir in sdirs:

    name = sdir.split("/")[-1].strip()

    # sim_id = name[2:].split("_")[0]

    # sim_ids = ["id74099", "id147479", "id242704"]

    # c = "tab:blue"

    # get the galaxy in HAGN
    intID = int(name[2:].split("_")[0])
    # gid = int(make_super_cat(197, outf="/data101/jlewis/hagn/super_cats")["gid"][0])

    hagn_tree_hids, hagn_tree_datas, hagn_tree_aexps = read_tree_rev(
        tgt_zed,
        [intID],
        tree_type="halo",
        # [gid],
        # tree_type="gal",
        target_fields=["m", "x", "y", "z", "r"],
        sim="hagn",
    )

    for key in hagn_tree_datas:
        hagn_tree_datas[key] = hagn_tree_datas[key][0][:]

    hagn_sim.init_cosmo()

    hagn_tree_times = hagn_sim.cosmo_model.age(1.0 / hagn_tree_aexps - 1.0).value * 1e3
    # print("halo id: ", intID)
    # print(intID)

    sim = ramses_sim(sdir, nml="cosmo.nml")

    sim_snaps = sim.snap_numbers
    sim_aexps = sim.get_snap_exps()
    sim.init_cosmo()
    sim_times = sim.cosmo_model.age(1.0 / sim_aexps - 1.0).value * 1e3

    # last sim_aexp
    # valid_steps = hagn_tree_aexps < (1.0 / (tgt_zed + 1.0))  # sim.aexp_end
    # nsteps = np.sum(valid_steps)

    nsteps = len(sim_snaps)
    nstep_tree_hagn = len(hagn_tree_aexps)

    mstel_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
    sfr_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
    time_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)

    mstel_zoom = np.zeros(nsteps, dtype=np.float32)
    sfr_zoom = np.zeros(nsteps, dtype=np.float32)
    time_zoom = np.zeros(nsteps, dtype=np.float32)

    # find last output with assoc files
    assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
    assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

    avail_aexps = sim.get_snap_exps(assoc_file_nbs, param_save=False)

    # hagn tree loop
    for istep, (aexp, time) in enumerate(zip(hagn_tree_aexps, hagn_tree_times)):

        hagn_snap = hagn_snaps[np.argmin(np.abs(hagn_aexps - aexp))]

        try:
            super_cat = make_super_cat(
                hagn_snap, outf="/data101/jlewis/hagn/super_cats"
            )  # , overwrite=True)
        except FileNotFoundError:
            continue

        gal_pties = get_cat_hids(super_cat, [int(hagn_tree_hids[0][istep])])

        # # gal_pties = get_cat_gids(super_cat, [int(hagn_tree_hids[0][istep])])

        if len(gal_pties["gid"]) == 0:
            continue

        # hagn_file = os.path.join(sim.path, f"stellar_history_{hagn_snap}.h5")

        # if not os.path.exists(hagn_file) or overwrite_hagn:

        # stars = gid_to_stars(
        #     gal_pties["gid"][0],
        #     hagn_snap,
        #     hagn_sim,
        #     ["mass", "birth_time", "metallicity"],
        # )

        # ages = stars["age"]
        # Zs = stars["metallicity"]

        # masses = sfhs.correct_mass(hagn_sim, ages, stars["mass"], Zs)

        # mstel_hagn[istep] = stars["mass"].sum()
        # sfr_hagn[istep] = np.sum(masses[ages < 1000] / 1e3)  # Msun/Myr
        # time_hagn[istep] = hagn_tree_times[istep]

        # # print(gal_pties)
        # if len(gal_pties["mgal"]) == 0:
        #     continue

        mstel_hagn[istep] = gal_pties["mgal"][0]
        sfr_hagn[istep] = gal_pties["sfr100"][0] * 1e6
        time_hagn[istep] = time

        # # sim_snap = sim.get_closest_snap(aexp=aexp)

        # print(aexp, avail_aexps)

    # # zoom loop
    for istep, (snap, aexp, time) in enumerate(zip(sim_snaps, sim_aexps, sim_times)):
        if time < hagn_tree_times.min() - 5:
            continue

        if np.all(np.abs(avail_aexps - aexp) > 1e-2):
            print("No assoc file for this aexp")
            continue

        assoc_file = assoc_files[assoc_file_nbs == snap]

        hagn_tree_arg = np.argmin(np.abs(hagn_tree_aexps - aexp))
        hagn_mass_sim_snap = hagn_tree_datas["m"][hagn_tree_arg]

        hagn_ctr, hagn_rvir = interpolate_tree_position(
            time,
            hagn_tree_times,
            hagn_tree_datas,
            l_hagn * aexp,
        )

        if hagn_ctr is None:
            continue

        hagn_ctr = decentre_coordinates(hagn_ctr, sim.path)

        try:
            tgt_hid, halo_dict, hosted_gals = find_zoom_tgt_halo(
                sim,
                snap,
                tgt_mass=hagn_mass_sim_snap,
                tgt_ctr=hagn_ctr,
                tgt_rad=hagn_rvir * 1.5,
            )
        except FileNotFoundError:
            print("No assoc file for this aexp")
            continue

        gid = int(hosted_gals["gids"][np.argmax(hosted_gals["mass"])])

        stars = read_zoom_stars(sim, snap, gid)

        ages = stars["agepart"]
        Zs = stars["Zpart"]

        masses = sfhs.correct_mass(hagn_sim, ages, stars["mpart"], Zs)

        mstel_zoom[istep] = stars["mpart"].sum()
        sfr_zoom[istep] = np.sum(masses[ages < 100] / 1e2)  # Msun/Myr
        time_zoom[istep] = time

    ssfr_hagn = sfr_hagn / mstel_hagn
    ssfr_zoom = sfr_zoom / mstel_zoom

    if np.any(mstel_hagn > 0) and np.any(mstel_zoom > 0):
        mstel_max = np.max(
            [
                mstel_max,
                # mstel_hagn[mstel_hagn > 0].max(),
                mstel_zoom[mstel_zoom > 0].max(),
            ]
        )
        mstel_min = np.min(
            [
                mstel_min,
                # mstel_hagn[mstel_hagn > 0].min(),
                mstel_zoom[mstel_zoom > 0].min(),
            ]
        )
        sfr_max = np.max(
            [
                sfr_max,
                #   sfr_hagn[mstel_hagn > 0].max(),
                sfr_zoom[mstel_zoom > 0].max(),
            ]
        )
        sfr_min = np.min(
            [
                sfr_min,
                #  sfr_hagn[mstel_hagn > 0].min(),
                sfr_zoom[mstel_zoom > 0].min(),
            ]
        )
        ssfr_max = np.max(
            [
                ssfr_max,
                # np.nanmax(ssfr_hagn[mstel_hagn > 0]),
                np.nanmax(ssfr_zoom[mstel_zoom > 0]),
            ]
        )
        ssfr_min = np.min(
            [
                ssfr_min,
                # np.nanmin(ssfr_hagn[mstel_hagn > 0]),
                np.nanmin(ssfr_zoom[mstel_zoom > 0]),
            ]
        )

    l = sfhs.plot_sf_stuff(
        ax,
        mstel_hagn,
        sfr_hagn,
        ssfr_hagn,
        time_hagn,
        0,
        hagn_sim.cosmo_model,
        # label=sim_id.split("_")[0],
        lw=1.5,
    )

    l = sfhs.plot_sf_stuff(
        ax,
        mstel_zoom,
        sfr_zoom,
        ssfr_zoom,
        time_zoom,
        0,
        sim.cosmo_model,
        label=sim.name,
        color=l[0].get_color(),
        lw=3.0,
    )

if np.isfinite(ssfr_min) and np.isfinite(ssfr_max):
    ax[0].set_ylim(ssfr_min * 0.5, ssfr_max * 1.5)
if np.isfinite(sfr_min) and np.isfinite(sfr_max):
    ax[1].set_ylim(sfr_min * 0.5, sfr_max * 1.5)
if np.isfinite(mstel_min) and np.isfinite(mstel_max):
    ax[2].set_ylim(mstel_min * 0.5, mstel_max * 1.5)

# tlim = time_zoom[np.where(sfr_zoom == 0)[-1]]
# ax[0].set_xlim(tlim, time_zoom[-1])
# ax[1].set_xlim(tlim, time_zoom[-1])
# ax[2].set_xlim(tlim, time_zoom[-1])

ax[0].set_xlim(hagn_tree_times.min() / 1e3, hagn_tree_times.max() / 1e3)


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

fig.savefig(f"sfhs_compHAGN_assoc.png")
