# from f90nml import read
from zoom_analysis.stars import sfhs
from zoom_analysis.stars.star_reader import yt_read_star_ball, read_part_ball

# from zoom_analysis.halo_maker.read_treebricks import (
#     read_brickfile,
#     convert_brick_units,
#     convert_star_units,
# )

from zoom_analysis.constants import ramses_pc

# from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory

from gremlin.read_sim_params import ramses_sim

# from scipy.spatial import cKDTree

import os
import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# from zoom_analysis.sink_reader import *

from hagn.association import gid_to_stars
from hagn.utils import get_hagn_sim  # , adaptahop_to_code_units
from hagn.tree_reader import read_tree_rev
from hagn.catalogues import get_halos_cat, make_super_cat, get_cat_hids

# get colour cycler
from cycler import cycler

# planck cosmo
from astropy.cosmology import Planck18 as cosmo

from hagn.catalogues import make_super_cat


# sim_ids = [
#     "id2757_995pcnt_mstel/",
#     "id19782_500pcnt_mstel/",pip install setuptools
#     "id13600_005pcnt_mstel/",
# ]
# sdir = "/home/jlewis/sims/proposal_zooms/ExtractZoom/mh1e12/"
hagn_snap = 197  # z=2 in the full res box
HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
# sim_ids = ["id147479", "id147479_low_nstar", "id242704"]
# sim_ids = ["id242704"]
sim_ids = ["id242704", "id147479"]  # , "id147479_high_nsink"]
# sim_ids = ["id74099", "id147479", "id242704"]

super_cat = make_super_cat(197, "/data101/jlewis/hagn/super_cats")  # , overwrite=True)


sdirs = [
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12",
]

tgt_zed = 2
tgt_time = cosmo.age(tgt_zed).value

# setup plot
fig, ax = sfhs.setup_sfh_plot()

# setup default colour cycler
ccclyer = cycler(
    "color",
    [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ],
)

yax2 = None

hagn_sim = get_hagn_sim()
l_hagn = hagn_sim.cosmo["unit_l"] / (ramses_pc * 1e6) / hagn_sim.aexp_stt
# colors = []

sfr_min, ssfr_min, sfh_min = np.inf, np.inf, np.inf
sfr_max, ssfr_max, sfh_max = -np.inf, -np.inf, -np.inf

for sim_id, c in zip(sim_ids, ccclyer):

    # get the galaxy in HAGN
    intID = int(sim_id[2:].split("_")[0])

    print("halo id: ", intID)

    gal_pties = get_cat_hids(super_cat, [intID])

    stars = gid_to_stars(
        gal_pties["gid"], hagn_snap, hagn_sim, ["mass", "birth_time", "metallicity"]
    )

    # print(stars["mass"].min())

    sfh, sfr, ssfr, t, cosmo = sfhs.get_sf_stuff(stars, tgt_zed, hagn_sim)

    sfr_min = min(sfr.min(), sfr_min)
    sfr_max = max(sfr.max(), sfr_max)
    ssfr_min = min(ssfr.min(), ssfr_min)
    ssfr_max = max(ssfr.max(), ssfr_max)
    sfh_min = min(sfh.min(), sfh_min)
    sfh_max = max(sfh.max(), sfh_max)

    # print(sim_id)
    # print(gal_pties["masses"], sfh[0])
    # print(gal_pties["sfrs"], sfr[0])
    # print(gal_pties["ssfrs"], ssfr[0])

    sfhs.plot_sf_stuff(
        ax,
        sfh,
        sfr,
        ssfr,
        t,
        0,
        cosmo,
        color=c["color"],
        label=sim_id,
        # label="id74099",
        lw=3,
    )

    hagn_tree_hids, hagn_tree_datas, hagn_tree_aexps = read_tree_rev(
        tgt_zed,
        gal_pties["hid"],
        tree_type="halo",
        target_fields=["m", "x", "y", "z", "r"],
    )

    sim = ramses_sim(os.path.join(sdirs[0], sim_id))

    sim_snaps = sim.snap_numbers
    sim_aexps = sim.get_snap_exps(param_save=False)

    # filter out tree steps that have no correponding outputs
    # prec = 1e-3
    # out_filter = np.asarray(
    #     [
    #         np.any(np.abs(sim_aexps - tree_aexp) / tree_aexp < prec)
    #         for tree_aexp in hagn_tree_aexps
    #     ]
    # )
    # hagn_tree_hids = hagn_tree_hids[0][out_filter]
    # hagn_tree_masses = hagn_tree_datas["m"][0][out_filter]
    # hagn_tree_aexps = hagn_tree_aexps[out_filter]

    # # close_tree_aexps = [
    # #     np.argmin(np.abs(sim_aexp - hagn_tree_aexps)) for sim_aexp in sim_aexps
    # # ]

    # # close_tree_hids = hagn_tree_hids[close_tree_aexps]
    # # close_tree_snaps = sim_snaps[close_tree_aexps]

    aexp = 1.0 / (tgt_zed + 1.0)

    cur_snap = sim_snaps[-1]
    cur_aexp = sim_aexps[-1]

    print("current snap: ", cur_snap)
    print("current aexp: ", cur_aexp)

    # # get hagn main branch galaxy id at this time
    tree_arg = np.argmin(np.abs(hagn_tree_aexps - cur_aexp))
    # print(1.0 / hagn_tree_aexps[tree_arg] - 1, 1.0 / cur_aexp - 1)
    # hagn_main_branch_id = hagn_tree_hids[np.argmin(np.abs(hagn_tree_aexps - cur_aexp))]
    hagn_ctr = np.asarray(
        [
            hagn_tree_datas["x"][0][tree_arg],
            hagn_tree_datas["y"][0][tree_arg],
            hagn_tree_datas["z"][0][tree_arg],
        ]
    )

    hagn_ctr += 0.5 * (l_hagn * cur_aexp)
    hagn_ctr /= l_hagn * cur_aexp

    hagn_rvir = hagn_tree_datas["r"][0][tree_arg] / (l_hagn * cur_aexp)  # * 3  # * 10

    print(hagn_rvir)

    # stars = yt_read_star_ball(sim, cur_snap, hagn_ctr, hagn_rvir)

    stars = read_part_ball(
        sim, cur_snap, hagn_ctr, hagn_rvir, ["metallicity", "mass", "birth_time"], fam=2
    )

    # print(stars)

    print(f"Found {stars['mass'].sum():.1e} Msun of stellar mass")

    # # # get hagn halo pos and rvir at this time
    # # super_cat = make_super_cat(cur_snap)

    # # hid_arg = super_cat["hid"] == hagn_main_branch_id

    # # hagn_ctr = np.asarray(
    # #     [super_cat["hx"][hid_arg], super_cat["hy"][hid_arg], super_cat["hz"][hid_arg]]
    # # )

    # # hagn_rvir = super_cat["rvir"][hid_arg]

    hm_path = os.path.join(
        sim.path,
        "HaloMaker_stars2_dp_rec_dust",
    )

    ## find brick files
    # bricks = [f for f in os.listdir(hm_path) if "brick" in f]
    # brick_snaps = [int(f[-3:]) for f in bricks]
    # order = np.argsort(brick_snaps)
    # brick_snaps = np.asarray(brick_snaps)[order]
    # bricks = np.asarray(bricks)[order]

    # tgt_brick_snap = brick_snaps[-1]

    # # time at snap
    # tsnap = sim.get_snap_times([tgt_brick_snap])[0]
    # get hagn stellar mass
    # snap_hagn_sm = sfh[np.argmin(np.abs(t - tsnap))]

    # # print("Reading brickfile")
    # gals = read_brickfile(os.path.join(hm_path, f"tree_bricks{tgt_brick_snap:03d}"))
    # convert_brick_units(gals, sim)  # so positions can be compared easily
    # # # closest to zoom centre
    # # # mass_filt = (gals["hosting info"]["hmass"] > snap_hagn_sm * 0.1) & (
    # # #     gals["hosting info"]["hmass"] < snap_hagn_sm * 10
    # # # )
    # gal_pos = np.asarray(
    #     [
    #         gals["positions"]["x"],  # [mass_filt],
    #         gals["positions"]["y"],  # [mass_filt],
    #         gals["positions"]["z"],  # [mass_filt],
    #     ]
    # )
    # # # search around HAGN galaxy position at current redshift!

    # # # get most massive of galaxies within factor 10 of HAGN and within 33% of rzoom
    # tree = cKDTree(gal_pos.T, boxsize=1.0 + 1e-6)

    # # print(hagn_ctr, hagn_rvir)
    # ball_arg = tree.query_ball_point(hagn_ctr, r=hagn_rvir)

    # print("%e" % (np.sum(gals["hosting info"]["hmass"][ball_arg])))

    # ball_masses = gals["hosting info"]["hmass"][ball_arg]
    # massive_arg = ball_arg[np.argmax(ball_masses)]

    # massive_id = gals["hosting info"]["hid"][massive_arg]

    # # massive_arg = np.argmax(gals["hosting info"]["hmass"])
    # # massive_id = gals["hosting info"]["hid"][massive_arg]

    # zoom_ctr = np.asarray(
    #     [
    #         sim.namelist["refine_params"]["xzoom"],
    #         sim.namelist["refine_params"]["yzoom"],
    #         sim.namelist["refine_params"]["zzoom"],
    #     ]
    # )

    # massive_gal_ctr = np.asarray(
    #     [
    #         gals["positions"]["x"][massive_arg],
    #         gals["positions"]["y"][massive_arg],
    #         gals["positions"]["z"][massive_arg],
    #     ]
    # )

    # rdist = np.linalg.norm(
    #     [
    #         zoom_ctr - massive_gal_ctr,
    #     ]
    # )

    # print("distance to zoom centre ", rdist)

    # print("Getting gal stars")
    # stars = get_gal_stars(
    #     tgt_brick_snap, massive_id, sim, fields=["Zpart", "mpart", "agepart"]
    # )

    # sim_z = 1.0 / sim.get_snap_exps([tgt_brick_snap]) - 1
    sfh, sfr, ssfr, t, cosmo = sfhs.get_sf_stuff(stars, 1.0 / cur_aexp - 1.0, sim)

    ssfr[np.isfinite(ssfr) == False] = 0

    sfr_min = min(sfr.min(), sfr_min)
    sfr_max = max(sfr.max(), sfr_max)
    ssfr_min = min(ssfr.min(), ssfr_min)
    ssfr_max = max(ssfr.max(), ssfr_max)
    sfh_min = min(sfh.min(), sfh_min)
    sfh_max = max(sfh.max(), sfh_max)

    #    print(list(zip(t, sfh)))

    sfhs.plot_sf_stuff(
        ax,
        sfh,
        sfr,
        ssfr,
        t,
        0,
        cosmo,
        color=c["color"],
        # label=sim_id + " zoom",
        ls="--",
        ticks=sim_id == sim_ids[-1],
        lw=3,
        # ticks=True,
        # marker="o",
    )


ax[0].set_ylim(ssfr_min * 0.5, ssfr_max * 1.5)
ax[1].set_ylim(sfr_min * 0.5, sfr_max * 1.5)
ax[2].set_ylim(sfh_min * 0.5, sfh_max * 1.5)

# add key to existing legend

ax[-1].plot([], [], "k-", label="HAGN")
ax[-1].plot([], [], "k--", label="zoom")

ax[-1].legend()
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

# add 2nd yaxis to last plot
# y2 = ax[-1].twiny()
# y2.set_xlim(ax[0].get_xlim())

fig.savefig("sfhs_comp.png")
