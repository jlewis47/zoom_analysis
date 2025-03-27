from zoom_analysis.stars import sfhs
from zoom_analysis.stars.star_reader import yt_read_star_ball, read_part_ball
from zoom_analysis.stars.leja_quench_fit import sfr_ridge_leja22
from zoom_analysis.halo_maker.read_treebricks import (
    read_brickfile,
    convert_brick_units,
    convert_star_units,
    read_zoom_stars,
)
from zoom_analysis.trees.tree_reader import read_tree_file_rev as read_tree_fev_sim
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import (
    get_halo_assoc_file,
    get_halo_props_snap,
    find_zoom_tgt_gal,
    find_zoom_tgt_halo,
    get_central_gal_for_hid,
    get_gal_props_snap,
)

from scipy.spatial import KDTree

from zoom_analysis.zoom_helpers import (
    find_starting_position,
    starting_hid_from_hagn,
    check_if_in_zoom,
    decentre_coordinates,
)


from hagn.association import gid_to_stars
from hagn.utils import get_hagn_sim, adaptahop_to_code_units
from hagn.tree_reader import read_tree_rev, interpolate_tree_position
from hagn.catalogues import get_halos_cat, make_super_cat, get_cat_hids, get_cat_gids


# from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


import os
import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# from zoom_analysis.sink_reader import *

# get colour cycler
# from cycler import cycler

# planck cosmo
from astropy.cosmology import Planck18 as cosmo

from hagn.catalogues import make_super_cat, get_cat_hids


# delta_t = 100  # Myr

# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id21892_leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id180130_leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id242704"  # _leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id242756"  # _leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id74099"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sdir = "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_drag"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel"
# sdir=     # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_lowerSFE"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_noAGN"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh"
# sdir=         # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/too_high_nsink/id242704"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/too_high_nsink/id242756"  # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704"  # _leastcoarse"
# sdir  "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756"  # _leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242704"  # _leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242756"  # _leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242704_lower_nstar"  # _leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242756_lower_nstar"  # _leastcoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel"  # _leastcoarse"
sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel"  # _leastcoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"  # _leastcoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380"  # _leastcoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289"  # _leastcoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag"  # _leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh"  # _leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099"

# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892"
# sdir =
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_evenlesscoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479"
# print(explr_dirs)


# setup plot
fig, ax = sfhs.setup_sfh_plot()

yax2 = None

tgt_zed = 2.0
tgt_time = cosmo.age(tgt_zed).value

delta_aexp = 0.005

mlim = 7e10

sfr_min, ssfr_min, sfh_min = np.inf, np.inf, np.inf
sfr_max, ssfr_max, sfh_max = -np.inf, -np.inf, -np.inf

last_hagn_id = -1
isim = 0

# zoom_ls = ["-", "--", ":", "-.", (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10))]
lines = []
labels = []


sfr_max = -np.inf
sfr_min = np.inf

ssfr_max = -np.inf
ssfr_min = np.inf

mstel_max = -np.inf
mstel_min = np.inf

last_simID = None
l = None

zoom_style = 0


name = sdir.split("/")[-1].strip()


sim = ramses_sim(sdir, nml="cosmo.nml")

sim_snaps = sim.snap_numbers
sim_aexps = sim.get_snap_exps()
sim.init_cosmo()
sim_times = sim.cosmo_model.age(1.0 / sim_aexps - 1.0).value * 1e3

# last sim_aexp
# valid_steps = hagn_tree_aexps < (1.0 / (tgt_zed + 1.0))  # sim.aexp_end
# nsteps = np.sum(valid_steps)

nsteps = len(sim_snaps)

mstel_zoom = np.zeros(nsteps, dtype=np.float32)
sfr_zoom = np.zeros(nsteps, dtype=np.float32)
time_zoom = np.zeros(nsteps, dtype=np.float32)

# find last output with assoc files
assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

avail_aexps = np.intersect1d(
    sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
)
avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

avail_snaps = np.intersect1d(sim_snaps, assoc_file_nbs)

avail_zeds = 1.0 / avail_aexps - 1.0

real_start_zed = avail_zeds[np.argmin(np.abs(avail_zeds - tgt_zed))]
real_start_snap = avail_snaps[np.argmin(np.abs(avail_zeds - tgt_zed))]

gal_props = get_gal_props_snap(sim.path, real_start_snap)

fpure = gal_props["host purity"] > 0.9999
# for k in gal_props:
#     gal_props[k] = gal_props[k][fpure]

# print(gal_props["mass"][fpure].max())

mass_ok = gal_props["mass"][fpure] > mlim
nb_gals_to_plot = np.sum(mass_ok)

# print(gal_props["gids"][fpure][mass_ok])


print(f"plotting {nb_gals_to_plot} galaxies above M*={mlim:.1e} Msun")
print(gal_props["gids"][fpure][mass_ok])

# mstel_zoom = np.zeros((len(avail_times),nb_gals_to_plot), dtype=np.float32)
mstel_zoom = np.zeros((len(avail_times)), dtype=np.float32)
# sfr_zoom = np.zeros((len(avail_times),nb_gals_to_plot), dtype=np.float32)
sfr_zoom = np.zeros((len(avail_times)), dtype=np.float32)
# time_zoom = np.zeros((len(avail_times),nb_gals_to_plot), dtype=np.float32)
time_zoom = np.zeros((len(avail_times)), dtype=np.float32)

fig, ax = sfhs.setup_sfh_plot()

hagn_sim = get_hagn_sim()

hagn_sim.init_cosmo()

# find closest_hagn snap
closest_hagn_snap = hagn_sim.get_closest_snap(zed=real_start_zed)
closest_hagn_zed = 1.0 / hagn_sim.get_snap_exps(closest_hagn_snap)[0] - 1


hagn_cat = make_super_cat(closest_hagn_snap)

hagn_gpos = np.transpose([hagn_cat["x"], hagn_cat["y"], hagn_cat["z"]])
hagn_hpos = np.transpose([hagn_cat["hx"], hagn_cat["hy"], hagn_cat["hz"]])

in_zoom = check_if_in_zoom(hagn_hpos, sim)

mass_cond = hagn_cat["mgal"] >= mlim

fig_pos, ax_pos = plt.subplots(1, 3, figsize=(12, 4))
for idim in range(3):

    i = int((idim) % 3)
    j = int((idim + 1) % 3)
    k = int((j + 1) % 3)

    gal_pos_all = gal_props["pos"].T[fpure][mass_ok]

    gal_pos_all = decentre_coordinates(gal_pos_all, sim.path)

    ax_pos[idim].scatter(gal_pos_all[:, i], gal_pos_all[:, j], c="b")
    ax_pos[idim].scatter(
        hagn_gpos[in_zoom * mass_cond, i], hagn_gpos[in_zoom * mass_cond, j], c="r"
    )

    for igal in range(len(gal_pos_all)):
        ax_pos[idim].annotate(
            gal_props["gids"][fpure][mass_ok][igal],
            (gal_pos_all[igal, i], gal_pos_all[igal, j]),
        )


fig_pos.savefig("debug_all_gals_pos.png")

for igal in range(nb_gals_to_plot):

    # get tree
    # gid = gal_props["gids"][mass_ok][igal]
    gid = gal_props["gids"][fpure][mass_ok][igal]
    zoom_gpos = gal_props["pos"].T[fpure][mass_ok][igal]
    hid = gal_props["host hid"][fpure][mass_ok][igal]
    zoom_mass = gal_props["mass"][fpure][mass_ok][igal]

    print(hid, gid)

    zoom_halo_props, hosted_gals = get_halo_props_snap(sim.path, real_start_snap, hid)

    zoom_hpos = zoom_halo_props["pos"]
    print(zoom_hpos)
    zoom_hpos = decentre_coordinates(zoom_hpos, sim.path)
    print(zoom_hpos)
    zoom_gpos = decentre_coordinates(zoom_gpos, sim.path)

    zoom_hrvir = zoom_halo_props["rvir"] * 1
    zoom_mvir = zoom_halo_props["mvir"]

    # gpos = np.transpose([hagn_cat["x"], hagn_cat["y"], hagn_cat["z"]])
    hpos = hagn_hpos

    dists = np.linalg.norm(hpos - zoom_hpos, axis=1)
    # dists = np.linalg.norm(gpos - zoom_hpos, axis=1)
    print(dists.min(), dists.max())

    # hagn_gal_tree = KDTree(gpos, boxsize=1.0 + 0.2 * gpos.min())
    hagn_gal_tree = KDTree(hpos, boxsize=1.0 + 0.2 * hpos.min())

    intID = int(name[2:].split("_")[0])

    hagn_hids = hagn_cat["hid"]

    # print(f"looking for hagn halo at z={closest_hagn_zed:.2f}")
    print(f"zoom halo mass is {zoom_mvir:.2e} Msun")
    print(f"zoom halo pos is {zoom_hpos}")
    print(f"zoom halo rvir is {zoom_hrvir*sim.cosmo.lcMpc*1e3:.2f} kpc")

    # tree_gids, tree_datas, tree_aexps = read_tree_fev_sim(
    tree_hids, tree_datas, tree_aexps = read_tree_fev_sim(
        # os.path.join(sim.path, "TreeMakerstars2_dp_rec_dust", "tree_rev.dat"),
        # fbytes=os.path.join(sim.path, "TreeMakerstars2_dp_rec_dust"),
        os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat"),
        fbytes=os.path.join(sim.path, "TreeMakerDM_dust"),
        zstart=real_start_zed,
        tgt_ids=[hid],
        # tgt_ids=[gid],
        star=False,
        # star=True,
    )

    # ok_step = tree_gids[0] > 0
    ok_step = tree_hids[0] > 0

    tree_hids = tree_hids[0][ok_step]
    # tree_gids = tree_gids[0][ok_step]
    tree_aexps = tree_aexps[ok_step]

    # print(tree_hids, tree_datas)
    tree_times = sim.cosmo_model.age(1.0 / tree_aexps - 1.0).value * 1e3  # Myr

    for k in tree_datas:
        tree_datas[k] = tree_datas[k][0][ok_step]

    for istep, (time, aexp, snap) in enumerate(
        zip(avail_times, avail_aexps, avail_snaps)
    ):

        if np.min(np.abs(aexp - tree_aexps)) > delta_aexp:
            continue

        tree_arg = np.argmin(np.abs(tree_aexps - aexp))

        hprops, hosted_gals = get_halo_props_snap(sim.path, snap, tree_hids[tree_arg])
        if hosted_gals == {}:
            print(f"step:{istep:d}, no hosted gals, skipping")
            continue
        cur_gid = hosted_gals["gids"][hosted_gals["mass"].argmax()]

        # cur_gid = tree_gids[tree_arg]

        _, cur_gal_props = get_gal_props_snap(sim.path, snap, gid=cur_gid)

        # print(snap, cur_gid)

        mstel_zoom[istep] = cur_gal_props["mass"]  # msun
        sfr_zoom[istep] = cur_gal_props["sfr100"]  # msun/Myr
        time_zoom[istep] = time  # Myr

    # print(mstel_zoom)

    ssfr_zoom = sfr_zoom / mstel_zoom

    # print(f"zoom style is {zoom_ls[zoom_style]}")
    l = sfhs.plot_sf_stuff(
        ax,
        mstel_zoom,
        sfr_zoom,
        ssfr_zoom,
        time_zoom,
        None,
        sim.cosmo_model,
        # label=sim.name,
        lw=2.0,
    )
    labels.append(gid)
    lines.append(l)

    # fig_pos, ax_pos = plt.subplots(1, 3, figsize=(12, 4))

    # for idim in range(3):
    #     i = int((idim) % 3)
    #     j = int((idim + 1) % 3)
    #     k = int((j + 1) % 3)

    #     # print(i, j, k)

    #     plane = np.abs(hpos[:, k] - zoom_hpos[k]) < zoom_hrvir * 1

    #     # ax_pos[idim].scatter(
    #     #     hpos[plane, i],
    #     #     hpos[plane, j],
    #     #     c="r",
    #     #     alpha=0.5,
    #     #     s=2,
    #     # s=np.sqrt(np.log10(hagn_cat["mhalo"][plane])) * 2,
    #     # )

    #     arg_plane = np.where(plane)[0]

    #     for iarg in arg_plane:
    #         ax_pos[idim].add_patch(
    #             plt.Circle(
    #                 (hpos[iarg, i], hpos[iarg, j]),
    #                 hagn_cat["rvir"][iarg],
    #                 fill=False,
    #                 color="r",
    #             )
    #         )

    #     # ax_pos[idim].scatter(zoom_hpos[i], zoom_hpos[j], c="b", marker="x")
    #     ax_pos[idim].add_patch(
    #         plt.Circle((zoom_hpos[i], zoom_hpos[j]), zoom_hrvir, fill=False, color="b")
    #     )

    #     # ax_pos[idim].set_xlim(
    #     #     zoom_hpos[i] - 5 * zoom_hrvir, zoom_hpos[i] + 5 * zoom_hrvir
    #     # )
    #     # ax_pos[idim].set_ylim(
    #     #     zoom_hpos[j] - 5 * zoom_hrvir, zoom_hpos[j] + 5 * zoom_hrvir
    #     # )

    # fig_pos.savefig(f"debug_pos_all_gals_{name:s}_{igal:d}.png")

    # found_hagn = False
    # rsearch = 0.1 * zoom_hrvir
    # while not found_hagn:  # and rsearch < zoom_hrvir:

    #     # ball_args = hagn_gal_tree.query_ball_point(zoom_hpos, rsearch)
    #     ball_args = hagn_gal_tree.query_ball_point(zoom_gpos, rsearch)

    #     print(rsearch, len(ball_args))

    #     if len(ball_args) > 0:

    #         halo_masses = hagn_cat["mhalo"][ball_args]
    #         # gal_masses = hagn_cat["mgal"][ball_args]

    #         found_hagn = True

    #         # found_hagn = np.any(np.abs(halo_masses - zoom_mvir) < 0.1 * zoom_mvir)
    #         # found_hagn = np.any(np.abs(gal_masses - zoom_mass) < 0.1 * zoom_mass)

    #         # hagn_gid = hagn_cat["gids"][ball_args][
    #         #     np.argmin(np.abs(halo_masses - zoom_mass))
    #         # ]
    #         if found_hagn:
    #             hagn_hid = hagn_cat["hid"][ball_args][
    #                 # np.argmin(np.abs(gal_masses - zoom_mvir))
    #                 np.argmin(np.abs(halo_masses - zoom_mvir))
    #             ]

    #     rsearch *= 2.0

    # if found_hagn:  # look for hagn history
    for hagn_hid in hagn_hids[in_zoom * mass_cond]:

        # print("found hagn - looking up sfh")

        hagn_tree_hids, hagn_tree_datas, hagn_tree_aexps = read_tree_rev(
            closest_hagn_zed,
            [hagn_hid],
            tree_type="halo",
            # [gid],
            # tree_type="gal",
            target_fields=["m", "x", "y", "z", "r"],
            sim="hagn",
        )

        for key in hagn_tree_datas:
            hagn_tree_datas[key] = hagn_tree_datas[key][0][:]

        hagn_tree_times = (
            hagn_sim.cosmo_model.age(1.0 / hagn_tree_aexps - 1.0).value * 1e3
        )

        nstes_hagn = len(hagn_tree_times)

        mstel_hagn = np.zeros(nstes_hagn, dtype=np.float32)
        sfr_hagn = np.zeros(nstes_hagn, dtype=np.float32)
        times_hagn = np.zeros(nstes_hagn, dtype=np.float32)

        for istep_hagn, (time_hagn, aexp_hagn) in enumerate(
            zip(hagn_tree_times, hagn_tree_aexps)
        ):

            snap_hagn = hagn_sim.get_closest_snap(zed=1.0 / aexp_hagn - 1.0)

            try:
                super_cat = make_super_cat(
                    snap_hagn, outf="/data101/jlewis/hagn/super_cats"
                )  # , overwrite=True)
            except FileNotFoundError:
                print("No super cat")
                continue

            gal_pties = get_cat_hids(super_cat, [int(hagn_tree_hids[0][istep_hagn])])

            if len(gal_pties["gid"]) == 0:
                print("No galaxy")
                continue

            mstel_hagn[istep_hagn] = gal_pties["mgal"][0]
            sfr_hagn[istep_hagn] = gal_pties["sfr100"][0] * 1e6
            times_hagn[istep_hagn] = time_hagn

        ssfr_hagn = sfr_hagn / mstel_hagn

        sfhs.plot_sf_stuff(
            ax,
            mstel_hagn,
            sfr_hagn,
            ssfr_hagn,
            times_hagn,
            0,
            hagn_sim.cosmo_model,
            ls=":",
            # color=,
            # label=sim_id.split("_")[0],
            lw=1.0,
        )


# if np.isfinite(ssfr_min) and np.isfinite(ssfr_max):
# ax[0].set_ylim(ssfr_min * 0.5, ssfr_max * 1.5)
# if np.isfinite(sfr_min) and np.isfinite(sfr_max):
# ax[1].set_ylim(sfr_min * 0.5, sfr_max * 1.5)
# if np.isfinite(mstel_min) and np.isfinite(mstel_max):
# ax[2].set_ylim(mstel_min * 0.5, mstel_max * 1.5)

# tlim = time_zoom[np.where(sfr_zoom == 0)[-1]]
# ax[0].set_xlim(tlim, time_zoom[-1])
# ax[1].set_xlim(tlim, time_zoom[-1])
# ax[2].set_xlim(tlim, time_zoom[-1])

# ax[0].set_xlim(
# sim_tree_times.min() / 1e3,
# sim_tree_times.max() / 1e3,
# min(hagn_tree_times.min(), sim_tree_times.min()) / 1e3,
# max(hagn_tree_times.max(), sim_tree_times.max()) / 1e3,
# )


ax[0].set_xlim(0.5, 3.2)
# ax[0].set_xlim(0.5, 2.5)

# add key to existing legend

# ax[-1].plot([], [], "k-", label="HAGN")
# ax[-1].plot([], [], "k-", label="zoom")


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
ax[0].text(xlim[0] + 0.1, 1.1e-5, "quenched", color="k", alpha=0.33, ha="left")
# ax[0].set_ylim(4e-6, 2e-2)
ylim = ax[0].get_ylim()
ax[0].fill_between(xlim, 1e-5, ylim[0], color="k", alpha=0.33)
ax[0].set_ylim(ylim)
for a in ax:
    ax[0].set_xlim(xlim)


# sfrs_leja = sfr_ridge_leja22(tgt_zed, mstel_zoom)
# ax[0].plot(mass_bins, sfrs_leja / 10.0 / mstel_zoom, "k--")
# ax[0].annotate(
#     mass_bins[0], sfrs_leja[0] / 10.0 / mstel_zoom, "0.1xMS of Leja+22", color="k"
# )


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

for a in ax:
    a.grid()

zspan_time = cosmo.age(zspan).value

for m, ssfr in zip(leja_masses, np.transpose(ssfr_leja_z2_lim)):
    # ax[0].axhline(ssfr, color="k", ls="--", alpha=0.3)
    finite = np.isfinite(ssfr)
    if finite.sum() < 2:
        continue
    ax[0].plot(zspan_time[finite], ssfr[finite], "k--", alpha=0.33)

    mid_point = np.where(finite)[0][int(0.5 * finite.sum())]

    ax[0].text(
        zspan_time[finite][mid_point],
        ssfr[finite][mid_point],
        f"Leja+22, M={m:.1e} $M_\odot$",
        color="k",
        alpha=0.33,
        ha="center",
    )

lines.append(
    [
        Line2D([0], [0], color="k", linestyle="-", lw=1),
        Line2D([0], [0], color="k", ls="-", lw=2),
    ]
)
labels.append(["HAGN", "zoom"])

ax[-1].legend(lines, labels, framealpha=0.0, ncol=2)  # , handlelength=3)

zstr = f"{tgt_zed:.1f}".replace(".", "p")
fig.savefig(f"figs/sfhs_all_gals_{name:s}_z{zstr}.png")
