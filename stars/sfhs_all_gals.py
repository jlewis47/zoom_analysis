from zoom_analysis.stars import sfhs
from zoom_analysis.zoom_helpers import decentre_coordinates
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


# from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# from scipy.spatial import cKDTree

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

tgt_zed = 2.05
tgt_time = cosmo.age(tgt_zed).value

delta_aexp = 0.005

mlim = 3e10

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

for igal in range(nb_gals_to_plot):

    # get tree
    # gid = gal_props["gids"][mass_ok][igal]
    gid = gal_props["gids"][fpure][mass_ok][igal]
    hid = gal_props["host hid"][fpure][mass_ok][igal]

    print(hid, gid)

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

    labels.append(f"z={1./aexp-1.:.2f}, {hid:d}:{gid:d}")
    lines.append(l)


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
