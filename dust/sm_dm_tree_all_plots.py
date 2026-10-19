# from f90nml import read
from f90_tools.star_reader import read_part_ball_NCdust
from zoom_analysis.read.read_data import read_data_ball
from zoom_analysis.stars import sfhs
from zoom_analysis.zoom_helpers import decentre_coordinates, find_starting_position
from zoom_analysis.halo_maker.read_treebricks import (
    read_zoom_stars,
)
from zoom_analysis.trees.tree_reader import read_tree_file_rev as read_tree_fev_sim
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import (
    # find_zoom_tgt_halo,
    get_central_gal_for_hid,
)
from zoom_analysis.dust.gas_reader import gas_pos_rad
from zoom_analysis.halo_maker.assoc_fcts import find_star_ctr_period

# from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# from scipy.spatial import cKDTree

import os
import numpy as np

from hagn.utils import get_hagn_sim
from hagn.tree_reader import read_tree_rev, interpolate_tree_position

# planck cosmo
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
from astropy import units as u
import h5py


def plot_dtms(ax, sim_times, md1_zoom, md2_zoom, md3_zoom, md4_zoom, mZ_zoom, l):
    ax[2].plot(
        sim_times / 1e3,
        md1_zoom / mZ_zoom,
        # ls=zoom_ls[isim],
        color=l.get_color(),
        ls=":",
        marker="$C$",
        markersize=5,
        markevery=5,
    )
    ax[2].plot(
        sim_times / 1e3,
        md2_zoom / mZ_zoom,
        # ls=zoom_ls[isim],
        color=l.get_color(),
        ls=":",
        marker="$C$",
        markersize=10,
        markevery=5,
    )
    ax[2].plot(
        sim_times / 1e3,
        md3_zoom / mZ_zoom,
        # ls=zoom_ls[isim],
        color=l.get_color(),
        ls=":",
        marker="$S$",
        markersize=5,
        markevery=5,
    )
    ax[2].plot(
        sim_times / 1e3,
        md4_zoom / mZ_zoom,
        # ls=zoom_ls[isim],
        color=l.get_color(),
        ls=":",
        marker="$S$",
        markersize=10,
        markevery=5,
    )
    ax[2].plot(
        sim_times / 1e3,
        (md1_zoom + md2_zoom + md3_zoom + md4_zoom) / mZ_zoom,
        # ls=zoom_ls[isim],
        color=l.get_color(),
        ls="-",
    )


def plot_dtgs(
    ax, sim_times, md1_zoom, md2_zoom, md3_zoom, md4_zoom, mG_zoom, mZ_zoom, l
):
    ax[1].plot(
        sim_times / 1e3,
        md1_zoom / mG_zoom,
        # ls=zoom_ls[isim],
        color=l.get_color(),
        ls=":",
        marker="$C$",
        markersize=5,
        markevery=5,
    )
    ax[1].plot(
        sim_times / 1e3,
        md2_zoom / mG_zoom,
        # ls=zoom_ls[isim],
        color=l.get_color(),
        ls=":",
        marker="$C$",
        markersize=10,
        markevery=5,
    )
    ax[1].plot(
        sim_times / 1e3,
        md3_zoom / mG_zoom,
        # ls=zoom_ls[isim],
        color=l.get_color(),
        ls=":",
        marker="$S$",
        markersize=5,
        markevery=5,
    )
    ax[1].plot(
        sim_times / 1e3,
        md4_zoom / mG_zoom,
        # ls=zoom_ls[isim],
        color=l.get_color(),
        ls=":",
        marker="$S$",
        markersize=10,
        markevery=5,
    )
    ax[1].plot(
        sim_times / 1e3,
        (md1_zoom + md2_zoom + md3_zoom + md4_zoom) / mG_zoom,
        # ls=zoom_ls[isim],
        color=l.get_color(),
        ls="-",
    )
    ax[1].plot(
        sim_times / 1e3,
        mZ_zoom / mG_zoom,
        # ls=zoom_ls[isim],
        color=l.get_color(),
        ls=":",
        marker="$Z$",
        markersize=10,
        markevery=5,
    )


def plot_masses(
    ax, name, sim_times, mstel_zoom, md1_zoom, md2_zoom, md3_zoom, md4_zoom, mZ_zoom
):
    (l,) = ax[0].plot(
        sim_times / 1e3,
        mstel_zoom,
        label=name,
        ls="-",
        marker="*",
        markevery=5,
        markersize=10,
    )  # , ls=zoom_ls[isim]
    ax[0].plot(
        sim_times / 1e3,
        mZ_zoom,
        # ls=zoom_ls[isim],
        color=l.get_color(),
        ls=":",
        marker="$Z$",
        markersize=10,
        markevery=5,
    )
    ax[0].plot(
        sim_times / 1e3,
        md1_zoom,
        # ls=zoom_ls[isim],
        color=l.get_color(),
        ls=":",
        marker="$C$",
        markersize=5,
        markevery=5,
    )
    ax[0].plot(
        sim_times / 1e3,
        md2_zoom,
        # ls=zoom_ls[isim],
        color=l.get_color(),
        ls=":",
        marker="$C$",
        markersize=10,
        markevery=5,
    )
    ax[0].plot(
        sim_times / 1e3,
        md3_zoom,
        # ls=zoom_ls[isim],
        color=l.get_color(),
        ls=":",
        marker="$S$",
        markersize=5,
        markevery=5,
    )
    ax[0].plot(
        sim_times / 1e3,
        md4_zoom,
        # ls=zoom_ls[isim],
        color=l.get_color(),
        ls=":",
        marker="$S$",
        markersize=10,
        markevery=5,
    )
    ax[0].plot(
        sim_times / 1e3,
        (md1_zoom + md2_zoom + md3_zoom + md4_zoom),
        # ls=zoom_ls[isim],
        color=l.get_color(),
        ls="-",
    )

    return l


# sim_ids = [
#     "id2757_995pcnt_mstel/",
#     "id19782_500pcnt_mstel/",pip install setuptools
#     "id13600_005pcnt_mstel/",
# ]
# sdir = "/home/jlewis/sims/proposal_zooms/ExtractZoom/mh1e12/"

HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
delta_t = 25  # Myr

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
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi/id180130_meanBondi_higher_nmax",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi/id180130_meanBondi_Sconstant",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288",
    "/data103/jlewis/sims/lvlmax_22/mh1e12/id180130",
]
# sdir =
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892"
# sdir =
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_evenlesscoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479"
# print(explr_dirs)

tgt_zed = 2.0

msun_to_g = 1.989e33

overwrite = False

# setup plot
fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
plt.subplots_adjust(hspace=0.0)

yax2 = None

last_hagn_id = -1
isim = 0

zoom_ls = ["-", "--", ":", "-."]

lines = []
labels = []

mstel_max = -np.inf
mstel_min = np.inf

hagn_sim = get_hagn_sim()

xmax = -np.inf
xmin = np.inf

for sdir in sdirs:
    name = sdir.split("/")[-1]
    dust_file = os.path.join(sdir, "dust_data.h5")
    sim = None

    if os.path.exists(dust_file) and not overwrite:
        with h5py.File(dust_file, "r") as f:
            mstel_zoom = f["mstel_zoom"][:]
            md1_zoom = f["md1_zoom"][:]
            md2_zoom = f["md2_zoom"][:]
            md3_zoom = f["md3_zoom"][:]
            md4_zoom = f["md4_zoom"][:]
            mZ_zoom = f["mZ_zoom"][:]
            mG_zoom = f["mG_zoom"][:]
            tgt_times = f["tgt_times"][:]
    else:
        intID = int(name[2:].split("_")[0])

        hagn_tree_hids, hagn_tree_datas, hagn_tree_aexps = read_tree_rev(
            tgt_zed,
            [intID],
            tree_type="halo",
            target_fields=["m", "x", "y", "z", "r"],
            sim="hagn",
        )

        for k in hagn_tree_datas.keys():
            hagn_tree_datas[k] = hagn_tree_datas[k][0]

        hagn_sim.init_cosmo()
        l_hagn = (
            hagn_sim.unit_l(hagn_sim.aexp_stt) / (ramses_pc * 1e6) / hagn_sim.aexp_stt
        )
        hagn_tree_times = (
            hagn_sim.cosmo_model.age(1.0 / hagn_tree_aexps - 1.0).value * 1e3
        )

        sim = ramses_sim(sdir, nml="cosmo.nml")

        sim_snaps = sim.snap_numbers
        sim_aexps = sim.get_snap_exps()
        sim.init_cosmo()
        sim_times = sim.cosmo_model.age(1.0 / sim_aexps - 1.0).value * 1e3

        tgt_times = np.arange(sim_times.min(), sim_times.max() + delta_t, delta_t)

        tgt_snaps = np.asarray(
            [sim.get_closest_snap(time=tgt_time) for tgt_time in tgt_times]
        )

        tgt_aexps = sim.get_snap_exps(tgt_snaps)
        tgt_times = sim.cosmo_model.age(1.0 / tgt_aexps - 1.0).value * 1e3

        nsteps = len(tgt_snaps)

        mstel_zoom = np.zeros(nsteps, dtype=np.float32)
        md1_zoom = np.zeros(nsteps, dtype=np.float32)
        md2_zoom = np.zeros(nsteps, dtype=np.float32)
        md3_zoom = np.zeros(nsteps, dtype=np.float32)
        md4_zoom = np.zeros(nsteps, dtype=np.float32)
        mZ_zoom = np.zeros(nsteps, dtype=np.float32)
        mG_zoom = np.zeros(nsteps, dtype=np.float32)

        assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
        assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

        avail_aexps = sim.get_snap_exps(assoc_file_nbs, param_save=False)
        if not hasattr(sim, "cosmo_model"):
            sim.init_cosmo()
        avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

        hid_start, halo_dict, hosted_gals, found, start_aexp = find_starting_position(
            sim,
            avail_aexps,
            hagn_tree_aexps,
            hagn_tree_datas,
            hagn_tree_times,
            avail_times,
        )
        if not found:
            continue

        sim_halo_tree_rev_fname = os.path.join(
            sim.path, "TreeMakerDM_dust", "tree_rev.dat"
        )
        if not os.path.exists(sim_halo_tree_rev_fname):
            print("No tree file")
            continue

        tgt_zeds = 1.0 / tgt_aexps - 1

        sim_tree_hids, sim_tree_datas, sim_tree_aexps = read_tree_fev_sim(
            sim_halo_tree_rev_fname,
            fbytes=os.path.join(sim.path, "TreeMakerDM_dust"),
            zstart=1.0 / start_aexp - 1.0,
            tgt_ids=[hid_start],
            star=False,
        )

        # print(sim_tree_hids)

        for k in sim_tree_datas.keys():
            sim_tree_datas[k] = sim_tree_datas[k][0]

        sim_tree_times = sim.cosmo_model.age(tgt_zeds).value * 1e3

        for istep, (snap, aexp, time) in enumerate(
            zip(tgt_snaps, tgt_aexps, tgt_times)
        ):
            if np.all(np.abs(avail_aexps - aexp) > 0.05):
                print("No assoc file for this aexp")
                continue

            assoc_file = assoc_files[assoc_file_nbs == snap]
            if len(assoc_file) == 0:
                print("No assoc file for this snap")
                continue

            sim_tree_arg = np.argmin(np.abs(sim_tree_aexps - aexp))
            cur_snap_hid = sim_tree_hids[0][sim_tree_arg]

            # print(snap, aexp, sim_tree_aexps[sim_tree_arg], cur_snap_hid)

            gid, gal_dict = get_central_gal_for_hid(sim, cur_snap_hid, snap)
            if gid == None:
                print("No central galaxy")
                continue

            tgt_pos = gal_dict["pos"]
            tgt_r = gal_dict["r50"]

            # stars = read_zoom_stars(sim, snap, gid)
            # stars = read_part_ball_NCdust(
            #     sim,
            #     snap,
            #     tgt_pos,
            #     tgt_r,
            #     tgt_fields=["mass", "birth_time", "metallicity"],
            #     fam=2,
            # )

            # stars = read_
            try:
                datas = read_data_ball(sim, snap, tgt_pos, tgt_r, host_halo=cur_snap_hid,
                tgt_fields=["mass","age","metallicity","density","dust_bin01","dust_bin02","dust_bin03","dust_bin04"],
                data_types=['stars','gas'])
            except (FileNotFoundError,AssertionError):
                print(f'snap {snap:d}: unavailable files')
                continue


            stars = datas['stars']

            ages = stars["age"]
            Zs = stars["metallicity"]

            masses = sfhs.correct_mass(hagn_sim, ages, stars["mass"], Zs)

            # star_pos, ctr_stars, extent_stars = find_star_ctr_period(stars["pos"])

            unit_d = sim.unit_d(aexp)

            mstel_zoom[istep] = masses.sum()

            # cells = gas_pos_rad(
            #     # sim, snap, [1, 6, 16, 17, 18, 19], ctr_stars, extent_stars
            #     sim,
            #     snap,
            #     [1, 6, 16, 17, 18, 19],
            #     tgt_pos,
            #     tgt_r,
            # )

            cells = datas["gas"]

            l_hagn_cm_comov = l_hagn * aexp * 1e6 * ramses_pc
            volumes = (2 ** -cells["ilevel"] * l_hagn_cm_comov) ** 3


            md1_zoom[istep] = (
                np.sum(cells["density"] * cells["dust_bin01"] * volumes)
                # * unit_d
                / msun_to_g
            )

            md2_zoom[istep] = (
                np.sum(cells["density"] * cells["dust_bin02"] * volumes)
                # * unit_d
                / msun_to_g
            )

            md3_zoom[istep] = (
                np.sum(cells["density"] * cells["dust_bin03"] * volumes)
                # * unit_d
                / msun_to_g
                / 0.163
            )

            md4_zoom[istep] = (
                np.sum(cells["density"] * cells["dust_bin04"] * volumes)
                # * unit_d
                / msun_to_g
                / 0.163
            )

            mZ_zoom[istep] = (
                np.sum(cells["density"] * cells["metallicity"] * volumes)
                # * unit_d
                / msun_to_g
            )

        with h5py.File(dust_file, "w") as f:
            f.create_dataset("mstel_zoom", data=mstel_zoom)
            f.create_dataset("md1_zoom", data=md1_zoom)
            f.create_dataset("md2_zoom", data=md2_zoom)
            f.create_dataset("md3_zoom", data=md3_zoom)
            f.create_dataset("md4_zoom", data=md4_zoom)
            f.create_dataset("mZ_zoom", data=mZ_zoom)
            f.create_dataset("mG_zoom", data=mG_zoom)
            f.create_dataset("tgt_times", data=tgt_times)

    l = plot_masses(
        ax, name, tgt_times, mstel_zoom, md1_zoom, md2_zoom, md3_zoom, md4_zoom, mZ_zoom
    )

    plot_dtgs(
        ax, tgt_times, md1_zoom, md2_zoom, md3_zoom, md4_zoom, mG_zoom, mZ_zoom, l
    )

    plot_dtms(ax, tgt_times, md1_zoom, md2_zoom, md3_zoom, md4_zoom, mZ_zoom, l)

    lines.append(Line2D([0, 1], [0, 1], ls="-", c=l.get_color()))
    labels.append(name)

    times_nonzero_stmass = tgt_times[mstel_zoom > 0]

    xmin = min(xmin, times_nonzero_stmass.min() / 1e3)
    xmax = max(xmax, times_nonzero_stmass.max() / 1e3)


ax[2].axhline(0.4, c="k", ls=":")
ax[2].text(0.6, 0.4, "MW", color="k", va="center", ha="center")

ax[-1].legend(
    [
        Line2D([0, 1], [0, 1], ls="-", c="k", marker="*", markersize=10),
        Line2D([0, 1], [0, 1], ls=":", c="k", marker="$Z$", markersize=10),
        Line2D([0, 1], [0, 1], ls="-", c="k"),
        Line2D([0, 1], [0, 1], ls=":", c="k", marker="$C$", markersize=5),
        Line2D([0, 1], [0, 1], ls=":", c="k", marker="$C$", markersize=10),
        Line2D([0, 1], [0, 1], ls=":", c="k", marker="$S$", markersize=5),
        Line2D([0, 1], [0, 1], ls=":", c="k", marker="$S$", markersize=10),
    ]
    + lines,
    [
        "Stars",
        "Metals",
        "Dust",
        "Carbon small",
        "Carbon large",
        "Silicate small",
        "Silicate large",
    ]
    + labels,
    framealpha=0.0,
    ncol=3,
)
ax[-1].axis("off")
# cur_leg = ax[-1].get_legend()
# ax[-1].legend(
#     cur_leg.legendHandles
#     + [
#         Line2D([0], [0], color="k", linestyle="-"),
#         Line2D([0], [0], color="k", ls="--"),
#     ],
#     sim_ids + ["HAGN", "zoom"],
# )

ax[0].set_ylim(
    1e2,
)

ax[0].set_xlim(xmin, xmax)

ax[0].tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=True,
    left=True,
    right=True,
    direction="in",
)
ax[1].set_ylim(
    1e-9,
)
ax[1].tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=True,
    right=True,
    direction="in",
)
ax[2].set_ylim(1e-6)
ax[2].tick_params(
    axis="both",
    which="both",
    bottom=True,
    top=False,
    left=True,
    right=True,
    direction="in",
)

for a in ax:
    a.set_yscale("log")
    a.grid()

xlim = ax[-1].get_xlim()
ax[-1].set_xlim(xlim)

ax[0].set_ylabel("X Mass [$M_{\odot}$]")
ax[1].set_ylabel("X-to-gas ratio")

ax[2].set_ylabel("X Dust-to-metal ratio")
ax[2].set_xlabel("Time [Gyr]")

y2 = ax[0].twiny()
y2.set_xlim(xlim)
# y2.set_xticks(ax.get_xticks())
y2.set_xticklabels([f"{cosmo.age(zed).value:.1f}" for zed in ax[-1].get_xticks()])
y2.set_xlabel("redshift")

fig.savefig(f"MdMst_trees.png")
