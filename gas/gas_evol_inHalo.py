# from f90nml import read
from f90_tools.star_reader import read_part_ball_NCdust
from zoom_analysis.stars import sfhs
from zoom_analysis.zoom_helpers import decentre_coordinates, find_starting_position
from zoom_analysis.halo_maker.read_treebricks import (
    read_zoom_stars,
)
from zoom_analysis.trees.tree_reader import read_tree_file_rev as read_tree_fev_sim
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import (
    # find_zoom_tgt_halo,
    # get_central_gal_for_hid,
    get_halo_props_snap,
)
from zoom_analysis.dust.gas_reader import code_to_cgs, gas_pos_rad

# from zoom_analysis.halo_maker.assoc_fcts import find_star_ctr_period

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
]


tgt_zed = 2.0

msun_to_g = 1.989e33

overwrite = False

# plot of mgas vs time for different sims
# fbaryon vs time for different sims
# max temperature vs time for different sims

# setup plot
fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
plt.subplots_adjust(hspace=0.0)


yax2 = None

last_hagn_id = -1
isim = 0

zoom_ls = ["-", "--", ":", "-."]

lines = []
labels = []

hagn_sim = get_hagn_sim()

xmax = -np.inf
xmin = np.inf


for sdir in sdirs:
    name = sdir.split("/")[-1]
    dust_file = os.path.join(sdir, "gas_data_inHalo.h5")
    sim = None

    if os.path.exists(dust_file) and not overwrite:
        with h5py.File(dust_file, "r") as f:
            mG_zoom = f["mG_zoom"][:]
            maxT_zoom = f["maxT_zoom"][:]
            meanT_zoom = f["meanT_zoom"][:]
            # print(maxT_zoom.min(), maxT_zoom.mean(), maxT_zoom.max())
            fractST_zoom = f["fracST_zoom"][:]
            fractG_zoom = f["fracG_zoom"][:]
            fractBaryon_zoom = f["fracBaryon_zoom"][:]
            time_zoom = f["time_zoom"][:]
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

        mG_zoom = np.zeros(nsteps, dtype=np.float32)
        maxT_zoom = np.zeros(nsteps, dtype=np.float32)
        meanT_zoom = np.zeros(nsteps, dtype=np.float32)
        fractST_zoom = np.zeros(nsteps, dtype=np.float32)
        fractG_zoom = np.zeros(nsteps, dtype=np.float32)
        fractBaryon_zoom = np.zeros(nsteps, dtype=np.float32)
        time_zoom = np.zeros(nsteps, dtype=np.float32)

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

            zed = 1.0 / aexp - 1.0

            sim_tree_arg = np.argmin(np.abs(sim_tree_aexps - aexp))
            cur_snap_hid = sim_tree_hids[0][sim_tree_arg]

            # print(snap, aexp, sim_tree_aexps[sim_tree_arg], cur_snap_hid)

            # gid, gal_dict = get_central_gal_for_hid(sim, cur_snap_hid, snap)
            # if gid == None:
            #     print("No central galaxy")
            #     continue

            hprops, hgals = get_halo_props_snap(sim.path, snap, cur_snap_hid)

            tgt_pos = hprops["pos"]
            tgt_r = hprops["rvir"]

            # stars = read_zoom_stars(sim, snap, gid)
            stars = read_part_ball_NCdust(
                sim,
                snap,
                tgt_pos,
                tgt_r,
                tgt_fields=["mass", "birth_time", "metallicity"],
                fam=2,
            )

            ages = stars["age"]
            Zs = stars["metallicity"]

            masses = sfhs.correct_mass(hagn_sim, ages, stars["mass"], Zs)
            dms = read_part_ball_NCdust(
                sim, snap, tgt_pos, tgt_r, tgt_fields=["mass"], fam=1
            )

            # star_pos, ctr_stars, extent_stars = find_star_ctr_period(stars["pos"])

            unit_d = sim.unit_d(aexp)

            code_cells = gas_pos_rad(
                sim, snap, ["density", "temperature"], tgt_pos, tgt_r
            )

            cells = code_to_cgs(sim, aexp, code_cells)

            l_hagn_cm_comov = l_hagn * aexp * 1e6 * ramses_pc
            volumes = (2 ** -cells["ilevel"] * l_hagn_cm_comov) ** 3

            mG_zoom[istep] = np.sum(cells["density"] * volumes) / msun_to_g

            maxT_zoom[istep] = np.max(cells["temperature"])
            meanT_zoom[istep] = np.average(
                cells["temperature"], weights=(cells["density"] * volumes)
            )

            st_mass = masses.sum()
            dm_mass = dms["mass"].sum()

            print(istep, zed, dm_mass, st_mass, mG_zoom[istep])

            tot_mass = st_mass + dm_mass + mG_zoom[istep]

            fractST_zoom[istep] = st_mass / tot_mass
            fractG_zoom[istep] = mG_zoom[istep] / tot_mass
            fractBaryon_zoom[istep] = (st_mass + mG_zoom[istep]) / tot_mass

            time_zoom[istep] = time

        with h5py.File(dust_file, "w") as f:

            f.create_dataset("mG_zoom", data=mG_zoom, compression="lzf")
            f.create_dataset("maxT_zoom", data=maxT_zoom, compression="lzf")
            f.create_dataset("meanT_zoom", data=meanT_zoom, compression="lzf")
            f.create_dataset("fracST_zoom", data=fractST_zoom, compression="lzf")
            f.create_dataset("fracG_zoom", data=fractG_zoom, compression="lzf")
            f.create_dataset(
                "fracBaryon_zoom", data=fractBaryon_zoom, compression="lzf"
            )
            f.create_dataset("time_zoom", data=time_zoom, compression="lzf")

    print((1 - fractBaryon_zoom))

    (l,) = ax[0].plot(time_zoom / 1e3, mG_zoom, ls=zoom_ls[isim], label=name)
    lines.append(Line2D([0, 1], [0, 1], ls="-", c=l.get_color()))
    labels.append(name)
    ax[0].set_ylim(
        1e5,
    )

    ax[1].plot(time_zoom / 1e3, fractG_zoom, ls=zoom_ls[isim], lw=2.0, c=l.get_color())
    ax[1].plot(
        time_zoom / 1e3, fractST_zoom, ls=zoom_ls[isim], lw=1.25, c=l.get_color()
    )
    ax[1].plot(
        time_zoom / 1e3, fractBaryon_zoom, ls=zoom_ls[isim], lw=3, c=l.get_color()
    )
    ax[1].set_ylim(1e-2, 1.05)

    ax[2].plot(time_zoom / 1e3, maxT_zoom, ls=zoom_ls[isim], c=l.get_color(), lw=2.5)
    ax[2].set_ylim(
        1e5,
    )

    ax[2].plot(time_zoom / 1e3, meanT_zoom, ls=zoom_ls[isim], c=l.get_color(), lw=1.5)
    ax[2].set_ylim(
        1e3,
    )

    times_nonzero = time_zoom[mG_zoom > 0]

    xmin = min(xmin, times_nonzero.min() / 1e3)
    xmax = max(xmax, times_nonzero.max() / 1e3)


ax[-1].legend(
    lines,
    labels,
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

# ax[0].set_ylim(
#     1e2,
# )

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
# ax[1].set_ylim(
#     1e-9,
# )
ax[1].tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=True,
    right=True,
    direction="in",
)
# ax[2].set_ylim(1e-6)
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

ax[0].set_ylabel("Gas Mass [$M_{\odot}$]")
ax[1].set_ylabel("Mass fraction")

ax[2].set_ylabel("Temperature [K]")
ax[2].set_xlabel("Time [Gyr]")

y2 = ax[0].twiny()
y2.set_xlim(xlim)
# y2.set_xticks(ax.get_xticks())
y2.set_xticklabels([f"{cosmo.age(zed).value:.1f}" for zed in ax[-1].get_xticks()])
y2.set_xlabel("redshift")

fig.savefig(f"gas_inHalo.png")
