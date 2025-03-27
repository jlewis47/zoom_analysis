# from f90nml import read
from time import time_ns
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
    get_central_gal_for_hid,
    get_halo_props_snap,
)
from zoom_analysis.dust.gas_reader import code_to_cgs, gas_pos_rad
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
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi/id180130_meanBondi_higher_nmax",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi/id180130_meanBondi_Sconstant",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_lowerSFE",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
]


tgt_zed = 2.0

msun_to_g = 1.989e33

overwrite = True
istep = 0
sim = None

# plot of mgas vs time for different sims
# fbaryon vs time for different sims
# max temperature vs time for different sims

# setup plot
# fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True)
fig, ax = plt.subplots(2, 2, figsize=(8, 8))
ax = ax.flatten()
plt.subplots_adjust(hspace=0.0)

lprof = 100

yax2 = None

last_hagn_id = -1
isim = 0

zoom_ls = ["-", "--", ":", "-."]

lines = []
labels = []

hagn_sim = get_hagn_sim()

xmax = -np.inf
xmin = np.inf

radial_bins = np.logspace(-10, 0.6, lprof)

for sdir in sdirs:
    name = sdir.split("/")[-1]
    dust_file = os.path.join(sdir, "profile_data.h5")
    sim = None

    if os.path.exists(dust_file) and not overwrite:
        with h5py.File(dust_file, "r") as f:
            mG_zoom = f["mG_zoom"][:]
            meanT_zoom = f["meanT_zoom"][:]
            # print(maxT_zoom.min(), maxT_zoom.mean(), maxT_zoom.max())
            fractST_zoom = f["fracST_zoom"][:]
            fractG_zoom = f["fracG_zoom"][:]
            fractBaryon_zoom = f["fracBaryon_zoom"][:]
            radial_bins = f["radial_bins"][:]
            time_zoom = f["time_zoom"][:]
    else:

        # radial_bins = np.logspace(-4, 1.0, lprof)  # in units of r50

        # print(radial_bins)

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

        mG_zoom = np.zeros((nsteps, lprof), dtype=np.float32)
        meanT_zoom = np.zeros((nsteps, lprof), dtype=np.float32)
        fractST_zoom = np.zeros((nsteps, lprof), dtype=np.float32)
        fractG_zoom = np.zeros((nsteps, lprof), dtype=np.float32)
        fractBaryon_zoom = np.zeros((nsteps, lprof), dtype=np.float32)
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
            zip(tgt_snaps[::-1], tgt_aexps[::-1], tgt_times[::-1])
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

            gid, gal_dict = get_central_gal_for_hid(sim, cur_snap_hid, snap)
            if gid == None:
                print("No central galaxy")
                continue

            # tgt_pos = gal_dict["pos"]
            # tgt_r = gal_dict["r50"]

            hprops, hgals = get_halo_props_snap(sim.path, snap, cur_snap_hid)

            tgt_pos = hprops["pos"]
            tgt_r = hprops["rvir"]

            print(tgt_r)

            # stars = read_zoom_stars(sim, snap, gid)
            print("loading stars")
            stars = read_part_ball_NCdust(
                sim,
                snap,
                tgt_pos,
                radial_bins[-1] * tgt_r,
                tgt_fields=["mass", "birth_time", "metallicity", "pos"],
                fam=2,
            )

            ages = stars["age"]
            Zs = stars["metallicity"]

            st_masses = sfhs.correct_mass(hagn_sim, ages, stars["mass"], Zs)

            print("loading dm")
            dms = read_part_ball_NCdust(
                sim,
                snap,
                tgt_pos,
                radial_bins[-1] * tgt_r,
                tgt_fields=["mass", "pos"],
                fam=1,
            )

            dm_masses = dms["mass"]

            # print(np.log10(dm_masses.sum()))

            # star_pos, ctr_stars, extent_stars = find_star_ctr_period(stars["pos"])

            unit_d = sim.unit_d(aexp)

            print("loading gas")
            code_cells = gas_pos_rad(
                sim, snap, ["density", "temperature"], tgt_pos, tgt_r * radial_bins[-1]
            )
            # code_cells = []

            cells = code_to_cgs(sim, aexp, code_cells)

            # print(cells)

            l_hagn_cm_comov = l_hagn * aexp * 1e6 * ramses_pc
            volumes = (2 ** -cells["ilevel"] * l_hagn_cm_comov) ** 3

            cell_masses = cells["density"] * volumes / msun_to_g

            # print(stars.keys(), dms.keys(), cells.keys())

            print("computing distances")

            cell_dist_to_ctr = np.linalg.norm(
                np.transpose([cells["x"], cells["y"], cells["z"]]) - tgt_pos, axis=1
            )
            stars_dist_to_ctr = np.linalg.norm(stars["pos"] - tgt_pos, axis=1)

            dm_dist_to_ctr = np.linalg.norm(dms["pos"] - tgt_pos, axis=1)

            print("looping over radial bins")

            for ir, (rmin, rmax) in enumerate(zip(radial_bins[:-1], radial_bins[1:])):

                if rmax * tgt_r < 1 / 2 ** cells["ilevel"].max():
                    # print("skipping", rmax * tgt_r, 1 / 2 ** cells["ilevel"].max())
                    continue

                cells_in_rbin = cell_dist_to_ctr / tgt_r < rmax

                st_in_rbin = stars_dist_to_ctr / tgt_r < rmax
                dm_in_rbin = dm_dist_to_ctr / tgt_r < rmax

                st_mass_rbin = st_masses[st_in_rbin].sum()
                dm_mass_rbin = dm_masses[dm_in_rbin].sum()
                cell_mass_rbin = cell_masses[cells_in_rbin].sum()

                print(st_mass_rbin, dm_mass_rbin, cell_mass_rbin)

                has_gas = cell_mass_rbin > 0
                has_stars = st_mass_rbin > 0
                has_dm = dm_mass_rbin > 0

                if has_gas:
                    mG_zoom[istep, ir] = cell_mass_rbin

                    meanT_zoom[istep, ir] = np.average(
                        cells["temperature"][cells_in_rbin],
                        weights=cell_masses[cells_in_rbin],
                    )

                tot_mass = st_mass_rbin + dm_mass_rbin + cell_mass_rbin

                if tot_mass > 0:
                    fractST_zoom[istep, ir] = st_mass_rbin / tot_mass
                    fractG_zoom[istep, ir] = cell_mass_rbin / tot_mass
                    fractBaryon_zoom[istep, ir] = (
                        st_mass_rbin + cell_mass_rbin
                    ) / tot_mass

                # print(ir, rmin, rmax, st_mass_rbin, dm_mass_rbin, cell_mass_rbin)

            time_zoom[istep] = time

            break

        with h5py.File(dust_file, "w") as f:

            f.create_dataset("mG_zoom", data=mG_zoom, compression="lzf")
            f.create_dataset("meanT_zoom", data=meanT_zoom, compression="lzf")
            f.create_dataset("fracST_zoom", data=fractST_zoom, compression="lzf")
            f.create_dataset("fracG_zoom", data=fractG_zoom, compression="lzf")
            f.create_dataset(
                "fracBaryon_zoom", data=fractBaryon_zoom, compression="lzf"
            )
            f.create_dataset("time_zoom", data=time_zoom, compression="lzf")
            f.create_dataset("radial_bins", data=radial_bins, compression="lzf")

    if sim == None:
        sim = ramses_sim(sdir, nml="cosmo.nml")

    omm = sim.cosmo["Omega_m"]
    omb = sim.cosmo["Omega_b"]

    ombaryon = omb / omm

    fractDM_zoom = 1 - fractST_zoom - fractG_zoom

    nonzero = fractBaryon_zoom[istep] > 0

    # ax[3].axvline(1.0 / 2 ** sim.namelist["amr_params"]["levelmax"], color="k", ls="--")

    # ax[0].plot(radial_bins[:-1], mG_zoom[istep], ls=zoom_ls[isim])
    ax[0].plot(
        radial_bins[nonzero], fractG_zoom[istep, nonzero], ls=zoom_ls[isim], lw=1.5
    )
    ax[1].plot(
        radial_bins[nonzero], fractST_zoom[istep, nonzero], ls=zoom_ls[isim], lw=1.5
    )
    # ax[1].plot(radial_bins[nonzero], fractDM_zoom[istep,nonzero], ls=zoom_ls[isim], lw=1.5)
    ax[2].plot(
        radial_bins[nonzero], fractBaryon_zoom[istep, nonzero], ls=zoom_ls[isim], lw=3
    )
    ax[2].axhline(ombaryon, color="k", ls="--")
    ax[3].plot(
        radial_bins[nonzero],
        fractBaryon_zoom[istep, nonzero] / ombaryon,
        ls=zoom_ls[isim],
        lw=3,
    )

    # ax[2].plot(radial_bins, meanT_zoom[-1], ls=zoom_ls[isim])

    # print((1 - fractBaryon_zoom))

# r"r/$\rm{r_{50}}$"

ax[0].set_ylabel("Gas fraction")
ax[1].set_ylabel("Star fraction")
ax[2].set_ylabel("Baryon fraction")
ax[3].set_ylabel("Baryon fraction/Cosmic baryon fraction")

ax[-1].legend(
    lines,
    labels,
    framealpha=0.0,
    ncol=3,
)
# ax[-1].axis("off")
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

# ax[0].set_xlim(xmin, xmax)

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

ax[3].tick_params(
    axis="both",
    which="both",
    bottom=True,
    top=False,
    left=True,
    right=True,
    direction="in",
)

for a in ax:
    # a.set_yscale("log")
    a.set_xscale("log")
    a.grid()

# ax[2].set_xlabel(r"r/$\rm{r_{50}}$")
ax[2].set_xlabel(r"r/$\rm{r_{vir}}$")
# ax[3].set_xlabel(r"r/$\rm{r_{50}}$")
ax[3].set_xlabel(r"r/$\rm{r_{vir}}$")


xlim = ax[-1].get_xlim()
ax[-1].set_xlim(xlim)

# ax[0].set_ylabel("Gas Mass [$M_{\odot}$]")
# ax[1].set_ylabel("Mass fraction")

# ax[2].set_ylabel("Max temperature [K]")
# ax[2].set_xlabel("Time [Gyr]")

# y2 = ax[0].twiny()
# y2.set_xlim(xlim)
# # y2.set_xticks(ax.get_xticks())
# y2.set_xticklabels([f"{cosmo.age(zed).value:.1f}" for zed in ax[-1].get_xticks()])

fig.savefig(f"profiles.png")
