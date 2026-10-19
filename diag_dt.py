import matplotlib
from matplotlib.colors import LogNorm
import numpy as np

from zoom_analysis.halo_maker.assoc_fcts import (
    find_zoom_tgt_halo,
    find_snaps_with_halos,
)
from zoom_analysis.dust.gas_reader import code_to_cgs, gas_pos_rad
from compress_zoom.read_compressd import read_compressed_target, check_for_compressd

from f90_tools.star_reader import read_part_ball_NCdust

import matplotlib.pyplot as plt

import os

from gremlin.read_sim_params import ramses_sim


# sim_dirs = ["/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"]
sim_dirs = [
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
]


def get_dt_from_logs(sim_dir):

    # find logs
    log_files = [
        f for f in os.listdir(sim_dir) if f.endswith(".log") and f.startswith("run_")
    ]

    dts = []
    aexps = []

    for log_file in log_files:

        with open(os.path.join(sim_dir, log_file), "r") as src:

            for line in src:

                if "Fine" in line and "dt" in line:

                    parts = line.split(" ")

                    dt = float(parts[parts.index("dt=") + 1])
                    aexp = float(parts[parts.index("a=") + 1])

                    dts.append(dt)
                    aexps.append(aexp)

    return dts, aexps


fig, ax = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
cb = None

skip = 10

for sim_dir in sim_dirs:

    sim = ramses_sim(sim_dir)

    snaps = find_snaps_with_halos(sim.snap_numbers, sim.path)

    snaps = snaps[::skip]

    log_dts, log_aexps = np.asarray(get_dt_from_logs(sim.path))

    snap_dts = np.zeros(len(snaps))
    max_dm_vel = np.zeros(len(snaps))
    max_star_vel = np.zeros(len(snaps))
    max_cell_temp = np.zeros(len(snaps))
    max_cell_vel = np.zeros(len(snaps))
    max_cell_cs = np.zeros(len(snaps))

    sim.init_cosmo()

    for isnap, snap in enumerate(snaps):

        snap_aexp = sim.get_snap_exps(snap)
        aexp_dist = np.abs(log_aexps - snap_aexp)
        if np.min(aexp_dist) < 0.05:
            snap_dts[isnap] = log_dts[np.argmin(aexp_dist)]
        else:
            continue

        aexp = sim.get_snap_exps(snap)
        zed = 1.0 / aexp - 1

        if zed > 10:
            continue

        time = sim.cosmo_model.age(zed).value * 1e3

        tgt_hid, halo_dict, hosted_galaxies = find_zoom_tgt_halo(sim, snap)

        # if check_for_compressd(sim, snap):

        #     compressd_data = read_compressed_target(sim, snap, hid=tgt_hid, gid=None)

        #     gas_temps = compressd_data["gas"]["temp"]
        #     max_cell_temp[isnap] = np.max(np.linalg.norm(gas_temps, axis=1))

        #     star_vels = compressd_data["stars"]["vel"]
        #     max_star_vel[isnap] = np.max(np.linalg.norm(star_vels, axis=1))

        #     dm_vels = compressd_data["dm"]["vel"]
        #     max_dm_vel[isnap] = np.max(np.linalg.norm(dm_vels, axis=1))

        # else:

        hpos = halo_dict["pos"]
        hrad = halo_dict["rvir"]

        gas_data_code = gas_pos_rad(
            sim,
            snap,
            ["temperature", "velocity_x", "velocity_y", "velocity_z"],
            hpos,
            hrad,
            verbose=False,
        )
        # gas_data_code["temperature"] = []

        cs2_code = (
            (sim.namelist["hydro_params"]["gamma"] - 1)
            * gas_data_code["pressure"]
            / gas_data_code["density"]
        )
        cs_kms = np.sqrt(cs2_code) * sim.unit_v(aexp) / 1e5

        max_cell_cs[isnap] = np.max(cs_kms)

        gas_data = code_to_cgs(sim, aexp, amrdata=gas_data_code)

        max_cell_temp[isnap] = np.max(gas_data["temperature"])
        max_cell_vel[isnap] = (
            np.max(
                [gas_data["velocity_x"], gas_data["velocity_y"], gas_data["velocity_z"]]
            )
            / 1e5
        )  # km/s

        star_data = read_part_ball_NCdust(
            sim, snap, hpos, hrad, tgt_fields=["vel"], fam=2
        )
        max_star_vel[isnap] = np.max(np.linalg.norm(star_data["vel"], axis=1))

        dm_data = read_part_ball_NCdust(
            sim, snap, hpos, hrad, tgt_fields=["vel"], fam=1
        )
        max_dm_vel[isnap] = np.max(np.linalg.norm(dm_data["vel"], axis=1))

        # print(snap, max_dm_vel[isnap], max_star_vel[isnap], max_cell_temp[isnap])

    snap_aexps = sim.get_snap_exps(snaps)
    snap_zeds = 1.0 / snap_aexps - 1

    lin_dt_scale = matplotlib.colors.Normalize(
        vmin=np.log10(np.min(snap_dts)),
        vmax=np.log10(np.max(snap_dts)),
        # LogNorm(vmin=np.min(snap_dts), vmax=np.max(snap_dts))
    )
    colors = plt.cm.jet(lin_dt_scale(np.log10(snap_dts)))

    ax[0].scatter(snap_zeds, max_dm_vel, c=colors)
    ax[0].set_ylabel("max DM vel, km/s")

    ax[1].scatter(snap_zeds, max_star_vel, c=colors)
    ax[1].set_ylabel("max star vel, km/s")

    ax[2].scatter(snap_zeds, max_cell_temp, c=colors)
    ax[2].set_ylabel("max cell temp, K")

    ax[3].scatter(snap_zeds, max_cell_vel, c=colors)
    ax[3].set_ylabel("max cell velocity, km/s")

    ax[4].scatter(snap_zeds, max_cell_cs, c=colors)
    ax[4].set_ylabel("max cell sound speed, km/s")

    if cb == None:
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=lin_dt_scale, cmap=plt.cm.jet),
            ax=ax,
            label="log10(dt)",
        )

for axis in ax:

    axis.set_xlabel("zed")
    axis.grid()
    # axis.set_yscale("log")

ax[2].set_yscale("log")
ax[3].set_yscale("log")
ax[0].invert_xaxis()

# xlim = ax[0].get_xlim()
# ax[0].set_xlim(xlim)
# xticks = ax[0].get_xticks()
# aexps = np.asarray(log_aexps)[np.digitize(log_dts, xticks)]
# # top axis is snap aexps
# y2 = ax[0].twiny()
# y2.set_xticks()


fig.savefig("diag_dt.png")
