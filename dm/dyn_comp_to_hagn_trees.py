from zoom_analysis.stars import sfhs

# from zoom_analysis.zoom_helpers import decentre_coordinates
# from zoom_analysis.stars.star_reader import yt_read_star_ball, read_part_ball

# from zoom_analysis.halo_maker.read_treebricks import (
#     # read_brickfile,
#     # convert_brick_units,
#     # convert_star_units,
#     # read_zoom_stars,
# )

from zoom_analysis.trees.tree_reader import read_tree_file_rev as read_tree_fev_sim
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import (
    # find_zoom_tgt_gal,
    # find_zoom_tgt_halo,
    get_central_gal_for_hid,
    get_gal_assoc_file,
    find_snaps_with_gals,
    get_halo_props_snap,
    compute_r200,
)

from zoom_analysis.stars.dynamics import extract_nh_kinematics

from zoom_analysis.zoom_helpers import find_starting_position

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


import os
import numpy as np
import h5py

from hagn.association import gid_to_stars
from hagn.utils import get_hagn_sim, adaptahop_to_code_units
from hagn.tree_reader import read_tree_rev, interpolate_tree_position
from hagn.catalogues import get_halos_cat, make_super_cat, get_cat_hids, get_cat_gids

from astropy.cosmology import Planck18 as cosmo

from hagn.catalogues import make_super_cat, get_cat_hids

from f90_tools.star_reader import read_part_ball_hagn, read_part_ball_NCdust


from zoom_analysis.halo_maker.read_treebricks import (
    read_zoom_brick,
    get_halos_properties,
)

# sim_ids = [
#     "id2757_995pcnt_mDM/",
#     "id19782_500pcnt_mDM/",pip install setuptools
#     "id13600_005pcnt_mDM/",
# ]
# sdir = "/home/jlewis/sims/proposal_zooms/ExtractZoom/mh1e12/"
hagn_snap = 197  # z=2 in the full res box
HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
delta_t = 100  # Myr

sdirs = [
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_coreen",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130_nh2",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_pdrag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_high_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_high_thermal_eff",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE_highAGNtherm",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_noAGN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
]


# setup plot
fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
ax = np.ravel(axs)

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

overwrite = False
plot_hagn = False

zoom_ls = ["-", "--", ":", "-.", (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10))] * 5
lines = []
labels = []


vrot_max = -np.inf
vrot_min = np.inf

sigma_max = -np.inf
sigma_min = np.inf

mDM_max = -np.inf
mDM_min = np.inf

xmin = np.inf

last_simID = None
l = None

zoom_style = 0

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

    if plot_hagn:
        mDM_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
        vrot_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
        sigma_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
        time_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)

    mDM_zoom = np.zeros(nsteps, dtype=np.float32)
    vrot_zoom = np.zeros(nsteps, dtype=np.float32)
    sigma_zoom = np.zeros(nsteps, dtype=np.float32)
    time_zoom = np.zeros(nsteps, dtype=np.float32)

    datas_path = os.path.join(sim.path, "computed_data")
    fout_h5 = os.path.join(datas_path, "DM_dynamics.h5")
    read_existing = True

    if not os.path.exists(datas_path):
        os.makedirs(datas_path)
        read_existing = False
    else:
        if not os.path.exists(fout_h5):
            read_existing = False

    if read_existing:

        # read the data
        with h5py.File(fout_h5, "r") as f:
            if plot_hagn:
                saved_mDM_hagn = f["mDM_hagn"][:]
                saved_vrot_hagn = f["vrot_hagn"][:]
                saved_sigma_hagn = f["sigma_hagn"][:]
                saved_time_hagn = f["time_hagn"][:]

            saved_mDM_zoom = f["mDM_zoom"][:]
            saved_vrot_zoom = f["vrot_zoom"][:]
            saved_sigma_zoom = f["sigma_zoom"][:]
            saved_time_zoom = f["time_zoom"][:]

        if saved_time_zoom.max() >= sim_times.max():
            # if we have the same number of values
            # then just read the data
            mDM_zoom = saved_mDM_zoom
            vrot_zoom = saved_vrot_zoom
            sigma_zoom = saved_sigma_zoom
            time_zoom = saved_time_zoom

            if plot_hagn:

                mDM_hagn = saved_mDM_hagn
                vrot_hagn = saved_vrot_hagn
                sigma_hagn = saved_vrot_hagn
                time_hagn = saved_time_hagn

        else:  # otherwise we need to recompute because something has changed
            read_existing = False

    # print(read_existing, len(saved_mDM_zoom), len(sim_snaps))

    if not read_existing or overwrite:

        # find last output with assoc files
        # assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
        # assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

        assoc_file_nbs = find_snaps_with_gals(sim_snaps, sim.path)

        avail_aexps = np.intersect1d(
            sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
        )
        avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

        if plot_hagn:
            assert False, "hagn dm dynamics not yet implemented"

        #     # hagn tree loop
        #     for istep, (aexp, time) in enumerate(zip(hagn_tree_aexps, hagn_tree_times)):

        #         hagn_snap = hagn_snaps[np.argmin(np.abs(hagn_aexps - aexp))]

        #         try:
        #             super_cat = make_super_cat(
        #                 hagn_snap, outf="/data101/jlewis/hagn/super_cats"
        #             )  # , overwrite=True)
        #         except FileNotFoundError:
        #             print("No super cat")
        #             continue

        #         gal_pties = get_cat_hids(super_cat, [int(hagn_tree_hids[0][istep])])

        #         # # gal_pties = get_cat_gids(super_cat, [int(hagn_tree_hids[0][istep])])

        #         if len(gal_pties["gid"]) == 0:
        #             print("No galaxy")
        #             continue

        #         # dm_ball_pos = [gal_pties["x"][0], gal_pties["y"][0], gal_pties["z"][0]]
        #         # dm_ball_pos =
        #         dm_ball_rad = gal_pties["rvir"]

        #         hagn_file = os.path.join(sim.path, f"stellar_history_{hagn_snap}.h5")

        #         dm_parts = read_part_ball_hagn(
        #             hagn_sim, hagn_snap, dm_ball_pos, dm_ball_rad, fam=1
        #         )

        #         vrot, disp = extract_nh_kinematics(
        #             masses,
        #             dm_parts["pos"],
        #             dm_parts["vel"],
        #             [gal_pties["x"][0], gal_pties["y"][0], gal_pties["z"][0]],
        #         )

        #         mDM_hagn[istep] = dm_parts["mass"].sum()
        #         vrot_hagn[istep] = vrot
        #         sigma_hagn[istep] = disp
        #         time_hagn[istep] = time

        #         # # sim_snap = sim.get_closest_snap(aexp=aexp)

        #         # print(aexp, avail_aexps)

        hid_start, halo_dict, start_hosted_gals, found, start_aexp = (
            find_starting_position(
                sim,
                avail_aexps,
                hagn_tree_aexps,
                hagn_tree_datas,
                hagn_tree_times,
                avail_times,
            )
        )

        if not found:
            continue

        # load sim tree
        sim_halo_tree_rev_fname = os.path.join(
            sim.path, "TreeMakerDM_dust", "tree_rev.dat"
        )
        if not os.path.exists(sim_halo_tree_rev_fname):
            print("No tree file")
            continue

        sim_zeds = 1.0 / avail_aexps - 1

        sim_tree_hids, sim_tree_datas, sim_tree_aexps = read_tree_fev_sim(
            sim_halo_tree_rev_fname,
            fbytes=os.path.join(sim.path, "TreeMakerDM_dust"),
            zstart=1.0 / start_aexp - 1.0,
            tgt_ids=[hid_start],
            star=False,
        )
        # print(sim_tree_hids)

        print(sim_tree_datas["m"])

        sim_tree_times = sim.cosmo_model.age(1.0 / sim_tree_aexps - 1).value * 1e3

        # # zoom loop
        for istep, (snap, aexp, time) in enumerate(
            zip(sim_snaps, sim_aexps, sim_times)
        ):

            zed = 1.0 / aexp - 1.0

            if np.all(np.abs(avail_aexps - aexp) > 1e-1):
                print(f"No assoc file for at z={1./aexp-1:.1f},snap={snap:d}")
                continue

            if not os.path.exists(get_gal_assoc_file(sim.path, snap)):
                continue

            sim_tree_arg = np.argmin(np.abs(sim_tree_aexps - aexp))
            cur_snap_hid = sim_tree_hids[0][sim_tree_arg]
            if cur_snap_hid in [0, -1]:
                continue
            # print(cur_snap_hid)

            # halo_props, _ = get_halo_props_snap(sim.path, snap, cur_snap_hid)

            # try:
            #     gid, gal_dict = get_central_gal_for_hid(sim, cur_snap_hid, snap)
            # except IndexError:
            #     print(f"No central galaxy at z={1./aexp-1:.1f},snap={snap:d}")
            #     continue

            # dm_ball_pos = halo_props['pos']

            treebrick = read_zoom_brick(
                snap, sim, "HaloMaker_DM_dust", galaxy=False, star=False
            )
            halo_props = get_halos_properties([cur_snap_hid], treebrick)

            # print(halo_props['positions'])

            dm_ball_pos = np.asarray(
                [
                    halo_props["positions"]["x"],
                    halo_props["positions"]["y"],
                    halo_props["positions"]["z"],
                ]
            )

            dm_ball_rad = halo_props["virial properties"]["rvir"]

            # print(dm_ball_pos,dm_ball_rad)

            # mvir_kg = halo_props['mvir'] * 2.0e30 #msun to kg
            # dm_ball_rad = compute_r200(sim.cosmo.H0_SI, sim.cosmo.Omega_m, sim.cosmo.Omega_b, zed, mvir_kg) #physical m
            # dm_ball_rad = (dm_ball_rad * 1e2)*sim.unit_d(aexp)/aexp #code units

            dm_parts = read_part_ball_NCdust(sim, snap, dm_ball_pos, dm_ball_rad, fam=1)

            vrot, disp = extract_nh_kinematics(
                dm_parts["mass"],
                dm_parts["pos"],
                dm_parts["vel"],
                dm_ball_pos,
            )

            mDM_zoom[istep] = dm_parts["mass"].sum()
            vrot_zoom[istep] = vrot
            sigma_zoom[istep] = disp
            time_zoom[istep] = time

            # print(f"snap {snap:d}, a={aexp:.5f}, mDM={mDM_zoom[istep]:.1e}")

            # after plot save the data to h5py file
        with h5py.File(fout_h5, "w") as f:
            if plot_hagn:
                f.create_dataset("mDM_hagn", data=mDM_hagn, compression="lzf")
                f.create_dataset("vrot_hagn", data=vrot_hagn, compression="lzf")
                f.create_dataset("sigma_hagn", data=sigma_hagn, compression="lzf")
                f.create_dataset("time_hagn", data=time_hagn, compression="lzf")

            f.create_dataset("mDM_zoom", data=mDM_zoom, compression="lzf")
            f.create_dataset("vrot_zoom", data=vrot_zoom, compression="lzf")
            f.create_dataset("sigma_zoom", data=sigma_zoom, compression="lzf")
            f.create_dataset("time_zoom", data=time_zoom, compression="lzf")

    if plot_hagn:
        stab_hagn = vrot_hagn / sigma_hagn
    stab_zoom = vrot_zoom / sigma_zoom

    if np.any(mDM_zoom > 0):  # np.any(mDM_hagn > 0) and
        mDM_max = np.max(
            [
                mDM_max,
                # mDM_hagn[mDM_hagn > 0].max(),
                mDM_zoom[mDM_zoom > 0].max(),
            ]
        )
        mDM_min = np.min(
            [
                mDM_min,
                # mDM_hagn[mDM_hagn > 0].min(),
                mDM_zoom[mDM_zoom > 0].min(),
            ]
        )
        vrot_max = np.max(
            [
                vrot_max,
                #   vrot_hagn[mDM_hagn > 0].max(),
                vrot_zoom[mDM_zoom > 0].max(),
            ]
        )
        vrot_min = np.min(
            [
                vrot_min,
                #  vrot_hagn[mDM_hagn > 0].min(),
                vrot_zoom[mDM_zoom > 0].min(),
            ]
        )
        sigma_max = np.max(
            [
                sigma_max,
                # np.nanmax(sigma_hagn[mDM_hagn > 0]),
                np.nanmax(sigma_zoom[mDM_zoom > 0]),
            ]
        )
        sigma_min = np.min(
            [
                sigma_min,
                # np.nanmin(sigma_hagn[mDM_hagn > 0]),
                np.nanmin(sigma_zoom[mDM_zoom > 0]),
            ]
        )

    if time_zoom[np.min(np.where(mDM_zoom > 0))] < xmin:
        xmin = time_zoom[np.min(np.where(mDM_zoom > 0))]

    color = None
    if last_simID == intID:
        color = l[0].get_color()
        zoom_style += 1
    else:
        zoom_style = 0

    if zoom_style == 0 and plot_hagn:
        ax[0].plot(
            time_hagn / 1e3,
            mDM_hagn,
            label="HAGN",
            color=color,
            lw=1.0,
            ls="-",
        )

        ax[1].plot(
            time_hagn / 1e3,
            np.abs(vrot_hagn),
            color=color,
            lw=1.0,
            ls="-",
        )

        ax[2].plot(
            time_hagn / 1e3,
            sigma_hagn,
            color=color,
            lw=1.0,
            ls="-",
        )

        ax[3].plot(
            time_hagn / 1e3,
            np.abs(stab_hagn),
            color=color,
            lw=1.0,
            ls="-",
        )

    # print(f"zoom style is {zoom_ls[zoom_style]}")
    l = ax[0].plot(
        time_zoom / 1e3,
        mDM_zoom,
        label="zoom",
        color=color,
        lw=2.5,
        ls=zoom_ls[zoom_style],
    )

    ax[1].plot(
        time_zoom / 1e3,
        np.abs(vrot_zoom),
        color=color,
        lw=2.5,
        ls=zoom_ls[zoom_style],
    )

    ax[2].plot(
        time_zoom / 1e3,
        sigma_zoom,
        color=color,
        lw=2.5,
        ls=zoom_ls[zoom_style],
    )

    ax[3].plot(
        time_zoom / 1e3,
        np.abs(stab_zoom),
        color=color,
        lw=2.5,
        ls=zoom_ls[zoom_style],
    )

    labels.append(sim.name)
    lines.append(l[0])

    # print(sim.name, intID, last_simID)
    # print(zoom_style, zoom_ls[zoom_style], l[0].get_color())

    last_simID = intID


# if np.isfinite(mDM_min) and np.isfinite(mDM_max):
#     ax[0].set_ylim(mDM_min * 0.5, mDM_max * 1.5)
# if np.isfinite(vrot_min) and np.isfinite(vrot_max):
#     ax[1].set_ylim(vrot_min * 0.5, vrot_max * 1.5)
# if np.isfinite(sigma_min) and np.isfinite(sigma_max):
#     ax[2].set_ylim(sigma_min * 0.5, sigma_max * 1.5)

ax[0].set_yscale("log")
ax[0].set_ylabel("DM mass [$M_\odot$]")

ax[1].set_ylabel("abs(Vrot) [km/s]")
ax[2].set_ylabel("Sigma [km/s]")
ax[3].set_ylabel("abs(Vrot/Sigma)")

ax[2].set_xlabel("Time [Gyr]")
ax[3].set_xlabel("Time [Gyr]")

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
if min(xlim) < xmin / 1e3:
    xlim = (xmin / 1e3, xlim[1])

ax[0].text(xlim[0] + 0.1, 1.1e-5, "quenched", color="k", alpha=0.33, ha="left")
# ax[0].set_ylim(4e-6, 2e-2)
ylim = ax[0].get_ylim()
ax[0].fill_between(xlim, 1e-5, ylim[0], color="k", alpha=0.33)
ax[0].set_ylim(ylim)

for a in ax:
    a.set_xlim(xlim)
    a.grid()


y2 = ax[0].twiny()
y2.set_xlim(xlim)
zlabels = [f"{cosmo.age(zed).value:.1f}" for zed in ax[0].get_xticks()]
# y2.set_xticks(ax[0].get_xticks())
y2.set_xticklabels(zlabels)
y2.set_xlabel("redshift")

y2 = ax[2].twiny()
y2.set_xlim(xlim)
zlabels = [f"{cosmo.age(zed).value:.1f}" for zed in ax[0].get_xticks()]
# y2.set_xticks(ax[0].get_xticks())
y2.set_xticklabels(zlabels)
y2.set_xlabel("redshift")


lines.extend(
    [
        Line2D([0], [0], color="k", linestyle="-", lw=1),
        Line2D([0], [0], color="k", ls="-", lw=2),
    ]
)
labels.extend(["HAGN", "zoom"])

ax[-1].legend(lines, labels, framealpha=0.0, ncol=2)  # , handlelength=3)

fig.savefig(f"dyn_DM_compHAGN_trees.png")
