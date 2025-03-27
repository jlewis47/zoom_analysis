# from zoom_analysis.stars import sfhs

# from zoom_analysis.halo_maker.read_treebricks import (
#     # read_brickfile,
#     # convert_brick_units,
#     # convert_star_units,
#     read_zoom_stars,
# )

import matplotlib.pyplot as plt

from zoom_analysis.trees.tree_reader import read_tree_file_rev as read_tree_fev_sim
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import (
    # find_zoom_tgt_gal,
    # find_zoom_tgt_halo,
    # get_central_gal_for_hid,
    # get_gal_assoc_file,
    get_halo_assoc_file,
    get_halo_props_snap,
    find_snaps_with_halos,
    # find_snaps_with_gals,
)

from zoom_analysis.sinks.sink_reader import (
    get_sink_mhistory,
    find_massive_sink,
    read_sink_bin,
    snap_to_coarse_step,
    check_if_superEdd,
    hid_to_sid,
)

from zoom_analysis.zoom_helpers import find_starting_position

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D

import os
import numpy as np
import h5py

from hagn.association import gid_to_stars
from hagn.utils import get_hagn_sim, adaptahop_to_code_units
from hagn.tree_reader import read_tree_rev, interpolate_tree_position
from hagn.catalogues import get_halos_cat, make_super_cat, get_cat_hids, get_cat_gids

from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u

from hagn.catalogues import make_super_cat, get_cat_hids

from f90_tools.star_reader import read_part_ball_hagn, read_part_ball_NCdust


hagn_snap = 197  # z=2 in the full res box
HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
delta_t = 100  # Myr

nsmooth = 5

sdirs = [
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_coreen",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_gravSink",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_DynBondiGravSinkMass",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_fullresIC",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_kicks",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_smooth_ref",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_low_dboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_minMassDynGrav",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_vlowSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_high_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_accBoost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE_lowNsink",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highSFE_lowSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE_lowSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_pdrag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nrg_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_high_SN",
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
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_noHydroDragB4z4merge",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
]


# setup plot
fig, ax = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

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

# overwrite = True
overwrite = False

zoom_ls = [
    "-",
    "--",
    ":",
    "-.",
    (0, (1, 10)),
    (0, (3, 5, 1, 5)),
    (0, (1, 2, 1, 2, 5, 3)),
] * 15
lines = []
labels = []


xlim = -np.inf, +np.inf

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

    mbh_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
    mdot_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
    mdotEdd_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
    time_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)

    mbh_zoom = np.zeros(nsteps, dtype=np.float32)
    mdot_zoom = np.zeros(nsteps, dtype=np.float32)
    mdotEdd_zoom = np.zeros(nsteps, dtype=np.float32)
    time_zoom = np.zeros(nsteps, dtype=np.float32)

    datas_path = os.path.join(sim.path, "computed_data")
    fout_h5 = os.path.join(datas_path, f"allHalo_bh_history.h5")
    read_existing = True

    if not os.path.exists(datas_path):
        os.makedirs(datas_path)
        read_existing = False
    else:
        if not os.path.exists(fout_h5):
            read_existing = False

    if overwrite:
        read_existing = False

    if read_existing:

        # read the data
        with h5py.File(fout_h5, "r") as f:
            saved_mbh_hagn = f["mbh_hagn"][:]
            saved_mdot_hagn = f["mdot_hagn"][:]
            saved_mdotEdd_hagn = f["mdotEdd_hagn"][:]
            saved_time_hagn = f["time_hagn"][:]

            saved_mbh_zoom = f["mbh_zoom"][:]
            saved_mdot_zoom = f["mdot_zoom"][:]
            saved_mdotEdd_zoom = f["mdotEdd_zoom"][:]
            saved_time_zoom = f["time_zoom"][:]

        if saved_time_zoom.max() >= sim_times.max():
            # if we have the same number of values
            # then just read the data
            mbh_zoom = saved_mbh_zoom
            mdot_zoom = saved_mdot_zoom
            mdotEdd_zoom = saved_mdotEdd_zoom
            time_zoom = saved_time_zoom

            mbh_hagn = saved_mbh_hagn
            mdot_hagn = saved_mdot_hagn
            mdotEdd_hagn = saved_mdotEdd_hagn
            time_hagn = saved_time_hagn
        else:  # otherwise we need to recompute because something has changed
            read_existing = False

    # print(read_existing, len(saved_mbh_zoom), len(sim_snaps))

    if not read_existing:

        # find last output with assoc files
        # assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
        # assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

        assoc_file_nbs = find_snaps_with_halos(sim_snaps, sim.path)

        avail_aexps = np.intersect1d(
            sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
        )
        avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

        prev_mass = -1
        # prev_pos = None

        # hagn tree loop
        for istep, (aexp, time) in enumerate(zip(hagn_tree_aexps, hagn_tree_times)):

            hagn_snap = hagn_snaps[np.argmin(np.abs(hagn_aexps - aexp))]

            try:
                super_cat = make_super_cat(
                    hagn_snap, outf="/data101/jlewis/hagn/super_cats"
                )  # , overwrite=True)
            except FileNotFoundError:
                print("No super cat")
                continue

            gal_pties = get_cat_hids(super_cat, [int(hagn_tree_hids[0][istep])])

            if len(gal_pties["hid"]) == 0:
                continue

            # # gal_pties = get_cat_gids(super_cat, [int(hagn_tree_hids[0][istep])])

            # if len(gal_pties["gid"]) == 0:
            #     print("No galaxy")
            #     continue

            # print(gal_pties.keys())
            tgt_pos = np.asarray(
                [gal_pties["hx"][0], gal_pties["hy"][0], gal_pties["hz"][0]]
            )
            tgt_rad = gal_pties["rvir"]

            sinks = find_massive_sink(
                tgt_pos,
                hagn_snap,
                hagn_sim,
                tgt_rad,
                all_sinks=True,
                tgt_fields=["position", "mass", "dMBH_coarse", "dMEd_coarse"],
            )

            # print(sinks)

            mbh_hagn[istep] = sinks["mass"].sum()
            mdot_hagn[istep] = sinks["dMBH_coarse"].sum()
            mdotEdd_hagn[istep] = sinks["dMEd_coarse"].sum()

            time_hagn[istep] = time

            # # sim_snap = sim.get_closest_snap(aexp=aexp)

            # print(aexp, avail_aexps)

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

        # print(hagn_ctr)

        # print(hagn_ctr)

        # hid_start = int(hosted_gals["hid"][np.argmax(hosted_gals["mass"])])

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

        # print(sim_tree_datas["m"])

        # print(sim_tree_hids)
        # print(sim_tree_datas)

        sim_tree_times = sim.cosmo_model.age(1.0 / sim_tree_aexps - 1).value * 1e3

        # # zoom loop
        for istep, (snap, aexp, time) in enumerate(
            zip(sim_snaps[:-1], sim_aexps[:-1], sim_times[:-1])
        ):
            # if time < sim_tree_times.min() - 5:
            # continue

            zed = 1.0 / aexp - 1.0

            if np.all(np.abs(avail_aexps - aexp) > 1e-1):
                print(f"No assoc file for at z={1./aexp-1:.1f},snap={snap:d}")
                continue

            if not os.path.exists(get_halo_assoc_file(sim.path, snap)):
                print("no assoc file")
                continue

            # assoc_file = assoc_files[assoc_file_nbs == snap]
            # if len(assoc_file) == 0:
            #     print(f"No assoc file for at z={1./aexp-1:.1f},snap={snap:d}")
            #     continue

            sim_tree_arg = np.argmin(np.abs(sim_tree_aexps - aexp))
            cur_snap_hid = sim_tree_hids[0][sim_tree_arg]
            if cur_snap_hid in [-1, 0]:
                continue

            hprops, hgals = get_halo_props_snap(sim.path, snap, hid=cur_snap_hid)

            tgt_pos = hprops["pos"]
            tgt_r = hprops["rvir"]

            zoom_sinks = find_massive_sink(
                tgt_pos,
                snap,
                sim,
                tgt_r,
                all_sinks=True,
                tgt_fields=["position", "mass", "dMBH_coarse", "dMEd_coarse"],
            )

            if len(zoom_sinks["mass"]) == 0:
                continue

            mbh_zoom[istep] = zoom_sinks["mass"].sum()
            mdot_zoom[istep] = zoom_sinks["dMBH_coarse"].sum()
            if mdot_zoom[istep] == 0:
                print(zoom_sinks)
            mdotEdd_zoom[istep] = zoom_sinks["dMEd_coarse"].sum()

            time_zoom[istep] = time

            # print(snap, 1.0 / aexp - 1, mbh_zoom[istep])

    # print(list(zip(sim_snaps, sim_times, 1.0 / sim_aexps - 1.0, mbh_zoom)))

    # filled_pos = time_zoom > 0

    # # print(time_zoom)

    # mbh_zoom = mbh_zoom[filled_pos]
    # mdot_zoom = mdot_zoom[filled_pos]
    # mdotEdd_zoom = mdotEdd_zoom[filled_pos]
    # time_zoom = time_zoom[filled_pos]

    if not np.sum(mbh_zoom) > 0:
        print(f"{sim.name}: No zoom bh")
        continue

    hagn_SE = mdot_hagn > mdotEdd_hagn
    if np.any(hagn_SE):
        mdot_hagn[hagn_SE] = mdotEdd_hagn[hagn_SE]

    if not check_if_superEdd(sim):
        print(f"{sim.name}: No super Eddinton accretion")
        mdot_zoom[mdot_zoom > mdotEdd_zoom] = mdotEdd_zoom[mdot_zoom > mdotEdd_zoom]

    print(sim.name, sim.namelist["smbh_params"]["mseed"])

    # smoothing

    if nsmooth > 1:

        # mbh_zoom = np.convolve(mbh_zoom, np.ones(nsmooth) / nsmooth, mode="same")
        # only smooth rates, these are very noisy
        mdot_zoom = np.convolve(mdot_zoom, np.ones(nsmooth) / nsmooth, mode="same")
        mdotEdd_zoom = np.convolve(
            mdotEdd_zoom, np.ones(nsmooth) / nsmooth, mode="same"
        )

    color = None
    if last_simID == intID:
        color = l.get_color()
        zoom_style += 1
    else:
        zoom_style = 0

    if zoom_style == 0:

        print(time_hagn, mbh_hagn)

        non_zero = time_hagn > 0

        zed_hagn = z_at_value(
            sim.cosmo_model.age,
            time_hagn[non_zero] * u.Myr,
            # method="Golden",
            # bracket=[1.0 / hagn_sim.aexp_stt - 1, 1e-8],
        )

        (l,) = ax[0].plot(zed_hagn, mbh_hagn[non_zero], lw=1.0)

        ax[1].plot(zed_hagn, mdot_hagn[non_zero], lw=1.0)

        ax[2].plot(zed_hagn, mdot_hagn[non_zero] / mdotEdd_hagn[non_zero], lw=1.0)

    # print(sim.name, mbh_zoom)

    xlim = max(xlim[0], zed_hagn[zed_hagn > 0].max()), min(xlim[1], zed_hagn.min())
    print(xlim)
    # print(f"zoom style is {zoom_ls[zoom_style]}")

    # print(time_zoom, mbh_zoom)
    non_zero = time_zoom > 0

    zeds_zoom = z_at_value(
        sim.cosmo_model.age,
        time_zoom[non_zero] * u.Myr,
        # method="Golden",
        # bracket=[1.0 / sim.aexp_stt - 1, 1e-8],
    )

    xlim = max(xlim[0], zeds_zoom[zeds_zoom > 0].max()), min(xlim[1], zeds_zoom.min())

    if mbh_zoom.sum() > 0:
        (l,) = ax[0].plot(
            zeds_zoom,
            mbh_zoom[non_zero],
            lw=2.0,
            color=l.get_color(),
            ls=zoom_ls[zoom_style],
        )

        ax[1].plot(
            zeds_zoom,
            mdot_zoom[non_zero],
            lw=2.0,
            color=l.get_color(),
            ls=zoom_ls[zoom_style],
        )

        ax[2].plot(
            zeds_zoom,
            mdot_zoom[non_zero] / mdotEdd_zoom[non_zero],
            lw=2.0,
            color=l.get_color(),
            ls=zoom_ls[zoom_style],
        )

        lvlmax = sim.namelist["amr_params"]["levelmax"]

        labels.append(sim.name + f" {lvlmax}")
        lines.append(l)

        # print(sim.name, intID, last_simID)
        # print(zoom_style, zoom_ls[zoom_style], l[0].get_color())

        last_simID = intID

    # after plot save the data to h5py file
    with h5py.File(fout_h5, "w") as f:
        f.create_dataset("mbh_hagn", data=mbh_hagn, compression="lzf")
        f.create_dataset("mdot_hagn", data=mdot_hagn, compression="lzf")
        f.create_dataset("mdotEdd_hagn", data=mdotEdd_hagn, compression="lzf")
        f.create_dataset("time_hagn", data=time_hagn, compression="lzf")

        f.create_dataset("mbh_zoom", data=mbh_zoom, compression="lzf")
        f.create_dataset("mdot_zoom", data=mdot_zoom, compression="lzf")
        f.create_dataset("mdotEdd_zoom", data=mdotEdd_zoom, compression="lzf")
        f.create_dataset("time_zoom", data=time_zoom, compression="lzf")


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

ax[2].tick_params(labelbottom=True)

ax[0].set_yscale("log")
ax[0].set_ylabel("$M_{BH}, M_\odot$")
ax[1].set_yscale("log")
ax[1].set_ylabel("$\dot{M}_{BH}, M_\odot/yr$")
ax[2].set_yscale("log")
ax[2].set_ylabel("Eddington ratio")

ax[2].set_xlabel("redshift")


ax[0].set_xlim(xlim[0], xlim[1])

y2 = ax[0].twiny()
y2.set_xticks(ax[0].get_xticks())
y2.set_xlim(xlim)
zlabels = [f"{cosmo.age(zed).value:.1f}" for zed in ax[0].get_xticks()]
y2.set_xticklabels(zlabels)
y2.set_xlabel("Time, Gyr")
y2.tick_params(labeltop=True)

ax[-1].axis("off")

lines.append(
    [
        Line2D([0], [0], color="k", linestyle="-", lw=1),
        Line2D([0], [0], color="k", ls="-", lw=2),
    ]
)
labels.append(["HAGN", "zoom"])

ax[-1].legend(lines, labels, framealpha=0.0, ncol=2, loc="center")  # , handlelength=3)

fig.subplots_adjust(hspace=0.0)

fig.savefig(f"agns_comp_allsnaps_inHalo.png")
