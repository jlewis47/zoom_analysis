from astropy.units.cgs import g
from matplotlib.lines import Line2D
from gremlin.read_sim_params import ramses_sim

from zoom_analysis.sinks.sink_reader import (
    get_sink_mhistory,
    find_massive_sink,
    read_sink_bin,
    snap_to_coarse_step,
    check_if_superEdd,
    hid_to_sid,
)
from zoom_analysis.halo_maker.assoc_fcts import (
    find_zoom_tgt_halo,
    get_central_gal_for_hid,
    get_gal_props_snap,
    find_snaps_with_halos,
)

from zoom_analysis.zoom_helpers import (
    starting_hid_from_hagn,
)

from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt
import os
import numpy as np
from zoom_analysis.constants import ramses_pc
from zoom_analysis.zoom_helpers import decentre_coordinates, find_starting_position
from zoom_analysis.trees.tree_reader import read_tree_file_rev as read_tree_file_rev_sim

from hagn.utils import get_hagn_sim

import h5py

# from hagn.IO import read_hagn_snap_brickfile, get_hagn_brickfile_stpids
from hagn.tree_reader import read_tree_rev, interpolate_tree_position
from hagn.catalogues import make_super_cat, get_cat_hids


hagn_snap = 197  # z=2 in the full res box
# HAGN_gal_dir = f"/data40b/Horizon-AGN/TREE_STARS/GAL_{hagn_snap:05d}"

nsmooth = 30

sim_paths = [
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_coreen",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_DynBondiGravSinkMass",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowSFE_gravSink",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowSFE_DynBondiGravSinkMass",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_kicks",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_smooth_ref",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_boostNH",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_boostNH_resimBoostFriction",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_frcAccrt",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_lowerSFE_drag",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_early_refine",
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
    # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_drag",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_lowerSFE_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_accBoost",
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
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",  # _leastcoarse",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE",  # _leastcoarse",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_lowerSFE_NHboost",  # _leastcoarse",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099",
    # "/data102/jlewis/sims/lvlmax_21/mh1e12/id180130_model6_eps0p05",
    # "/data103/jlewis/sims/lvlmax_21/mh1e12/id180130_256",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_noHydroDragB4z4merge",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
]

fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

hagn_sim = get_hagn_sim()
hagn_sim.init_cosmo()
super_cat = make_super_cat(197, "hagn")  # , overwrite=True)

l_hagn = hagn_sim.unit_l(hagn_sim.aexp_stt) / (ramses_pc * 1e6) / hagn_sim.aexp_stt
# l_hagn = hagn_sim.unit_l(1.0 / tgt_zed - 1.0) / (ramses_pc * 1e6)  # / hagn_sim.aexp_stt
# print(l_hagn)

overwrite = False
# overwrite = True

zstt = 2.0

done_hagn_ids = []
done_hagn_colors = []
lines = []
labels = []

lss = [
    "-",
    "--",
    ":",
    "-.",
    (0, (1, 10)),
    (0, (3, 5, 1, 5)),
    (0, (1, 2, 1, 2, 5, 3)),
] * 15

zmin_plot = 10
zmax_plot = 0


for isim, sim_path in enumerate(sim_paths):

    # print(sim_path)

    sim_name = sim_path.split("/")[-1]
    sdir = sim_path.split(sim_name)[0]

    print(sim_name)

    sim_id = int(sim_name[2:].split("_")[0])

    # print(sim_id, done_hagn_ids, not sim_id in done_hagn_ids)

    gal_pties = get_cat_hids(super_cat, [sim_id])

    # sim_path = os.path.join(sdir, sim_name)

    sim = ramses_sim(sim_path, nml="cosmo.nml")

    if not hasattr(sim, "cosmo_model"):
        sim.init_cosmo()

    snap = sim.get_closest_snap(zed=zstt)
    snap_aexp = sim.get_snap_exps(snap)

    avail_snaps = find_snaps_with_halos(sim.snap_numbers, sim.path)
    avail_aexps = sim.get_snap_exps(param_save=False)
    avail_times = sim.get_snap_times(param_save=False)

    zmin_plot = min(zmin_plot, 1.0 / avail_aexps.max() - 1.0)
    zmax_plot = max(zmax_plot, 1.0 / avail_aexps.min() - 1.0)

    datas_path = os.path.join(sim.path, "computed_data")
    fout_h5 = os.path.join(datas_path, f"central_bh_history.h5")
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
            saved_zeds_zoom = f["zeds_zoom"][:]
            saved_mass_zoom = f["mass_zoom"][:]
            saved_mdot_zoom = f["mdot_zoom"][:]
            saved_mdot_edd_zoom = f["mdot_edd_zoom"][:]

            saved_zeds_hagn = f["zeds_hagn"][:]
            saved_mass_hagn = f["mass_hagn"][:]
            saved_mdot_hagn = f["mdot_hagn"][:]
            saved_mdot_edd_hagn = f["mdot_edd_hagn"][:]

        halo_snaps = find_snaps_with_halos(sim.snap_numbers, sim.path)

        sim_snap = halo_snaps[-2]

        last_sink_file = snap_to_coarse_step(sim_snap, sim)
        aexp_last_sink_file = read_sink_bin(
            os.path.join(sim.sink_path, f"sink_{last_sink_file:05d}.dat"),
        )["aexp"]
        if 1.0 / aexp_last_sink_file - 1.0 >= saved_zeds_zoom.min():

            zeds_zoom = saved_zeds_zoom
            mass_zoom = saved_mass_zoom
            mdot_data_zoom = saved_mdot_zoom
            mdot_edd_data_zoom = saved_mdot_edd_zoom

            zeds_hagn = saved_zeds_hagn
            mass_hagn = saved_mass_hagn
            mdot_data_hagn = saved_mdot_hagn
            mdot_edd_data_hagn = saved_mdot_edd_hagn
        else:
            read_existing = False

    if not read_existing or overwrite:

        pos = np.asarray(
            [
                sim.namelist["refine_params"]["xzoom"],
                sim.namelist["refine_params"]["yzoom"],
                sim.namelist["refine_params"]["zzoom"],
            ]
        )

        hagn_tree_hids, hagn_tree_datas, hagn_tree_aexps = read_tree_rev(
            zstt,
            gal_pties["hid"],
            tree_type="halo",
            target_fields=["m", "x", "y", "z", "r"],
        )

        for key in hagn_tree_datas:
            hagn_tree_datas[key] = hagn_tree_datas[key][0][:]

        hagn_tree_times = (
            hagn_sim.cosmo_model.age(1.0 / hagn_tree_aexps - 1).value * 1e3
        )  # Myr

        # aexp = 1.0 / (tgt_zed + 1.0)

        # cur_snap = sim.snaps[-1]
        # cur_aexp = sim.aexps[-1]

        # # get hagn main branch galaxy id at this time
        snap_time = sim.cosmo_model.age(1.0 / snap_aexp - 1).value * 1e3  # Myr

        tree_arg = np.argmin(np.abs(snap_aexp - hagn_tree_aexps))

        hagn_aexp = hagn_tree_aexps[tree_arg]
        hagn_time = hagn_tree_times[tree_arg]

        hagn_ctr, hagn_rvir = interpolate_tree_position(
            hagn_time,
            hagn_tree_times,
            hagn_tree_datas,
            l_hagn * hagn_aexp,
        )
        if hagn_ctr is None:
            continue

        # massive_sink = find_massive_sink(
        #     # pos, snap, sim, rmax=sim.namelist["refine_params"]["rzoom"]
        #     # pos,
        #     hagn_ctr,
        #     snap,
        #     sim,
        #     # rmax=0.3 * sim.namelist["refine_params"]["rzoom"],
        #     rmax=hagn_rvir,
        # )

        # if len(massive_sink) == 0:
        #     continue

        # sid = massive_sink["identity"]

        # print(sid)

        # central_sink = get_sink_mhistory(
        #     sid,
        #     snap,
        #     sim,
        # )

        # print(massive_sink)
        # zeds = central_sink["zeds"]

        # times = sim.cosmo_model.age(zeds).value * 1e3  # Myr

        # print(list(zip(central_sink["mass"], central_sink["zeds"])))

        # (l,) = ax.plot(central_sink["zeds"], central_sink["mass"], label="", lw=3, ls="--")

        # gal_pties = get_tgt_HAGN_pties(hids=[intID])

        # now get equivalent for hagn
        hagn_snap = hagn_sim.snap_numbers[
            np.argmin(np.abs(hagn_sim.get_snap_exps(param_save=False) - snap_aexp))
        ]
        # hagn_snap = hagn_sim.get_closest_snap(zed=tgt_zed)
        # # hagn_gals = read_hagn_snap_brickfile(hagn_snap, hagn_sim)
        # pos, rvir, _ = get_hagn_brickfile_stpids(
        #     f"/data40b/Horizon-AGN/TREE_STARS/tree_bricks{hagn_snap:03d}",
        #     gal_pties["gid"],
        #     sim,
        # )

        # print(pos, rvir)
        # hagn_massive_sid = find_massive_sink(pos, hagn_snap, hagn_sim, rmax=rvir * 2)[

        found = False

        rsearch = hagn_rvir * 0.1

        while not found and (rsearch < hagn_rvir):

            try:
                hagn_massive_sid = find_massive_sink(
                    hagn_ctr, hagn_snap, hagn_sim, rmax=rsearch, hagn=True
                )["identity"]
                found = hagn_massive_sid != []
                # print(found, rsearch,hagn_massive_sid)
                
            except ValueError:
                pass
            if not found:
                rsearch*=1.5

        if not found:
            print(f"no massive sink found in hagn for halo {sim.name.split('_')[0]}")
            continue

        print(f"hagn massive sink has id: {hagn_massive_sid}")
        hagn_sink_hist = get_sink_mhistory(
            hagn_massive_sid, hagn_snap, hagn_sim, hagn=True
        )

        print(hagn_sink_hist.keys())

        # print(list(zip(hagn_sink_hist["zeds"], hagn_sink_hist["mass"])))

        zeds_hagn = hagn_sink_hist["zeds"]

        if not hasattr(hagn_sim, "cosmo_model"):
            hagn_sim.init_cosmo()

        times = hagn_sim.cosmo_model.age(zeds_hagn).value * 1e3  # Myr

        mass_hagn = hagn_sink_hist["mass"]

        if "dMsmbhdt_coarse" in hagn_sink_hist.keys():
            mdot_data_hagn = hagn_sink_hist["dMsmbhdt_coarse"]
        else:
            mdot_data_hagn = hagn_sink_hist["dMBH_coarse"]
            mdot_edd_data_hagn = hagn_sink_hist["dMEd_coarse"]

            if not check_if_superEdd(hagn_sim):
                mdot_data_hagn = np.min([mdot_data_hagn, mdot_edd_data_hagn], axis=0)
            # else:
                # mdot_data_hagn = mdot_data_hagn

        # now get sim thing

        # decal = -1
        # found = False

        # while not found:

        #     # last available snap
        #     halo_snaps = find_snaps_with_halos(sim.snap_numbers, sim.path)
        #     sim_snap = halo_snaps[decal]
        #     sim_aexp = sim.get_snap_exps(sim_snap)[0]
        #     sim_time = sim.cosmo_model.age(1.0 / sim_aexp - 1).value * 1e3

        #     # # find hagn halo position
        #     sim_hagn_ctr, sim_hagn_rvir = interpolate_tree_position(
        #         sim_time, hagn_tree_times, hagn_tree_datas, l_hagn * sim_aexp
        #     )

        #     if sim_hagn_ctr is None:
        #         decal -= 1
        #         # print("position interpolation failed")
        #         continue

        #     sim_hagn_ctr = decentre_coordinates(sim_hagn_ctr, sim.path)

        #     # # find galaxy in halo
        #     # hid_zoom_tgt, hprops_zoom_tgt, hosted_gal_props = find_zoom_tgt_halo(
        #     #     sim, 1.0 / sim_aexp - 1.0, tgt_ctr=sim_hagn_ctr, tgt_rad=sim_hagn_rvir
        #     # )

        #     # gid, gprops = get_central_gal_for_hid(sim, hid_zoom_tgt, sim_snap)
        #     # rmax = gprops["rmax"]

        #     # rmax = hprops_zoom_tgt['rvir']

        #     cur_hid, hprops, hosted_gal_props = find_zoom_tgt_halo(
        #         sim, sim_snap, tgt_ctr=sim_hagn_ctr, tgt_rad=sim_hagn_rvir
        #     )

        #     # print(hosted_gal_props)

        #     if hosted_gal_props == {}:
        #         decal -= 1
        #         continue
        #     else:
        #         found = True

        # gid, gdict = get_central_gal_for_hid(sim, cur_hid, sim_snap)

        # print(cur_hid, hprops, hosted_gal_props)

        # rmax = gdict["rmax"]
        # r50 = gdict["rmax"]
        # gpos = gdict["pos"]

        # # hctr = hprops["pos"]
        # # hrvir = hprops["rvir"]

        # # hagn_massive_sid = find_massive_sink(pos, hagn_snap, hagn_sim, rmax=rvir * 2)[
        # found = False
        # rsearch = r50 * 0.1

        # while (not found) and (rsearch < rmax):
        #     print(found, rsearch)
        #     try:
        #         sim_massive_sid = find_massive_sink(
        #             gpos,
        #             sim_snap,
        #             sim,
        #             rmax=rsearch,
        #             # sim_hagn_ctr,
        #             # sim_snap,
        #             # sim,
        #             # rmax=rmax,
        #         )["identity"]
        #         found = True
        #     except ValueError:
        #         rsearch *= 1.5
        #         continue

        # if not found:
        #     print(f"no massive sink found in zoom: {sim.name}")
        #     continue

        # print(sim_snap, sim.get_closest_snap(aexp=snap_aexp))

        hid_start, halos, galaxies, start_aexp, found = starting_hid_from_hagn(
            zstt,
            sim,
            hagn_sim,
            sim_id,
            avail_aexps[:-1],
            avail_times[:-1],
            # ztgt=tgt_zed,
        )

        sim_snap = sim.get_closest_snap(aexp=start_aexp)

        sim_massive_sid, found = hid_to_sid(sim, hid_start, sim_snap, debug=True)

        print(sim_snap, hid_start, sim_massive_sid, found)

        if not found:
            print(f"no massive sink found in zoom: {sim.name}")
            continue

        print(f"zoom massive sink has id: {sim_massive_sid}")
        sim_sink_hist = get_sink_mhistory(sim_massive_sid, sim_snap, sim)

        # print(list(zip(hagn_sink_hist["zeds"], hagn_sink_hist["mass"])))

        zeds_zoom = sim_sink_hist["zeds"]

        if not hasattr(sim, "cosmo_model"):
            sim.init_cosmo()

        times = sim.cosmo_model.age(zeds_zoom).value * 1e3  # Myr

        # print(sim_sink_hist)
        mass_zoom = sim_sink_hist["mass"]

        if "dMsmbhdt_coarse" in sim_sink_hist.keys():
            min_mdot_data_zoom = sim_sink_hist["dMsmbhdt_coarse"]
        else:

            mdot_data_zoom = sim_sink_hist["dMBH_coarse"]
            mdot_edd_data_zoom = sim_sink_hist["dMEd_coarse"]
            if not check_if_superEdd(sim):
                min_mdot_data_zoom = np.min([mdot_data_zoom, mdot_edd_data_zoom], axis=0)
            else:
                min_mdot_data_zoom = mdot_data_zoom

    #     # print(mdot_data, mdot_edd_data)

    #     # print(len(mdot_data), len(mdot_edd_data))
    # super_edd = check_if_superEdd(sim)

    if not check_if_superEdd(sim):
        min_mdot_data_zoom = np.min(
            [mdot_data_zoom, mdot_edd_data_zoom], axis=0
        )  # take minimum to only have relavent rate
        # print(len(zeds), len(min_mdot_data))
    else:
        min_mdot_data_zoom = mdot_data_zoom

    # smooth using convolution
    if min_mdot_data_zoom.shape[0] > nsmooth:
        min_mdot_data_zoom = np.convolve(
            min_mdot_data_zoom, np.ones(nsmooth) / nsmooth, mode="same"
        )
    else:
        min_mdot_data_zoom = min_mdot_data_zoom

    # min_mdot_data_hagn = np.min([mdot_data_hagn, mdot_edd_data_hagn], axis=0)
    # # smooth using convolution
    # if min_mdot_data_hagn.shape[0] > nsmooth:
    #     min_mdot_data_hagn = np.convolve(
    #         min_mdot_data_hagn, np.ones(nsmooth) / nsmooth, mode="same"
    #     )
    # else:
    #     min_mdot_data_hagn = min_mdot_data_hagn

    min_mdot_data_hagn = mdot_data_hagn

    non_zero_hagn = np.where(mass_hagn > 0)[0]

    if sim_id in done_hagn_ids:
        c = done_hagn_colors[done_hagn_ids.index(sim_id)]
        # (l,) = axs[0].plot(zeds_hagn, mass_hagn, lw=1, c=c)  # , c=l.get_color())
    elif non_zero_hagn.sum() > 0:
        (l,) = axs[0].plot(
            zeds_hagn[non_zero_hagn], mass_hagn[non_zero_hagn], lw=1
        )  # , c=l.get_color())
        c = l.get_color()
        axs[1].plot(
            zeds_hagn[non_zero_hagn], min_mdot_data_hagn[non_zero_hagn], lw=1, c=c
        )
        axs[2].plot(
            zeds_hagn[non_zero_hagn],
            min_mdot_data_hagn[non_zero_hagn] / mdot_edd_data_hagn[non_zero_hagn],
            lw=1,
            c=c,
        )

    nb_times_done = int(np.sum(np.in1d(done_hagn_ids, sim_id)))
    zoom_ls = lss[nb_times_done]

    non_zero = np.where(mass_zoom > 0)[0]

    if len(non_zero) == 0:
        continue

    # sim plots
    axs[0].plot(
        zeds_zoom[non_zero], mass_zoom[non_zero], lw=2, c=c, label=sim.name, ls=zoom_ls
    )
    # min_mdot_data = np.convolve(mdot_data, np.ones(nsmooth) / nsmooth, mode="same")

    axs[1].plot(
        zeds_zoom[non_zero], min_mdot_data_zoom[non_zero], lw=2, c=c, ls=zoom_ls
    )
    axs[2].plot(
        zeds_zoom[non_zero],
        min_mdot_data_zoom[non_zero] / mdot_edd_data_zoom[non_zero],
        lw=2,
        c=c,
        ls=zoom_ls,
    )

    lvlmax = sim.namelist["amr_params"]["levelmax"]

    labels.append(sim.name + f" {lvlmax:d}")
    lines.append(Line2D([0], [0], color=c, lw=2, ls=zoom_ls))

    done_hagn_colors.append(c)
    done_hagn_ids.append(sim_id)

    if not read_existing or overwrite:

        with h5py.File(fout_h5, "w") as f:
            f.create_dataset("zeds_zoom", data=zeds_zoom)
            f.create_dataset("mass_zoom", data=mass_zoom)
            f.create_dataset("mdot_zoom", data=mdot_data_zoom)
            f.create_dataset("mdot_edd_zoom", data=mdot_edd_data_zoom)

            f.create_dataset("zeds_hagn", data=zeds_hagn)
            f.create_dataset("mass_hagn", data=mass_hagn)
            f.create_dataset("mdot_hagn", data=mdot_data_hagn)
            f.create_dataset("mdot_edd_hagn", data=mdot_edd_data_hagn)


labels.append(["HAGN", "zoom"])
lines.append(
    [
        Line2D([0, 0], [0, 0], color="k", lw=1, ls="-"),
        Line2D([0, 0], [0, 0], color="k", lw=2, ls="-"),
    ]
)

# axs[0].legend(framealpha=0.0)

axs[1].legend([], [], title=f"smoothed X{nsmooth}", framealpha=0.0)
axs[2].legend([], [], title=f"smoothed X{nsmooth}", framealpha=0.0)

axs[-1].legend(lines, labels, framealpha=0.0, ncols=2, loc="center")
axs[-1].axis("off")

# ax2.invert_xaxis()

axs[2].set_xlabel("z")

axs[0].set_ylabel("BH mass [M$_\odot$]")
axs[1].set_ylabel("Mdot [M$_\odot$/yr]")
axs[2].set_ylabel("Mdot/Mdot_Edd")

axs[2].set_ylim(
    1e-4,
)
# ax.grid()

# axs[0].invert_xaxis()
axs[0].set_xlim(10, zmin_plot)

for ax in axs[:-1]:
    ax.tick_params(direction="in", top=True, right=True)
for ax in axs[:-1]:
    ax.set_yscale("log")

axs[-2].tick_params(direction="in", top=True, right=True, bottom=True, labelbottom=True)

plt.subplots_adjust(hspace=0.0)

# second xaxis times...
ticks = axs[0].get_xticks()
axs[0].set_xticks(ticks)
ax2 = axs[0].twiny()
ax2.set_xlim(axs[0].get_xlim())
ax2.set_xticks(ticks)
ax2.set_xticklabels([f"{sim.cosmo_model.age(xtick).value:.2f}" for xtick in ticks])
ax2.set_xlabel("time [Gyr]")
ax2.tick_params(direction="in", top=True, right=True)

fig.savefig("sink_comp_one_snap")
