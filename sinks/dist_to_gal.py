from matplotlib.lines import Line2D
from gremlin.read_sim_params import ramses_sim
from zoom_analysis.sinks.sink_reader import (
    get_sink_mhistory,
    find_massive_sink,
    read_sink_bin,
    snap_to_coarse_step,
    check_if_superEdd,
)
from zoom_analysis.halo_maker.assoc_fcts import (
    find_snaps_with_halos,
    get_halo_props_snap,
    find_star_ctr_period,
    # find_zoom_tgt_halo,
    # get_central_gal_for_hid,
    # get_gal_props_snap,
)

from zoom_analysis.zoom_helpers import find_starting_position, decentre_coordinates
from gremlin.read_sim_params import ramses_sim
from zoom_analysis.constants import ramses_pc
from zoom_analysis.trees.tree_reader import read_tree_file_rev as read_tree_fev_sim
from zoom_analysis.trees.tree_reader import convert_adaptahop_pos

from zoom_analysis.halo_maker.read_treebricks import read_zoom_stars

# from hagn.IO import read_hagn_snap_brickfile, get_hagn_brickfile_stpids
from hagn.tree_reader import read_tree_rev, interpolate_tree_position
from hagn.catalogues import make_super_cat, get_cat_hids
from hagn.utils import get_hagn_sim

import matplotlib.pyplot as plt
import os
import numpy as np

import h5py


tgt_zed = 2.0

hagn_snap = 197  # z=2 in the full res box
# HAGN_gal_dir = f"/data40b/Horizon-AGN/TREE_STARS/GAL_{hagn_snap:05d}"

nsmooth = 60

sim_paths = [
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_coreen",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_low_dboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_minMassDynGrav",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_vlowSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_high_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd",
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
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
]


def hagn_reffs_file(snap):
    path = "/data102/dubois/BigSimsCatalogs/H-AGN/Catalogs/DataRelease/HAGN/ReffOfGal"

    files = [f for f in os.listdir(path) if f"output_{snap:05d}" in f]
    if len(files) == 0:
        return None
    else:
        return os.path.join(path, files[0])


fig, axs = plt.subplots(5, 1, figsize=(8, 12), sharex=True)

hagn_sim = get_hagn_sim()
hagn_sim.init_cosmo()
super_cat = make_super_cat(197, "hagn")  # , overwrite=True)

l_hagn = hagn_sim.unit_l(hagn_sim.aexp_stt) / (ramses_pc * 1e6) / hagn_sim.aexp_stt
# l_hagn = hagn_sim.unit_l(1.0 / tgt_zed - 1.0) / (ramses_pc * 1e6)  # / hagn_sim.aexp_stt
# print(l_hagn)

overwrite = False

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
    (0, (3, 5, 1, 5, 1, 5)),
] * 15

xmin = np.inf
xmax = -np.inf


for isim, sim_path in enumerate(sim_paths):

    found_sim_sink = False

    sim_name = sim_path.split("/")[-1]
    sdir = sim_path.split(sim_name)[0]

    # print(sim_name)

    sim_id = int(sim_name[2:].split("_")[0])

    # print(sim_id, done_hagn_ids, not sim_id in done_hagn_ids)

    gal_pties = get_cat_hids(super_cat, [sim_id])

    # sim_path = os.path.join(sdir, sim_name)

    sim = ramses_sim(sim_path, nml="cosmo.nml")

    if not hasattr(sim, "cosmo_model"):
        sim.init_cosmo()

    snap = sim.get_closest_snap(zed=tgt_zed)
    snap_aexp = sim.get_snap_exps(snap)

    datas_path = os.path.join(sim.path, "computed_data")
    fout_h5 = os.path.join(datas_path, f"central_bh_dist.h5")
    read_existing = True

    if not os.path.exists(datas_path) or overwrite:
        if not os.path.exists:
            os.makedirs(datas_path)
        read_existing = False
    else:
        if not os.path.exists(fout_h5):
            read_existing = False

    if read_existing:

        # read the data
        with h5py.File(fout_h5, "r") as f:
            saved_zeds_zoom = f["zeds_zoom"][:]
            saved_mass_zoom = f["mass_zoom"][:]
            saved_dens_zoom = f["dens_zoom"][:]
            saved_vrel_zoom = f["vrel_zoom"][:]
            saved_gal_dist_zoom = f["gal_dist_zoom"][:]
            saved_sim_tree_aexps = f["sim_tree_aexps"][:]

            saved_zeds_hagn = f["zeds_hagn"][:]
            saved_mass_hagn = f["mass_hagn"][:]
            saved_dens_hagn = f["dens_hagn"][:]
            saved_vrel_hagn = f["vrel_hagn"][:]
            saved_gal_dist_hagn = f["gal_dist_hagn"][:]
            saved_hagn_tree_aexps = f["hagn_tree_aexps"][:]

        sim_snap = sim.snap_numbers[-2]
        last_sink_file = snap_to_coarse_step(sim_snap, sim)
        aexp_last_sink_file = read_sink_bin(
            os.path.join(sim.sink_path, f"sink_{last_sink_file:05d}.dat"),
        )["aexp"]
        if 1.0 / aexp_last_sink_file - 1.0 >= saved_zeds_zoom.min():

            zeds_zoom = saved_zeds_zoom
            mass_zoom = saved_mass_zoom
            vrel_data_zoom = saved_vrel_zoom
            dens_data_zoom = saved_dens_zoom
            gal_dist_zoom = saved_gal_dist_zoom
            sim_tree_aexps = saved_sim_tree_aexps

            zeds_hagn = saved_zeds_hagn
            mass_hagn = saved_mass_hagn
            dens_data_hagn = saved_dens_hagn
            vrel_data_hagn = saved_vrel_hagn
            gal_dist_hagn = saved_gal_dist_hagn
            hagn_tree_aexps = saved_hagn_tree_aexps

            xmin = min(xmin, zeds_zoom.min())
            xmax = max(xmax, zeds_zoom.max())

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
            tgt_zed,
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

        try:
            hagn_massive_sid = find_massive_sink(
                hagn_ctr, hagn_snap, hagn_sim, rmax=hagn_rvir * 2, hagn=True
            )["identity"]
            print(f"hagn massive sink has id: {hagn_massive_sid}")
        except ValueError:
            print(f"no massive sink found in hagn for halo {sim.name.split('_')[0]}")
            continue
        # print()

        hagn_sink_hist = get_sink_mhistory(
            hagn_massive_sid,
            hagn_snap,
            hagn_sim,
            hagn=True,
            out_keys=[("position", 3), ("dens", 1), ("vrel", 1), ("mass", 1)],
        )

        hagn_sink_aexps = 1.0 / (1 + hagn_sink_hist["zeds"])

        gal_dist_hagn = np.zeros(len(hagn_tree_hids[0]))

        for i, (hid, aexp) in enumerate(zip(hagn_tree_hids[0], hagn_tree_aexps)):

            hagn_sink_aexp_arg = np.argmin(np.abs(hagn_sink_aexps - aexp))
            if np.abs(hagn_sink_aexps[hagn_sink_aexp_arg] - aexp) > 0.005:
                print("aexp interpolation failed")
                continue

            cur_snap = hagn_sim.get_closest_snap(zed=1.0 / aexp - 1)
            cur_super_cat = make_super_cat(cur_snap, "hagn")

            super_arg = np.where(cur_super_cat["hid"] == hid)[0]

            if len(super_arg) == 0:
                continue

            elif len(super_arg) > 1:
                gal_masses = cur_super_cat["mgal"][super_arg]
                arg_max_mass = np.argmax(gal_masses)
                gal_ctr = [
                    cur_super_cat["x"][super_arg][arg_max_mass],
                    cur_super_cat["y"][super_arg][arg_max_mass],
                    cur_super_cat["z"][super_arg][arg_max_mass],
                ]
                hagn_gid = cur_super_cat["gid"][super_arg][arg_max_mass]
            else:
                gal_ctr = [
                    cur_super_cat["x"][super_arg[0]],
                    cur_super_cat["y"][super_arg[0]],
                    cur_super_cat["z"][super_arg[0]],
                ]
                hagn_gid = cur_super_cat["gid"][super_arg[0]]

            # print(gal_ctr, hagn_sink_hist["position"][hagn_sink_aexp_arg])

            gal_dist_hagn[i] = (
                np.linalg.norm(gal_ctr - hagn_sink_hist["position"][hagn_sink_aexp_arg])
                * sim.cosmo.lcMpc
                * 1e3
                * aexp
            )  # ckpc

            gal_reffs = np.genfromtxt(hagn_reffs_file(cur_snap))
            reff_gids = gal_reffs[:, 0]
            reffs_kpc = gal_reffs[:, -1]

            gid_arg = np.where(reff_gids == hagn_gid)[0]

            # print(gal_dist_hagn[i], reffs_kpc[gid_arg] * aexp)

            gal_dist_hagn[i] /= reffs_kpc[gid_arg]

        # print(list(zip(hagn_sink_hist["zeds"], hagn_sink_hist["mass"])))

        zeds_hagn = hagn_sink_hist["zeds"]

        if not hasattr(hagn_sim, "cosmo_model"):
            hagn_sim.init_cosmo()

        times = hagn_sim.cosmo_model.age(zeds_hagn).value * 1e3  # Myr

        mass_hagn = hagn_sink_hist["mass"]

        dens_data_hagn = np.convolve(
            hagn_sink_hist["dens"], np.ones(nsmooth) / nsmooth, mode="same"
        )
        vrel_data_hagn = np.convolve(
            hagn_sink_hist["vrel"], np.ones(nsmooth) / nsmooth, mode="same"
        )

        # now get sim thing

        # last available snap
        sim_snap = sim.snap_numbers[-2]
        sim_aexp = sim.get_snap_exps(sim_snap)[0]
        sim_time = sim.cosmo_model.age(1.0 / sim_aexp - 1).value * 1e3

        # # find hagn halo position
        # sim_hagn_ctr, sim_hagn_rvir = interpolate_tree_position(
        #     sim_time, hagn_tree_times, hagn_tree_datas, l_hagn * sim_aexp
        # )

        # if sim_hagn_ctr is None:
        #     print("position interpolation failed")
        #     continue

        # sim_hagn_ctr = decentre_coordinates(sim_hagn_ctr, sim.path)

        # # find galaxy in halo
        # hid_zoom_tgt, hprops_zoom_tgt, hosted_gal_props = find_zoom_tgt_halo(
        #     sim, 1.0 / sim_aexp - 1.0, tgt_ctr=sim_hagn_ctr, tgt_rad=sim_hagn_rvir
        # )

        # gid, gprops = get_central_gal_for_hid(sim, hid_zoom_tgt, sim_snap)
        # rmax = gprops["rmax"]

        # rmax = hprops_zoom_tgt['rvir']

        avail_snaps = find_snaps_with_halos(sim.snap_numbers, sim.path)
        avail_aexps = sim.get_snap_exps(avail_snaps)
        avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1).value * 1e3  # Myr

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

        sim_halo_tree_rev_fname = os.path.join(
            sim.path, "TreeMakerDM_dust", "tree_rev.dat"
        )

        sim_tree_hids, sim_tree_datas, sim_tree_aexps = read_tree_fev_sim(
            sim_halo_tree_rev_fname,
            fbytes=os.path.join(sim.path, "TreeMakerDM_dust"),
            zstart=1.0 / start_aexp - 1.0,
            tgt_ids=[hid_start],
            star=False,
        )

        halo_pos_last = np.transpose(
            [
                sim_tree_datas["x"][0][0],
                sim_tree_datas["y"][0][0],
                sim_tree_datas["z"][0][0],
            ]
        )
        convert_adaptahop_pos(sim.cosmo.lcMpc * sim_aexp, halo_pos_last)

        halo_rvir_last = sim_tree_datas["r"][0][0] / sim.cosmo.lcMpc * sim_aexp

        # hagn_massive_sid = find_massive_sink(pos, hagn_snap, hagn_sim, rmax=rvir * 2)[
        try:
            sim_massive_sid = find_massive_sink(
                halo_pos_last,
                sim_snap,
                sim,
                rmax=halo_rvir_last * 2,
                # sim_hagn_ctr,
                # sim_snap,
                # sim,
                # rmax=rmax,
            )["identity"]
            print(f"zoom massive sink has id: {sim_massive_sid}")
        except ValueError:
            print(f"no massive sink found in zoom: {sim.name}")
            continue
        # print()

        found_sim_sink = True

        sim_sink_hist = get_sink_mhistory(
            sim_massive_sid,
            sim_snap,
            sim,
            out_keys=[("position", 3), ("dens", 1), ("vrel", 1), ("mass", 1)],
        )

        # print(list(zip(hagn_sink_hist["zeds"], hagn_sink_hist["mass"])))

        zeds_zoom = sim_sink_hist["zeds"]

        if not hasattr(sim, "cosmo_model"):
            sim.init_cosmo()

        times = sim.cosmo_model.age(zeds_zoom).value * 1e3  # Myr

        # print(sim_sink_hist)
        mass_zoom = sim_sink_hist["mass"]

        dens_data_zoom = np.convolve(
            sim_sink_hist["dens"], np.ones(nsmooth) / nsmooth, mode="same"
        )
        vrel_data_zoom = np.convolve(
            sim_sink_hist["vrel"], np.ones(nsmooth) / nsmooth, mode="same"
        )

        sim_sink_hist_zeds = sim_sink_hist["zeds"]
        sim_sink_hist_aexps = 1.0 / (1 + sim_sink_hist_zeds)

        gal_dist_zoom = np.empty(len(sim_tree_hids[0]))

        for i, (hid, aexp) in enumerate(zip(sim_tree_hids[0], sim_tree_aexps)):

            if hid in [0, -1]:
                continue

            sink_aexp_arg = np.argmin(np.abs(sim_sink_hist_aexps - aexp))
            if np.abs(sim_sink_hist_aexps[sink_aexp_arg] - aexp) > 0.005:
                print("aexp interpolation failed")
                continue

            cur_snap = sim.get_closest_snap(zed=1.0 / aexp - 1)
            hprops, gal_props = get_halo_props_snap(sim.path, cur_snap, hid)

            if gal_props != {}:

                gal_masses = gal_props["mass"]
                gal_pos = gal_props["pos"]

                argmaxmass = np.argmax(gal_masses)

                gal_ctr = gal_pos[:, argmaxmass]

                zoom_gid = int(gal_props["gids"][argmaxmass])

                gal_dist_zoom[i] = np.linalg.norm(
                    gal_ctr - sim_sink_hist["position"][sink_aexp_arg]
                )  # /gal_props["r50"]

                zoom_stars = read_zoom_stars(sim, cur_snap, zoom_gid)
                stars_pos, star_ctr, star_ext = find_star_ctr_period(zoom_stars["pos"])

                reff = 0

                for inds in [[0, 1], [0, 2], [1, 2]]:

                    reff += (
                        np.percentile(
                            np.linalg.norm(
                                stars_pos[:, inds] - star_ctr[None, inds], axis=1
                            ),
                            50,
                        )
                        / 3
                    )

                gal_dist_zoom[i] /= reff

        xmin = min(xmin, zeds_zoom.min())
        xmax = max(xmax, zeds_zoom.max())

        # print(mdot_data, mdot_edd_data)

        # print(len(mdot_data), len(mdot_edd_data))

    if sim_id in done_hagn_ids:
        c = done_hagn_colors[done_hagn_ids.index(sim_id)]
        # (l,) = axs[0].plot(zeds_hagn, mass_hagn, lw=1, c=c)  # , c=l.get_color())
    else:
        (l,) = axs[0].plot(zeds_hagn, mass_hagn, lw=1)  # , c=l.get_color())
        c = l.get_color()
        axs[1].plot(zeds_hagn, dens_data_hagn, lw=1, c=c)
        axs[2].plot(zeds_hagn, vrel_data_hagn, lw=1, c=c)
        axs[3].plot(1.0 / hagn_tree_aexps - 1, gal_dist_hagn, lw=1, c=c)

    nb_times_done = int(np.sum(np.in1d(done_hagn_ids, sim_id)))
    zoom_ls = lss[nb_times_done]

    # sim plots
    axs[0].plot(zeds_zoom, mass_zoom, lw=2, c=c, label=sim.name, ls=zoom_ls)
    # min_mdot_data = np.convolve(mdot_data, np.ones(nsmooth) / nsmooth, mode="same")

    gal_dist_zoom[gal_dist_zoom < 1e-3] = 1e-3

    axs[1].plot(zeds_zoom, dens_data_zoom, lw=2, c=c, ls=zoom_ls)
    axs[2].plot(zeds_zoom, vrel_data_zoom, lw=2, c=c, ls=zoom_ls)
    axs[3].plot(1.0 / sim_tree_aexps - 1, gal_dist_zoom, lw=2, c=c, ls=zoom_ls)

    labels.append(sim.name)
    lines.append(Line2D([0], [0], color=c, lw=2, ls=zoom_ls))

    done_hagn_colors.append(c)
    done_hagn_ids.append(sim_id)

    if not read_existing or overwrite:

        with h5py.File(fout_h5, "w") as f:
            f.create_dataset("zeds_zoom", data=zeds_zoom)
            f.create_dataset("mass_zoom", data=mass_zoom)
            f.create_dataset("dens_zoom", data=dens_data_zoom)
            f.create_dataset("vrel_zoom", data=vrel_data_zoom)
            f.create_dataset("gal_dist_zoom", data=gal_dist_zoom)
            f.create_dataset("sim_tree_aexps", data=sim_tree_aexps)

            f.create_dataset("zeds_hagn", data=zeds_hagn)
            f.create_dataset("mass_hagn", data=mass_hagn)
            f.create_dataset("dens_hagn", data=dens_data_hagn)
            f.create_dataset("vrel_hagn", data=vrel_data_hagn)
            f.create_dataset("gal_dist_hagn", data=gal_dist_hagn)
            f.create_dataset("hagn_tree_aexps", data=hagn_tree_aexps)


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
axs[1].set_ylabel("cloud density")
axs[2].set_ylabel("cloud vrel")
axs[3].set_ylabel(r"$\mathrm{d_{sink2gal}/R_{eff}}$ ")

# axs[-2].set_ylim(0, 5)

# axs[1].set_ylim(1e-23, 1e-17)
# ax.grid()

# axs[0].set_xlim(3.6, 10)

for ax in axs[:-1]:
    ax.tick_params(direction="in", top=True, right=True)
for ax in axs[:-1]:
    ax.set_yscale("log")

axs[-2].tick_params(direction="in", top=True, right=True, bottom=True, labelbottom=True)

plt.subplots_adjust(hspace=0.0)

if found_sim_sink:
    axs[0].set_xlim(xmax, xmin)
# axs[0].invert_xaxis()

# second xaxis times...
ticks = axs[0].get_xticks()
axs[0].set_xticks(ticks)
ax2 = axs[0].twiny()
ax2.set_xlim(axs[0].get_xlim())
ax2.set_xticks(ticks)
ax2.set_xticklabels([f"{sim.cosmo_model.age(xtick).value:.2f}" for xtick in ticks])
ax2.set_xlabel("time [Gyr]")
ax2.tick_params(direction="in", top=True, right=True)

fig.savefig("dist_to_gal.png")
