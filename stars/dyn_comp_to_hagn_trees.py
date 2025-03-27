from zoom_analysis.stars import sfhs

# from zoom_analysis.zoom_helpers import decentre_coordinates
# from zoom_analysis.stars.star_reader import yt_read_star_ball, read_part_ball
# from zoom_analysis.stars.leja_quench_fit import sfr_ridge_leja22
from zoom_analysis.read.read_data import read_data_ball

# from zoom_analysis.halo_maker.read_treebricks import (
# read_brickfile,
# convert_brick_units,
# convert_star_units,
# read_zoom_stars,
# )

from zoom_analysis.trees.tree_reader import (
    read_tree_file_rev_correct_pos as read_tree_fev_sim,
)
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import (
    # find_zoom_tgt_gal,
    # find_zoom_tgt_halo,
    get_assoc_pties_in_tree,
    get_central_gal_for_hid,
    get_gal_assoc_file,
    find_snaps_with_gals,
    # smooth_radii_tree,
    smooth_props,
)

from dynamics import extract_nh_kinematics

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

# from f90_tools.star_reader import read_part_ball_hagn, read_part_ball_NCdust


# sim_ids = [
#     "id2757_995pcnt_mstel/",
#     "id19782_500pcnt_mstel/",pip install setuptools
#     "id13600_005pcnt_mstel/",
# ]
# sdir = "/home/jlewis/sims/proposal_zooms/ExtractZoom/mh1e12/"
hagn_snap = 197  # z=2 in the full res box
HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
delta_t = 100  # Myr

sdirs = [
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_coreen",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_kicks",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
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
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_boostNH",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_boostNH_resimBoostFriction",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_frcAccrt",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_lowerSFE_drag",
]


# setup plot
fig, axs = plt.subplots(2, 3, figsize=(14, 10), sharex=True)
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

overwrite = True
plot_hagn = False
only_main_stars = True

zoom_ls = ["-", "--", ":", "-.", (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10))] * 5
lines = []
labels = []


vrot_max = -np.inf
vrot_min = np.inf

sigma_max = -np.inf
sigma_min = np.inf

mstel_max = -np.inf
mstel_min = np.inf

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
        mstel_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
        vrot_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
        sigma_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
        time_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)

    mstel_zoom = np.zeros(nsteps, dtype=np.float32)
    vrot_zoom = np.zeros(nsteps, dtype=np.float32)
    sigma_zoom = np.zeros(nsteps, dtype=np.float32)
    time_zoom = np.zeros(nsteps, dtype=np.float32)
    fdisk_zoom = np.zeros(nsteps, dtype=np.float32)
    fbulge_zoom = np.zeros(nsteps, dtype=np.float32)

    datas_path = os.path.join(sim.path, "computed_data")
    fout_h5 = os.path.join(datas_path, f"stellar_dynamics.h5")
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
            if plot_hagn:
                saved_mstel_hagn = f["mstel_hagn"][:]
                saved_vrot_hagn = f["vrot_hagn"][:]
                saved_sigma_hagn = f["sigma_hagn"][:]
                saved_time_hagn = f["time_hagn"][:]

            saved_mstel_zoom = f["mstel_zoom"][:]
            saved_vrot_zoom = f["vrot_zoom"][:]
            saved_sigma_zoom = f["sigma_zoom"][:]
            saved_time_zoom = f["time_zoom"][:]
            saved_fdisk_zoom = f["fdisk_zoom"][:]
            saved_fbulge_zoom = f["fbulge_zoom"][:]

        if saved_time_zoom.max() >= sim_times.max():
            # if we have the same number of values
            # then just read the data
            mstel_zoom = saved_mstel_zoom
            vrot_zoom = saved_vrot_zoom
            sigma_zoom = saved_sigma_zoom
            time_zoom = saved_time_zoom
            fbulge_zoom = saved_fbulge_zoom
            fdisk_zoom = saved_fdisk_zoom

            if plot_hagn:

                mstel_hagn = saved_mstel_hagn
                vrot_hagn = saved_vrot_hagn
                sigma_hagn = saved_vrot_hagn
                time_hagn = saved_time_hagn

        else:  # otherwise we need to recompute because something has changed
            read_existing = False

    # print(read_existing, len(saved_mstel_zoom), len(sim_snaps))

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

                # # gal_pties = get_cat_gids(super_cat, [int(hagn_tree_hids[0][istep])])

                if len(gal_pties["gid"]) == 0:
                    print("No galaxy")
                    continue

                hagn_file = os.path.join(sim.path, f"stellar_history_{hagn_snap}.h5")

                # this is super slow... shold put in separate file and only do once per unique halo id
                stars = gid_to_stars(
                    gal_pties["gid"][0],
                    hagn_snap,
                    hagn_sim,
                    ["mass", "birth_time", "metallicity", "pos", "vel"],
                )

                # r50 and !!!

                ages = stars["age"]
                Zs = stars["metallicity"]

                masses = sfhs.correct_mass(hagn_sim, ages, stars["mass"], Zs)
                rot_props, kin_props = extract_nh_kinematics(
                    masses,
                    stars["pos"],
                    stars["vel"],
                    [gal_pties["x"][0], gal_pties["y"][0], gal_pties["z"][0]],
                )

                vrot = rot_props["Vrot"]
                disp = rot_props["disp"]

                vrot = np.abs(vrot)

                mstel_hagn[istep] = stars["mass"].sum()
                vrot_hagn[istep] = vrot
                sigma_hagn[istep] = disp
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

        start_snap = sim.get_closest_snap(aexp=start_aexp)

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
            sim,
            fbytes=os.path.join(sim.path, "TreeMakerDM_dust"),
            snap=start_snap,
            zstart=1.0 / start_aexp - 1.0,
            tgt_ids=[hid_start],
            star=False,
        )

        sim_tree_hids = sim_tree_hids[0]

        sim_tree_times = sim.cosmo_model.age(1.0 / sim_tree_aexps - 1).value * 1e3

        sim.get_snap_times()

        gal_props_tree = get_assoc_pties_in_tree(
            sim,
            sim_tree_aexps,
            sim_tree_hids,
            assoc_fields=["pos", "r50", "rmax", "mass"],
        )

        print(gal_props_tree.keys())

        # r50s_smooth, rmax_smooth, rvir_smooth = smooth_radii_tree(
        #     gal_props_tree["r50"], gal_props_tree["rmax"], gal_props_tree["rvir"]
        # )
        smooth_gal_props_tree = smooth_props(gal_props_tree)

        fig_test, ax_test = plt.subplots()
        ax_test.plot(
            gal_props_tree["aexps"], gal_props_tree["pos"], label="rmax", marker="o"
        )
        ax_test.plot(
            gal_props_tree["aexps"],
            gal_props_tree["hpos"],
            label="r50",
            ls="--",
            marker="o",
        )
        fig_test.savefig("test_pos.png")
        fig_test, ax_test = plt.subplots()
        ax_test.plot(
            1.0 / smooth_gal_props_tree["aexps"] - 1,
            smooth_gal_props_tree["pos"],
            label="rmax",
            marker="o",
        )
        ax_test.plot(
            1.0 / smooth_gal_props_tree["aexps"] - 1,
            smooth_gal_props_tree["hpos"],
            label="r50",
            ls="--",
            marker="o",
        )
        fig_test.savefig("test_smooth_pos.png")
        fig_test, ax_test = plt.subplots()
        ax_test.plot(
            1.0 / gal_props_tree["aexps"] - 1,
            gal_props_tree["mass"],
            label="rmax",
            marker="o",
        )
        ax_test.plot(
            1.0 / gal_props_tree["aexps"] - 1,
            gal_props_tree["mvir"],
            label="r50",
            ls="--",
            marker="o",
        )
        ax_test.set_yscale("log")
        fig_test.savefig("test_smooth_m.png")

        prev_mass = -1
        prev_pos = -1
        prev_rad = -1
        prev_time = -1

        # # zoom loop
        for istep, (snap, aexp, time) in enumerate(
            zip(sim_snaps[::-1], sim_aexps[::-1], sim_times[::-1])
        ):

            if np.all(np.abs(avail_aexps - aexp) > 1e-1):
                print(f"No assoc file for at z={1./aexp-1:.1f},snap={snap:d}")
                prev_time = time
                continue

            if not os.path.exists(get_gal_assoc_file(sim.path, snap)):
                prev_time = time
                continue

            sim_tree_arg = np.argmin(np.abs(sim_tree_aexps - aexp))
            cur_snap_hid = sim_tree_hids[sim_tree_arg]

            if cur_snap_hid in [0, -1]:
                prev_time = time
                continue

            gid, gal_dict = get_central_gal_for_hid(
                sim,
                cur_snap_hid,
                snap,
                main_stars=only_main_stars,
                prev_mass=prev_mass,
                prev_pos=prev_pos,
                prev_rad=prev_rad,
                # debug=True,
                verbose=True,
            )
            if gid == None:
                print(f"No central galaxy at z={1./aexp-1:.1f},snap={snap:d}")
                prev_time = time
                continue

            # tgt_pos = gal_dict["pos"]
            # tgt_r = gal_dict["r50"]

            # tgt_pos = gal_dict["pos"]
            # tgt_r = gal_dict["r50"]

            smooth_tree_arg = np.argmin(np.abs(smooth_gal_props_tree["aexps"] - aexp))

            tgt_pos = gal_props_tree["pos"][
                smooth_tree_arg
            ]  # smoothing on position is super dangerous...
            tgt_r = smooth_gal_props_tree["r50"][smooth_tree_arg] * 1.0

            prev_pos = tgt_pos
            prev_mass = gal_dict["mass"]

            # stars = read_zoom_stars(sim, snap, gid)

            # ages = stars["agepart"]
            # Zs = stars["Zpart"]

            # masses = sfhs.correct_mass(hagn_sim, ages, stars["mpart"], Zs)
            stars = read_data_ball(
                sim,
                snap,
                tgt_pos,
                tgt_r,
                host_halo=cur_snap_hid,
                data_types=["stars"],
                tgt_fields=[
                    "pos",
                    "vel",
                    "mass",
                    "age",
                    "metallicity",
                ],
            )

            if stars == None:
                prev_time = time
                continue

            ages = stars["age"]
            Zs = stars["metallicity"]

            masses = sfhs.correct_mass(hagn_sim, ages, stars["mass"], Zs)

            if len(masses) == 0:
                prev_time = time
                continue

            # star_bulk_vel = np.linalg.norm(
            #     np.average(stars["vel"], axis=0, weights=masses)
            # )  # km/s
            # star_bulk_vel = (
            #     star_bulk_vel * (3600 * 24 * 365 * 1e6) * (1e3 / 3.08e16 / 1e3)
            # )  # kpc/Myr

            # print("star bulk vel", star_bulk_vel)
            if istep > 0:
                # prev_rad = abs(star_bulk_vel * (time - sim_times[istep - 1]) * 1.5)
                # prev_rad = prev_rad / sim.cosmo.lcMpc / 1e3  # ckpc->code
                # print(f"this is: {prev_rad:.2e} code units")
                dt = abs(time - prev_time) * (1e6 * 3600 * 24 * 365)  # s
                max_vel = 5e3  # km/s
                prev_rad = max_vel * dt * 1e3 / (3.08e16 * 1e3)  # kpc
                print(
                    f"max gal velocity between snaps : {max_vel:.1e}, equivalent search radius : {prev_rad:.2f} kpc"
                )
                prev_rad = prev_rad / (sim.cosmo.lcMpc * 1e3)

                prev_rad = max(prev_rad, tgt_r * 2)
                print(
                    f"adopted search radius: {prev_rad* sim.cosmo.lcMpc * 1e3:.2f} kpc"
                )

            prev_time = time

            rot_props, kin_props = extract_nh_kinematics(
                masses,
                stars["pos"],
                stars["vel"] - np.average(stars["vel"], axis=0, weights=masses),
                gal_dict["pos"],
            )

            vrot = rot_props["Vrot"]
            disp = rot_props["disp"]

            fdisks = kin_props["fdisk"]
            fbulge = kin_props["fbulge"]

            vrot = np.abs(vrot)

            print(vrot, disp, vrot / disp)

            mstel_zoom[istep] = masses.sum()
            vrot_zoom[istep] = vrot
            sigma_zoom[istep] = disp
            time_zoom[istep] = time
            fdisk_zoom[istep] = fdisks
            fbulge_zoom[istep] = fbulge

            # after plot save the data to h5py file
        with h5py.File(fout_h5, "w") as f:
            if plot_hagn:
                f.create_dataset("mstel_hagn", data=mstel_hagn, compression="lzf")
                f.create_dataset("vrot_hagn", data=vrot_hagn, compression="lzf")
                f.create_dataset("sigma_hagn", data=sigma_hagn, compression="lzf")
                f.create_dataset("time_hagn", data=time_hagn, compression="lzf")

            f.create_dataset("mstel_zoom", data=mstel_zoom, compression="lzf")
            f.create_dataset("vrot_zoom", data=vrot_zoom, compression="lzf")
            f.create_dataset("sigma_zoom", data=sigma_zoom, compression="lzf")
            f.create_dataset("time_zoom", data=time_zoom, compression="lzf")
            f.create_dataset("fdisk_zoom", data=fdisk_zoom, compression="lzf")
            f.create_dataset("fbulge_zoom", data=fbulge_zoom, compression="lzf")

    if plot_hagn:
        stab_hagn = vrot_hagn / sigma_hagn
    stab_zoom = vrot_zoom / sigma_zoom

    if np.any(mstel_zoom > 0):  # np.any(mstel_hagn > 0) and
        mstel_max = np.max(
            [
                mstel_max,
                # mstel_hagn[mstel_hagn > 0].max(),
                mstel_zoom[mstel_zoom > 0].max(),
            ]
        )
        mstel_min = np.min(
            [
                mstel_min,
                # mstel_hagn[mstel_hagn > 0].min(),
                mstel_zoom[mstel_zoom > 0].min(),
            ]
        )
        vrot_max = np.max(
            [
                vrot_max,
                #   vrot_hagn[mstel_hagn > 0].max(),
                vrot_zoom[mstel_zoom > 0].max(),
            ]
        )
        vrot_min = np.min(
            [
                vrot_min,
                #  vrot_hagn[mstel_hagn > 0].min(),
                vrot_zoom[mstel_zoom > 0].min(),
            ]
        )
        sigma_max = np.max(
            [
                sigma_max,
                # np.nanmax(sigma_hagn[mstel_hagn > 0]),
                np.nanmax(sigma_zoom[mstel_zoom > 0]),
            ]
        )
        sigma_min = np.min(
            [
                sigma_min,
                # np.nanmin(sigma_hagn[mstel_hagn > 0]),
                np.nanmin(sigma_zoom[mstel_zoom > 0]),
            ]
        )

    if time_zoom[np.min(np.where(mstel_zoom > 0))] < xmin:
        xmin = time_zoom[np.min(np.where(mstel_zoom > 0))]

    color = None
    if last_simID == intID:
        color = l[0].get_color()
        zoom_style += 1
    else:
        zoom_style = 0

    if zoom_style == 0 and plot_hagn:
        ax[0].plot(
            time_hagn / 1e3,
            mstel_hagn,
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

    order = np.argsort(time_zoom)
    # non_zero = np.where((mstel_zoom > 0) * (time_zoom > 0))[0]
    non_zero = np.where((time_zoom > 0))[0]
    order = order[non_zero]

    # print(f"zoom style is {zoom_ls[zoom_style]}")
    l = ax[0].plot(
        time_zoom[order] / 1e3,
        mstel_zoom[order],
        label="zoom",
        color=color,
        lw=2.5,
        ls=zoom_ls[zoom_style],
    )

    ax[1].plot(
        time_zoom[order] / 1e3,
        np.abs(vrot_zoom[order]),
        color=color,
        lw=2.5,
        ls=zoom_ls[zoom_style],
    )

    ax[2].plot(
        time_zoom[order] / 1e3,
        sigma_zoom[order],
        color=color,
        lw=2.5,
        ls=zoom_ls[zoom_style],
    )

    ax[3].plot(
        time_zoom[order] / 1e3,
        np.abs(stab_zoom[order]),
        color=color,
        lw=2.5,
        ls=zoom_ls[zoom_style],
    )

    ax[4].plot(
        time_zoom[order] / 1e3,
        fdisk_zoom[order],
        color=color,
        lw=2.5,
        ls=zoom_ls[zoom_style],
    )

    labels.append(sim.name)
    lines.append(l[0])

    # print(sim.name, intID, last_simID)
    # print(zoom_style, zoom_ls[zoom_style], l[0].get_color())

    last_simID = intID

    fig_test, ax_test = plt.subplots()
    ax_test.plot(
        gal_props_tree["aexps"], gal_props_tree["pos"], label="rmax", marker="o"
    )
    ax_test.plot(
        gal_props_tree["aexps"],
        gal_props_tree["hpos"],
        label="r50",
        ls="--",
        marker="o",
    )
    fig_test.savefig("test_pos.png")
    fig_test, ax_test = plt.subplots()
    ax_test.plot(
        smooth_gal_props_tree["aexps"],
        smooth_gal_props_tree["pos"],
        label="rmax",
        marker="o",
    )
    ax_test.plot(
        smooth_gal_props_tree["aexps"],
        smooth_gal_props_tree["hpos"],
        label="r50",
        ls="--",
        marker="o",
    )
    fig_test.savefig("test_smooth_pos.png")
    fig_test, ax_test = plt.subplots()
    ax_test.plot(
        gal_props_tree["aexps"], gal_props_tree["mass"], label="rmax", marker="o"
    )
    ax_test.plot(
        gal_props_tree["aexps"],
        gal_props_tree["mvir"],
        label="r50",
        ls="--",
        marker="o",
    )
    ax_test.plot(sim_aexps, mstel_zoom[::-1])
    ax_test.set_yscale("log")
    fig_test.savefig("test_smooth_m.png")

# if np.isfinite(mstel_min) and np.isfinite(mstel_max):
#     ax[0].set_ylim(mstel_min * 0.5, mstel_max * 1.5)
# if np.isfinite(vrot_min) and np.isfinite(vrot_max):
#     ax[1].set_ylim(vrot_min * 0.5, vrot_max * 1.5)
# if np.isfinite(sigma_min) and np.isfinite(sigma_max):
#     ax[2].set_ylim(sigma_min * 0.5, sigma_max * 1.5)

ax[0].set_yscale("log")
ax[0].set_ylabel("Stellar mass [$M_\odot$]")

ax[1].set_ylabel("abs(Vrot) [km/s]")
ax[2].set_ylabel("Sigma [km/s]")
ax[3].set_ylabel("abs(Vrot/Sigma)")
ax[4].set_ylabel("fdisk")

ax[2].set_xlabel("Time [Gyr]")
ax[4].set_xlabel("Time [Gyr]")
# ax[5].set_xlabel("Time [Gyr]")

ax[4].set_ylim(0, 1)
ax[5].set_ylim(0, 1)

plt.subplots_adjust(hspace=0.0)

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
# if min(xlim) < xmin / 1e3:
#     xlim = (xmin / 1e3, xlim[1])

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

y2 = ax[1].twiny()
y2.set_xlim(xlim)
zlabels = [f"{cosmo.age(zed).value:.1f}" for zed in ax[1].get_xticks()]
# y2.set_xticks(ax[0].get_xticks())
y2.set_xticklabels(zlabels)
y2.set_xlabel("redshift")

y2 = ax[2].twiny()
y2.set_xlim(xlim)
zlabels = [f"{cosmo.age(zed).value:.1f}" for zed in ax[2].get_xticks()]
# y2.set_xticks(ax[0].get_xticks())
y2.set_xticklabels(zlabels)
y2.set_xlabel("redshift")

if plot_hagn:
    labels.extend(["HAGN", "zoom"])
    lines.extend([Line2D([], [], lw=1.0, c="k"), Line2D([], [], lw=3.0, c="k")])


ax[-1].legend(lines, labels, framealpha=0.0, ncol=2)  # , handlelength=3)

ax[-1].axis("off")

fig.savefig(f"dyn_compHAGN_trees.png")
