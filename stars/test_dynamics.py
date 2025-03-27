from re import I
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic_2d
from zoom_analysis.read.read_data import read_data_ball
from zoom_analysis.stars import sfhs

# from zoom_analysis.zoom_helpers import decentre_coordinates
# from zoom_analysis.stars.star_reader import yt_read_star_ball, read_part_ball
from zoom_analysis.stars.leja_quench_fit import sfr_ridge_leja22
from zoom_analysis.halo_maker.read_treebricks import (
    # read_brickfile,
    # convert_brick_units,
    # convert_star_units,
    read_zoom_stars,
)
from zoom_analysis.trees.tree_reader import read_tree_file_rev as read_tree_fev_sim
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

from dynamics import *

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
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE",
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
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
]


# setup plot
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
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


last_hagn_id = -1
isim = 0

overwrite = True


zoom_ls = ["-", "--", ":", "-.", (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10))] * 5
lines = []
labels = []


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

    sim = ramses_sim(sdir, nml="cosmo.nml")

    sim_snaps = sim.snap_numbers
    sim_aexps = sim.get_snap_exps()
    sim.init_cosmo()
    sim_times = sim.cosmo_model.age(1.0 / sim_aexps - 1.0).value * 1e3

    # last sim_aexp
    # valid_steps = hagn_tree_aexps < (1.0 / (tgt_zed + 1.0))  # sim.aexp_end
    # nsteps = np.sum(valid_steps)

    nsteps = len(sim_snaps)

    # find last output with assoc files
    # assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
    # assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

    assoc_file_nbs = find_snaps_with_gals(sim_snaps, sim.path)

    avail_aexps = np.intersect1d(
        sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
    )
    avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

    hid_start, halo_dict, start_hosted_gals, found, start_aexp = find_starting_position(
        sim,
        avail_aexps,
        hagn_tree_aexps,
        hagn_tree_datas,
        hagn_tree_times,
        avail_times,
    )

    if not found:
        continue

    # load sim tree
    sim_halo_tree_rev_fname = os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat")
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

    sim_tree_hids = sim_tree_hids[0]

    sim_tree_times = sim.cosmo_model.age(1.0 / sim_tree_aexps - 1).value * 1e3

    sim.get_snap_times()

    gal_props_tree = get_assoc_pties_in_tree(
        sim,
        sim_tree_aexps,
        sim_tree_hids,
        assoc_fields=["pos", "r50", "rmax"],
    )

    gal_smoothed_props_tree = smooth_props(gal_props_tree)

    fig, ax = plt.subplots()

    # # zoom loop
    for istep, (snap, aexp, time) in enumerate(
        zip(sim_snaps[-1:], sim_aexps[-1:], sim_times[-1:])
    ):

        if np.all(np.abs(avail_aexps - aexp) > 1e-1):
            print(f"No assoc file for at z={1./aexp-1:.1f},snap={snap:d}")
            continue

        if not os.path.exists(get_gal_assoc_file(sim.path, snap)):
            continue

        sim_tree_arg = np.argmin(np.abs(sim_tree_aexps - aexp))
        cur_snap_hid = sim_tree_hids[sim_tree_arg]

        gid, gal_dict = get_central_gal_for_hid(sim, cur_snap_hid, snap)
        if gid == None:
            print(f"No central galaxy at z={1./aexp-1:.1f},snap={snap:d}")
            continue

        tgt_pos = gal_dict["pos"]
        # tgt_r = gal_dict["r50"]

        smooth_tree_arg = np.argmin(np.abs(gal_smoothed_props_tree["aexps"] - aexp))

        tgt_r = gal_smoothed_props_tree["r50"][smooth_tree_arg] * 0.5

        # stars = read_zoom_stars(sim, snap, gid)

        # ages = stars["agepart"]
        # Zs = stars["Zpart"]

        # masses = sfhs.correct_mass(hagn_sim, ages, stars["mpart"], Zs)

        # stars = read_part_ball_NCdust(
        #     sim,
        #     snap,
        #     tgt_pos,
        #     tgt_r,
        #     tgt_fields=[
        #         "pos",
        #         "vel",
        #         "mass",
        #         "birth_time",
        #         "metallicity",
        #     ],
        #     fam=2,
        # )

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

        ages = stars["age"]
        Zs = stars["metallicity"]

        masses = sfhs.correct_mass(hagn_sim, ages, stars["mass"], Zs)

        if len(masses) == 0:
            continue

        ang_mom = compute_ang_mom(masses, stars["pos"], stars["vel"], tgt_pos)

        ang_dir = ang_mom / np.linalg.norm(ang_mom)

        bulk_vel = np.average(stars["vel"], axis=0, weights=masses)

        gal_ref_vels = stars["vel"] - bulk_vel

        # projected map of velocities in xy xz yz planes
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        pos_mins = np.min(stars["pos"], axis=0)
        pos_maxs = np.max(stars["pos"], axis=0)

        npos_bins = 30
        pos_bins = [
            np.linspace(pos_mins[0], pos_maxs[0], npos_bins),
            np.linspace(pos_mins[1], pos_maxs[1], npos_bins),
            np.linspace(pos_mins[2], pos_maxs[2], npos_bins),
        ]

        vmin = 0.1
        vmax = 500

        plane_dist_tol = 1.0 / (sim.cosmo.lcMpc * 1e3)

        for i in range(3):

            ind0 = i
            ind1 = i + 1
            ind2 = i + 2

            ind1 = ind1 % 3
            ind2 = ind2 % 3

            print(f"Plotting {ind0} vs {ind1}, view from {ind2}")

            plane_stars = np.abs(stars["pos"][:, ind2] - tgt_pos[ind2]) < plane_dist_tol

            plane_vels = np.linalg.norm(gal_ref_vels[:, [ind0, ind1]], axis=1)[
                plane_stars
            ]

            # means, _, _, _ = binned_statistic_2d(
            #     stars["pos"][plane_stars, ind0],
            #     stars["pos"][plane_stars, ind1],
            #     plane_vels,
            #     bins=(pos_bins[ind0], pos_bins[ind1]),
            #     statistic="mean",
            # )

            # img = ax[i].imshow(
            #     means.T,
            #     origin="lower",
            #     extent=(pos_mins[ind0], pos_maxs[ind0], pos_mins[ind1], pos_maxs[ind1]),
            #     # norm=LogNorm(vmin=vmin, vmax=vmax),
            #     vmin=vmin,
            #     vmax=vmax,

            mean_vel_0, _, _, _ = binned_statistic_2d(
                stars["pos"][plane_stars, ind0],
                stars["pos"][plane_stars, ind1],
                gal_ref_vels[plane_stars, ind0],
                bins=(pos_bins[ind0], pos_bins[ind1]),
            )
            mean_vel_1, _, _, _ = binned_statistic_2d(
                stars["pos"][plane_stars, ind0],
                stars["pos"][plane_stars, ind1],
                gal_ref_vels[plane_stars, ind1],
                bins=(pos_bins[ind0], pos_bins[ind1]),
            )

            rho_dirs = np.transpose(
                [
                    stars["pos"][plane_stars, ind0],
                    stars["pos"][plane_stars, ind1],
                    np.zeros_like(stars["pos"][plane_stars, ind0]),
                ]
            )
            rho_dirs /= np.linalg.norm(rho_dirs, axis=1)[:, None]
            z_dirs = np.asarray([0, 0, 1])
            phi_dirs = np.cross(rho_dirs, z_dirs[None, :])
            phi_dirs /= np.linalg.norm(phi_dirs, axis=1)[:, None]

            tan_vels = mass_dot_product(gal_ref_vels[plane_stars], phi_dirs)
            rad_vels = mass_dot_product(gal_ref_vels[plane_stars], rho_dirs)
            vert_vels = mass_dot_product(gal_ref_vels[plane_stars], z_dirs)

            print(np.average(tan_vels, weights=masses[plane_stars]))
            print(
                np.sqrt(
                    (
                        weighted_variance(rad_vels, masses[plane_stars])
                        + weighted_variance(tan_vels, masses[plane_stars])
                        + weighted_variance(vert_vels, masses[plane_stars])
                    )
                    / 3.0
                )
            )

            ax[i].quiver(
                pos_bins[ind0][:-1], pos_bins[ind1][:-1], mean_vel_0.T, mean_vel_1.T
            )  # , scale=1e-2)

            # print(
            #     np.average(
            #         np.linalg.norm(
            #             [
            #                 gal_ref_vels[plane_stars, ind0],
            #                 gal_ref_vels[plane_stars, ind1],
            #             ],
            #             axis=0,
            #         ),
            #         weights=masses[plane_stars],
            #     )
            # )

            # )
            # ax[0, i].set_xlabel(f"pos {ind0}")
            # ax[0, i].set_ylabel(f"pos {ind1}")
            # ax[1, i].quiver(
            #     stars["pos"][plane_stars, ind0],
            #     stars["pos"][plane_stars, ind1],
            #     gal_ref_vels[plane_stars, ind0],
            #     gal_ref_vels[plane_stars, ind1],
            # )

        # plt.colorbar(img, ax=ax)

        fig.savefig(f"velocities_{snap:d}.png")

        pos_stars = stars["pos"]

        vrot, disp, fbulge, fdist = extract_nh_kinematics(
            masses, pos_stars, gal_ref_vels, tgt_pos, debug=False
        )

        print(f"vrot={vrot}, disp={disp}")
        print(f"vrot/disps={vrot/disp}")

        vel_rad, vel_tan, vel_vert = project_vels(
            masses, stars["pos"] - tgt_pos, stars["vel"], ang_mom, debug=True
        )

        f_bulge = masses[vel_tan > 0].sum() * 2 / masses.sum()
        f_disk = 1 - f_bulge

        print(f"f_bulge={f_bulge}, f_disk={f_disk}")
