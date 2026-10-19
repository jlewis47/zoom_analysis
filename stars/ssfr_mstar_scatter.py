from scipy.spatial import KDTree

# from zoom_analysis.stars import sfhs

# from zoom_analysis.zoom_helpers import decentre_coordinates
# from zoom_analysis.stars.star_reader import yt_read_star_ball, read_part_ball
# from zoom_analysis.stars.leja_quench_fit import sfr_ridge_leja22
# from zoom_analysis.halo_maker.read_treebricks import (
# read_brickfile,
# convert_brick_units,
# convert_star_units,
# read_zoom_stars,
# )
# from zoom_analysis.trees.tree_reader import read_tree_file_rev as read_tree_fev_sim
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import (
    # find_zoom_tgt_gal,
    # find_zoom_tgt_halo,
    # get_central_gal_for_hid,
    get_gal_assoc_file,
    find_snaps_with_gals,
    get_gal_props_snap,
    get_halo_props_snap,
)

from zoom_analysis.sinks.sink_reader import (
    find_massive_sink,
    # read_sink_bin,
    # snap_to_coarse_step,
)

from zoom_analysis.zoom_helpers import (
    decentre_coordinates,
    # find_starting_position,
    # starting_hid_from_hagn,
)

from gremlin.read_sim_params import ramses_sim

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_stuff import setup_plots

import os
import numpy as np

# import h5py

# from hagn.association import gid_to_stars
from hagn.utils import get_hagn_sim, adaptahop_to_code_units

# from hagn.tree_reader import read_tree_rev, interpolate_tree_position
from hagn.catalogues import get_halos_cat, make_super_cat, get_cat_hids, get_cat_gids

from astropy.cosmology import Planck18 as cosmo

from hagn.catalogues import make_super_cat, get_cat_hids

# from f90_tools.star_reader import read_part_ball_hagn, read_part_ball_NCdust


def calculate_reines_mbh(mstar_msun, alpha=7.45, beta=1.05):
    return 10**alpha * (mstar_msun / 1e11) ** beta  # msun


# sim_ids = [
#     "id2757_995pcnt_mstel/",
#     "id19782_500pcnt_mstel/",pip install setuptools
#     "id13600_005pcnt_mstel/",
# ]
# sdir = "/home/jlewis/sims/proposal_zooms/ExtractZoom/mh1e12/"
hagn_snap = 197  # z=2 in the full res box
HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
delta_t = 100  # Myr


ztgt = 2.0

sdirs = [
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_coreen",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_DynBondiGravSinkMass",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowSFE_gravSink",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowSFE_DynBondiGravSinkMass",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_fullresIC",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_kicks",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_smooth_ref",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_boostNH",
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
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
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
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",  # _leastcoarse",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel",  # _leastcoarse",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",  # _leastcoarse",
    # # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099",
    # "/data102/jlewis/sims/lvlmax_21/mh1e12/id180130_model6_eps0p05",
    # "/data103/jlewis/sims/lvlmax_21/mh1e12/id180130_256",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_noHydroDragB4z4merge",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
]

setup_plots()

fig, ax = plt.subplots(
    1,
    2,
    figsize=(8, 4),
    layout="constrained",
    sharex=True,
    sharey=True,
)
plt.subplots_adjust(wspace=0.0, hspace=0.0)


fig_pos, ax_pos = plt.subplots(
    1, 3, figsize=(8, 3), layout="constrained", sharex=True, sharey=True
)

fig_hmsmr, ax_hmsmr = plt.subplots(
    1,
    2,
    figsize=(8, 4),
    layout="constrained",
    sharex=True,
    sharey=True,
)
plt.subplots_adjust(wspace=0.0, hspace=0.0)

fig_bhmsmr, ax_bhmsmr = plt.subplots(
    1,
    2,
    figsize=(8, 4),
    layout="constrained",
    sharex=True,
    sharey=True,
)
plt.subplots_adjust(wspace=0.0, hspace=0.0)

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

mass_bins = [10**9.5, 10**10.6, np.inf]
# mass_bins = [10**9.5, 10**10., np.inf]
ssfr_bins = [1e-9, 1e-10, 1e-11, 0.0]

markers = [
    "o",
    "s",
    "D",
    "P",
    "X",
    "v",
    "^",
    "<",
    ">",
    "d",
    "p",
    "*",
    "h",
    "H",
    "+",
    "x",
]

# zed_bins = []

isim = 0

main_stars = False

if main_stars:
    main_star_str = "mainStars"
else:
    main_star_str = ""

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


sfr_max = -np.inf
sfr_min = np.inf

ssfr_max = -np.inf
ssfr_min = np.inf

mstel_max = -np.inf
mstel_min = np.inf

last_simID = None
l = None

zoom_style = 0
vol_tot = 0

for sdir in sdirs:

    name = sdir.split("/")[-1].strip()

    # sim_id = name[2:].split("_")[0]

    # sim_ids = ["id74099", "id147479", "id242704"]

    # c = "tab:blue"

    # get the galaxy in HAGN

    intID = int(name[2:].split("_")[0])
    # gid = int(make_super_cat(197, outf="/data101/jlewis/hagn/super_cats")["gid"][0])

    sim = ramses_sim(sdir, nml="cosmo.nml")

    sim_snaps = sim.snap_numbers
    sim_aexps = sim.get_snap_exps()
    sim.init_cosmo()
    sim_times = sim.cosmo_model.age(1.0 / sim_aexps - 1.0).value * 1e3

    # last sim_aexp
    # valid_steps = hagn_tree_aexps < (1.0 / (tgt_zed + 1.0))  # sim.aexp_end
    # nsteps = np.sum(valid_steps)

    hagn_sim = get_hagn_sim()

    nsteps = len(sim_snaps)
    nstep_hagn = len(hagn_aexps)

    # find last output with assoc files
    # assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
    # assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

    assoc_file_nbs = find_snaps_with_gals(sim_snaps, sim.path)

    avail_aexps = np.intersect1d(
        sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
    )

    if len(avail_aexps) == 0:
        continue

    avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

    prev_mass = -1
    # prev_pos = None

    zoom_ctr = [
        sim.namelist["refine_params"]["xzoom"],
        sim.namelist["refine_params"]["yzoom"],
        sim.namelist["refine_params"]["zzoom"],
    ]
    if "rzoom" in sim.namelist["refine_params"]:
        check_zoom = (
            lambda coords: np.linalg.norm(coords - zoom_ctr, axis=1)
            < sim.namelist["refine_params"]["rzoom"]
        )

        vol_cMpc = (
            4.0
            / 3
            * np.pi
            * (sim.namelist["refine_params"]["rzoom"] * sim.cosmo.lcMpc) ** 3
        )

    elif "azoom" in sim.namelist["refine_params"]:
        if "zoom_shape" in sim.namelist["refine_params"]:
            ellipse = True
            if sim.namelist["refine_params"]["zoom_shape"] == "rectangle":
                ellipse = False
        else:
            ellipse = True

        a = sim.namelist["refine_params"]["azoom"]
        b = sim.namelist["refine_params"]["bzoom"]
        c = sim.namelist["refine_params"]["czoom"]

        if ellipse:
            check_zoom = (
                lambda coords: (
                    ((coords[:, 0] - zoom_ctr[0]) / a) ** 2
                    + ((coords[:, 1] - zoom_ctr[1]) / b) ** 2
                    + ((coords[:, 2] - zoom_ctr[2]) / c) ** 2
                )
                < 1
            )

            vol_cMpc = 4 / 3 * np.pi * a * b * c * sim.cosmo.lcMpc**3

        else:
            check_zoom = (
                lambda coords: (np.abs(coords[:, 0] - zoom_ctr[0]) < a)
                * ((np.abs(coords[:, 1] - zoom_ctr[1])) < b)
                * (np.abs(coords[:, 2] - zoom_ctr[2]) < c)
            )

            vol_cMpc = a * b * c * sim.cosmo.lcMpc**3

    last_sim_aexp = avail_aexps[-1]

    # hagn tree loop
    for istep, (aexp, time) in enumerate(zip(hagn_aexps, hagn_times)):

        if np.abs(aexp - last_sim_aexp) > 0.01:
            continue

        hagn_snap = hagn_snaps[np.argmin(np.abs(hagn_aexps - aexp))]

        try:
            super_cat = make_super_cat(
                hagn_snap, outf="/data101/jlewis/hagn/super_cats"
            )  # , overwrite=True)
        except FileNotFoundError:
            print("No super cat")
            continue

        # print(super_cat.keys())

        hagn_pos = np.transpose([super_cat["x"], super_cat["y"], super_cat["z"]])

        # hagn_pos = decentre_coordinates(hagn_pos, sim.path)

        hagn_in_zoom = np.where(check_zoom(hagn_pos))[0]

        # print(len(in_zoom), len(hagn_pos))

        # print(super_cat.keys())

        sfr_hagn = super_cat["sfr100"][hagn_in_zoom]
        mstel_hagn = super_cat["mgal"][hagn_in_zoom]
        mdm_hagn = super_cat["mhalo"][hagn_in_zoom]
        rvir_hagn = super_cat["rvir"][hagn_in_zoom]

        # coarse_step = snap_to_coarse_step(hagn_snap)
        # sinks = read_sink_bin(
        #     os.path.join(sim.sink_path, f"sink_{coarse_step:05d}.dat"), hagn=True
        # )

        mbh_hagn = np.zeros_like(mstel_hagn, dtype=float)
        dmbh_hagn = np.zeros_like(mstel_hagn, dtype=float)
        hagn_zoom_pos = hagn_pos[hagn_in_zoom]

        for i, gid in enumerate(super_cat["gid"][hagn_in_zoom]):
            sink = find_massive_sink(
                hagn_zoom_pos[i],
                hagn_snap,
                hagn_sim,
                rmax=rvir_hagn[i] * 1.0,
                hagn=True,
                tgt_fields=["mass", "dMBH_coarse", "position"],
            )

            if type(sink["mass"]) in [np.float32, np.float64]:
                mbh_hagn[i] = sink["mass"]
                dmbh_hagn[i] = sink["dMBH_coarse"]

        ssfr_hagn = sfr_hagn / mstel_hagn

        # print(
        #     hagn_pos[in_zoom].mean(axis=0),
        #     hagn_pos[in_zoom].min(axis=0),
        #     hagn_pos[in_zoom].max(axis=0),
        #     zoom_ctr,
        #     a,
        #     b,
        #     c,
        # )
    dist_to_hagn_aexps = np.abs(sim_aexps[:, np.newaxis] - hagn_aexps)

    # # zoom loop
    for istep, (snap, aexp, time) in enumerate(
        zip(sim_snaps[:-1], sim_aexps[:-1], sim_times[:-1])
    ):
        # if time < sim_tree_times.min() - 5:
        # continue

        cur_dists = dist_to_hagn_aexps[istep, :]
        is_min_dist = (
            cur_dists.min() == dist_to_hagn_aexps[:, np.argmin(cur_dists)].min()
        )

        if not is_min_dist:
            continue

        zed = 1.0 / aexp - 1.0

        if np.all(np.abs(avail_aexps - aexp) > 1e-1):
            print(f"No assoc file for at z={1./aexp-1:.1f},snap={snap:d}")
            continue

        if not os.path.exists(get_gal_assoc_file(sim.path, snap)):
            print("no assoc file")
            continue

        # assoc_file = assoc_files[assoc_file_nbs == snap]
        # if len(assoc_file) == 0:
        #     print(f"No assoc file for at z={1./aexp-1:.1f},snap={snap:d}")
        #     continue

        gprops = get_gal_props_snap(sim.path, snap)

        all_pos = gprops["pos"].T
        all_pos_decentre = decentre_coordinates(all_pos, sim.path)

        pure = gprops["host purity"] > (1 - 1e-4)

        hprops = get_halo_props_snap(sim.path, snap)

        hids = hprops["hid"]

        central_gids = np.zeros(len(hids), dtype=np.int64)
        rvirs = np.zeros(len(hids), dtype=np.float64)
        hpos = np.zeros((len(hids), 3), dtype=np.float64)

        gal_tree = KDTree(all_pos_decentre, boxsize=1 + 1e-6)

        for i, hid in enumerate(hids):

            hpos[i] = decentre_coordinates(hprops[f"halo_{hid:07d}"]["pos"], sim.path)
            rvirs[i] = hprops[f"halo_{hid:07d}"]["rvir"]
            gals_in_rvir = gal_tree.query_ball_point(hpos[i], rvirs[i])
            # print(np.log10(hprops["mvir"][i]), gals_in_rvir)
            if len(gals_in_rvir) > 1:

                masses_in_rvir = gprops["mass"][gals_in_rvir]
                gids_in_rvir = gprops["gids"][gals_in_rvir]

                central_gids[i] = gids_in_rvir[masses_in_rvir.argmax()]

        # in_zoom = np.where(check_zoom(all_pos) * pure)[0]
        in_zoom = pure
        # print(gprops.keys())

        central = gprops["central"] == 1

        gal_cond = central * in_zoom

        mstel_zoom = gprops["mass"][gal_cond]
        mdm_zoom = gprops["host mass"][gal_cond]
        # rvir_zoom = gprops["rvir"][gal_cond]
        sfr_zoom = gprops["sfr100"][gal_cond] / 1e6
        # bhm_zoom = gprops["bh mass"][in_zoom*central]
        ssfr_zoom = sfr_zoom / mstel_zoom

        gids = gprops["gids"][gal_cond]

        zoom_pos_centred = all_pos[gal_cond]
        zoom_pos = all_pos_decentre[gal_cond]

        mbh_zoom = np.zeros_like(mstel_zoom, dtype=float)
        dmbh_zoom = np.zeros_like(mstel_zoom, dtype=float)

        for i in range(gal_cond.sum()):

            gid = gids[i]
            hid_arg = gid == central_gids
            if hid_arg.sum() == 0:
                continue
            hid = hids[hid_arg][0]

            rvir_zoom = hprops[f"halo_{hid:07d}"]["rvir"]

            # if mstel_zoom[i] > 1e6:
            sink = find_massive_sink(
                zoom_pos_centred[i],
                snap,
                sim,
                rmax=rvir_zoom * 1.0,
                hagn=False,
                tgt_fields=["mass", "dMBH_coarse", "position"],
            )

            if type(sink["mass"]) in [np.float32, np.float64]:
                mbh_zoom[i] = sink["mass"]
                dmbh_zoom[i] = sink["dMBH_coarse"]

        # print(ssfr.min())

    # print(vol_cMpc)

    vol_tot += vol_cMpc

    color = None
    # print(sim.name, color, l)

    if last_simID == intID:
        color = l.get_edgecolor()
        zoom_style += 1
    else:
        zoom_style = 0

    plot_mbin = 0
    plot_ssfr_bin = -1

    if zoom_style == 0:
        l = ax[0].scatter(
            mstel_hagn, sfr_hagn, marker=markers[zoom_style], s=10, alpha=0.3
        )
        ax_hmsmr[0].scatter(
            mdm_hagn, mstel_hagn, marker=markers[zoom_style], s=10, alpha=0.3
        )
        ax_bhmsmr[0].scatter(
            mstel_hagn, mbh_hagn, marker=markers[zoom_style], s=10, alpha=0.3
        )

        ax_pos[0].scatter(
            hagn_pos[hagn_in_zoom, 0],
            hagn_pos[hagn_in_zoom, 1],
            s=1,
            c=l.get_edgecolor(),
        )
        ax_pos[1].scatter(
            hagn_pos[hagn_in_zoom, 1],
            hagn_pos[hagn_in_zoom, 2],
            s=1,
            c=l.get_edgecolor(),
        )
        ax_pos[2].scatter(
            hagn_pos[hagn_in_zoom, 0],
            hagn_pos[hagn_in_zoom, 2],
            s=1,
            c=l.get_edgecolor(),
        )

    ax[1].scatter(
        mstel_zoom,
        sfr_zoom,
        marker=markers[zoom_style],
        s=10,
        color=l.get_edgecolor(),
        alpha=0.3,
    )
    ax_hmsmr[1].scatter(
        mdm_zoom,
        mstel_zoom,
        marker=markers[zoom_style],
        s=10,
        color=l.get_edgecolor(),
        alpha=0.3,
    )
    l_style = ax_bhmsmr[1].scatter(
        mstel_zoom,
        mbh_zoom,
        marker=markers[zoom_style],
        s=10,
        color=l.get_edgecolor(),
        alpha=0.3,
    )

    ax_pos[0].scatter(
        all_pos_decentre[:, 0], all_pos_decentre[:, 1], s=1, c=l.get_edgecolor()
    )
    ax_pos[1].scatter(
        all_pos_decentre[:, 1], all_pos_decentre[:, 2], s=1, c=l.get_edgecolor()
    )
    ax_pos[2].scatter(
        all_pos_decentre[:, 0], all_pos_decentre[:, 2], s=1, c=l.get_edgecolor()
    )

    lvlmax = sim.namelist["amr_params"]["levelmax"]

    labels.append(sim.name + f" {lvlmax}")
    lines.append(l_style)

    last_simID = intID


ax[0].legend(lines, labels, framealpha=0.0)  # , handlelength=3)
ax_hmsmr[0].legend(lines, labels, framealpha=0.0)  # , handlelength=3)
ax_bhmsmr[0].legend(lines, labels, framealpha=0.0)  # , handlelength=3)

for iter_axis in ax:
    iter_axis.grid()
    iter_axis.tick_params(which="both", direction="in", right=True, top=True)

    iter_axis.set_xscale("log")
    iter_axis.set_yscale("log")

    iter_axis.set_xlabel(r"$M_{\star}$ [$M_{\odot}$]")

ax[0].set_ylabel(r"$\log_{10}(\mathrm{SFR})$ [$M_{\odot}.yr^{-1}$]")

ax[0].set_title("HAGN galaxies in zoom regions")
ax[1].set_title("Zoom galaxies, fpure>0.9999")

ax_hmsmr[0].set_xscale("log")
ax_hmsmr[1].set_xscale("log")
ax_hmsmr[1].set_yscale("log")
ax_hmsmr[0].set_yscale("log")


ax_hmsmr[0].set_ylabel(r"$M_{\star}$ [$M_{\odot}$]")
for ax_iter in ax_hmsmr:
    ax_iter.grid()
    ax_iter.tick_params(which="both", direction="in", right=True, top=True)

    ax_iter.set_xlabel(r"$M_{halo}$ [$M_{\odot}$]")

xlim = np.asarray(list(ax_hmsmr[1].get_xlim()))
ylim = np.asarray(list(ax_hmsmr[1].get_ylim()))
ax_hmsmr[1].set_xlim(xlim)
ax_hmsmr[1].set_ylim(ylim)


for iax, ax_iter in enumerate(ax_hmsmr):
    for hmsmr, hmsmr_ls in zip([1.0, 0.1, 0.01], ["-", "--", ":"]):
        p1 = ax_iter.transData.transform_point([xlim[0], xlim[0]])
        p0 = ax_iter.transData.transform_point([xlim[1], xlim[1]])
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        if angle < 0:
            angle = 180 + angle
        ax_iter.plot(xlim, xlim * hmsmr, ls=hmsmr_ls, color="k", lw=0.5, zorder=np.inf)
        if iax == 1:
            ax_iter.text(
                xlim[0] * 2,
                xlim[0] * 2 * hmsmr * 1.05,
                f"HMSMR = {hmsmr:0.2f}",
                verticalalignment="bottom",
                horizontalalignment="left",
                fontsize=6,
                rotation=angle,
                # rotation_mode="anchor",
                # transform_rotates_text=True,
                zorder=np.inf,
            )

ax_hmsmr[0].set_title("HAGN galaxies in zoom regions")
ax_hmsmr[1].set_title("Zoom galaxies, centrals with fpure>0.9999")

ax_bhmsmr[0].set_title("HAGN galaxies in zoom regions")
ax_bhmsmr[1].set_title("Zoom galaxies, centrals with fpure>0.9999")

ax_bhmsmr[0].set_xscale("log")
ax_bhmsmr[0].set_yscale("log")

xlim = np.asarray(list(ax_bhmsmr[1].get_xlim()))
ylim = np.asarray(list(ax_bhmsmr[1].get_ylim()))

ax_bhmsmr[1].set_xlim(xlim)
ax_bhmsmr[1].set_ylim(ylim)


reines_mbh = calculate_reines_mbh(xlim)


ax_bhmsmr[0].set_ylabel(r"$M_{BH}$ [$M_{\odot}$]")
for ax_iter in ax_bhmsmr:
    ax_iter.grid()
    ax_iter.tick_params(which="both", direction="in", right=True, top=True)

    ax_iter.set_xlabel(r"$M_{\star}$ [$M_{\odot}$]")

    ax_iter.plot(xlim, reines_mbh, ls="--", color="k", lw=0.5, zorder=np.inf)

# angle = np.arctan(np.diff(np.log10(reines_mbh)) / np.diff(np.log10(xlim))) * 180 / np.pi

p0 = ax_bhmsmr[1].transData.transform_point([xlim[0], reines_mbh[0]])
p1 = ax_bhmsmr[1].transData.transform_point([xlim[1], reines_mbh[1]])
dx = p1[0] - p0[0]
dy = p1[1] - p0[1]

angle = np.degrees(np.arctan2(dy, dx))


ax_bhmsmr[0].text(
    4e8,
    calculate_reines_mbh(4e8) * 1.05,
    r"Reines+ 2015 $M_{BH}/M_{\star}$",
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=6,
    rotation=angle,
    # rotation_mode="anchor",
    # transform_rotates_text=True,
    zorder=np.inf,
)


# line of constant sSFR = 1e-11 yr^-1
# x = np.logspace(7, 12, 100)
xlim = np.asarray(ax[0].get_xlim())
ax[0].set_xlim()
y = 1e-11 * xlim


ax[0].plot(xlim, y, "k--", lw=1)
ax[1].plot(xlim, y, "k--", lw=1)
y = 1e-10 * xlim

ax[0].plot(xlim, y, "k:", lw=1)
ax[1].plot(xlim, y, "k:", lw=1)


fig_dir = os.path.join("figs", main_star_str)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
zstr = f"{ztgt:.1f}".replace(".", "p")
f_fig = os.path.join(fig_dir, f"scatter_sfrVSmstar_allzeds.png")

fig.savefig(f_fig)
fig_pos.legend(lines, labels)
fig_pos.savefig(f_fig.replace(".png", "_pos.png"))
fig_hmsmr.savefig(f_fig.replace("sfrVSmstar", "hmsmr"))
fig_bhmsmr.savefig(f_fig.replace("sfrVSmstar", "bhmsmr"))
