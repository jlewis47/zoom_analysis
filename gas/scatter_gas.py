# from f90nml import read
import h5py
from plot_constraints import plot_MZR
from zoom_analysis.read.read_data import read_data_ball
from zoom_analysis.sinks import sink_histories, sink_reader
from zoom_analysis.constants import Msun_cgs, ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import get_gal_assoc_file

from scipy.stats import binned_statistic

# from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

import os
import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# from zoom_analysis.sink_reader import *

from hagn.association import gid_to_stars
from hagn.utils import get_hagn_sim, adaptahop_to_code_units
from hagn.tree_reader import read_tree_rev
from hagn.catalogues import get_halos_cat, make_super_cat, get_cat_hids

# get colour cycler
# from cycler import cycler

# planck cosmo
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
from astropy import units as u

from zoom_analysis.stars import sfhs
from zoom_analysis.zoom_helpers import check_in_all_sims_vol

from zoom_analysis.visu.visu_fct import get_gal_props_snap

# from hagn.catalogues import make_super_cat

# fig, ax = sink_histories.setup_bh_plot()

tgt_zed = 4.0


stellar_bins = np.logspace(5.5, 12, 8)


fpure = 1.0 - 1e-4

HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
delta_t = 100  # Myr

sim_dirs = [
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_coreen",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_fullresIC",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_gravSink",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_DynBondiGravSinkMass",
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
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE_stgNHboost_stricterSF/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_vlowSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_high_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd_drag",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_frcAccrt",
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
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_lowerSFE_NHboost",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_smallICs",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_stricterSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_stricterSF_radioHeavy",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictestSF_lowSNe/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_medSFE_stgNHboost_stricterSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_highSFE_stgNHboost_strictestSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_XtremeSF_lowSNe",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_MegaSF_lowSNe",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_SuperMegaSF_midSNe",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_XtremeLowSFE_stgNHboost_strictSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_SuperLowSFE_stgNHboost_strictSF",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictestSF_lowSNe",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictestSF_lowSNe_highAGNeff",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_XtremeLowSFE_stgNHboost_strictSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id180130_256",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id180130_model6_eps0p05",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id180130_model5",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id112288",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_noHydroDragB4z4merge",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data103/jlewis/sims/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF/",
    # "/data103/jlewis/sims/lvlmax_22/mh1e12/id52380_NClike",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288_novrel_lowerSFE_stgNHboost_strictSF/",
]
# list of all available pyplot markers:
# https://matplotlib.org/stable/api/markers_api.html
markers = [
    "o",
    "v",
    "^",
    "<",
    ">",
    "s",
    "p",
    "P",
    "*",
    "h",
    "X",
    "D",
    "d",
    "|",
    "_",
]

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fig_ratio, ax_ratio = plt.subplots(1, 1, figsize=(8, 8))

sim_names = []
all_same = True
prev_ID = int(ramses_sim(sim_dirs[0], nml="cosmo.nml").name.split("_")[0][2:])

for isim, sim_dir in enumerate(sim_dirs[1:]):

    sim = ramses_sim(sim_dir, nml="cosmo.nml")

    name = sim.name

    intID = int(sim.name.split("_")[0][2:])
    all_same = all_same * (intID == prev_ID)

for isim, sim_dir in enumerate(sim_dirs):

    sim = ramses_sim(sim_dir, nml="cosmo.nml")

    name = sim.name

    intID = int(sim.name.split("_")[0][2:])

    sim_names.append(intID)
    # hagn_sim = get_hagn_sim()
    # hagn_snap = hagn_sim.get_closest_snap(zed=tgt_zed)

    # # super_cat = make_super_cat(
    # #     hagn_snap, outf="/data101/jlewis/hagn/super_cats"
    # # )  # , overwrite=True)

    # # gal_pties = get_cat_hids(super_cat, [intID])

    # hagn_zed = 1.0 / hagn_sim.aexps[hagn_sim.snap_numbers == hagn_snap][0] - 1

    # mbh, dmbh, t_sink, fmerger, t_merge = sink_histories.get_bh_stuff(
    #     hagn_sim, intID, hagn_zed
    # )

    # l = sink_histories.plot_bh_stuff(
    #     ax,
    #     mbh,
    #     dmbh,
    #     t_sink,
    #     fmerger,
    #     t_merge,
    #     0,
    #     cosmo,
    #     label=name,
    #     lw=1,
    # )

    # c = l[0].get_color()
    # find last assoc_file
    found = False
    decal = 0

    z_dists = np.abs(1.0 / sim.aexps - 1.0 - tgt_zed)
    args = np.argsort(z_dists)

    while not found and decal < len(sim.snap_numbers):
        arg = args[decal]
        sim_snap = sim.snap_numbers[arg]
        sim_zed = 1.0 / sim.aexps[arg] - 1

        gfile = get_gal_assoc_file(sim_dir, sim_snap)

        found_cond = os.path.isfile(gfile) * (z_dists[arg] < 0.1)
        print(sim_snap, sim_zed, decal, gfile)

        if found_cond:
            found = True
        else:
            decal += 1

    if not found:
        continue

    gal_dict = get_gal_props_snap(sim.path, sim_snap, main_stars=False)
    if all_same:
        vol_args, min_vol = check_in_all_sims_vol(gal_dict["pos"].T, sim, sim_dirs)
    else:
        vol_args = np.full(gal_dict["host purity"].shape, True)

    # fig,ax = plt.subplots(1,1)

    # ax.scatter((gal_dict["pos"][0,:]-0.5)+sim.zoom_ctr[0],(gal_dict["pos"][1,:]-0.5)+sim.zoom_ctr[1])

    # fig.savefig('test_gal_coords')

    # print(gal_dict.keys())
    # print(min_vol,vol_args.sum(),len(vol_args))
    print(sim.zoom_ctr, gal_dict["pos"].mean(axis=1))

    host_purity = gal_dict["host purity"]
    central_cond = gal_dict["central"] == 1
    pure_cond = (host_purity > fpure) * central_cond * vol_args
    gal_mass = gal_dict["mass"][pure_cond]
    gal_pos = gal_dict["pos"][:, pure_cond]
    rmax = gal_dict["rmax"][pure_cond]
    r50 = gal_dict["r50"][pure_cond]
    halo_mass = gal_dict["host mass"][pure_cond]
    hids = gal_dict["host hid"][pure_cond]

    print(f"I found {pure_cond.sum():d} galaxies with purity > {fpure:f},")
    print(f"and {len(host_purity)-pure_cond.sum()} galaxies with purity < {fpure:f}")

    label = sim.name
    if len(sim_dirs) > 1:
        label += f", z={sim_zed:.4f}"

    mgas = np.zeros(len(hids), dtype=np.float32)
    mstels = np.zeros(len(hids), dtype=np.float32)

    for igal in range(len(hids)):

        tgt_pos = gal_pos[:, igal]
        tgt_r = r50[igal] * 3
        hid = hids[igal]

        try:
            datas = read_data_ball(
                sim,
                sim_snap,
                tgt_pos,
                tgt_r,
                hid,
                data_types=["gas", "stars"],
                tgt_fields=[
                    "ilevel",
                    "density",
                    # "chem_O",
                    "mass",
                    "age",
                    "metallicity",
                ],
            )
        except (KeyError, FileNotFoundError):
            continue

        gas = datas["gas"]
        if gas is None:
            continue
        if len(gas["ilevel"]) < 1:
            continue

        mgas[igal] = (
            gas["density"]
            * (
                ((sim.cosmo.lcMpc * ramses_pc * 1e6) / 2 ** gas["ilevel"]) ** 3
                / Msun_cgs
            )
        ).sum()  # g/cc

        stars = datas["stars"]
        if stars is None:
            continue
        if len(stars["mass"]) < 1:
            continue

        # print(stars)

        masses = sfhs.correct_mass(
            sim, stars["age"], stars["mass"], stars["metallicity"]
        )

        mstels[igal] = np.sum(masses)

        print(igal, np.log10(mstels[igal]), np.log10(mgas[igal]))

    points = ax.scatter(mstels, mgas, label=label, alpha=0.5, marker=markers[isim])
    # points = ax.scatter(
    #     mstels, 12+np.log10(ZOH), label=label, alpha=0.5, marker=markers[isim]
    # )

    # print(halo_mass,gal_mass,ZOH)

    # get avg values in bins
    bin_means, bin_edges, binnumber = binned_statistic(
        mstels, mgas, bins=stellar_bins, statistic="mean"
    )
    # bin_means, bin_edges, binnumber = binned_statistic(
    #     mstels, 12+np.log10(ZOH), bins=stellar_bins, statistic="mean"
    # )
    bin_centers, bin_edges, binnumber = binned_statistic(
        mstels, mstels, bins=stellar_bins, statistic="median"
    )
    # bin_width = bin_edges[1:] - bin_edges[:-1]
    # bin_centers = bin_edges[:-1] + bin_width * 0.5

    # plot
    # ax.errorbar(bin_centers, bin_means, xerr=bin_width/2, color=points.get_facecolor(), ls="none",alpha=1.0)
    ax.plot(bin_centers, bin_means, color=points.get_facecolor(), ls="-", alpha=1.0)


# now for HAGN

# hagn_sim = get_hagn_sim()

# hagn_snap = hagn_sim.get_closest_snap(zed=tgt_zed)
# hagn_aexp = hagn_sim.get_snap_exps(hagn_snap)[0]
# hagn_zed = 1.0 / hagn_aexp - 1

# has_hagn_snap=False

# try:
#     hagn_gals = make_super_cat(hagn_snap)
#     has_hagn_snap=True
# except FileNotFoundError:
#     pass

# if has_hagn_snap:

#     # print(hagn_gals.keys())

#     hagn_gal_pos = np.transpose([hagn_gals["x"], hagn_gals["y"], hagn_gals["z"]])

#     vol_args,min_vol = check_in_all_sims_vol(hagn_gal_pos, hagn_sim, sim_dirs)

#     if vol_args.sum() > 0:
#         mhalos = hagn_gals['mhalo'][vol_args]
#         mgals= hagn_gals['mgal'][vol_args]


#         ax.scatter(mhalos, mgals, alpha=0.5, marker="x",color='k')#, label=f"HAGN, z={hagn_zed:.2f}")

#         #now avg stats
#         avg, _, _ = binned_statistic(
#             mhalos,mgals, bins=stellar_bins, statistic=np.nanmean
#         )
#         avg_ratio, _, _ = binned_statistic(
#             mhalos,mgals/mhalos, bins=stellar_bins, statistic=np.nanmean
#         )
#         med_bin, _, _ = binned_statistic(
#             mhalos,mhalos , bins=stellar_bins, statistic=np.nanmedian
#         )
#         ax.plot(med_bin, avg, c="k", lw=1, ls="--",label=f"HAGN, z={hagn_zed:.2f}")


ax.tick_params(
    axis="both",
    which="both",
    bottom=True,
    # top=True,
    top=False,
    left=True,
    right=True,
    direction="in",
)

# ax.grid()

ax.set_xlabel("Stellar mass, [M$_\odot$]")
ax.set_ylabel("Gas mass,  [M$_\odot$]")

# ax.set_ylim(max(6, min(ax.get_ylim())), max(max(ax.get_ylim()), 9))

ax.set_yscale("log")
ax.set_xscale("log")

title_txt = ""
if len(sim_dirs) == 1:
    title_txt += f"z={sim_zed:.2f}"
    # ax.text(0.05, 0.95, f"z={sim_zed:.2f}", transform=ax.transAxes)
# also print purity threshold
# ax.text(0.05, 0.9, f"fpure > {fpure:.3f}", transform=ax.transAxes)
title_txt += f" fpure > {fpure:.3f}"


ax.legend(framealpha=0.0, title=title_txt)

fig_name = "zoom_gas_scatter"

if len(np.unique(sim_names)) == 1:
    fig_name += f"_id{sim_names[0]:d}"

if tgt_zed != None:
    fig.savefig(f"{fig_name:s}_z{tgt_zed:.2f}.png")
else:
    fig.savefig(f"{fig_name:s}.png")
