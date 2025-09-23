from socket import INADDR_MAX_LOCAL_GROUP
from matplotlib import markers
from matplotlib.patches import Patch
from zoom_analysis.stars import sfhs
from zoom_analysis.zoom_helpers import decentre_coordinates, check_in_all_sims_vol
from plot_constraints.plot_PQRO25_SMHMR import zed_to_data

# from zoom_analysis.stars.star_reader import yt_read_star_ball, read_part_ball
from zoom_analysis.stars.leja_quench_fit import sfr_ridge_leja22
from zoom_analysis.halo_maker.read_treebricks import (
    read_brickfile,
    convert_brick_units,
    convert_star_units,
    read_zoom_stars,
)
from zoom_analysis.trees.tree_reader import read_tree_file_rev as read_tree_fev_sim
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import (
    get_gal_assoc_file,
    get_halo_assoc_file,
    get_halo_props_snap,
    find_zoom_tgt_gal,
    find_zoom_tgt_halo,
    get_central_gal_for_hid,
    get_gal_props_snap,
)


from matplotlib import colors
from cycler import cycler

# from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

# from scipy.spatial import cKDTree

import os
import numpy as np


def nedkova21_size_fit(mass, z):

    zbins = [0.2, 0.5, 1.0, 1.0, 1.5, 2.0]
    if z > zbins[-1] + 0.5 or z < zbins[0] - 0.5:
        return np.zeros_like(mass)
    alphas = [0.04, -0.03, -0.33, -0.17]
    betas = [1.82, 1.6, 1.25, 1.84]
    log10_gammas = [-0.24, 0.48, 3.24, 1.76]
    deltas = [10.94, 10.95, 10.63, 11.09]
    log10_mlim = [7.0, 7.44, 9.2, 9.8]

    iz = np.digitize(z, zbins) - 1
    iz = np.clip(iz, 0, len(alphas) - 1)

    alpha = alphas[iz]
    beta = betas[iz]
    gamma = 10 ** log10_gammas[iz]
    delta = deltas[iz]
    mlim = 10 ** log10_mlim[iz]

    rs = gamma * (mass) ** alpha * (1.0 + mass / (10**delta)) ** (beta - alpha)

    rs = np.where(mass > mlim, rs, 0.0)

    return rs


def mowla19_size_fit(mass, z):

    zbins = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]
    if z > zbins[-1] + 0.5 or z < zbins[0] - 0.5:
        return np.zeros_like(mass)
    alphas = [0.27, 0.17, 0.13, 0.09, 0.13, 0.11]
    log_A = [0.74, 0.65, 0.6, 0.53, 0.49, 0.48]

    iz = np.digitize(z, zbins) - 1
    iz = np.clip(iz, 0, len(alphas) - 1)

    alpha = alphas[iz]
    A = 10 ** log_A[iz]

    rs = A * (mass / 7e10) ** alpha

    rs = np.where(mass > 10 ** (9.75), rs, 0.0)

    return rs


def yang_size_fit(mass, z, sftype="sf", obs="opt"):
    "return sizes in kpc from Yang+25, https://arxiv.org/pdf/2504.07185"

    assert sftype in ["sf", "quiescent"], "type must be either 'sf' or 'quiescent'"
    assert obs in ["opt", "uv"], "type must be either 'opt' or 'uv'"

    if obs == "opt":
        if sftype == "sf":
            zbins = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
            alpha = [0.2, 0.15, 0.18, 0.14, 0.19, 0.1, -0.13, 0.37]
            logA = [0.47, 0.33, 0.27, 0.16, 0.16, -0.19, -0.61, 0.10]
            sigLogReff = [0.19, 0.22, 0.22, 0.21, 0.17, 0.25, 0.26, 0.28]
        else:
            zbins = [2.5, 3.5, 4.5]
            alpha = [0.52, 0.78, 0.64]
            logA = [-0.05, -0.13, -0.31]
            sigLogReff = [0.16, 0.24, 0.25]
    else:
        if sftype == "sf":
            zbins = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
            alpha = [0.19, 0.19, 0.18, 0.12, 0.25, 0.13, 0.17, 0.7]
            logA = [0.41, 0.35, 0.2, 0.06, 0.21, -0.17, -0.18, 0.5]
            sigLogReff = [0.22, 0.23, 0.25, 0.25, 0.19, 0.24, 0.22, 0.09]
        else:
            zbins = [2.5, 3.5, 4.5]
            alpha = [0.61, 0.94, 0.31]
            logA = [-0.1, -0.1, -0.27]
            sigLogReff = [0.15, 0.2, 0.07]

    if z > zbins[-1] + 0.5 or z < zbins[0] - 0.5:
        return np.zeros_like(mass), 0.0

    iz = np.digitize(z, zbins) - 1
    iz = np.clip(iz, 0, len(alpha) - 1)

    rs = 10 ** logA[iz] * (mass / 5e10) ** alpha[iz]
    rs = np.where(mass > 10 ** (9), rs, 0.0)

    return rs, sigLogReff[iz]


sim_dirs = [
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_kicks",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE_lowSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_vlowSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_high_SN",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_lowerSFE_drag",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_frcAccrt",
    # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_DynBondiGravSinkMass",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel",/
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_stricterSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictestSF_lowSNe",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_MegaSF_lowSNe",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_SuperMegaSF_midSNe",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_XtremeSF_lowSNe",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictestSF_lowSNe_highAGNeff",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_SuperMegaSF_midSNe",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_XtremeLowSFE_stgNHboost_strictSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_SuperLowSFE_stgNHboost_strictSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_stricterSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_lowerSFE_NHboost",
    # "/data102/jlewis/sims/lvlmax_21/mh1e12/id180130_model6_eps0p05",
    # "/data103/jlewis/sims/lvlmax_21/mh1e12/id180130_256",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756",
    # "/data103/jlewis/sims/lvlmax_22/mh1e12/id52380_NClike",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_lowSN_strictSF",
    # "/data103/jlewis/sims/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF_lowSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130_novrel_lowerSFE_stgNHboost_strictSF",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288",
    # "/data103/jlewis/sims/lvlmax_22/mh1e12/id180130",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE",
]

main_stars = True

bin_size = 0.5
zbins = np.arange(1.5, 8.0, bin_size)
zbin_err = 0.2
sims_per_zbins = np.zeros((len(zbins)))
# find_smallest z with 2 sims !
smallest_z = np.inf

sim_names = []
all_same = True
prev_ID = int(ramses_sim(sim_dirs[0], nml="cosmo.nml").name.split("_")[0][2:])

for isim, sim_dir in enumerate(sim_dirs[1:]):

    sim = ramses_sim(sim_dir, nml="cosmo.nml")

    name = sim.name

    intID = int(sim.name.split("_")[0][2:])
    sim_names.append(intID)
    all_same = all_same * (intID == prev_ID)

    # print(all_same)

for sim_dir in sim_dirs:

    sim = ramses_sim(sim_dir)

    sim_snaps = sim.snap_numbers
    sim_aexps = sim.get_snap_exps(sim_snaps)

    for iz, z in enumerate(zbins):

        snap = sim.get_closest_snap(zed=z)

        # assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
        # assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

        # if not snap in assoc_file_nbs:
        #     continue

        assoc_file = get_gal_assoc_file(sim.path, snap)
        if not os.path.exists(assoc_file):
            continue

        aexp = sim.get_snap_exps(snap)
        zed_sim = 1.0 / aexp - 1
        if np.abs(zed_sim - z) > zbin_err:
            continue

        sims_per_zbins[iz] += 1


smallest_z = np.where(sims_per_zbins >= min(1, len(sim_dirs)))[0][0]
zbins = zbins[smallest_z:]


mhalo_bins = np.logspace(8, 13, 7)
mgal_bins = np.logspace(5.5, 12, 8)

ned_done = np.full(len(zbins), False)
mow_done = np.full(len(zbins), False)
yang_done = np.full(len(zbins), False)
ssfr_stuff_done = np.full(len(zbins), False)
PQRO_done = np.full(len(zbins), False)

# zbins = [6, 5, 4, 3, 2]
# zbins = [7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0]  # , 3.5]  # , 3.0, 2.5, 2.0]

n_zbins = len(zbins)
n_plots = n_zbins

nrows = int(np.sqrt(n_plots))
ncols = np.ceil(n_plots / nrows).astype(int)
overlap = nrows * ncols - n_plots
row_fact = 0
height_ratios = [1 for i in range(nrows)]
if overlap == 0:
    nrows += 1
    row_fact = 0.6
    height_ratios.append(row_fact)
overlap = nrows * ncols - n_plots

avail_colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "gold",
    "tab:brown",
    "tab:purple",
    "tab:olive",
    "tab:cyan",
    "tab:pink",
]
default_cycler = (
    cycler(color=avail_colors) * cycler(markers=["o"]) * cycler(linestyles=["-"])
)
if len(sim_dirs) > len(avail_colors):
    default_cycler *= cycler(markers=["o", "D"]) + cycler(linestyles=["-", ":"])

labels_size = []
labels_sfsm = []
labels_hmsmr = []

plot_colors = [c["color"] for c in default_cycler]
plot_linestyles = [c["linestyles"] for c in default_cycler]
plot_markers = [c["markers"] for c in default_cycler]

lines_hmsmr = []
lines_size = []
lines_sfsm = []

row_aspect = nrows - row_fact
col_aspect = ncols

figsize = (col_aspect * 4, row_aspect * 4)

legend_args = {
    # "handles": lines,
    # "labels": labels,
    "ncol": 1,
    "loc": "lower right",
    "framealpha": 0.0,
}

subplot_args = {
    "nrows": nrows,
    "ncols": ncols,
    "sharex": True,
    "sharey": True,
    "figsize": figsize,
    "height_ratios": height_ratios,
    # "layout": "constrained",
}

smhmr_fig, smhmr_ax = plt.subplots(**subplot_args)
smhmr_ax = np.ravel(smhmr_ax)
plt.subplots_adjust(hspace=0.0, wspace=0.0)
size_fig, size_ax = plt.subplots(**subplot_args)
size_ax = np.ravel(size_ax)
plt.subplots_adjust(hspace=0.0, wspace=0.0)
sfsm_fig, sfsm_ax = plt.subplots(**subplot_args)
sfsm_ax = np.ravel(sfsm_ax)
plt.subplots_adjust(hspace=0.0, wspace=0.0)

size_ax[0].set_ylim(0.01, 10)
sfsm_ax[0].set_ylim(100, 1e9)
smhmr_ax[0].set_ylim(4e-5, 0.5)

size_ax[0].set_xlim(mgal_bins[0], mgal_bins[-1])
sfsm_ax[0].set_xlim(mgal_bins[0], mgal_bins[-1])
smhmr_ax[0].set_xlim(mhalo_bins[0], mhalo_bins[-1])

for isim, sim_dir in enumerate(sim_dirs):

    sim = ramses_sim(sim_dir)

    print(sim.name)

    sim_snaps = sim.snap_numbers
    sim_aexps = sim.get_snap_exps()
    sim.init_cosmo()
    sim_times = sim.cosmo_model.age(1.0 / sim_aexps - 1.0).value * 1e3

    for iz, z in enumerate(zbins):

        zbin_txt = f"z={z:.1f} $\pm$ {zbin_err:.1f}"

        smhmr_ax[iz].text(
            0.05,
            0.95,
            zbin_txt,
            transform=smhmr_ax[iz].transAxes,
            fontsize=10,
            verticalalignment="top",
        )

        if not PQRO_done[iz]:
            PQRO_data = zed_to_data(z)

            PQRO_bin_edges = PQRO_data["bins"][np.isfinite(PQRO_data["bins"])]

            PQRO_bins = np.diff(PQRO_bin_edges) + PQRO_bin_edges[:-1]

            pqro_err = smhmr_ax[iz].errorbar(  # 10**PQRO_bins,
                10 ** PQRO_data["logMstarMh_lower_err"],
                10 ** PQRO_data["logMh"],
                # xerr=[10**(PQRO_data["logMstarMh_lower_err"]-PQRO_data["logMh_lower_err"]),
                #         10**(PQRO_data["logMstarMh_lower_err"]+PQRO_data["logMh_upper_err"])],
                # yerr=[10**(PQRO_data["logMh"]-PQRO_data["logMstarMh"]),
                #         10**(PQRO_data["logMh"]+PQRO_data["logMstarMh"])],
                # xerr=[10**(PQRO_data["logMstarMh_lower_err"])-10**(PQRO_data["logMh_lower_err"]),
                #         10**(PQRO_data["logMstarMh_lower_err"])+10**(PQRO_data["logMh_upper_err"])],
                # yerr=[10**(PQRO_data["logMh"])-10**(PQRO_data["logMstarMh"]),
                #         10**(PQRO_data["logMh"])+10**(PQRO_data["logMstarMh"])],
                color="k",
                ls="none",
                lw=0.0,
                elinewidth=0.5,
                marker="*",
            )

            PQRO_done[iz] = True

        size_ax[iz].text(
            0.95,
            0.1,
            zbin_txt,
            transform=size_ax[iz].transAxes,
            fontsize=10,
            verticalalignment="top",
            ha="right",
        )

        sfsm_ax[iz].text(
            0.95,
            0.1,
            zbin_txt,
            transform=sfsm_ax[iz].transAxes,
            fontsize=10,
            verticalalignment="top",
            ha="right",
        )

        snap = sim.get_closest_snap(zed=z)

        assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
        assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

        if not snap in assoc_file_nbs:
            continue

        aexp = sim.get_snap_exps(snap)
        zed_sim = 1.0 / aexp - 1
        if np.abs(zed_sim - z) > zbin_err:
            print(f"no close enough snaps to zbin {z:.1f}")
            continue

        time = sim.cosmo_model.age(1.0 / aexp - 1.0).value * 1e3

        gal_props = get_gal_props_snap(sim_dir, snap, main_stars=True)

        central = gal_props["central"]

        gal_pos = gal_props["pos"].T[central]

        # vol_args,min_vol=check_in_all_sims_vol(gal_pos, sim, sim_dirs)
        if all_same:
            vol_args, min_vol = check_in_all_sims_vol(gal_props["pos"].T, sim, sim_dirs)
        else:
            vol_args = np.full(gal_props["host purity"].shape, True)

        # r50s = gal_props["r50"][central*vol_args] * sim.cosmo.lcMpc * 1e3 * aexp
        reffs = gal_props["reff"][central * vol_args] * sim.cosmo.lcMpc * 1e3 * aexp
        mstels = gal_props["mass"][central * vol_args]
        sfrs = gal_props["sfr100"][central * vol_args]
        mvir = gal_props["host mass"][central * vol_args]

        if len(mstels) == 0:
            print("no galaxies")
            continue

        # sizes
        avg_size, _, _ = binned_statistic(
            mstels, reffs, bins=mgal_bins, statistic="mean"
        )
        errs_size, _, _ = binned_statistic(
            mstels, reffs, bins=mgal_bins, statistic="std"
        )
        counts_size, _, _ = binned_statistic(
            mstels, reffs, bins=mgal_bins, statistic="count"
        )
        binctr_size, _, _ = binned_statistic(
            mstels, mstels, bins=mgal_bins, statistic="median"
        )
        # avg_size, _, _ = binned_statistic(
        #     mstels, r50s, bins=mgal_bins, statistic="mean"
        # )
        # errs_size, _, _ = binned_statistic(
        #     mstels, r50s, bins=mgal_bins, statistic="std"
        # )
        # counts_size, _, _ = binned_statistic(
        #     mstels, r50s, bins=mgal_bins, statistic="count"
        # )
        # binctr_size, _, _ = binned_statistic(
        #     mstels, mstels, bins=mgal_bins, statistic="median"
        # )

        if np.any(counts_size == 0):
            empty_bins = np.where(counts_size == 0)

            binctrs_static = np.diff(mgal_bins) * 0.5 + mgal_bins[:-1]

            binctr_size[empty_bins] = binctrs_static[empty_bins]

        color_rgb = colors.to_rgb(plot_colors[isim])
        color_rgba = (*color_rgb, 0.4)
        mstel_colors = np.array([color_rgba for i in range(len(mstels))])

        arg_first_full = np.where(counts_size > 0)[0][0]
        arg_empty = np.where(counts_size == 0)[0]
        if len(arg_empty) > 1 and arg_empty.max() >= arg_first_full:
            max_full_bin = (binctr_size + 0.5 * np.diff(mgal_bins))[
                arg_empty[arg_empty > arg_first_full][0] - 1
            ]
            points_full_alpha = np.where(mstels > max_full_bin)[0]
            mstel_colors[points_full_alpha, 3] = 1

        size_ax[iz].scatter(
            mstels,
            # r50s,
            reffs,
            label=sim_dir,
            marker=plot_markers[isim],
            facecolors="none",
            edgecolors=mstel_colors,
            # alpha=alphas,
        )

        size_ax[iz].errorbar(
            binctr_size,
            avg_size,  # * (counts_size > 3),
            yerr=errs_size,  # * (counts_size > 3),
            color=plot_colors[isim],
            ls=plot_linestyles[isim],
        )

        if not ned_done[iz]:
            nedkova_sizes = nedkova21_size_fit(mgal_bins, z)
            size_ax[iz].plot(
                mgal_bins[nedkova_sizes > 0],
                nedkova_sizes[nedkova_sizes > 0],
                "k--",
                alpha=0.5,
            )
            ned_done[iz] = True

        if not mow_done[iz]:
            mowla_sizes = mowla19_size_fit(mgal_bins, z)
            size_ax[iz].plot(
                mgal_bins[mowla_sizes > 0],
                mowla_sizes[mowla_sizes > 0],
                "k-.",
                alpha=0.5,
            )
            mow_done[iz] = True

        if not yang_done[iz]:
            yang_sizes_opt_sf, yang_sigmas_opt_sf = yang_size_fit(
                mgal_bins, z, sftype="sf", obs="opt"
            )
            size_ax[iz].plot(
                mgal_bins[yang_sizes_opt_sf > 0],
                yang_sizes_opt_sf[yang_sizes_opt_sf > 0],
                ls="--",
                color="olive",
                alpha=1.0,
            )
            # size_ax[iz].fill_between(mgal_bins[yang_sizes_opt_sf>0],
            #                          yang_sizes_opt_sf[yang_sizes_opt_sf>0]-yang_sigmas_opt_sf,
            #                          yang_sizes_opt_sf[yang_sizes_opt_sf>0]+yang_sigmas_opt_sf,
            #                          color='olive',
            #                          alpha=0.3)

            yang_sizes_opt_quiescent, yang_sigmas_opt_quiescent = yang_size_fit(
                mgal_bins, z, sftype="quiescent", obs="opt"
            )
            size_ax[iz].plot(
                mgal_bins[yang_sizes_opt_quiescent > 0],
                yang_sizes_opt_quiescent[yang_sizes_opt_quiescent > 0],
                ls=":",
                color="olive",
                alpha=1.0,
            )
            # size_ax[iz].fill_between(mgal_bins[yang_sizes_opt_quiescent>0],
            #                          yang_sizes_opt_quiescent[yang_sizes_opt_quiescent>0]-yang_sigmas_opt_quiescent,
            #                          yang_sizes_opt_quiescent[yang_sizes_opt_quiescent>0]+yang_sigmas_opt_quiescent,
            #                          color='olive',
            #                          alpha=0.3)

            yang_done[iz] = True

        if not ssfr_stuff_done[iz] == True:

            sfr_line = 1e-11 * mgal_bins * 1e6

            sfsm_ax[iz].plot(mgal_bins, sfr_line, "--k")

            quiesc = 0.2 / time  # Myr^-1

            sfsm_ax[iz].plot(mgal_bins, quiesc * mgal_bins, ":k")

            ssfr_stuff_done[iz] = True

        # hmsmr

        avg_mstel, _, _ = binned_statistic(
            mvir, mstels / mvir, bins=mhalo_bins, statistic="mean"
        )
        errs_mstel, _, _ = binned_statistic(
            mvir, mstels / mvir, bins=mhalo_bins, statistic="std"
        )
        counts_mstel, _, _ = binned_statistic(
            mvir, mstels / mvir, bins=mhalo_bins, statistic="count"
        )
        binctrs_mstel, _, _ = binned_statistic(
            mvir, mvir, bins=mhalo_bins, statistic="median"
        )

        if np.any(counts_mstel == 0):
            empty_bins = np.where(counts_mstel == 0)

            binctrs_static = np.diff(mhalo_bins) * 0.5 + mhalo_bins[:-1]

            binctrs_mstel[empty_bins] = binctrs_static[empty_bins]

        color_rgb = colors.to_rgb(plot_colors[isim])
        color_rgba = (*color_rgb, 0.4)
        mvir_colors = np.array([color_rgba for i in range(len(mvir))])

        arg_first_full = np.where(counts_mstel > 0)[0][0]
        arg_empty = np.where(counts_mstel == 0)[0]
        if len(arg_empty) > 1 and arg_empty.max() >= arg_first_full:
            max_full_bin = (binctrs_mstel + np.diff(mhalo_bins) * 0.5)[
                arg_empty[arg_empty > arg_first_full][0] - 1
            ]
            points_full_alpha = np.where(mvir > max_full_bin)[0]
            mvir_colors[points_full_alpha, 3] = 1

        smhmr_ax[iz].scatter(
            mvir,
            mstels / mvir,
            label=sim_dir,
            marker=plot_markers[isim],
            facecolors="none",
            edgecolors=mvir_colors,
        )

        smhmr_ax[iz].errorbar(
            binctrs_mstel,
            avg_mstel,  # * (counts_mstel > 3),
            yerr=errs_mstel,  # * (counts_mstel > 3),
            ls=plot_linestyles[isim],
            color=plot_colors[isim],
        )

        # sfsm

        avg_sfr, _, _ = binned_statistic(mstels, sfrs, bins=mgal_bins, statistic="mean")
        errs_sfr, _, _ = binned_statistic(mstels, sfrs, bins=mgal_bins, statistic="std")
        counts_sfr, _, _ = binned_statistic(
            mstels, sfrs, bins=mgal_bins, statistic="count"
        )
        binctrs_mstel, _, _ = binned_statistic(
            mstels, mstels, bins=mgal_bins, statistic="median"
        )

        color_rgb = colors.to_rgb(plot_colors[isim])
        color_rgba = (*color_rgb, 0.4)
        mstel_colors = np.array([color_rgba for i in range(len(mstels))])

        arg_first_full = np.where(counts_sfr > 0)[0][0]
        arg_empty = np.where(counts_sfr == 0)[0]
        if len(arg_empty) > 1 and arg_empty.max() >= arg_first_full:
            max_full_bin = (binctrs_mstel + np.diff(mgal_bins) * 0.5)[
                arg_empty[arg_empty > arg_first_full][0] - 1
            ]
            points_full_alpha = np.where(mstels > max_full_bin)[0]
            mstel_colors[points_full_alpha, 3] = 1

        sfsm_ax[iz].scatter(
            mstels,
            sfrs,
            label=sim_dir,
            marker=plot_markers[isim],
            facecolors="none",
            edgecolors=mvir_colors,
        )

        sfsm_ax[iz].errorbar(
            binctrs_mstel,
            avg_sfr,  # * (counts_mstel > 3),
            yerr=errs_sfr,  # * (counts_mstel > 3),
            ls=plot_linestyles[isim],
            color=plot_colors[isim],
        )

        if sim.name not in labels_size:
            labels_size.append(sim.name)
            l = [
                Line2D(
                    [],
                    [],
                    markeredgecolor=c["color"],
                    markerfacecolor="none",
                    color=c["color"],
                    ls=c["linestyles"],
                    marker=c["markers"],
                )
                for ic, c in enumerate(default_cycler)
                if ic == isim
            ]
            lines_size.extend(l)
        if sim.name not in labels_hmsmr:
            labels_hmsmr.append(sim.name)
            l = [
                Line2D(
                    [],
                    [],
                    markeredgecolor=c["color"],
                    markerfacecolor="none",
                    color=c["color"],
                    ls=c["linestyles"],
                    marker=c["markers"],
                )
                for ic, c in enumerate(default_cycler)
                if ic == isim
            ]
            lines_hmsmr.extend(l)
        if sim.name not in labels_sfsm:
            labels_sfsm.append(sim.name)
            l = [
                Line2D(
                    [],
                    [],
                    markeredgecolor=c["color"],
                    markerfacecolor="none",
                    color=c["color"],
                    ls=c["linestyles"],
                    marker=c["markers"],
                )
                for ic, c in enumerate(default_cycler)
                if ic == isim
            ]
            lines_sfsm.extend(l)


if "Nedkova+21" not in labels_size:
    labels_size.append("Nedkova+21")
    lines_size.append(Line2D([], [], color="k", ls="--", alpha=0.5))
if "Mowla+19" not in labels_size:
    labels_size.append("Mowla+19")
    lines_size.append(Line2D([], [], color="k", ls="-.", alpha=0.5))
if "Yang+25, SF" not in labels_size:
    labels_size.append("Yang+25, SF")
    lines_size.append(Line2D([], [], color="olive", ls="--", alpha=1.0))
    labels_size.append("Yang+25, quiescent")
    lines_size.append(Line2D([], [], color="olive", ls=":", alpha=1.0))
if f"$sSFR<10^{-11} yr^{-1}$" not in labels_sfsm:
    labels_sfsm.append("sSFR<$10^{-11}$ yr$^{-1}$")
    labels_sfsm.append("$sSFR<0.2/T")
    lines_sfsm.append(Line2D([], [], color="k", ls="--", alpha=1.0))
    lines_sfsm.append(Line2D([], [], color="k", ls=":", alpha=1.0))
if "Paquereau+25" not in labels_hmsmr:
    labels_hmsmr.append("Paquereau+25")
    lines_hmsmr.append(pqro_err)

for plot in [smhmr_ax, size_ax, sfsm_ax]:
    for iax, ax in enumerate(plot):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid()
        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            right=True,
            top=True,
            bottom=True,
            # labelbottom=True,
        )
        # ax.set_aspect("auto")
        if iax < ((nrows * ncols) - overlap):
            ax.set_box_aspect(1)

    if overlap > 0:
        for ax in plot[-overlap:]:
            ax.axis("off")

        legend_args["ncol"] = 1
        if len(sim_dirs) > 4 and overlap > 1:
            legend_args["ncol"] = 2
            # nlegcols = max(int(len(sim_dirs) / 4), 2)


size_ax[-1].legend(labels=labels_size, handles=lines_size, **legend_args)
smhmr_ax[-1].legend(labels=labels_hmsmr, handles=lines_hmsmr, **legend_args)
sfsm_ax[-1].legend(labels=labels_sfsm, handles=lines_sfsm, **legend_args)


for iax in range(ncols * nrows):

    if iax % ncols == 0:
        smhmr_ax[iax].set_ylabel(r"HMSMR")
        # size_ax[iax].set_ylabel(r"R$_{50}$ (kpc)")
        size_ax[iax].set_ylabel(r"R$_{eff}$ (kpc)")
        sfsm_ax[iax].set_ylabel(r"SFR$_{100\, Myr}$ (M$_\odot$/Myr)")
    elif (iax + 1) % ncols == 0:
        ax.tick_params(labelright=True, right=True)

    if iax + 1 > n_plots - ncols:
        for ax in [smhmr_ax[iax], size_ax[iax], sfsm_ax[iax]]:
            ax.tick_params(labelbottom=True, bottom=True)
        size_ax[iax].set_xlabel(r"M$_{\star}$ ($M_{\odot}$)")
        sfsm_ax[iax].set_xlabel(r"M$_{\star}$ ($M_{\odot}$)")
        smhmr_ax[iax].set_xlabel(r"M$_{\rm halo}$ ($M_{\odot}$)")
    #         maj_xticks = ax.get_xticks()
    #         ax.set_xticklabels([f"{xt:.1g}" for xt in maj_xticks])
    elif iax > n_plots - ncols:
        ax.tick_params(labeltop=True, top=True)


fig_name = "figs/"

if len(np.unique(sim_names)) == 1:
    fig_name += f"id{sim_names[0]:d}_"

smhmr_fig.savefig(fig_name + "smhmr.png")
size_fig.savefig(fig_name + "size_mr.png")
sfsm_fig.savefig(fig_name + "sf_sm.png")
