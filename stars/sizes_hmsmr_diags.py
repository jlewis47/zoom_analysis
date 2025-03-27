from ast import arg
from math import gamma
from matplotlib import markers
from matplotlib.patches import Patch
from traitlets import default
from zoom_analysis.stars import sfhs
from zoom_analysis.zoom_helpers import decentre_coordinates, check_in_all_sims_vol

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
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictBH",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_stricterSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_lowerSFE_NHboost",
    # "/data102/jlewis/sims/lvlmax_21/mh1e12/id180130_model6_eps0p05",
    # "/data103/jlewis/sims/lvlmax_21/mh1e12/id180130_256",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
    # # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE",
]

main_stars = True

bin_size = 0.5
zbins = np.arange(1.5, 8.0, bin_size)
zbin_err = 0.2
sims_per_zbins = np.zeros((len(zbins)))
# find_smallest z with 2 sims !
smallest_z = np.inf
for sim_dir in sim_dirs:

    sim = ramses_sim(sim_dir)

    sim_snaps = sim.snap_numbers
    sim_aexps = sim.get_snap_exps(sim_snaps)

    for iz, z in enumerate(zbins):

        snap = sim.get_closest_snap(zed=z)

        assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
        assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

        if not snap in assoc_file_nbs:
            continue

        aexp = sim.get_snap_exps(snap)
        zed_sim = 1.0 / aexp - 1
        if np.abs(zed_sim - z) > zbin_err:
            continue

        sims_per_zbins[iz] += 1


smallest_z = np.where(sims_per_zbins >= min(2, len(sim_dirs)))[0][0]
zbins = zbins[smallest_z:]


mhalo_bins = np.logspace(9, 13, 8)
mgal_bins = np.logspace(6, 12, 12)

ned_done = np.full(len(zbins), False)
mow_done = np.full(len(zbins), False)

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
labels_hmsmr = []

plot_colors = [c["color"] for c in default_cycler]
plot_linestyles = [c["linestyles"] for c in default_cycler]
plot_markers = [c["markers"] for c in default_cycler]

lines_hmsmr = []
lines_size = []

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

size_ax[0].set_ylim(0.01, 10)
smhmr_ax[0].set_ylim(4e-5, 0.5)

size_ax[0].set_xlim(mgal_bins[0], mgal_bins[-1])
smhmr_ax[0].set_xlim(mhalo_bins[0], mhalo_bins[-1])

for isim, sim_dir in enumerate(sim_dirs):

    sim = ramses_sim(sim_dir)

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

        size_ax[iz].text(
            0.95,
            0.1,
            zbin_txt,
            transform=size_ax[iz].transAxes,
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
            continue

        time = sim.cosmo_model.age(1.0 / aexp - 1.0).value * 1e3

        gal_props = get_gal_props_snap(sim_dir, snap, main_stars=True)

        central = gal_props["central"]

        gal_pos = gal_props["pos"].T[central]

        vol_args,min_vol=check_in_all_sims_vol(gal_pos, sim, sim_dirs)

        r50s = gal_props["r50"][central][vol_args] * sim.cosmo.lcMpc * 1e3 * aexp
        mstels = gal_props["mass"][central][vol_args]
        # sfrs = gal_props["sfr1000"][central][vol_args]
        mvir = gal_props["host mass"][central][vol_args]

        # sizes

        avg_size, _, _ = binned_statistic(
            mstels, r50s, bins=mgal_bins, statistic="mean"
        )
        errs_size, _, _ = binned_statistic(
            mstels, r50s, bins=mgal_bins, statistic="std"
        )
        counts_size, _, _ = binned_statistic(
            mstels, r50s, bins=mgal_bins, statistic="count"
        )
        binctr_size, _, _ = binned_statistic(
            mstels, mstels, bins=mgal_bins, statistic="median"
        )

        color_rgb = colors.to_rgb(plot_colors[isim])
        color_rgba = (*color_rgb, 0.4)
        mstel_colors = np.array([color_rgba for i in range(len(mstels))])

        arg_first_full = np.where(counts_size > 0)[0][0]
        arg_empty = np.where(counts_size == 0)[0]
        if len(arg_empty) > 1:
            max_full_bin = (binctr_size + 0.5 * np.diff(mgal_bins))[
                arg_empty[arg_empty > arg_first_full][0] - 1
            ]
            points_full_alpha = np.where(mstels > max_full_bin)[0]
            mstel_colors[points_full_alpha, 3] = 1

        size_ax[iz].scatter(
            mstels,
            r50s,
            label=sim_dir,
            marker=plot_markers[isim],
            facecolors="none",
            edgecolors=mstel_colors,
            # alpha=alphas,
        )

        size_ax[iz].errorbar(
            binctr_size,
            avg_size * (counts_size > 3),
            yerr=errs_size * (counts_size > 3),
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

        color_rgb = colors.to_rgb(plot_colors[isim])
        color_rgba = (*color_rgb, 0.4)
        mvir_colors = np.array([color_rgba for i in range(len(mvir))])

        arg_first_full = np.where(counts_mstel > 0)[0][0]
        arg_empty = np.where(counts_mstel == 0)[0]
        if len(arg_empty) > 1:
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
            avg_mstel * (counts_mstel > 3),
            yerr=errs_mstel * (counts_mstel > 3),
            ls=plot_linestyles[isim],
            color=plot_colors[isim],
        )

        # smhmr_ax[iz].plot(
        #     smhmr_ax[iz].get_xlim(),
        #     np.asarray(smhmr_ax[iz].get_xlim()) * 0.01,
        #     "k--",
        #     alpha=0.3,
        # )

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


if "Nedkova+21" not in labels_size:
    labels_size.append("Nedkova+21")
    lines_size.append(Line2D([], [], color="k", ls="--", alpha=0.5))
if "Mowla+19" not in labels_size:
    labels_size.append("Mowla+19")
    lines_size.append(Line2D([], [], color="k", ls="-.", alpha=0.5))


for plot in [smhmr_ax, size_ax]:
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


for iax in range(ncols * nrows):

    if iax % ncols == 0:
        smhmr_ax[iax].set_ylabel(r"HMSMR")
        size_ax[iax].set_ylabel(r"$R_{50}$ (kpc)")

    if iax + 1 > n_plots - ncols:
        for ax in [smhmr_ax[iax], size_ax[iax]]:
            ax.tick_params(labelbottom=True)
        size_ax[iax].set_xlabel(r"$M_{\star}$ ($M_{\odot}$)")
        smhmr_ax[iax].set_xlabel(r"$M_{\rm halo}$ ($M_{\odot}$)")
    #         maj_xticks = ax.get_xticks()
    #         ax.set_xticklabels([f"{xt:.1g}" for xt in maj_xticks])


smhmr_fig.savefig("figs/smhmr.png")
size_fig.savefig("figs/size_mr.png")
