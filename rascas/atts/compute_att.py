from tkinter.ttk import Separator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.optimize import curve_fit


import os
import h5py
from numpy._typing import _128Bit

from gremlin.read_sim_params import ramses_sim

# def rascas_SMC_ext():


def plot_lephare_ext_laws(ax, names=None):

    ext_dir = "/home/mgarnichey/LEPHARE/ext/"

    if names is None:
        names = ["SB_calzetti.dat"]
    elif names == "all":

        names = [f for f in os.listdir(ext_dir) if f.endswith(".dat")]

    for name in names:

        try:
            data = np.loadtxt(os.path.join(ext_dir, name))
        except:
            data = np.loadtxt(os.path.join(ext_dir, name), delimiter=",")

        ax.plot(1.0 / data[:, 0] * 1e4, data[:, 1], label=name)  # inverse micrometers


def fit_fct(x, a, b):
    return a * np.log10(x) + b


def fit_attenuation(inv_wav, att):

    # use poly fit to fit attenuation curve
    # return the coefficients of the fit

    isfinite = np.isfinite(att)

    # def fit_fct(x, a, b, c):
    #     return abs(a) * x**2 + abs(b) * x + c

    return curve_fit(fit_fct, inv_wav[isfinite], att[isfinite], p0=[1, 1])[0]

    # return np.polyfit(inv_wav[isfinite], att[isfinite], deg=3)


def detect_mocks(path):

    # print(path)

    snaps = []
    galaxies = []
    mock_paths = []

    for snap in os.listdir(path):

        snaps.append(snap)
        # print(snap, os.listdir(os.path.join(path, snap)))

        for gal in os.listdir(os.path.join(path, snap)):

            galaxies.append(gal)
            # print(gal, os.listdir(os.path.join(path, snap, gal)))

            mock_list = np.asarray(
                [
                    f
                    for f in os.listdir(os.path.join(path, snap, gal))
                    if f.endswith("h5") and "spectrum" in f
                ]
            )

            mock_list_dir_nb = [int(f.split("_")[-1].split(".")[0]) for f in mock_list]

            for mocks in mock_list[np.argsort(mock_list_dir_nb)]:

                # print(mocks)

                mock_paths.append(os.path.join(path, snap, gal, mocks))

    mock_path_snaps = np.array([int(f.split("/")[-3]) for f in mock_paths])
    mock_path_galaxies = np.array([f.split("/")[-2] for f in mock_paths])

    return mock_path_snaps, mock_path_galaxies, mock_paths


def read_h5_band_data(mock_path):

    with h5py.File(mock_path, "r") as f:

        spe = f["speectral_data"]

        bands = spe["bands"][()]
        band_ctrs = spe["lam"][()] / 1e4  # micrometers
        mags = spe["mag"][()]

    return bands, band_ctrs, mags


def compute_att(mock_path1, mock_path2):

    bands1, band_ctrs1, mags1 = read_h5_band_data(mock_path1)
    bands2, band_ctrs2, mags2 = read_h5_band_data(mock_path2)

    atts = mags1 - mags2

    assert np.all(bands1 == bands2), "Band names do not match"
    assert np.all(band_ctrs1 == band_ctrs2), "Band centers do not match"

    return bands1, band_ctrs1, atts


def plot_att(ax, band_ctrs=None, atts=None):

    if band_ctrs is not None and atts is not None:
        ax.plot(1 / band_ctrs, atts)

    ax.set_xscale("log")
    ax.grid()

    ax.set_xlabel(f"Inverse wavelength, $\mu$m^{-1}")
    ax.set_ylabel("Attenuation/A(1.0 microns), AB magnitude")


if __name__ == "__main__":

    # sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id74099"

    # path_nodust = (
    #     "/data101/jlewis/mock_spe/zoom_tgt_bpass_kroupa100_wlya_noDust/id_novrel/"
    # )
    # path_dust = "/data101/jlewis/mock_spe/zoom_tgt_bpass_kroupa100_wlya/id_novrel/"
    path_nodust = "/data101/jlewis/mock_spe/zoom_tgt_bc03_chabrier100_wlya_noDust/id242756_novrel/"
    path_dust = (
        "/data101/jlewis/mock_spe/zoom_tgt_bc03_chabrier100_wlya/id242756_novrel/"
    )

    sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel/"
    # path_nodust = (
    #     "/data101/jlewis/mock_spe/zoom_tgt_bpass_kroupa100_wlya_noDust/id242756_novrel/"
    # )
    # path_dust = (
    #     "/data101/jlewis/mock_spe/zoom_tgt_bpass_kroupa100_wlya/id242756_novrel/"
    # )

    skip = 5

    max_z = 5.0

    sim = ramses_sim(sim_path)

    ndirs_per_gal = 12

    snaps_nodust, galaxies_nodust, mock_paths_nodust = detect_mocks(path_nodust)
    snaps_dust, galaxies_dust, mock_paths_dust = detect_mocks(path_dust)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharex=False, sharey=True)

    u_snaps_nodust = np.unique(snaps_nodust)
    u_snaps_dust = np.unique(snaps_dust)
    matching_snaps = np.intersect1d(u_snaps_nodust, u_snaps_dust)

    ncolors = int(len(matching_snaps) / skip)
    redshifts = np.zeros(ncolors, dtype=float)

    for isnap, snap in enumerate(matching_snaps[::skip]):

        zed = 1.0 / sim.get_snap_exps(snap) - 1
        redshifts[isnap] = zed

    # cmap = plt.cm.viridis
    cmap = plt.cm.Spectral
    # get colors from redshifts
    colors = cmap(np.linspace(0, 1, ncolors))

    all_atts = []
    all_inv_wav = []

    for isnap, snap in enumerate(matching_snaps[::skip]):

        zed = redshifts[isnap]
        if zed > max_z:
            continue
        print(f"Snap: {snap}, z: {zed}")

        mock_files_nodust = np.array(mock_paths_nodust)[snaps_nodust == snap]
        mock_files_dust = np.array(mock_paths_dust)[snaps_dust == snap]

        galaxies_nodust_snap = np.array(galaxies_nodust)[snaps_nodust == snap]
        galaxies_dust_snap = np.array(galaxies_dust)[snaps_dust == snap]

        matching_galaxies = np.intersect1d(
            np.unique(galaxies_nodust_snap), np.unique(galaxies_dust_snap)
        )

        for gal in matching_galaxies:

            mock_files_dust = np.array(mock_paths_dust)[
                (snaps_dust == snap) & (galaxies_dust == gal)
            ]
            mock_files_nodust = np.array(mock_paths_nodust)[
                (snaps_nodust == snap) & (galaxies_nodust == gal)
            ]

            atts_all_dir = []

            for idir in range(ndirs_per_gal):
                band_names, band_ctrs, atts = compute_att(
                    mock_files_dust[idir], mock_files_nodust[idir]
                )

                atts_all_dir.append(atts)
                all_atts.append(atts)
                all_inv_wav.append(1.0 / band_ctrs)

            one_micron_arg = np.argmin(np.abs(band_ctrs - 1.0))

            for atts in all_atts:
                atts /= atts[one_micron_arg]

            for atts in atts_all_dir:
                atts /= atts[one_micron_arg]

            med_atts = np.nanmedian(atts_all_dir, axis=0)
            p95_atts = np.nanpercentile(atts_all_dir, 95, axis=0)
            p05_atts = np.nanpercentile(atts_all_dir, 5, axis=0)

            ax[0].plot(
                1 / band_ctrs, med_atts, color=colors[isnap]
            )  # , label=f"{snap}_{gal}")
            # ax[0].fill_between(
            #     1 / band_ctrs,
            #     p05_atts,
            #     p95_atts,
            #     alpha=0.33,
            #     # label=f"{snap}_{gal}",
            #     color=colors[isnap],
            # )

    ax[0].set_ylim(np.nanmin(p05_atts) - 0.25, np.nanmax(p95_atts) + 0.25)
    # ax[0].set_ylim(np.nanmin(p05_atts) - 0.25, np.nanmax(p95_atts) + 0.25)
    ax[0].set_xlim(np.nanmin(1 / band_ctrs) * 0.9, np.nanmax(1 / band_ctrs) * 1.1)

    for name, ctr in zip(band_names, band_ctrs):

        ax[0].text(
            1 / ctr,
            5.0,
            name,
            rotation=45,
            fontsize=8,
            ha="center",
            va="bottom",
        )

    plot_att(ax[0])
    plot_att(ax[1])

    ax[0].grid()

    # create colorbar axis on the right side of the plot using plot divider
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # color bar using colors with correct redshift values
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax)
    cb.set_label("Redshift")
    cb.set_ticks(np.linspace(0, 1, ncolors))
    cb.set_ticklabels(np.round(redshifts, decimals=2))

    # plot_lephare_ext_laws(ax, names=None)
    plot_lephare_ext_laws(ax[0], names="all")

    # make a legend outside of the plot
    ax[0].legend(loc="center left", bbox_to_anchor=(0, 0.5), framealpha=0.0)

    # make a fit to all of the data
    fit_params = fit_attenuation(
        np.concatenate(all_inv_wav), att=np.concatenate(all_atts)
    )

    # ax[1].plot(1.0 / band_ctrs, np.polyval(fit_params, 1.0 / band_ctrs), label="Fit")
    ax[1].scatter(np.concatenate(all_inv_wav), np.concatenate(all_atts), s=1)

    # ax.legend(framealpha=0.0)

    wavs_lephare_ext, ext_lephare_ext_calzetii = wavs_ang = np.loadtxt(
        os.path.join("/home/mgarnichey/LEPHARE/ext/", "SB_calzetti.dat")
    ).T
    ax[1].plot(
        1.0 / wavs_lephare_ext * 1e4,
        # np.polyval(fit_params, 1.0 / wavs_lephare_ext * 1e4),
        fit_fct(1.0 / wavs_lephare_ext * 1e4, *fit_params),
        # label="Fit",
    )

    ax[1].legend(framealpha=0.0)
    ax[1].plot(1.0 / wavs_lephare_ext * 1e4, ext_lephare_ext_calzetii, label="Calzetti")

    # plt.tight_layout()

    fig.savefig("test_att")
