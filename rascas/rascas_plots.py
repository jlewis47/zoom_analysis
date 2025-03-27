import os
from turtle import color
from astropy.extern.ply.yacc import YaccError
import numpy as np

# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

# import napari
from microfilm.microplot import microshow

from scipy.ndimage import convolve

import h5py

from zoom_analysis.rascas.errs import get_cl_err

from zoom_analysis.rascas.filts.filts import (
    norm_band,
    read_transmissions,
    convolve_spe,
    convolve_cube,
    flamb_to_mAB,
    flamb_fnu,
    m_to_M,
)


# def gkern(l=5, sig=1.0):
#     """\
#     creates gaussian kernel with side length `l` and a sigma of `sig`
#     """
#     ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
#     gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
#     kernel = np.outer(gauss, gauss)
#     return kernel / np.sum(kernel)


def plot_spec(wave, spectrum):

    fig = plt.figure(figsize=(10, 6))

    ax = fig.add_subplot(111)

    ax.plot(wave / 1e4, spectrum["spectrum"], label="Spectrum")

    ax.set_xlabel(r"Wavelength ($\mu m$)")

    ax.set_ylabel("Flux")

    ax.grid()

    return fig, ax


def gen_spec(aexp, spec, lambda_bins, filts, rf=False):

    filts_names, filts_wavs, filts_trans = filts
    # filts_names, filts_wavs, filts_trans = read_transmissions()

    mab_spec = flamb_to_mAB(spec, lambda_bins * aexp)
    # print(filts_names)
    # print(mab_spec)

    if rf:

        nircam_spe_wav, nircam_spe = convolve_spe(
            lambda_bins * aexp, spec, filts_wavs, filts_trans
        )
        nircam_mab = flamb_to_mAB(nircam_spe, nircam_spe_wav * aexp)

    else:

        nircam_spe_wav, nircam_spe = convolve_spe(
            lambda_bins, spec, filts_wavs, filts_trans
        )
        nircam_mab = flamb_to_mAB(nircam_spe, nircam_spe_wav * aexp)

    # print(nircam_mab)
    # print(nircam_spe_wav, wav_ctrs)

    print(list(zip(filts_names, nircam_mab)))

    # mag_err = dumb_constant_mag(band_mags, val=0.01)
    # mag_err = dumb_constant_mag(band_mags, val=0.01)
    # _, mag_err, _, _ = get_cl_err(nircam_mab, filts_names)
    new_mag, mag_err, _, _ = get_cl_err(nircam_mab, filts_names)

    # print(list(zip(new_mag, nircam_mab)))
    # print(list(zip(new_mag, nircam_mab)))

    return (filts_names, mab_spec, nircam_mab, new_mag, mag_err)


def gen_imgs(z, lambda_bins, cube, filts):

    filts_names, filts_wavs, filts_trans = filts

    # wav_ctrs = np.asarray([wav[int(0.5 * len(wav))] for wav in filts_wavs])

    aexp = 1.0 / (1.0 + z)

    max_filt, min_filt = 0, np.inf

    for wav in filts_wavs:
        if wav.max() > max_filt:
            max_filt = wav.max()
        if wav.min() < min_filt:
            min_filt = wav.min()

    # print(cube)

    imgs = np.empty((len(filts_names), int(cube["img_npix"]), int(cube["img_npix"])))
    band_ctrs = np.empty(len(filts_names))

    for iband, name in enumerate(filts_names):

        print(f"Getting flux in band {name:s}")
        # if iband != 5:
        # continue
        # if iband == 9:
        band_ctrs[iband], img = convolve_cube(
            lambda_bins,  # aexp,
            # lambda_bins * (1.0 + z),
            cube["cube"],
            [filts_wavs[iband]],
            [filts_trans[iband]],
        )

        imgs[iband, :, :] = img  # norm_band(img, name)

    return filts_names, band_ctrs, imgs


def plot_spe_bands(
    aexp,
    rgal,
    l,
    lambda_bins,
    spec_mag,
    band_mags,
    band_mag_errs,
    imgs,
    band_ctrs,
    filts,
    **kwargs,
):

    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)

    fig = plt.figure(figsize=(20, 7))

    filts_names, filts_wavs, filts_trans = filts

    max_filt, min_filt = 0, np.inf

    for wav in filts_wavs:
        if wav.max() > max_filt:
            max_filt = wav.max()
        if wav.min() < min_filt:
            min_filt = wav.min()

    wav_ctrs = np.asarray([wav[int(0.5 * len(wav))] for wav in filts_wavs])
    norm_wav_ctrs = (wav_ctrs - wav_ctrs.min()) / (wav_ctrs.max() - wav_ctrs.min())
    filt_colors = plt.get_cmap("coolwarm")(norm_wav_ctrs)

    gs = GridSpec(3, len(filts_names), height_ratios=[2, 3, 1], hspace=0.0)
    ax0 = [fig.add_subplot(gs[0, i]) for i in range(len(filts_names))]
    ax1 = fig.add_subplot(gs[1, :])
    ax2 = fig.add_subplot(gs[2, :], sharex=ax1)
    comb_ax = [ax0, ax1, ax2]

    z = 1.0 / aexp - 1.0

    for iband, (name, img) in enumerate(zip(filts_names, imgs)):

        yband = np.linspace(-rgal[0], +rgal[0], img.shape[0]) * l * 1e3 * aexp
        zband = np.linspace(-rgal[0], +rgal[0], img.shape[1]) * l * 1e3 * aexp

        # print(yband.min(), yband.max())
        # print(zband.min(), zband.max())

        # print(yband)

        # print(np.min(img), np.max(img), img.sum())

        ok_vals = img > 0

        if ok_vals.sum() > 0:
            img[ok_vals == False] = img[ok_vals].min()
            mab_img = flamb_to_mAB(img, band_ctrs[iband])  # /aexp?

            # print(np.min(mab_img), np.max(mab_img), mab_img.sum())

            # for visibility, re-normalize between transmissions

            ok_vals = mab_img > -999

            print(mab_img[ok_vals].max(), mab_img[ok_vals].min())

            hist = np.histogram(mab_img[ok_vals], bins=20)
            # hist = np.histogram(np.log10(img[ok_vals]), bins=50)
            # print(list(zip(hist[0], hist[1])))
            cdf = np.cumsum(hist[0]) / np.sum(hist[0])
            if vmin == None:
                vmin = hist[1][np.argmin(np.abs(cdf - 0.1))]
            if vmax == None:
                vmax = hist[1][np.argmin(np.abs(cdf - 0.9))]
            # vmin = 10 ** hist[1][np.argmin(np.abs(cdf - 0.2))]
            # vmax = 10 ** hist[1][np.argmin(np.abs(cdf - 0.8))]
            # vmin = mab_img[ok_vals].min()
            # vmax = mab_img[ok_vals].max()
            # vmax = img[ok_vals].max()

            # hax.plot(hist[1][:-1], hist[0], marker="o", label=name)

            # print(vmin, vmax)

            # mab_img[ok_vals == False] = np.nan

            # mab_img[ok_vals == False] = mab_img[ok_vals].max()

            # print(img.shape)

            # if np.sum(img) > 0:

            if np.any(img < vmax):

                img = comb_ax[0][iband].imshow(
                    # ok_vals,
                    mab_img.T,
                    # img[0],
                    origin="lower",
                    # vmax=0,
                    # vmin=-80,
                    # norm=LogNorm(vmin=vmin, vmax=vmax),
                    vmax=vmax,
                    vmin=vmin,
                    cmap="Greys",
                    extent=[yband[0], yband[-1], zband[0], zband[-1]],
                )

            # print(img.get_clim())

        # colorbar
        # cbar = fig.colorbar(img, ax=comb_ax[0][iband])

        comb_ax[0][iband].set_xlim(yband[0], yband[-1])
        comb_ax[0][iband].set_ylim(zband[0], zband[-1])

        comb_ax[0][iband].set_aspect("equal")

        # bad pixels black
        comb_ax[0][iband].set_facecolor("black")

        comb_ax[0][iband].set_title(name)

        if iband == 0:
            comb_ax[0][iband].tick_params(
                axis="both",
                which="both",
                bottom=False,
                labelbottom=False,
            )
            comb_ax[0][iband].set_ylabel(r"$z, \mathrm{kpc}$")
        else:
            comb_ax[0][iband].tick_params(
                axis="both",
                which="both",
                bottom=False,
                labelbottom=False,
                left=False,
                labelleft=False,
            )

    # hax.legend(framealpha=0.0)
    # hfig.savefig(f"hist_img.png")

    ok = spec_mag > -999
    # ok = mab_spec > 0
    comb_ax[1].plot(lambda_bins[ok] / 1e4, spec_mag[ok], c="k")
    # comb_ax[1].plot(lambda_bins, spe_MAB)
    # comb_ax[1].set_xlabel(r"$\lambda_{obs}, \AA$")
    comb_ax[1].set_ylabel("$\mathrm{m_{AB}}$")
    # comb_ax[1].set_ylabel("AB magnitude")
    # comb_ax[1].set_yscale("log")
    # comb_ax[1].set_ylim(1, np.max(spec["spectrum"]) * 1.5)
    # comb_ax[1].set_xlim(lambda_bins[0] / 1e4, lambda_bins[-1] / 1e4)
    comb_ax[1].set_xlim(min_filt / 1e4, max_filt / 1e4)
    # comb_ax[1].set_xlim(lambda_bins[0] / 1e4, max_filt / 1e4)
    # comb_ax[1].set_ylim(spec_mag[ok].min() - 0.5, spec_mag[ok].max() + 0.5)

    spec_min = spec_mag[np.argmin(np.abs(lambda_bins[0] - min_filt))]

    comb_ax[1].set_ylim(
        band_mags[np.isfinite(band_mags)].min() - 0.5,
        max(band_mags[np.isfinite(band_mags)].max(), spec_min) + 0.5,
    )
    comb_ax[1].grid()

    comb_ax[1].tick_params(bottom=False, labelbottom=False)

    # nircam_spe_wav, nircam_spe = convolve_spe(
    #     lambda_bins, spec["spectrum"], filts_wavs, filts_trans
    # )

    # nircam_mab = flamb_to_mAB(nircam_spe, nircam_spe_wav * aexp)

    # # mag_err = dumb_constant_mag(band_mags, val=0.01)
    # # mag_err = dumb_constant_mag(band_mags, val=0.01)
    # new_mag, mag_err, _, _ = get_cl_err(nircam_mab, filts_names)

    # nircam_spe_mab = flamb_to_mAB(nircam_spe, nircam_spe_wav)

    # print(nircam_spe_wav, nircam_spe, nircam_spe_mab)

    for nirwav, nirmag, nirmag_err, color in zip(
        band_ctrs, band_mags, band_mag_errs, filt_colors
    ):
        ok = nirmag > -999
        comb_ax[1].errorbar(
            nirwav[ok] / 1e4,
            nirmag[ok],
            yerr=nirmag_err[ok],
            ls="none",
            marker="o",
            mfc="none",
            color=color,
            lw=5,
            elinewidth=1.5,
            ms=15,
        )
    # comb_ax[0].plot(nircam_spe_wav, nircam_spe_mab, ls="none", marker="o", mfc="none")

    comb_ax[1].invert_yaxis()

    comb_ax[2].set_xlabel(r"$\lambda_{obs}, \mu m$")
    comb_ax[2].set_ylabel("Transmission")
    comb_ax[2].set_yscale("log")

    comb_ax[2].grid()

    for name, wav, tran, color in zip(
        filts_names, filts_wavs, filts_trans, filt_colors
    ):
        # norm = simps(tran, wav)
        # print(np.max(tran / norm), tran.max(), norm)
        # print(comb_ax[1].get_xlim(), wav.min(), wav.max())
        line = comb_ax[2].plot(wav / 1e4, tran, label=name, color=color)
        if wav.min() / 1e4 < min(comb_ax[1].get_xlim()) or wav.max() / 1e4 > max(
            comb_ax[1].get_xlim()
        ):
            continue
        arg = int(len(wav) / 2.0)
        comb_ax[2].text(
            wav[arg] / 1e4,
            tran[arg] * 0.9,
            name,
            # color=line[0].get_color(),
            color=color,
            size=8,
            ha="center",
        )

    comb_ax[2].tick_params(
        axis="both", which="both", direction="in", left=False, labelleft=False
    )

    comb_ax[1].set_xscale("log")
    comb_ax[2].set_xscale("log")

    plt.tight_layout()
    return fig, comb_ax


def plot_mock_composite(b, g, r, yrascas=None, zrascas=None, **kwargs):

    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

    # pl_img = axs[0].imshow(
    #     bw_img.T, origin="lower", cmap="gray", norm=LogNorm(vmin=1), interpolation=None
    # )
    # print(yrascas[0] - ctr_gal[1], l)

    # brightest_mag = np.min([r, g, b])

    if yrascas is None or zrascas is None:
        extent = None
        axs.imshow(
            # color_img_norm,
            np.transpose([r.T, g.T, b.T]),
            origin="lower",
            # vmin=0,
            # vmax=np.max(),
            # norm=LogNorm(vmin=1),
            **kwargs,
        )
    else:
        extent = [yrascas[0], yrascas[-1], zrascas[0], zrascas[-1]]
        axs.imshow(
            # color_img_norm,
            np.transpose([r.T, g.T, b.T]),
            origin="lower",
            # vmin=0,
            # vmax=np.max(),
            # norm=LogNorm(vmin=1),
            extent=extent,
            **kwargs,
        )

        axs.set_xlabel(r"$y, \mathrm{ckpc/h}$")
        axs.set_ylabel(r"$z, \mathrm{ckpc/h}$")

        delta_txt = 5
        ruler_len = 5
        axs.plot(
            [yrascas[0] + delta_txt, yrascas[0] + delta_txt + ruler_len],
            [zrascas[0] + delta_txt, zrascas[0] + delta_txt],
            c="white",
            lw=3,
        )
        axs.text(
            yrascas[0] + delta_txt + ruler_len * 0.5,
            zrascas[0] + delta_txt + 2,
            "5 kpc/h",
            c="white",
            fontsize=8,
            ha="center",
        )

    # axs.imshow(
    #     r,
    #     alpha=0.33,
    #     vmax=1,
    #     vmin=0,
    #     origin="lower",
    #     # norm=LogNorm(vmin=1),
    #     extent=[yrascas[0], yrascas[-1], zrascas[0], zrascas[-1]],
    #     cmap="Reds",
    # )
    # axs.imshow(
    #     b,
    #     alpha=0.33,
    #     vmax=1,
    #     vmin=0,
    #     origin="lower",
    #     # norm=LogNorm(vmin=1),
    #     extent=[yrascas[0], yrascas[-1], zrascas[0], zrascas[-1]],
    #     cmap="Blues",
    # )
    # axs.imshow(
    #     g,
    #     alpha=0.33,
    #     vmax=1,
    #     vmin=0,
    #     origin="lower",
    #     # norm=LogNorm(vmin=1),
    #     extent=[yrascas[0], yrascas[-1], zrascas[0], zrascas[-1]],
    #     cmap="Greens",
    # )

    axs.set_facecolor("black")

    # axs.imshow(color_img.T, origin="lower", norm=LogNorm())
    # pl_img = ax.imshow(bw_img.T, origin="lower", cmap="gray")

    # colorbar
    # cbar = fig.colorbar(pl_img, ax=axs)

    return fig, axs


# def plot_mock_composite_napari(bands, names, colors):

#     # create viewer instance
#     viewer = napari.Viewer()

#     # add channels one by one
#     for iband, (band, c, n) in enumerate(zip(bands, colors, names)):
#         viewer.add_image(band, colormap=c, name=n, blending="additive")

#     img_RGB = viewer.layers.RGB_image()

#     fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

#     img_RGB = img_RGB.data / np.max(img_RGB.data)

#     axs.imshow(
#         # color_img_norm,
#         np.transpose(img_RGB),
#         origin="lower",
#         vmin=0,
#         vmax=1,
#         # norm=LogNorm(vmin=1),
#     )

#     axs.set_facecolor("black")

#     fig.savefig('mock_test_napari.png')


def plot_mock_composite_microfilm(bands, colors):

    microim = microshow(
        images=bands[:, :, :],
        fig_scaling=5,
        cmaps=colors,
        # unit="um",
        # scalebar_size_in_units=0,
        # # scalebar_unit_per_pix=0.065,
        # # scalebar_font_size=20,
        # label_text="",
        # # label_font_size=0.04,
    )

    microim.savefig("./microfilm_test", bbox_inches="tight", pad_inches=0, dpi=600)


def mags_from_hdf5(fname, bands=None, rest_frame=False):

    if rest_frame:
        rf_key = "_rf"
    else:
        rf_key = ""

    with h5py.File(fname, "r") as src:

        spe_data = src["speectral_data"]

        band_fnames = spe_data["bands"][()]
        if bands != None:
            band_fnames = [name.decode("utf-8") for name in band_fnames]
        else:
            bands = band_fnames

        mags, mag_errs = np.empty(len(bands)), np.empty(len(bands))

        for iband, band in enumerate(bands):

            band_arg = np.where(np.in1d(band_fnames, band))[0]

            # print(type(band), type(band_fnames[0]))

            # print(band, band_fnames, band_arg)

            if len(band_arg) == 0:
                print(f"Band {band:s} not found in file")
                continue

            mags[iband] = spe_data["mag" + rf_key][band_arg]
            mag_errs[iband] = spe_data["mag_err" + rf_key][band_arg]

    return list(zip(bands, mags, mag_errs))


# In [90]: nuvs=np.empty(len(files_to_read))
#     ...: js=np.empty(len(files_to_read))
#     ...: rs=np.empty(len(files_to_read))
#     ...: zs=np.empty(len(files_to_read))
#     ...:
#     ...: for ifile,fname in enumerate(files_to_read):
#     ...:     with h5py.File(fname,'r') as src:
#     ...:         zs[ifile]=src['header']['z'][()]
#     ...:     mags=mags_from_hdf5(fname,["NUV","rHSC","K"])
#     ...:     nuvs[ifile],rs[ifile],js[ifile]=mags[0][1],mags[1][1],mags[2][1]
