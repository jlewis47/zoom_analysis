from multiprocessing import process
from mpl_toolkits.axes_grid1.parasite_axes import parasite_axes_class_factory
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import os

# from astropy.visualization import make_lupton_rgb

from scipy.signal import convolve2d

from zoom_analysis.rascas.rascas_plots import plot_mock_composite

from scipy.ndimage import gaussian_filter
from gremlin.read_sim_params import ramses_sim
from scipy.stats import binned_statistic_2d

import argparse


def micron_to_fwhm_as(micron):
    x1, x2 = 0.7, 4.5
    y1, y2 = 0.025, 0.14

    sl = (y2 - y1) / (x2 - x1)
    b = y1 - sl * x1

    return sl * micron + b


def mags_for_img(mag, depth, zero, norm):

    # mag[mag > depth] = depth
    mag[mag > depth] = np.inf
    mag[np.isfinite(mag) == False] = depth
    mag = -mag

    

    # print(zero,norm)

    # if zero is None: zero = np.nanmin(mag)
    if zero is None: zero = np.nanpercentile(mag,0.05)
    mag = mag - zero
    # if norm is None: norm = np.nanmax(mag)
    if norm is None: norm = np.nanpercentile(mag,99.95)
    mag = mag / norm

    return mag, zero, norm


def plot_and_save_mock_composite(
    field1, field2, field3, save_path, tag, px_as, z, ang_per_ckpc, overwrite=False, **kwargs
):

    outf = os.path.join(save_path, f"mock_rgb_{tag}.png")

    if not overwrite and os.path.exists(outf):
        print(f"Found existing {outf}, skipping...")
        return

    fig, ax = plot_mock_composite(
        field1,
        field2,
        field3,
        fig=None,
        ax=None,
        **kwargs
    )

    # print(field1.min(),field2.max())


    # print(field1.shape)

    # Background color black
    ax.set_facecolor("black")
    clean = kwargs.get('clean',False)

    px_len_1as = 1.0 / px_as

    # print(px_len_1as)

    px_size = field1.shape[0]

    xspot = int(0.1 * px_size)
    yspot = int(0.05 * px_size)

    # print(xspot, px_len_1as)
    # print(yspot)

    # print(field1.shape, px_len_1as, px_as)

    if not clean:

        ax.set_xlabel(r"$y, \mathrm{ckpc/h}$")
        ax.set_ylabel(r"$z, \mathrm{ckpc/h}$")

        delta_txt = 5
        ruler_len = 5
        ax.plot(
            [delta_txt, delta_txt + ruler_len],
            [delta_txt, delta_txt],
            c="white",
            lw=3,
        )
        ax.text(
            delta_txt + ruler_len * 0.5,
            delta_txt + 2,
            "5 kpc/h",
            c="white",
            fontsize=8,
            ha="center",
        )

        ax.text(
            xspot + 0.5 * px_len_1as * 0.5 + 1,
            yspot,
            f'0.5"/{0.5/ang_per_ckpc:.1f} ckpc',
            color="white",
            ha="center",
            va="center",
        )
        ax.plot(
            [xspot, xspot + px_len_1as * 0.5],
            [yspot * 0.7, yspot * 0.7],
            color="white",
            lw=2,
        )

        ax.text(
            0.875,
            0.95,
            "F150",
            color="lightskyblue",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=12,
            alpha=1.0,
        )
        ax.text(
            0.875,
            0.92,
            "F277",
            color="palegreen",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=12,
            alpha=1.0,
        )
        ax.text(
            0.875,
            0.89,
            "F444",
            color="lightcoral",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=12,
            alpha=1.0,
        )
        ax.text(
            0.1,
            0.95,
            f"z={z:.2f}",
            color="white",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=12,
            alpha=1.0,
        )

    ax.axis("off")


    if clean:
        outf=outf[:-4]+'_clean.png'

    print(outf)

    fig.savefig(outf, bbox_inches="tight", pad_inches=-0.1)
    plt.close()

    return fig,ax


def upscale(image, factor):

    new = np.zeros((int(image.shape[0] * factor), int(image.shape[1] * factor)))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new[i * factor : (i + 1) * factor, j * factor : (j + 1) * factor] = image[
                i, j
            ]

    return new


def rebin_image(image, bin_factor):
    # shape = (image.shape[0] // bin_factor, bin_factor, image.shape[1] // bin_factor, bin_factor)
    # return image.reshape(shape).mean(-1).mean(1)

    cur_shape = image.shape
    print(cur_shape, bin_factor)
    tgt_xbins = np.arange(int(np.ceil(cur_shape[0] // bin_factor)))
    tgt_ybins = np.arange(int(np.ceil(cur_shape[1] // bin_factor)))

    X, Y = np.meshgrid(
        np.arange(cur_shape[0]),
        np.arange(cur_shape[1]),
    )

    new_img = binned_statistic_2d(
        np.ravel(X) / bin_factor,
        np.ravel(Y) / bin_factor,
        np.ravel(image),
        statistic=np.nansum,
        bins=[tgt_xbins, tgt_ybins],
    )[0]

    return new_img.T


def make_snap_gal_mock(
    sim,
    model_name,
    snap,
    gal,
    depths=35,
    spath="/data101/jlewis/mock_spe",
    tgt_img_size_as=3.0,
    los_nbs=np.arange(12),
    suffixes=["wIGM.h5", "wIGM_wPSF_rebinned.h5"],
    overwrite=False,
    zstr='Z1p0',
    **kwargs
):

    if type(depths) == int or type(depths) == float:
        depths = [depths, depths, depths]

    norms = kwargs.get('norms',None)
    zeros = kwargs.get('zeros',None)

    if type(norms) not in [list,np.ndarray]:
        norms = [norms]*3
    if type(zeros) not in [list,np.ndarray]:
        zeros = [zeros]*3

    sim_name = sim.name

    ztest = "ZTEST" in model_name

    if sim_name == "NH_for_rascas" and not ztest:
        mock_sim_name = "NewHorizon"
    else:
        mock_sim_name = sim_name


    if not ztest:
        out_dir = os.path.join(
            "/data101/jlewis/color_mocks",
            model_name,
            mock_sim_name,
            f"{snap:d}",
            f"gal_{gal:07d}",
        )
        in_dir =os.path.join(spath,
                        model_name,
                        mock_sim_name,
        )
    else:
        
        out_dir = os.path.join(
            "/data101/jlewis/color_mocks",
            model_name,
            mock_sim_name,
            zstr,
            f"{snap:d}",
            f"gal_{gal:07d}",
        )
        in_dir =os.path.join(spath,
                        model_name,
                        mock_sim_name,
                        zstr,
        )


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not ztest:
        cur_aexp = sim.get_snap_exps(snap)
        cur_z = 1.0 / cur_aexp - 1.0
    else:
        cur_z = [float(zstr.replace('p','.')[1:])]



    sim.init_cosmo()

    ang_per_ckpc = sim.cosmo_model.arcsec_per_kpc_comoving(cur_z).value[0]

    for los_nb in los_nbs:
        for isuffix, mjy_suffix in enumerate(suffixes):

            depth = depths[isuffix]

            tgt = os.path.join(
                in_dir,
                f"{snap:d}",
                f"gal_{gal:07d}",
                "MJy_images",
                f"mock_images_{mock_sim_name}_{snap:d}_{gal:d}_{los_nb:d}_{mjy_suffix}",
            )

            tag = f"{los_nb:d}_{mjy_suffix.replace('.h5','')}"
            outf = os.path.join(tgt, f"mock_rgb_{tag}.png")

            if not overwrite and os.path.exists(outf):
                print(f"Found existing {outf}, skipping...")
                continue

            print(tgt)

            with h5py.File(tgt, "r") as f:
                # print(f.keys())
                print(f["header"].attrs.keys())
                aper_sr = f["header"].attrs["aperture_sr"][()]
                px_as = f["header"].attrs["pixel_res_as"][()]
                band_names = f["header"].attrs["band_names"][()]
                f115 = f["NIRCAM-F115"][:]
                f150 = f["NIRCAM-F150"][:]
                f277 = f["NIRCAM-F277"][:]
                f444 = f["NIRCAM-F444"][:]

            # print(px_as, band_names)

            wavs = [1.15, 1.5, 2.77, 4.44]  # microns

            flux_mjy = [f115, f150, f277, f444]

            mag_lims = [
                27.45,
                27.66,
                28.28,
                28.17,
            ]  # from https://cosmos.astro.caltech.edu/page/cosmosweb after 4 visits
            lim_area = 0.15  # as
            # mag_lims = [+2.5*np.log(mag_times[i]/1e4) + mag_lims_10ks[i] for i in range(len(mag_lims_10ks))]

            uni_res = type(px_as) == np.float64

            if uni_res:
                # px_sr = aper_sr / np.prod(np.shape(f115))
                # px_as = np.sqrt(px_sr) * 3600 * 180 / np.pi
                px_sr = (px_as * 180 / np.pi / 3600.0) ** 2
                # aper_sr = np.prod(flux_mjy[0].shape) * px_sr

                mag_lims = [
                    -2.5
                    * np.log10((10 ** (ml / -2.5)) / lim_area**2 / 4 / np.pi * px_as**2)
                    for i, ml in enumerate(mag_lims)
                ]
                apers = aper_sr

                mags = np.asarray(
                    [-2.5 * np.log10(fl * 1e6 * px_sr) + 8.90 for fl in flux_mjy]
                )

                # print(
                #     np.nanmax(mags[np.isfinite(mags)]),
                #     np.nanmin(mags),
                #     mag_lims,
                #     depths,
                # )

            else:
                # px_sr = [
                #     aper_sr[-4:][ifl] / np.prod(np.shape(fl))
                #     for ifl, fl in enumerate(flux_mjy)
                # ]
                # px_as = np.sqrt(px_sr) * 3600 * 180 / np.pi
                pick_bands = np.asarray(
                    [
                        "NIRCAM-F115",
                        "NIRCAM-F150",
                        "NIRCAM-F277",
                        "NIRCAM-F444",
                    ]
                )
                # print(band_names)
                # print(px_as)
                px_as = [px_as[np.where(band == band_names)[0]] for band in pick_bands]
                aper_sr = [
                    aper_sr[np.where(band == band_names)[0]] for band in pick_bands
                ]
                # print(px_as)
                # print(aper_sr)

                # px_sr = (px_as * 180 / np.pi / 3600.0) ** 2
                # aper_sr = np.prod(flux_mjy[0].shape) * px_sr

                mag_lims = [
                    -2.5
                    * np.log10(
                        (10 ** (ml / -2.5)) / lim_area**2 / 4 / np.pi * px_as[i] ** 2
                    )
                    for i, ml in enumerate(mag_lims)
                ]

                apers = aper_sr[-4:]

                mags = np.asarray(
                    [
                        -2.5 * np.log10(fl * 1e6 * aper / np.prod(np.shape(fl))) + 8.90
                        for fl, aper in zip(flux_mjy, apers)
                    ]
                )

                # print(
                #     np.nanmax(mags[np.isfinite(mags)]),
                #     np.nanmin(mags),
                #     mag_lims,
                #     depths,
                # )

            # print(mag_lims)
            # print(np.sum(np.isinf(mags), axis=(1, 2)))
            # print(np.sum(np.isnan(mags), axis=(1, 2)))

            fl_names = ["NIRCAM-F115", "NIRCAM-F150", "NIRCAM-F277", "NIRCAM-F444"]

            vmin = 5 # lower to get rid of noise?
            # vmax = 28  

            processed_mags = []

            for iband, (mag, ml, fl_name) in enumerate(zip(mags, mag_lims, fl_names)):
                single_out = os.path.join(
                    out_dir,
                    fl_name,
                )
                if not os.path.exists(single_out):
                    os.makedirs(single_out)

                fig, ax = plt.subplots(1, 1, figsize=(20, 20), layout="tight")
                i = ax.imshow(mag.T, cmap="Greys", vmax=ml)
                plt.colorbar(i, ax=ax)
                ax.set_title(f"{fl_name}")
                ax.set_facecolor("black")
                fig.savefig(
                    os.path.join(
                        single_out,
                        f"{fl_name}_{los_nb:d}_{mjy_suffix.replace('.h5','.png')}",
                    )
                )

                # print(np.nanmin(mag), np.nanmax(mag))
                mag[mag < vmin] = vmin
                mag[mag > ml] = ml
                mag[np.isfinite(mag) == False] = ml
                # print(np.nanmin(mag), np.nanmax(mag))

                if uni_res:
                    cur_px_as = px_as
                else:
                    cur_px_as = px_as[iband]

                img_size_px = mag.shape[0]
                half_size_px = int(0.5 * img_size_px)
                img_size_as = cur_px_as * img_size_px

                tgt_frac = tgt_img_size_as / img_size_as

                # print(img_size_as, cur_px_as, tgt_frac)

                tgt_size_px = tgt_img_size_as / cur_px_as

                # print(tgt_size_px, tgt_frac, img_size_px)
                half_tgt_size_px = int(0.5 * tgt_size_px)

                mag = mag[
                    half_size_px - half_tgt_size_px : half_size_px + half_tgt_size_px,
                    half_size_px - half_tgt_size_px : half_size_px + half_tgt_size_px,
                ]  # cut to target size!

                real_size_as = mag.shape[0] * cur_px_as
                # print(cur_px_as)
                # print(real_size_as, tgt_img_size_as, img_size_as)
                # print(mag.shape[0], img_size_px)
                # print()
                processed_mags.append(mag)

            f115_mag, f150_mag, f277_mag, f444_mag = processed_mags

            # print(f115_mag.min(), f115_mag.max())

            # mag_img_f115 = mags_for_img(f115_mag, depth)
            mag_img_f150, zeros[0], norms[0] = mags_for_img(f150_mag, depth, zeros[0], norms[0])
            mag_img_f277, zeros[1], norms[1] = mags_for_img(f277_mag, depth, zeros[1], norms[1])
            mag_img_f444, zeros[2], norms[2] = mags_for_img(f444_mag, depth, zeros[2], norms[2])

            print(zeros,norms)

            if type(px_as) != np.float64:
                if np.all(np.diff(px_as) == 0.0):
                    px_as = px_as[0]
                if len(px_as) == 1:
                    px_as = px_as[0]

            fig,ax=plot_and_save_mock_composite(
                mag_img_f150,
                mag_img_f277,
                mag_img_f444,
                save_path=out_dir,
                tag=tag,
                px_as=px_as,
                z=cur_z[0],
                ang_per_ckpc=ang_per_ckpc,
                overwrite=overwrite,
                **kwargs
            )

            plt.close()

            return fig,ax


if __name__ == "__main__":

    # spath = "/data101/jlewis/mock_spe"
    # # sim_path = f"/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/"
    # sim_path = f"/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/"
    # # model_name = "zoom_tgt_bc03_chabrier100_wlya"
    # # model_name = "zoom_tgt_bc03_chabrier100_wlya_noDust"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_SMC"
    # # model_name = "zoom_tgt_bc03_chabrier100_wlya_ndust_SMC"

    # sim_name = "id52380"
    # # snap=205
    # # snap = 207
    # # gal = 4
    # # snap=206
    # # gal=2
    # # snap=205
    # # gal=1
    # # snap=204
    # # gal=3
    # # snap=203
    # # gal=1
    # snap = 200
    # gal = 1
    # # snap=201
    # # gal=4

    # # sim_name = "id180130"
    # # snap = 143
    # # gal = 2

    # # snap=179
    # # gal=1

    # # sim_name = "id26646"
    # # snap=194
    # # gal=1

    # # sim_name = "id242756_novrel_lowerSFE_stgNHboost_strictSF"
    # # snap=309
    # # gal=3
    # # sim_name = "id242756_novrel_lowerSFE_stgNHboost"
    # # snap=291
    # # gal=3
    # # sim_name = "id242756_novrel_lowerSFE"
    # # snap=284
    # # gal=4
    # # sim_name = "id242756_novrel"
    # # snap=347
    # # gal=14

    # # sim_name = "id26646"
    # # snap = 194
    # # gal = 1
    # # snap = 172
    # # gal = 7

    # # sim_name = "id52380"
    # # snap=205
    # # gal=1
    # # snap = 179
    # # gal = 1

    # # nircam_res_as=0.01
    # # nircam_res_as=[0.03,0.03,0.06,0.06]
    # # nircam_res_as=[0.06,0.06,0.06,0.06]
    # nircam_res_as = [0.03, 0.03, 0.03, 0.03]
    # # nircam_res_as=0.15

    # tgt_img_size_as = 3.0

    # cut = 75
    # depth = 35

    # los_nbs = np.arange(12)

    # out_dir = os.path.join(
    #     "/data101/jlewis/color_mocks", model_name, sim_name, f"{snap:d}", f"gal_{gal:07d}"
    # )

    argParse = argparse.ArgumentParser()

    argParse.add_argument("sim_path", type=str, help="path to sim")
    argParse.add_argument(
        "model",
        type=str,
        help="rascas model, e.g. zoom_tgt_bc03_chabrier100_wlya_ndust_SMC",
    )
    argParse.add_argument(
        "gal_id",
        type=int,
        help="integer id of galaxy in galaxy finder catalogue at this snapshot",
    )
    argParse.add_argument("snap", type=int, help="integer id of this snapshot")
    argParse.add_argument(
        "--size_as",
        type=float,
        help="desired arcseconds aperture of images",
        default=3.0,
    )
    argParse.add_argument(
        "--depth",
        type=float,
        help="desired depth for image color scale",
        default=35,
    )

    args = argParse.parse_args()

    make_snap_gal_mock(
        args.sim_path,
        args.model,
        args.snap,
        args.gal_id,
        args.depth,
        tgt_img_size_as=args.size_as,
        depth=args.depth,
        # los_nbs,
        # out_dir,
    )
