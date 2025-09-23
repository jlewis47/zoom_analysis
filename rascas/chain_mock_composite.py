from mpl_toolkits.axes_grid1.parasite_axes import parasite_axes_class_factory
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import os

# from astropy.visualization import make_lupton_rgb

from scipy.signal import convolve2d

from rascas_plots import plot_mock_composite

from scipy.ndimage import gaussian_filter
from gremlin.read_sim_params import ramses_sim
from scipy.stats import binned_statistic_2d


def micron_to_fwhm_as(micron):
    x1, x2 = 0.7, 4.5
    y1, y2 = 0.025, 0.14

    sl = (y2 - y1) / (x2 - x1)
    b = y1 - sl * x1

    return sl * micron + b


def mags_for_img(mag, depth):

    mag[mag > depth] = depth
    mag[np.isfinite(mag) == False] = depth
    mag = -mag

    mag = mag - np.min(mag)
    mag = mag / np.max(mag)

    return mag


def plot_and_save_mock_composite(
    field1, field2, field3, save_path, tag, px_as, zed, ang_per_ckpc
):
    fig, ax = plot_mock_composite(
        field1,
        field2,
        field3,
        fig=None,
        ax=None,
    )

    # Background color black
    ax.set_facecolor("black")

    px_len_1as = 1.0 / px_as

    begin_bar = 30
    end_bar = 30 + px_len_1as * 0.5
    y_bar = 20
    x_text = begin_bar + (end_bar - begin_bar) / 2
    y_text = y_bar + 10

    nb_pixels = field1.shape[0]

    ax.text(
        x_text / nb_pixels,
        y_text / nb_pixels,
        f'0.5"/{0.5/ang_per_ckpc:.1f} ckpc',
        color="white",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.plot(
        [begin_bar / nb_pixels, end_bar / nb_pixels],
        [y_bar / nb_pixels, y_bar / nb_pixels],
        color="white",
        lw=2,
        transform=ax.transAxes,
    )

    ax.text(
        0.1,
        0.95,
        f"z={zed:.2f}",
        color="white",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=16,
        alpha=1.0,
    )

    ax.text(
        0.875,
        0.95,
        "F150",
        color="lightskyblue",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=14,
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
        fontsize=14,
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
        fontsize=14,
        alpha=1.0,
    )

    ax.axis("off")

    fname = os.path.join(save_path, f"mock_rgb_{tag}.png")
    print(f"writing {fname}")
    fig.savefig(fname)
    plt.close()


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
        statistic="sum",
        bins=[tgt_xbins, tgt_ybins],
    )[0]

    return new_img.T


spath = "/data101/jlewis/mock_spe"
# sim_path = f"/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/"
sim_path = f"/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/"
# model_name = "zoom_tgt_bc03_chabrier100_wlya"
# model_name = "zoom_tgt_bc03_chabrier100_wlya_noDust"
model_name = "zoom_tgt_bc03_chabrier100_wlya_SMC"
# model_name = "zoom_tgt_bc03_chabrier100_wlya_ndust_SMC"

sim_name = "id180130"
# sim_name = "id52380"
# snap=205
# snap=207
# gal=4
# snap=206
# gal=2
# snap=205
# gal=1
# snap=204
# gal=3
# snap=203
# gal=1
# snap=202
# gal=1
# snap=201
# gal=4

# snap=179
# gal=1

# sim_name = "id26646"
# snap=194
# gal=1

# sim_name = "id242756_novrel_lowerSFE_stgNHboost_strictSF"
# snap=309
# gal=3
# sim_name = "id242756_novrel_lowerSFE_stgNHboost"
# snap=291
# gal=3
# sim_name = "id242756_novrel_lowerSFE"
# snap=284
# gal=4
# sim_name = "id242756_novrel"
# snap=347
# gal=14

# sim_name = "id26646"
# snap = 194
# gal = 1
# snap = 172
# gal = 7

# sim_name = "id52380"
# snap=205
# gal=1
# snap = 179
# gal = 1

# nircam_res_as=0.01
# nircam_res_as=[0.03,0.03,0.06,0.06]
# nircam_res_as=[0.06,0.06,0.06,0.06]
nircam_res_as = [0.03, 0.03, 0.03, 0.03]
# nircam_res_as=0.15

cut = 75
depth = 30

los_nbs = np.arange(12)
# los_nbs= [2]


sim = ramses_sim(os.path.join(sim_path, sim_name))

snaps = os.listdir(os.path.join(sim.path, "rascas", model_name))
snap_nbs = np.sort([int(s.split("_")[1]) for s in snaps])[::-1]


for snap in snap_nbs:

    gals = os.listdir(
        os.path.join(sim.path, "rascas", model_name, f"output_{snap:05d}")
    )
    gal_nbs = [int(g.split("_")[1]) for g in gals if "vignette" not in g]

    for gal in gal_nbs:

        out_dir = os.path.join(
            "/data101/jlewis/color_mocks",
            model_name,
            sim_name,
            f"{snap:d}",
            f"gal_{gal:07d}",
        )

        if not os.path.exists(
            f"/data101/jlewis/mock_spe/{model_name:s}/{sim.name:s}/{snap:d}/gal_{gal:07d}/MJy_images/"
        ):
            continue

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        cur_aexp = sim.get_snap_exps(snap)[0]
        cur_z = 1.0 / cur_aexp - 1.0

        sim.init_cosmo()

        ang_per_ckpc = sim.cosmo_model.arcsec_per_kpc_comoving(cur_z).value

        for los_nb in los_nbs:

            tgt = os.path.join(
                spath,
                model_name,
                sim_name,
                f"{snap:d}",
                f"gal_{gal:07d}",
                "MJy_images",
                f"mock_images_{sim_name}_{snap:d}_{gal:d}_{los_nb:d}_wIGM.h5",
            )

            with h5py.File(tgt, "r") as f:
                print(f.keys())

                print(f["header"].attrs.keys())
                aper_sr = f["header"].attrs["aperture_sr"]
                px_as = f["header"].attrs["pixel_res_as"]
                print(f["header"].keys())

                if not "F444" in f.keys():
                    print(f"warning, missing F444 band in {tgt}")
                    continue

                f115 = f["F115"][:]
                f150 = f["F150"][:]
                f277 = f["F277"][:]
                f444 = f["F444"][:]

            # tgt_size = 400 #ckpc, side of square
            print(aper_sr)  # aperture for rgal but not full image
            tgt_size_as = 15

            wavs = [1.15, 1.5, 2.77, 4.44]  # microns

            px_sr = aper_sr / np.prod(np.shape(f115))
            # px_as = np.sqrt(px_sr) * 3600 * 180 / np.pi
            # px_as = np.sqrt(aper_sr / np.pi) / np.pi * 3600 * 180
            # print(px_as)
            # px_as = tgt_res * ang_per_ckpc
            px_ckpc = px_as / ang_per_ckpc

            # tgt_size_px = tgt_size / px_ckpc
            tgt_size_px = tgt_size_as / px_as

            print(tgt_size_px)

            # chop the images to the same size starting from the center
            f115 = f115[
                int(f115.shape[0] / 2 - tgt_size_px) : int(
                    f115.shape[0] / 2 + tgt_size_px
                ),
                int(f115.shape[1] / 2 - tgt_size_px) : int(
                    f115.shape[1] / 2 + tgt_size_px
                ),
            ]
            f150 = f150[
                int(f150.shape[0] / 2 - tgt_size_px) : int(
                    f150.shape[0] / 2 + tgt_size_px
                ),
                int(f150.shape[1] / 2 - tgt_size_px) : int(
                    f150.shape[1] / 2 + tgt_size_px
                ),
            ]
            f277 = f277[
                int(f277.shape[0] / 2 - tgt_size_px) : int(
                    f277.shape[0] / 2 + tgt_size_px
                ),
                int(f277.shape[1] / 2 - tgt_size_px) : int(
                    f277.shape[1] / 2 + tgt_size_px
                ),
            ]
            f444 = f444[
                int(f444.shape[0] / 2 - tgt_size_px) : int(
                    f444.shape[0] / 2 + tgt_size_px
                ),
                int(f444.shape[1] / 2 - tgt_size_px) : int(
                    f444.shape[1] / 2 + tgt_size_px
                ),
            ]

            print(f115.shape, f150.shape, f277.shape, f444.shape)

            # aperture is smaller now
            aper_sr = (tgt_size_px * px_as * 2 * np.pi / 3600 / 180) ** 2 * np.pi

            bin_facts = nircam_res_as / px_as

            psf = np.asarray([micron_to_fwhm_as(w) / 2.355 for w in wavs[:]])
            psf_pxs = np.asarray([p / px_as for p in psf])

            bin_facts = np.int16(np.ceil(bin_facts))  # Ensure bin_fact is an integer
            # print(bin_fact)
            flux_mjy = [f115, f150, f277, f444]
            rebinned_flux_mjy = [
                rebin_image(fl, bin_fact) for fl, bin_fact in zip(flux_mjy, bin_facts)
            ]

            max_res = np.min(nircam_res_as)
            res_ratios = np.int16(np.ceil(nircam_res_as / max_res))

            print(bin_facts)
            print(max_res, res_ratios)

            # print(f115.shape, f150.shape, f277.shape, f444.shape)
            #
            # print(len(rebinned_flux_mjy), len(rebinned_flux_mjy[0]), len(rebinned_flux_mjy[2]))
            # make sure the rebinned images are the same size by tiling
            # rebinned_flux_mjy = [upscale(fl,res_ratios[i]) for i,fl in enumerate(rebinned_flux_mjy)]
            #
            # print(len(rebinned_flux_mjy), len(rebinned_flux_mjy[0]), len(rebinned_flux_mjy[2]))
            #
            # min_l = np.min(list(map(len, rebinned_flux_mjy)))
            # rebinned_flux_mjy = [fl[:min_l,:min_l] for fl in rebinned_flux_mjy]
            #
            # rebinned_flux_mjy=[]

            # dont rebin like above but use a flat kernel to sum the pixels
            # for i,fl in enumerate(flux_mjy):
            #     kernel_size = res_ratios[i]
            #     kernel = np.ones((kernel_size, kernel_size))
            #     print(flux_mjy[i].shape)
            #     rebinned_flux_mjy.append(convolve2d(fl, kernel, mode='same'))
            #     print(rebinned_flux_mjy[i].shape)

            # f115, f150, f277, f444 = rebinned_flux_mjy

            rebinned_psfd_mjy = [
                gaussian_filter(fl, fwmh_px)
                for fwmh_px, fl in zip(psf_pxs, rebinned_flux_mjy)
            ]

            # mag_lims_10ks=[28.8,29.0,28.7,28.3]

            # mag_times= [74727.78,
            #         66481.95599999999,
            #         22160.652,
            #         20099.196]

            print(px_as, nircam_res_as)

            mag_lims = [
                27.45,
                27.66,
                28.28,
                28.17,
            ]  # from https://cosmos.astro.caltech.edu/page/cosmosweb after 4 visits
            lim_area = 0.15  # as
            # mag_lims = [+2.5*np.log(mag_times[i]/1e4) + mag_lims_10ks[i] for i in range(len(mag_lims_10ks))]

            mag_lims = [
                -2.5
                * np.log10((10 ** (ml / -2.5)) / lim_area**2 * nircam_res_as[i] ** 2)
                for i, ml in enumerate(mag_lims)
            ]

            print(mag_lims)

            fl_names = ["F115", "F150", "F277", "F444"]

            vmin = -50
            vmax = 35  # lower to get rid of noise

            pretty_mags = []
            mags = []
            mags_rebined = []
            mags_rebined_psfd = []

            for (fl_name, fl, fl_rebined, fl_rebined_psfd), ml in zip(
                zip(fl_names, flux_mjy, rebinned_flux_mjy, rebinned_psfd_mjy), mag_lims
            ):

                mag = -2.5 * np.log10(fl * 1e6 * aper_sr / np.prod(np.shape(fl))) + 8.90
                mag_rebinned = (
                    -2.5
                    * np.log10(
                        fl_rebined * 1e6 * aper_sr / np.prod(np.shape(fl_rebined))
                    )
                    + 8.90
                )
                mag_rebinned_psfd = (
                    -2.5
                    * np.log10(
                        fl_rebined_psfd
                        * 1e6
                        * aper_sr
                        / np.prod(np.shape(fl_rebined_psfd))
                    )
                    + 8.90
                )

                single_out = os.path.join(
                    out_dir,
                    fl_name,
                )
                if not os.path.exists(single_out):
                    os.makedirs(single_out)

                fig, ax = plt.subplots(1, 1, figsize=(20, 20), layout="tight")
                i = ax.imshow(mag.T, cmap="Greys", vmax=vmax)
                plt.colorbar(i, ax=ax)
                ax.set_title(f"{fl_name}")
                ax.set_facecolor("black")
                fig.savefig(
                    os.path.join(single_out, f"{fl_name}_{los_nb:d}_pretty.png")
                )
                fig, ax = plt.subplots(1, 1, figsize=(20, 20), layout="tight")
                i = ax.imshow(mag_rebinned.T, cmap="Greys", vmax=ml)
                plt.colorbar(i, ax=ax)
                ax.set_title(f"{fl_name}")
                ax.set_facecolor("black")
                fig.savefig(
                    os.path.join(single_out, f"{fl_name}_{los_nb:d}_rebinned.png")
                )
                fig, ax = plt.subplots(1, 1, figsize=(20, 20), layout="tight")
                i = ax.imshow(mag_rebinned_psfd.T, cmap="Greys", vmax=ml)
                plt.colorbar(i, ax=ax)
                ax.set_title(f"{fl_name}")
                ax.set_facecolor("black")
                fig.savefig(
                    os.path.join(single_out, f"{fl_name}_{los_nb:d}_rebinned_psfd.png")
                )

                pretty_mag = np.copy(mag)
                pretty_mag[pretty_mag < vmin] = vmin
                pretty_mag[pretty_mag > vmax] = vmax
                pretty_mag[np.isfinite(pretty_mag) == False] = vmax
                pretty_mags.append(pretty_mag)

                # print(mag.min(), mag.max())

                mag[mag < vmin] = vmin
                mag[mag > ml] = ml
                mag[np.isfinite(mag) == False] = ml
                mags.append(mag)

                mag_rebinned[mag_rebinned < vmin] = vmin
                mag_rebinned[mag_rebinned > ml] = ml
                mag_rebinned[np.isfinite(mag_rebinned) == False] = ml
                mags_rebined.append(mag_rebinned)

                mag_rebinned_psfd[mag_rebinned_psfd < vmin] = vmin
                mag_rebinned_psfd[mag_rebinned_psfd > ml] = ml
                mag_rebinned_psfd[np.isfinite(mag_rebinned_psfd) == False] = ml
                mags_rebined_psfd.append(mag_rebinned_psfd)

            f115_mag, f150_mag, f277_mag, f444_mag = mags
            (
                f115_mag_rebinned,
                f150_mag_rebinned,
                f277_mag_rebinned,
                f444_mag_rebinned,
            ) = mags_rebined
            (
                f115_mag_rebinned_psfd,
                f150_mag_rebinned_psfd,
                f277_mag_rebinned_psfd,
                f444_mag_rebinned_psfd,
            ) = mags_rebined_psfd

            print(f115_mag.min(), f115_mag.max())

            # fig = plt.figure()
            # plt.hist(np.log10(f444[f444 > 0]), bins=100, histtype="step")
            # plt.yscale("log")
            # fig.savefig(os.path.join(out_dir,"f444_hist.png"))

            # fig = plt.figure()
            # plt.hist(f444_mag.flatten(), bins=100, histtype="step")
            # plt.yscale("log")
            # fig.savefig(os.path.join(out_dir,"f444_mag_hist.png"))

            # fig = plt.figure(figsize=(20, 20), layout="tight")
            # ax = fig.add_subplot(111)
            # ax.set_position([0, 0, 1, 1])  # This line ensures the image spans the full canvas

            # img = ax.imshow(f444_mag, cmap="inferno")
            # cb = plt.colorbar(img)

            # max_norm = np.max([f150_mag, f277_mag, f444_mag])

            # img = make_lupton_rgb(f444, f277, f150, Q=10.0, stretch=0.25)  # , minimum=vmin)

            # print(img.shape)

            mag_img_f115 = mags_for_img(f115_mag, depth)
            mag_img_f150 = mags_for_img(f150_mag, depth)
            mag_img_f277 = mags_for_img(f277_mag, depth)
            mag_img_f444 = mags_for_img(f444_mag, depth)

            # Replace the $SELECTION_PLACEHOLDER$ with the function call
            plot_and_save_mock_composite(
                mag_img_f150,
                mag_img_f277,
                mag_img_f444,
                save_path=out_dir,
                tag=f"{los_nb:d}",
                px_as=px_as,
                zed=cur_z,
                ang_per_ckpc=ang_per_ckpc,
            )

            plt.close()

            mag_img_f115_rebinned = mags_for_img(f115_mag_rebinned, depth)
            mag_img_f150_rebinned = mags_for_img(f150_mag_rebinned, depth)
            mag_img_f277_rebinned = mags_for_img(f277_mag_rebinned, depth)
            mag_img_f444_rebinned = mags_for_img(f444_mag_rebinned, depth)

            print(
                mag_img_f150_rebinned.shape,
                mag_img_f277_rebinned.shape,
                mag_img_f444_rebinned.shape,
            )

            plot_and_save_mock_composite(
                mag_img_f150_rebinned,
                mag_img_f277_rebinned,
                mag_img_f444_rebinned,
                save_path=out_dir,
                tag=f"{los_nb:d}_rebinned",
                px_as=px_as * bin_facts[0],
                zed=cur_z,
                ang_per_ckpc=ang_per_ckpc,
            )

            mag_img_f115_rebinned_psfd = mags_for_img(f115_mag_rebinned_psfd, depth)
            mag_img_f150_rebinned_psfd = mags_for_img(f150_mag_rebinned_psfd, depth)
            mag_img_f277_rebinned_psfd = mags_for_img(f277_mag_rebinned_psfd, depth)
            mag_img_f444_rebinned_psfd = mags_for_img(f444_mag_rebinned_psfd, depth)

            plot_and_save_mock_composite(
                mag_img_f150_rebinned_psfd,
                mag_img_f277_rebinned_psfd,
                mag_img_f444_rebinned_psfd,
                save_path=out_dir,
                tag=f"{los_nb:d}_rebinned_psfd",
                px_as=px_as * bin_facts[0],
                zed=cur_z,
                ang_per_ckpc=ang_per_ckpc,
            )

            pretty_mag_img_f115 = mags_for_img(pretty_mags[0], depth)
            pretty_mag_img_f150 = mags_for_img(pretty_mags[1], depth)
            pretty_mag_img_f277 = mags_for_img(pretty_mags[2], depth)
            pretty_mag_img_f444 = mags_for_img(pretty_mags[3], depth)

            plot_and_save_mock_composite(
                pretty_mag_img_f150,
                pretty_mag_img_f277,
                pretty_mag_img_f444,
                save_path=out_dir,
                tag=f"{los_nb:d}_pretty",
                px_as=px_as,
                zed=cur_z,
                ang_per_ckpc=ang_per_ckpc,
            )
