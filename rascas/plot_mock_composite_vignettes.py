from mpl_toolkits.axes_grid1.parasite_axes import parasite_axes_class_factory
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import os
from astropy.visualization import make_lupton_rgb

from rascas_plots import plot_mock_composite_vignettes

from scipy.ndimage import gaussian_filter
from gremlin.read_sim_params import ramses_sim
from scipy.stats import binned_statistic_2d

spath = "/data101/jlewis/mock_spe"
# sim_path = f"/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/"
sim_path = f"/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/"
# model_name = "zoom_tgt_bc03_chabrier100_wlya"
model_name = "zoom_tgt_bc03_chabrier100_wlya_SMC"


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


def plot_and_save_mock_composite_vignette(
    field1, field2, field3, save_path, tag, px_as, ang_per_ckpc
):
    fig, ax = plot_mock_composite_vignettes(
        field1,
        field2,
        field3,
        fig=None,
        ax=None,
    )

    # Background color black
    ax[0].set_facecolor("black")

    px_len_1as = 1.0 / px_as

    ax[0].text(
        25 + 0.5 * px_len_1as * 0.5 + 1,
        30,
        f'0.5"/{0.5/ang_per_ckpc:.1f} ckpc',
        color="white",
        ha="center",
        va="center",
    )
    ax[0].plot([25, 25 + px_len_1as * 0.5], [20, 20], color="white", lw=2)

    ax[0].text(
        0.875,
        0.95,
        "F150",
        color="lightskyblue",
        ha="left",
        va="top",
        transform=ax[0].transAxes,
        fontsize=12,
        alpha=1.0,
    )
    ax[0].text(
        0.875,
        0.92,
        "F277",
        color="palegreen",
        ha="left",
        va="top",
        transform=ax[0].transAxes,
        fontsize=12,
        alpha=1.0,
    )
    ax[0].text(
        0.875,
        0.89,
        "F444",
        color="lightcoral",
        ha="left",
        va="top",
        transform=ax[0].transAxes,
        fontsize=12,
        alpha=1.0,
    )

    # ax[0].axis("off")

    fig.savefig(os.path.join(save_path, f"mock_rgb_{tag}.png"))
    plt.close()


def rebin_image(image, bin_factor):
    # shape = (image.shape[0] // bin_factor, bin_factor, image.shape[1] // bin_factor, bin_factor)
    # return image.reshape(shape).mean(-1).mean(1)

    cur_shape = image.shape
    tgt_xbins = np.arange(int(cur_shape[0] // bin_factor))
    tgt_ybins = np.arange(int(cur_shape[1] // bin_factor))

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


# # sim_name = "id52380"
# # snap=205
# # gal=1
# snap=207
# gal=4
sim_name = "id180130"
snap = 143
gal = 2

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

nircam_res_as = 0.031

cut = 75
depth = 35

los_nbs = np.arange(12)


out_dir = os.path.join(
    "/data101/jlewis/color_mocks",
    model_name,
    sim_name,
    f"{snap:d}",
    f"gal_{gal:07d}",
    "vignettes",
)

print(out_dir)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

sim = ramses_sim(os.path.join(sim_path, sim_name))

cur_aexp = sim.get_snap_exps(snap)
cur_z = 1.0 / cur_aexp - 1.0

sim.init_cosmo()

ang_per_ckpc = sim.cosmo_model.arcsec_per_kpc_comoving(cur_z).value[0]


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
        f115 = f["F115"][:]
        f150 = f["F150"][:]
        f277 = f["F277"][:]
        f444 = f["F444"][:]

    wavs = [1.15, 1.5, 2.77, 4.44]  # microns

    px_sr = aper_sr / np.prod(np.shape(f115))
    px_as = np.sqrt(px_sr) * 3600 * 180 / np.pi

    bin_fact = nircam_res_as / px_as

    psf = np.asarray([micron_to_fwhm_as(w) / 2.355 for w in wavs[:]])
    psf_pxs = np.asarray([p / px_as for p in psf])

    bin_fact = int(np.round(bin_fact))  # Ensure bin_fact is an integer
    print(bin_fact)
    flux_mjy = [f115, f150, f277, f444]
    rebinned_flux_mjy = [rebin_image(fl, bin_fact) for fl in flux_mjy]
    rebinned_psfd_mjy = [
        gaussian_filter(fl, fwmh_px) for fwmh_px, fl in zip(psf_pxs, rebinned_flux_mjy)
    ]
    # f115, f150, f277, f444 = rebinned_flux_mjy

    mag_lims = [28.8, 29.0, 28.7, 28.3]

    vmin = -50
    vmax = 50  # lower to get rid of noise

    pretty_mags = []
    mags = []
    mags_rebined = []
    mags_rebined_psfd = []

    for (fl, fl_rebined, fl_rebined_psfd), ml in zip(
        zip(flux_mjy, rebinned_flux_mjy, rebinned_psfd_mjy), mag_lims
    ):

        mag = -2.5 * np.log10(fl * 1e6 * aper_sr / np.prod(np.shape(fl))) + 8.90
        mag_rebinned = (
            -2.5 * np.log10(fl_rebined * 1e6 * aper_sr / np.prod(np.shape(fl_rebined)))
            + 8.90
        )
        mag_rebinned_psfd = (
            -2.5
            * np.log10(
                fl_rebined_psfd * 1e6 * aper_sr / np.prod(np.shape(fl_rebined_psfd))
            )
            + 8.90
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
    f115_mag_rebinned, f150_mag_rebinned, f277_mag_rebinned, f444_mag_rebinned = (
        mags_rebined
    )
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
    plot_and_save_mock_composite_vignette(
        mag_img_f150,
        mag_img_f277,
        mag_img_f444,
        save_path=out_dir,
        tag=f"{los_nb:d}",
        px_as=px_as,
        ang_per_ckpc=ang_per_ckpc,
    )

    plt.close()

    mag_img_f115_rebinned = mags_for_img(f115_mag_rebinned, depth)
    mag_img_f150_rebinned = mags_for_img(f150_mag_rebinned, depth)
    mag_img_f277_rebinned = mags_for_img(f277_mag_rebinned, depth)
    mag_img_f444_rebinned = mags_for_img(f444_mag_rebinned, depth)

    plot_and_save_mock_composite_vignette(
        mag_img_f150_rebinned,
        mag_img_f277_rebinned,
        mag_img_f444_rebinned,
        save_path=out_dir,
        tag=f"{los_nb:d}_rebinned",
        px_as=px_as * bin_fact,
        ang_per_ckpc=ang_per_ckpc,
    )

    mag_img_f115_rebinned_psfd = mags_for_img(f115_mag_rebinned_psfd, depth)
    mag_img_f150_rebinned_psfd = mags_for_img(f150_mag_rebinned_psfd, depth)
    mag_img_f277_rebinned_psfd = mags_for_img(f277_mag_rebinned_psfd, depth)
    mag_img_f444_rebinned_psfd = mags_for_img(f444_mag_rebinned_psfd, depth)

    plot_and_save_mock_composite_vignette(
        mag_img_f150_rebinned_psfd,
        mag_img_f277_rebinned_psfd,
        mag_img_f444_rebinned_psfd,
        save_path=out_dir,
        tag=f"{los_nb:d}_rebinned_psfd",
        px_as=px_as * bin_fact,
        ang_per_ckpc=ang_per_ckpc,
    )

    pretty_mag_img_f115 = mags_for_img(pretty_mags[0], depth)
    pretty_mag_img_f150 = mags_for_img(pretty_mags[1], depth)
    pretty_mag_img_f277 = mags_for_img(pretty_mags[2], depth)
    pretty_mag_img_f444 = mags_for_img(pretty_mags[3], depth)

    plot_and_save_mock_composite_vignette(
        pretty_mag_img_f150,
        pretty_mag_img_f277,
        pretty_mag_img_f444,
        save_path=out_dir,
        tag=f"{los_nb:d}_pretty",
        px_as=px_as,
        ang_per_ckpc=ang_per_ckpc,
    )
