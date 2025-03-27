import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import os
from astropy.visualization import make_lupton_rgb

from rascas_plots import plot_mock_composite


spath = "/data101/jlewis/mock_spe"
model_name = "zoom_tgt_bc03_chabrier100_wlya"


def mags_for_img(mag, depth):

    mag[mag > depth] = depth
    mag[np.isfinite(mag) == False] = depth
    mag = -mag

    mag = mag - np.min(mag)
    mag = mag / np.max(mag)

    return mag

sim_name = "id242756_novrel_lowerSFE_stgNHboost_strictSF"
snap=169
gal=31

# sim_name = "id26646"
# snap = 194
# gal = 1
# snap = 172
# gal = 7

# sim_name = "id52380"
# snap = 179
# gal = 1

cut = 75
depth = 35

los_nbs = np.arange(12)

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
        f444 = f["F277"][:]

    px_sr = aper_sr / np.prod(np.shape(f115))
    px_as = np.sqrt(px_sr) * 3600 * 180 / np.pi

    flux_mjy = [f115, f150, f277, f444]

    vmin = 1
    vmax = 50

    mags = []

    for fl in flux_mjy:

        mag = -2.5 * np.log10(fl * 1e6 * aper_sr / np.prod(np.shape(fl))) + 8.90

        mag[mag < vmin] = vmin
        mag[mag > vmax] = vmax

        mag[np.isfinite(mag) == False] = vmax

        mags.append(mag)

    f115_mag, f150_mag, f277_mag, f444_mag = mags

    print(f115_mag.min(), f115_mag.max())

    fig = plt.figure()
    plt.hist(np.log10(f444[f444 > 0]), bins=100, histtype="step")
    plt.yscale("log")
    fig.savefig("f444_hist.png")

    fig = plt.figure()
    plt.hist(f444_mag.flatten(), bins=100, histtype="step")
    plt.yscale("log")
    fig.savefig("f444_mag_hist.png")

    fig = plt.figure(figsize=(10, 10), layout="tight")
    ax = fig.add_subplot(111)
    ax.set_position([0, 0, 1, 1])  # This line ensures the image spans the full canvas

    img = ax.imshow(f444_mag, cmap="inferno")
    cb = plt.colorbar(img)

    max_norm = np.max([f150_mag, f277_mag, f444_mag])

    img = make_lupton_rgb(f444, f277, f150, Q=10.0, stretch=0.25)  # , minimum=vmin)

    print(img.shape)

    mag_img_f115 = mags_for_img(f115_mag, depth)
    mag_img_f150 = mags_for_img(f150_mag, depth)
    mag_img_f277 = mags_for_img(f277_mag, depth)
    mag_img_f444 = mags_for_img(f444_mag, depth)

    # fig, ax = plot_mock_composite(f150, f277, f444, vmin=1e-4, vmax=1e1)
    fig, ax = plot_mock_composite(
        mag_img_f115,
        mag_img_f150,
        mag_img_f277,
        # mag_img_f444,
        # f150_mag / f150_mag.max(),
        # f277_mag / f277_mag.max(),
        # f444_mag / f444_mag.max(),
        # img[cut:-cut, cut:-cut, 0],
        # img[cut:-cut, cut:-cut, 1],
        # img[cut:-cut, cut:-cut, 2],
    )

    # bg color black
    ax.set_facecolor("black")

    px_len_1as = 1.0 / px_as

    ax.text(
        10 + 0.5 * px_len_1as + 2, 17, f'1"', color="white", ha="center", va="center"
    )
    ax.plot([10, 10 + px_len_1as], [10, 10], color="white", lw=2)

    ax.text(
        0.875,
        0.95,
        "F115",
        color="lightskyblue",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.8,
    )
    ax.text(
        0.875,
        0.92,
        "F150",
        color="palegreen",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.8,
    )
    ax.text(
        0.875,
        0.89,
        "F277",
        color="lightcoral",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.8,
    )

    ax.axis("off")

    fig.savefig(f"mock_rgb_{los_nb:d}")

    plt.close()
