import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import h5py

from gremlin.read_sim_params import ramses_sim

from zoom_analysis.rascas.rascas_steps import get_directions_cart

from zoom_analysis.read.read_data import read_data_ball
from zoom_analysis.visu.visu_fct import (
    plot_fields,
    plot_stars,
    basis_from_vect,
    plot_zoom_gals,
)
from zoom_analysis.halo_maker.assoc_fcts import get_gal_props_snap

# test_sim = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"
test_sim = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130"
# test_sim = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380"
model_name = "zoom_tgt_bc03_chabrier100_wlya_SMC"

sim = ramses_sim(test_sim, nml="cosmo.nml")

sim.init_cosmo()

import healpy as hp

rascas_dirs = get_directions_cart(hp.npix2nside(12))


fig, axs = plt.subplots(
    len(rascas_dirs),
    2,
    figsize=(
        2 * 6,
        int(12 * 6),
    ),
    layout="constrained",
)
snap = 160
# snap = 158
# snap = 193
# snap = 199
aexp = sim.get_snap_exps(snap)

# gprops = get_gal_props_snap(sim.path, snap)

# arg = np.argmax(gprops["mass"])

# tgt_pos = gprops["pos"][:, arg]
# tgt_r = gprops["rmax"][arg]
# hid = gprops["host hid"][arg]
# gid = gprops["gids"][arg]

# print(gid)

gid = 2
gprops = get_gal_props_snap(sim.path, snap, gid=gid)

tgt_pos = gprops["pos"]
tgt_r = gprops["rmax"]

outp = os.path.join(
    "/data101/jlewis",
    "mock_spe",
    model_name,
    f"{sim.name:s}",
    f"{snap:d}",
    f"gal_{gid:07d}",
)

img_dir = os.path.join(outp, "MJy_images")

for idir in range(len(rascas_dirs)):
    # for idir in range(3):

    cur_dir = rascas_dirs[idir]
    cur_dir[np.abs(cur_dir) < 1e-10] = 0.0
    cur_axs = axs[idir, :]

    cur_axs[0].grid()
    cur_axs[1].grid()

    basis = basis_from_vect(cur_dir)
    # print(cur_dir)
    # print(basis)

    # plot_fields(  # most time spent in loaded data and make_img_stars and make_img_cic
    #     "stellar mass",
    #     fig,
    #     cur_axs[0],
    #     aexp,
    #     basis,
    #     tgt_pos,
    #     tgt_r,
    #     sim,
    #     hid=hid,
    #     cb=False,
    #     transpose=True,
    #     origin="lower",
    #     vmax=1e8,
    #     vmin=1e5,
    # )

    # plot_fields(  # all time is spent in fill_pixels
    #     "density",
    #     fig,
    #     cur_axs[1],
    #     aexp,
    #     basis,
    #     tgt_pos,
    #     tgt_r,
    #     sim,
    #     hid=hid,
    #     cb=False,
    #     transpose=True,
    #     origin="lower",
    #     # vmax=1e7,
    #     # vmin=1e5,
    # )

    fname = os.path.join(
        img_dir,
        # f"mock_images_{sim.name:s}_{snap}_{gid}_{idir:d}_noIGM.h5",
        # f"mock_images_{sim.name:s}_{snap}_{gid}_{idir:d}_wIGM_wPSF.h5",
        f"mock_images_{sim.name:s}_{snap}_{gid}_{idir:d}_wIGM_wPSF_rebinned.h5",
    )

    band_name = "NIRCAM-F444"

    print(fname)

    with h5py.File(fname, "r") as src:
        # print(list(src["header"].attrs.items()))
        f444 = src[band_name][()]
        rhostar = src["stellar_surface density"][()]
        flux_res_as = src["header"].attrs["pixel_res_as"]
        star_res_as = src["header"].attrs["pixel_res_stars_as"]
        band_names = src["header"].attrs["band_names"]

    print(band_names, flux_res_as)

    band_arg = band_names.tolist().index(band_name)

    if type(flux_res_as) in [list, np.ndarray]:
        flux_res_as = flux_res_as[band_arg]

    shape_star = rhostar.shape[0]
    shape_flux = f444.shape[0]

    # print(
    #     tgt_r
    #     * 2
    #     * sim.cosmo.lcMpc
    #     * 1e3
    #     * sim.cosmo_model.arcsec_per_kpc_comoving(1.0 / aexp - 1.0).value
    #     / shape_star,
    #     star_res_as,
    #     flux_res_as,
    # )

    code2ckpc = sim.cosmo.lcMpc * 1e3
    ckpc2as = sim.cosmo_model.arcsec_per_kpc_comoving(1.0 / aexp - 1.0).value
    code2as = code2ckpc * ckpc2as

    lim_window = [-0.5, 0.5]

    lim_flux = -0.5 * shape_flux * flux_res_as, 0.5 * shape_flux * flux_res_as
    lim_star = -0.5 * shape_star * star_res_as, 0.5 * shape_star * star_res_as

    cur_axs[0].imshow(
        rhostar.T,
        origin="lower",
        norm=LogNorm(vmin=1e7, vmax=5e13),
        extent=[*lim_star, *lim_star],
        cmap="viridis",
    )
    cur_axs[0].set_xlim(lim_window)
    cur_axs[0].set_ylim(lim_window)

    cur_axs[1].imshow(
        f444.T,
        origin="lower",
        norm=LogNorm(vmin=1e-2, vmax=5),
        extent=[*lim_flux, *lim_flux],
        cmap="Greys_r",
    )
    cur_axs[1].set_facecolor("k")
    cur_axs[1].set_xlim(lim_window)
    cur_axs[1].set_ylim(lim_window)

    # axs[idir, 0].set_title(cur_dir)

    for ax in cur_axs:

        # ax.set_xlim(-tgt_r * sim.cosmo.lcMpc * 1e3, tgt_r * sim.cosmo.lcMpc * 1e3)
        # ax.set_ylim(-tgt_r * sim.cosmo.lcMpc * 1e3, tgt_r * sim.cosmo.lcMpc * 1e3)

        plot_zoom_gals(
            ax,
            snap,
            sim,
            tgt_pos,
            tgt_r,
            tgt_r,
            direction=cur_dir,
            hm="HaloMaker_stars2_dp_rec_dust",
            units=code2as,
            color="r",
        )

    fig.savefig("test_mock_directions")
