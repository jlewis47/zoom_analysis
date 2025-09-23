from multiprocessing import resource_sharer
from hagn.association import gid_to_stars
from hagn.catalogues import get_cat_hids, make_super_cat
from hagn.tree_reader import read_tree_rev
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import zoom

# from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic_2d
import os

from skimage.transform import rescale


from scipy.ndimage import gaussian_filter, convolve
from scipy.signal import convolve2d, fftconvolve

from zoom_analysis.rascas.rascas_steps import (
    filt_file_to_dict,
    get_directions_cart,
    micron_to_fwhm_as,
)
from zoom_analysis.read.read_data import read_data_ball
from zoom_analysis.visu.visu_fct import plot_fields, plot_stars, basis_from_vect

from zoom_analysis.halo_maker.read_treebricks import (
    convert_star_units,
    read_zoom_brick,
    read_gal_stars,
    read_zoom_stars,
)
from zoom_analysis.stars import sfhs

from hagn.utils import get_hagn_sim, get_nh_sim

from zoom_analysis.zoom_helpers import starting_hid_from_hagn

import healpy as hp

# from zoom_analysis.stars.star_reader import read_part_ball,
from f90_tools.star_reader import read_part_ball_NCdust, read_part_ball_NH

from gremlin.read_sim_params import ramses_sim

from zoom_analysis.halo_maker.assoc_fcts import (
    get_halo_props_snap,
    get_gal_props_snap,
    find_zoom_tgt_halo,
    get_halo_assoc_file,
    get_assoc_pties_in_tree,
    smooth_props,
)

from zoom_analysis.rascas.field_fits import (
    load_psf,
    # rebin_factor
    # rebin
    dilate,
)

from zoom_analysis.rascas.filts.filts import (
    flamb_fnu,
    fnu_to_mAB,
)

# from zoom_analysis.trees.tree_reader import read_tree_file_rev
from zoom_analysis.trees.tree_reader import read_tree_file_rev_correct_pos


# from matplotlib.gridspec import GridSpec

# from astropy.visualization import make_lupton_rgb

# from scipy.integrate import simps

from zoom_analysis.constants import *

from f90_tools.star_reader import read_part_ball_NCdust

# from zoom_analysis.rascas.errs import dumb_constant_mag, get_cl_err

from zoom_analysis.rascas.rascas_steps import read_params
from zoom_analysis.rascas.read_rascas import (
    cube_to_erg_s_A_cm2_as2,
    read_mock_dump,
    spec_to_erg_s_A_cm2,
    # spec_to_erg_s_A_cm2_rf,
    mock_spe_dump,
)

from zoom_analysis.rascas.igm import T_IGM_Inoue2014

from zoom_analysis.rascas.filts.filts import (
    read_transmissions,
)

from zoom_analysis.rascas.rascas_plots import *


def rescale_image_flux_conserving(image, scale_factor):
    # Rescale image using zoom
    rescaled_image = zoom(image, 1 / scale_factor, order=1)  # linear interpolation

    # Conserve flux: scale intensity by (scale_factor)^2
    rescaled_image *= scale_factor**2

    return rescaled_image


def pad_to_multiple(img, factor):
    ny, nx = img.shape
    ny_pad = (factor - ny % factor) % factor
    nx_pad = (factor - nx % factor) % factor
    return np.pad(img, ((0, ny_pad), (0, nx_pad)), mode="constant", constant_values=0)


def rebin_flux_conserving(img, cur_res_as, tgt_res_as):
    """
    Rebin a 2D image from current resolution to target resolution,
    conserving total flux and avoiding aliasing artifacts.

    Parameters
    ----------
    img : 2D ndarray
        Input image with square pixels.
    cur_res_as : float
        Current resolution per pixel (in arcseconds).
    tgt_res_as : float
        Target resolution per pixel (in arcseconds).

    Returns
    -------
    img_rebinned : 2D ndarray
        Rebinned image, with total flux preserved.
    """

    if tgt_res_as <= cur_res_as:
        # No rebinning needed
        return img.copy()

    # Compute integer rebinning factor
    rebin_factor = tgt_res_as / cur_res_as
    int_factor = int(np.round(rebin_factor))

    if not np.isclose(rebin_factor, int_factor, rtol=1e-2):
        print(
            f"Warning: rebin factor {rebin_factor:.3f} is not integer — rounding to {int_factor}"
        )

    rebin_factor = int_factor

    # Apply Gaussian smoothing to suppress high-frequency noise (anti-aliasing)
    sigma = rebin_factor / 2.355  # ~match FWHM of target resolution
    img_smoothed = gaussian_filter(img, sigma=sigma)

    # Trim image to be divisible by rebin_factor
    ny, nx = img_smoothed.shape
    ny_trim = ny - (ny % rebin_factor)
    nx_trim = nx - (nx % rebin_factor)
    img_trimmed = img_smoothed[:ny_trim, :nx_trim]

    # Reshape and sum blocks to conserve flux
    img_rebinned = img_trimmed.reshape(
        ny_trim // rebin_factor, rebin_factor, nx_trim // rebin_factor, rebin_factor
    ).sum(axis=(1, 3))

    return img_rebinned


def write_mock_imgs(fname, aexp, time, z, aper_sr, names, imgs, st_img):

    with h5py.File(
        fname,
        "w",
    ) as f:
        hdr = f.create_group("header")

        hdr.attrs["redshift"] = z
        hdr.attrs["aexp"] = aexp
        hdr.attrs["time_Myr"] = time
        hdr.attrs["aperture_sr"] = aper_sr
        hdr.attrs["img_shape"] = imgs[0].shape
        hdr.attrs["flux_units"] = "MJy.sr^-1"
        hdr.attrs["stellar_surface_density_units"] = "M_sun.as^-2"
        hdr.attrs["stellar_surface_density_help"] = (
            "computed as a 2d histogram of the stellar particles in the aperture, taking stars that are in [-0.5,0.5] around the image in the projection direction"
        )

        for i, (band_name, img) in enumerate(zip(names, imgs)):
            f.create_dataset(band_name, data=img, compression="lzf")

        f.create_dataset("stellar_surface density", data=st_img, compression="lzf")


if __name__ == "__main__":

    hagn_sim = get_hagn_sim()

    hm = "TREE_STARS_AdaptaHOP_dp_SCnew_gross/"

    # dsims = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/"
    # sim_id = "id242756_novrel_lowerSFE_stgNHboost_strictSF"
    # snap_targets = [309]
    # gid=3

    # dsims = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/"
    # sim_id = "id242756_novrel_lowerSFE_stgNHboost"
    # snap_targets = [291]
    # gid=3

    # dsims = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/"
    # sim_id = "id242756_novrel_lowerSFE"
    # snap_targets = [284]
    # gid=4

    # dsims = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/"
    # sim_id = "id242756_novrel"
    # snap_targets = [347]
    # gid=14

    # dsims = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal"
    # sim_id = "id74099"
    # sim_id = "id242756_novrel_lowerSFE_stgNHboost_strictSF"
    # sim_id = "id242704_novrel"
    # sim_id = "id21892_novrel"
    # dsims = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/"
    # sim_id = "id26646"
    # sim_id = "id52380"
    # sim_id = "id180130"
    dsims = "/data101/jlewis/sims/"
    sim_id = "NH_for_rascas"
    sim_path = os.path.join(dsims, sim_id)
    zed_targets = None
    snap_targets = None
    # snap_targets=[205]
    # gid=1
    # snap_targets=[206]
    # gid=2
    # snap_targets=[207]
    # snap_targets=np.arange(150,208)
    # snap_targets=np.arange(100,144)
    # gid=4
    # snap_targets=[169]
    # snap = 203
    # gid = 2
    # snap = 232
    # gid = 2
    # snap = 257
    # gid = 1
    # snap = 267
    # gid = 1
    # snap = 294
    # gid = 10
    # hid = 203
    # zed_target = 2.64  # bc03 all sauces
    # zed_target = 3.5  # bc03 nodust
    # zed_target = 3.5  # bc03 nodust no lya
    # zed_target = 2
    zmax = 6.0
    # zmax = 4.0
    deltaT = 50  # Myr
    ndir = 12
    # ndir = 12
    rfact = 1.1
    zstt = 2.0
    tgt_hid = 21742

    overwrite = True

    nist_pc = 3.085678e18  # cm
    dist_10pc = 10.0 * nist_pc

    filter_select = [
        "u_new",
        "gHSC",
        "iHSC",
        "rHSC",
        "yHSC",
        "zHSC",
        "F115",
        "F150",
        "F277",
        "F444",
        "f814_res",
    ]

    # model_name = "LMC"
    # model_name = "SMC"
    # model_name = "ignoreDust"
    # model_name = "zoom_tgt_ignore_Dust"
    # model_name = "zoom_tgt"
    # model_name = "zoom_tgt_bpass_kroupa300_bpassV23"
    # model_name = "zoom_tgt_bpass_kroupa100_broken_metals"
    # model_name = "zoom_tgt_bpass_chabrier100_new_wlya"
    # model_name = "zoom_tgt_bpass_kroupa100_wlya_old_pos"
    # model_name = "zoom_tgt_bpass_chabrier100_new"
    # model_name = "zoom_tgt_bpass_kroupa100"
    # model_name = "zoom_tgt_bpass_kroupa100_wlya"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_ndust"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_ndust_SMC"
    model_name = "zoom_tgt_bc03_chabrier100_wlya_SMC"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_SMC_draine"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_YD"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_MW"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_noDust"
    # model_name = "zoom_tgt_bc03_chabrier100"
    # model_name = "zoom_tgt_bpass_kroupa100_wlya_noDust"
    # model_name = "zoom_tgt_bpass_chabrier100"
    # model_name = "zoom_tgt_bpass_chabrier100_ignoreDust"

    filt_dict = filt_file_to_dict()

    print(filt_dict)

    # sim = ramses_sim(sim_path, nml="cosmo.nml")

    sim = get_nh_sim()

    name = sim.name
    # intID = int(name[2:].split("_")[0])

    tree_name = os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat")
    byte_file = os.path.join(sim.path, "TreeMakerDM_dust")  # , "tree_rev_nbytes")

    # tree_name = os.path.join(sim.path, "TreeMakerTreeHalo", "tree_rev.dat")
    # byte_file = os.path.join(sim.path, "TreeMakerTreeHalo", "tree_rev_nbytes")
    # tree_name = os.path.join(sim.path, "TreeMakerTreeGal", "tree_rev.dat")
    # byte_file = os.path.join(sim.path, "TreeMakerTreeGal", "tree_rev_nbytes")

    # zed_target = 2.0

    if snap_targets is not None:
        snap_target = max(snap_targets)
    else:
        snap_target = sim.snap_numbers.max()
    # snap_target = sim.get_closest_snap(zed=zed_target)
    true_zed_snap = 1.0 / sim.get_snap_exps(snap_target, param_save=False)[0] - 1

    sim.init_cosmo()

    sim_times = sim.get_snap_times(param_save=False)  # [::-1]
    sim_aexps = sim.get_snap_exps(param_save=False)  # [::-1]
    sim_snaps = sim.snap_numbers  # [::-1]
    # look for treebricks
    if not np.any(hm == d for d in os.listdir(sim.path)):
        print(f"Treebricks not found in {sim.path}")

    last_time = sim_times[-1] + 2 * deltaT

    tgt_gid = None

    last_snap = 446
    last_aexp = sim.get_snap_exps(446, param_save=False)[0]

    nh_tree_hids, nh_tree_datas, nh_tree_aexps = read_tree_rev(
        1.0 / last_aexp - 1.0,
        [tgt_hid],
        tree_type="halo",
        # [gid],
        # tree_type="gal",
        target_fields=["m", "x", "y", "z", "r"],
        sim="nh",
        float_dtype=np.float32,
    )

    nh_tree_times = sim.cosmo_model.age(1.0 / nh_tree_aexps - 1.0).value * 1e3  # Myr

    filt = nh_tree_datas["x"][0] != -1

    for key in nh_tree_datas:
        nh_tree_datas[key] = nh_tree_datas[key][0][filt]
    tree_hids = nh_tree_hids[0][filt]
    tree_aexps = nh_tree_aexps[filt]
    tree_times = nh_tree_times[filt]

    tree_hids = tree_hids[0]
    if not hasattr(sim, "cosmo_model"):
        sim.init_cosmo()

    tree_times = sim.cosmo_model.age(1.0 / tree_aexps - 1.0).value * 1e3  # Myr

    tree_snaps = np.asarray([sim.get_closest_snap(aexp=a) for a in tree_aexps])

    nsteps = len(tree_times)

    reffs = np.zeros(nsteps)
    rvirs = np.zeros(nsteps)
    mass = np.zeros(nsteps)
    pos = np.zeros((nsteps, 3))

    # sim_aexps = sim.get_snap_exps(param_save=False)  # [::-1]
    # sim_times = sim.get_snap_times(param_save=False)  # [::-1]
    sim_snaps = sim.snap_numbers  # [::-1]

    rascas_path = os.path.join(dsims, sim_id, "rascas", model_name)

    rascas_snaps = np.asarray(
        [
            int(f.split("_")[-1])
            for f in os.listdir(rascas_path)
            if f.startswith("output_")
        ]
    )

    final_gid = None

    names_all, wavs_all, trans_all = read_transmissions()

    for i, (aexp, time) in enumerate(zip(nh_tree_aexps, nh_tree_times)):

        snap = sim_snaps[np.argmin(np.abs(sim_aexps - aexp))]

        if abs(aexp - sim_aexps[np.argmin(np.abs(sim_aexps - aexp))]) > 1e-3:
            continue

        try:
            super_cat = make_super_cat(
                snap, outf="/data101/jlewis/nh/super_cats", sim="nh", overwrite=True
            )
            # print(istep, 1.0 / aexp - 1)
        except FileNotFoundError:
            print("No super cat")
            continue

        # super_cat=convert_cat_units(super_cat,sim, snap)

        gal_pties = get_cat_hids(super_cat, [int(nh_tree_hids[0][i])])

        gid = int(gal_pties["gid"][0])

        rascas_dir = os.path.join(rascas_path, f"output_{snap:05d}", f"gal_{gid:07d}")

        try:
            stars = gid_to_stars(
                gid, snap, sim, ["pos", "mass", "birth_time", "metallicity"]
            )
        except (FileNotFoundError, ValueError):
            print(f"missing file for snapshot {snap}... skipping")
            continue

        star_pos = np.abs(np.unwrap(stars["pos"], period=1.0, axis=1))

        # tgt_pos = np.transpose([gal_pties['x'],gal_pties['y'],gal_pties['z']])
        tgt_pos = np.median(star_pos, axis=0)
        rmax = np.max(np.linalg.norm(star_pos - tgt_pos, axis=1))

        assert rmax < 0.1, "huge rmax, something probably went very wrong!"

        # gal_pties = read_zoom_brick(
        # snap, sim, hm, sim_path="/data7b/NewHorizon", star=True, galaxy=True
        # )

        # print(istep,len(gal_pties['mgal']))

        stmasses = sfhs.correct_mass(
            sim, stars["age"], stars["mass"], stars["metallicity"]
        )

        if len(gal_pties["mgal"]) == 0:
            continue

        mass[i] = gal_pties["mgal"]
        pos[i, :] = [gal_pties["x"][0], gal_pties["y"][0], gal_pties["z"][0]]
        reffs[i] = gal_pties["rgal"]
        rvirs[i] = gal_pties["rvir"]

        # cur_rmax = rvirs[i]
        cur_rmax = rmax
        cur_gid = int(gal_pties["gid"][0])
        cur_pos = pos[i, :]

        try:
            rascas_params = read_params(os.path.join(rascas_dir, "params_rascas.cfg"))
        except FileNotFoundError:
            print(f"no params_rascas.cfg in {rascas_dir}")
            continue
        print(f"found rascas dir or params_rascas.cfg in: {rascas_dir}")

        if not os.path.exists(os.path.join(rascas_dir, "mock_test.spectrum")):
            print(f"no mock_test.spectrum in {rascas_dir}")
            continue
        print(f"found mock_test.spectrum in {rascas_dir}")

        last_time = time

        # print(f"ok snap {snap:d}")

        # continue
        rgal = rmax
        mgal = stmasses.sum()
        agegal = np.max(stars["age"])

        # ctr_gal = [0.63813084, 0.7364341052162953, 0rgl.42615680278854773]
        # rgal = 0.0003
        # l = 100e3  # ckpc/h
        [aexp] = sim.get_snap_exps(snap, param_save=False)
        z = 1.0 / aexp - 1.0
        if not hasattr(sim, "cosmo_model"):
            sim.init_cosmo()
        t = sim.cosmo_model.age(z).value  # Gyr
        l = (
            sim.cosmo.boxlen
            * sim.unit_l(sim.aexp_stt)
            / (ramses_pc * 1e6)
            / sim.aexp_stt
        )

        pfs_path = os.path.join(rascas_dir, "PFSDump", f"pfsdump")
        #

        outp = os.path.join(
            "/data101/jlewis",
            "mock_spe",
            model_name,
            sim_id,
            f"{snap:d}",
            f"gal_{gid:07d}",
        )

        if not os.path.exists(outp):
            os.makedirs(outp, exist_ok=True)

        mocks_spe = read_mock_dump(
            os.path.join(rascas_dir, "mock_test.spectrum"), ndir=ndir
        )
        mocks_cube = read_mock_dump(
            os.path.join(rascas_dir, "mock_test.cube"), ndir=ndir
        )

        aper_cm = rgal * 2 * l * (1e6 * ramses_pc)
        # distl = sim.cosmo_model.luminosity_distance(z).value * (1e6 * ramses_pc)
        # aper_as = np.arctan(aper_cm / distl) * 180.0 / np.pi * 3600.0
        aper_as = (rgal * 2 * l * 1e3) * sim.cosmo_model.arcsec_per_kpc_comoving(
            z
        ).value

        # fstar = os.path.join(sim.path, hm, f"GAL_{snap:05d}", f"gal_stars_{gid:07d}")
        # print(fstar)
        # stars = read_gal_stars(fstar)
        # convert_star_units(stars, snap, sim)

        sfh, sfr, ssfr, t_sfr, _ = sfhs.get_sf_stuff(stars, z, sim, deltaT=10.0)

        # print(cur_gal_props)
        # now get in ball of size aperture like rascas does
        # stars_ball = read_part_ball_NCdust(
        #     sim,
        #     snap,
        #     gal_props_tree["pos"][aexp_arg],
        #     gal_props_tree["rmax"][aexp_arg] * rfact * np.sqrt(2),
        #     ["mass", "birth_time", "metallicity", "pos"],
        #     fam=2,
        # )

        # print(gal_props_tree.keys())
        try:
            stars_ball = read_part_ball_NH(
                sim,
                snap,
                cur_pos,
                cur_rmax,
                tgt_fields=["age", "mass", "birth_time", "metallicity", "pos"],
                fam=2,
            )
        except FileNotFoundError:

            continue
        # print(stars_ball.keys())

        stmasses = sfhs.correct_mass(
            sim, stars_ball["age"], stars_ball["mass"], stars_ball["metallicity"]
        )

        sfh_ball, sfr_ball, ssfr_ball, t_sfr_ball, _ = sfhs.get_sf_stuff(
            stars_ball, z, sim, deltaT=10.0
        )

        # break

        gal_dict = {}
        gal_dict["mass"] = sfh
        gal_dict["sfr"] = sfr
        gal_dict["ssfr"] = ssfr
        gal_dict["mass_aper"] = sfh_ball
        gal_dict["sfr_aper"] = sfr_ball
        gal_dict["ssfr_aper"] = ssfr_ball
        gal_dict["time"] = t_sfr
        gal_dict["age"] = np.max(stars["age"])
        gal_dict["age_wmstar"] = np.average(stars["age"], weights=stars["mass"])

        rascas_dirs = get_directions_cart(hp.npix2nside(ndir))[::-1]
        # rascas_dirs = hp.npix2nside(ndir)
        # directions = basis_from_vect(rascas_dir)[::-1]

        # ctrd_star_pos = stars["pos"]-cur_gal_props["pos"]

        if not hasattr(sim, "cosmo_model"):
            sim.init_cosmo()

        ang_per_ckpc = sim.cosmo_model.arcsec_per_kpc_comoving(z).value

        for idir in range(ndir):

            outf = os.path.join(outp, f"mock_spectrum_{idir}.h5")

            if not overwrite and os.path.exists(outf):
                continue

            spec = mocks_spe[f"direction_{idir:d}"]
            # spec2 = mocks_spe[f"direction_{idir:d}"]
            cube = mocks_cube[f"direction_{idir:d}"]

            # load spectrum
            # aper_cm = spec["aperture"] * l
            aper_sr = np.pi * (aper_as / 3600 / 180 * np.pi) ** 2

            wavs = [1.15, 1.5, 2.77, 4.44]  # microns

            px_sr = aper_sr / np.prod(np.shape(cube["cube"])[:2])
            px_as = aper_as / cube["cube"].shape[0]
            px_ckpc = px_as / ang_per_ckpc

            # print("aper_sr:", aper_sr)

            cur_dir = rascas_dirs[idir]

            basis = basis_from_vect(cur_dir)[::-1]
            # print(cur_dir, basis)

            rgal_ckpc = 2 * rgal * l * 1e3
            tgt_res = 0.5 * (l * 1e3 / 2 ** sim.namelist["amr_params"]["levelmax"])
            tgt_nbins = int(rgal_ckpc / tgt_res)

            print(tgt_nbins, rgal_ckpc)

            # real_res = rgal_ckpc / tgt_nbins
            # pixel_as = (
            #     (
            #         np.arctan(
            #             rgal_ckpc / sim.cosmo_model.luminosity_distance(z).value / 1e3
            #         )
            #         * 180.0
            #         / np.pi
            #         * 3600.0
            #     )
            #     / tgt_nbins
            #     * 2
            # )

            pixel_as = tgt_res * ang_per_ckpc

            # print(pixel_as)

            fig, ax = plt.subplots(1, 1)

            # vmin = 3 * np.min(stmasses) / pixel_as**2
            # vmax = 1e3 * np.max(stmasses) / pixel_as**2

            vmin = 1e7
            vmax = 1e14  # msun/as^2

            # basis_order = [basis[2], basis[1], basis[0]]
            # basis_order = [basis[0], basis[1], basis[2]]
            # print(basis, basis_order)

            # # print(fig,ax,sim,aexp,basis[::-1],tgt_nbins,stmasses, stars_ball["pos"], cur_gal_props["pos"], rgal)
            st_img = plot_stars(
                fig,
                ax,
                sim,
                aexp,
                basis,
                tgt_nbins,
                stmasses / pixel_as**2,
                stars_ball["pos"],
                cur_pos,
                cur_rmax * l * 1e3,
                cb=True,
                cmap="gray",
                label="Stellar surface density [M$_\odot$/as$^2$]",
                vmin=vmin,
                vmax=vmax,
                transpose=True,
                # transpose=False,
                mode="sum",  # sum or mean in cells
                binning="cic",  # simple just use a histogram
            )
            # st_img = plot_fields("stellar mass",fig,ax,aexp,)

            # # print(np.log10([st_img.sum(), stmasses.sum(), sfh[0]]))

            # # print(st_img.min(), st_img.max(), np.mean(st_img), np.median(st_img))

            st_fig = os.path.join(outp, f"stars_{idir:d}.png")
            print(st_img.shape)

            # fig.savefig(st_fig)

            # print(f"saved {st_fig}")

            # spec_rf = spec_to_erg_s_A_cm2_rf(spec, pfs_path,z=z)
            lmin = spec["lambda_min"]
            lmax = spec["lambda_max"]
            nlamb = int(spec["lambda_npix"])
            lambda_bins_rf = np.linspace(lmin, lmax, nlamb)
            lambda_bins = lambda_bins_rf * (1.0 + z)

            # print(lmin, lmax)
            # print(lmin * (1 + z), lmax * (1 + z))

            spec_obs = spec_to_erg_s_A_cm2(spec, pfs_path, z)

            spec_rf = spec_to_erg_s_A_cm2(spec, pfs_path, 0.0)
            # spec_rf = spec_rf / (
            #     4 * np.pi * dist_10pc**2
            # )  # at 10 pc from the galaxy for absolute magnitudes ?????

            # print(lmin, lmax)
            # print(spec["spectrum"].max(), spec["spectrum"].min())

            # fig, ax = plt.subplots(1, 1)

            # ax.plot(lambda_bins, spec["spectrum"])
            # apply IGM absorption

            spec_obs_igm = spec_obs * T_IGM_Inoue2014(lambda_bins, z)

            # fig.savefig("test")

            # print(lambda_bins[0], lambda_bins[-1])

            # ditch filters outside of spectral coverage
            names = [
                names_all[i]
                for i in range(len(names_all))
                if np.all(wavs_all[i] <= lmax * (1.0 + z))
                and np.all(wavs_all[i] >= lmin * (1 + z))
                and names_all[i] in filter_select
            ]

            trans = [
                trans_all[i]
                for i in range(len(names_all))
                if np.all(wavs_all[i] <= lmax * (1.0 + z))
                and np.all(wavs_all[i] >= lmin * (1 + z))
                and names_all[i] in filter_select
            ]

            wavs = [
                wavs_all[i]
                for i in range(len(names_all))
                if np.all(wavs_all[i] <= lmax * (1.0 + z))
                and np.all(wavs_all[i] >= lmin * (1 + z))
                and names_all[i] in filter_select
            ]

            # print(lmax * (1.0 + z))
            # print(lmin * (1.0 + z))

            # print(list(zip(names_all,wavs_all)))
            # print(names_all)
            # print(names)

            wav_ctrs = np.asarray([wav[int(0.5 * len(wav))] for wav in wavs])

            wav_order = np.argsort(wav_ctrs)

            wavs = [wavs[order] for order in wav_order]
            names = [names[order] for order in wav_order]
            trans = [trans[order] for order in wav_order]

            # load cube

            dx_box = cube["cube_side"] / cube["cube"].shape[0]
            dx_cm = dx_box * sim.unit_l(sim.aexp_stt) / sim.aexp_stt

            # print(cube["cube"].max())
            cube_to_erg_s_A_cm2_as2(cube, pfs_path, z, aper_as)
            # print(cube["cube"].max())

            # print(np.max(spec["spectrum"]))

            # print(list(zip(spec_obs_igm, spec_obs)))
            #
            filt_names, mag_spec, mags_no_err, mags, mag_errs = gen_spec(
                aexp, spec_obs_igm, lambda_bins, [names, wavs, trans], rf=False
            )
            filt_names_rf, mag_spec_rf, mags_no_err_rf, mags_rf, mag_errs_rf = gen_spec(
                1.0, spec_obs, lambda_bins_rf, [names, wavs, trans], rf=True
            )

            print("spec mags:", list(zip(names, mags_no_err)))

            # print(list(zip(spec_obs_igm, spec_rf)))
            # print(list(zip(names, mags, mags_rf, lambda_bins)))

            # lambda_ctr_args = np.asarray(
            #     [np.argmin(np.abs(wav_ctr - lambda_bins)) for wav_ctr in wav_ctrs]
            # )

            # print(list(zip(mag_spec[lambda_ctr_args], mag_spec_rf[lambda_ctr_args])))

            print("done integrating on spectrum")

            _, band_ctrs, imgs = gen_imgs(z, lambda_bins, cube, [names, wavs, trans])
            # fnus = flamb_fnu(imgs.sum(axis=(1, 2)), band_ctrs * aexp)
            # ms = fnu_to_mAB(fnus / np.prod(imgs[0].shape))
            # print("integrated img map:", list(zip(names, ms)))

            imgs_MJy = [
                flamb_fnu(img, band_ctr) * aper_as**2 / aper_sr / 1e-23 / 1e6
                for band_ctr, img in zip(band_ctrs, imgs)
            ]  # MJy/sr
            imgs_MJy_wIGM = [
                flamb_fnu(img, band_ctr)
                * aper_as**2
                / aper_sr
                / 1e-23
                / 1e6
                * T_IGM_Inoue2014(band_ctr, z)
                for img, band_ctr in zip(imgs, band_ctrs)
            ]

            Ts = np.asarray([T_IGM_Inoue2014(band_ctr, z) for band_ctr in band_ctrs])

            mags_img_MJy = (
                -2.5
                * np.log10(
                    np.nansum(imgs_MJy, axis=(1, 2))
                    * Ts
                    * (aper_sr / np.prod(imgs[0].shape))
                    * 1e6
                )
                + 8.9
            )
            # print("integrated img mags", mags_img_MJy)
            fnu_imgs = (
                flamb_fnu(np.nansum(imgs, axis=(1, 2)), band_ctrs)
                * Ts
                * (aper_as**2 / np.prod(imgs[0].shape))
            )
            mags_imgs_fnu = -2.5 * np.log10(fnu_imgs) - 48.60
            print("integrated img mags:", mags_imgs_fnu)

            print("done integrating on cube")

            img_dir = os.path.join(outp, "MJy_images")

            if not os.path.exists(img_dir):
                os.makedirs(img_dir, exist_ok=True)

            fname = os.path.join(
                img_dir,
                f"mock_images_{sim_id}_{snap}_{gid}_{idir:d}_noIGM.h5",
            )
            write_mock_imgs(fname, aexp, time, z, aper_sr, names, imgs_MJy, st_img)

            fname = os.path.join(
                outp,
                "MJy_images",
                f"mock_images_{sim_id}_{snap}_{gid}_{idir:d}_wIGM.h5",
            )
            write_mock_imgs(fname, aexp, time, z, aper_sr, names, imgs_MJy_wIGM, st_img)

            print("applying psf")

            # psf = np.asarray([filt_dict[name]['psf_as']/2.355 for name,w in zip(names[:],wavs[:])])
            psfs = [load_psf(filt_dict[name]["psf_fname"]) for name in names[:]]
            print([(names[ipsf], psf[1]) for ipsf, psf in enumerate(psfs)])
            psfs_rebinned = [
                dilate(psf[0], psf[1] / px_as) / (psf[1] / px_as) ** 2
                for ipsf, psf in enumerate(psfs)
            ]
            # psfs_rebinned = [
            #     dilate(psf, filt_dict[names[ipsf]]["res_as"] / px_as)
            #     / (filt_dict[names[ipsf]]["res_as"] / px_as) ** 2
            #     for ipsf, psf in enumerate(psfs)
            # ]
            # psf_pxs = np.asarray([p / px_as for p in psf])

            print("loaded and rebinned psfs")

            fname = os.path.join(
                outp,
                "MJy_images",
                f"mock_images_{sim_id}_{snap}_{gid}_{idir:d}_wIGM_wPSF.h5",
            )
            # imgs_MJy_wIGM_wPSF = [gaussian_filter(img, fwhm_px) for img,fwhm_px in zip(imgs_MJy_wIGM,psf_pxs)]
            # imgs_MJy_wIGM_wPSF = [convolve(img,psf) for img,psf in zip(imgs_MJy_wIGM,psfs_rebinned)]
            imgs_MJy_wIGM_wPSF = [
                fftconvolve(img, psf) for img, psf in zip(imgs_MJy_wIGM, psfs_rebinned)
            ]

            print("convolved with psfs")

            write_mock_imgs(
                fname, aexp, time, z, aper_sr, names, imgs_MJy_wIGM_wPSF, st_img
            )

            print("rebinning to correct resolution")

            imgs_MJy_wIGM_wPSF_rebinned = []

            # ress = np.asarray(
            #     [filt_dict[name]["res_as"] for name, w in zip(names[:], wavs[:])]
            # )
            ress = [psf[1] for psf in psfs]
            for iband in range(len(names)):

                img = imgs_MJy_wIGM_wPSF[iband]
                tgt_res = ress[iband]

                cur_size = np.asarray(list(img.shape))
                cur_res_as = px_as

                # print(tgt_res,cur_res_as)

                if cur_res_as < tgt_res:

                    bin_fact = tgt_res / cur_res_as
                    tgt_size = np.int32(np.ceil(cur_size / bin_fact))

                    int_fact = int(np.ceil(bin_fact))

                    # pad_size = int_fact * tgt_size

                    # diff_pad = (pad_size - cur_size)[0]

                    # pad_left = int(diff_pad*0.5)
                    # pad_right = pad_left
                    # if pad_left+pad_right<diff_pad:pad_right+=1

                    # padded_img = np.pad(img,((pad_left,pad_right),(pad_left,pad_right)))

                    # bin_fact_round = int(np.ceil(bin_fact))

                    # cur_X,cur_Y = np.mgrid[0:pad_size[0],0:pad_size[1]]
                    # cur_X,cur_Y = np.mgrid[0:cur_size[0],0:cur_size[1]]

                    # xbins = np.linspace(0,int(pad_size[0]),int(tgt_size[0]))
                    # ybins = np.linspace(0,int(pad_size[1]),int(tgt_size[1]))
                    # xbins = np.linspace(0,int(cur_size[0]),int(tgt_size[0]))
                    # ybins = np.linspace(0,int(cur_size[1]),int(tgt_size[1]))

                    # xbins = np.arange(int(np.floor(tgt_size[0])))
                    # ybins = np.arange(int(np.floor(tgt_size[0])))

                    # img_smooth = gaussian_filter(img,tgt_res / cur_res_as / 2.355)

                    # imgs_MJy_wIGM_wPSF_rebinned.append(rescale(img,cur_res_as/tgt_res))

                    # imgs_MJy_wIGM_wPSF_rebinned.append(binned_statistic_2d(np.ravel(cur_X),np.ravel(cur_Y),np.ravel(padded_img),"sum",bins=[xbins,ybins])[0])
                    # imgs_MJy_wIGM_wPSF_rebinned.append(binned_statistic_2d(np.ravel(cur_X),np.ravel(cur_Y),np.ravel(img),"sum",bins=[xbins,ybins])[0])
                    # imgs_MJy_wIGM_wPSF_rebinned.append(binned_statistic_2d(np.ravel(cur_X),np.ravel(cur_Y),np.ravel(img_smooth),"sum",bins=tgt_size)[0])
                    # imgs_MJy_wIGM_wPSF_rebinned.append(rebin_flux_conserving(img,cur_res_as,tgt_res))
                    imgs_MJy_wIGM_wPSF_rebinned.append(
                        rescale_image_flux_conserving(img, bin_fact)
                    )

                else:

                    imgs_MJy_wIGM_wPSF_rebinned.append(img)

            # print([img.shape for img in imgs_MJy_wIGM_wPSF_rebinned])
            # print([img.shape for img in imgs_MJy_wIGM_wPSF])

            fname = os.path.join(
                outp,
                "MJy_images",
                f"mock_images_{sim_id}_{snap}_{gid}_{idir:d}_wIGM_wPSF_rebinned.h5",
            )
            write_mock_imgs(
                fname,
                aexp,
                time,
                z,
                aper_sr,
                names,
                imgs_MJy_wIGM_wPSF_rebinned,
                st_img,
            )

            print("saved mock images")

            mag_lims = [
                -2.5
                * np.log10(
                    (10 ** (filt_dict[band]["depth_aper"] / -2.5))
                    / filt_dict[band]["aper_depth_as"] ** 2
                    * filt_dict[band]["res_as"] ** 2
                )
                for i, band in enumerate(names)
            ]

            fig_spe, ax_spe = plot_spe_bands(
                aexp,
                [rgal],
                l,
                lambda_bins,
                mag_spec,
                mags,
                mag_errs,
                -2.5
                * np.log10(
                    [
                        img_MJy * 1e6 * T * (aper_sr / np.prod(imgs[0].shape))
                        for T, img_MJy in zip(Ts, imgs_MJy)
                    ]
                )
                + 8.9,
                band_ctrs,
                [names, wavs, trans],
                vmin=np.full_like(mag_lims, 20),
                vmax=mag_lims,
            )

            weighted_age = gal_dict["age_wmstar"]

            ztxt = f"z={z:.2f}"
            agetxt = f"age U={t:.2f} Gyr"
            agegaltxt = f"age gal={weighted_age/1e3:.2f} Gyr"
            masstxt = "M$_\star$=" + f"{mgal:.2e}" + " M$_\odot$"
            txt = "\n".join([ztxt, agetxt, agegaltxt, masstxt])
            ax_spe[1].text(
                0.825,
                0.05,
                txt,
                transform=ax_spe[1].transAxes,
                fontsize=12,
                va="bottom",
                ha="left",
            )

            ax_spe[0][-1].axis("off")

            png_fig = os.path.join(
                outp, f"mock_spectrum_{sim_id}_{snap}_{gid}_{idir:d}.png"
            )

            fig_spe.savefig(png_fig)

            # fig_spe.savefig(png_fig.replace(".png", ".pdf"), format="pdf")

            mag_imgs_wpsf_rebinned = [
                -2.5
                * np.log10(img_MJy * 1e6 * T * (aper_sr / np.prod(img_MJy[0].shape)))
                + 8.9
                for T, img_MJy in zip(Ts, imgs_MJy_wIGM_wPSF_rebinned)
            ]

            # cut to right sizes
            mag_imgs_wpsf_rebinned_right_size = []

            for iband in range(len(names)):

                # pixel_size_as = filt_dict[names[iband]]["res_as"]
                pixel_size_as = psfs[iband][1]

                d_A = sim.cosmo_model.angular_diameter_distance(1.0 / aexp - 1.0).value
                theta = pixel_size_as
                theta_radian = theta * np.pi / 180 / 3600
                pixel_size_kpc = d_A * theta_radian * 1e3 / aexp

                half_size = int(0.5 * mag_imgs_wpsf_rebinned[iband].shape[0])

                cur_image = np.copy(mag_imgs_wpsf_rebinned[iband])

                nb_pixels = int(0.5 * rgal_ckpc / pixel_size_kpc)

                cur_image = cur_image[
                    half_size - nb_pixels : half_size + nb_pixels,
                    half_size - nb_pixels : half_size + nb_pixels,
                ]

                mag_imgs_wpsf_rebinned_right_size.append(cur_image)

            fig_spe, ax_spe = plot_spe_bands(
                aexp,
                [rgal],
                l,
                lambda_bins,
                mag_spec,
                mags,
                mag_errs,
                mag_imgs_wpsf_rebinned_right_size,
                band_ctrs,
                [names, wavs, trans],
                vmax=mag_lims,
                vmin=np.full_like(mag_lims, 20),
            )

            weighted_age = gal_dict["age_wmstar"]

            ztxt = f"z={z:.2f}"
            agetxt = f"age U={t:.2f} Gyr"
            agegaltxt = f"age gal={weighted_age/1e3:.2f} Gyr"
            masstxt = "M$_\star$=" + f"{mgal:.2e}" + " M$_\odot$"
            txt = "\n".join([ztxt, agetxt, agegaltxt, masstxt])
            ax_spe[1].text(
                0.825,
                0.05,
                txt,
                transform=ax_spe[1].transAxes,
                fontsize=12,
                va="bottom",
                ha="left",
            )

            ax_spe[0][-1].axis("off")

            png_fig = os.path.join(
                outp, f"mock_spectrum_wPSF_rebinned_{sim_id}_{snap}_{gid}_{idir:d}.png"
            )

            fig_spe.savefig(png_fig)

            # fig_spe.savefig(png_fig.replace(".png", ".pdf"), format="pdf")

            print(f"spec plot done: {png_fig}")

            print("output magnitudes dump")

            mock_spe_dump(
                outf,
                z,
                aper_as,
                names,
                lambda_bins,
                mag_spec,
                mags,
                mag_errs,
                mags_rf,
                mag_errs_rf,
                band_ctrs,
                gal_data=gal_dict,
            )
