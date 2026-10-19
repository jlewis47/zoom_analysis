from datetime import date
from time import ctime, strftime
import numpy as np
import matplotlib.pyplot as plt

from time import ctime

from scipy.ndimage import zoom

from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic_2d
import os

from skimage.transform import rescale


from scipy.ndimage import gaussian_filter, convolve, rotate
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

from hagn.utils import get_hagn_sim

from zoom_analysis.zoom_helpers import starting_hid_from_hagn

import healpy as hp

# from zoom_analysis.stars.star_reader import read_part_ball,
from f90_tools.star_reader import read_part_ball_NCdust

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

import astropy.units as u

# from zoom_analysis.trees.tree_reader import read_tree_file_rev
from zoom_analysis.trees.tree_reader import read_tree_file_rev_correct_pos


# from matplotlib.gridspec import GridSpec

# from astropy.visualization import make_lupton_rgb

# from scipy.integrate import simps

from zoom_analysis.constants import *

from f90_tools.star_reader import read_part_ball_NCdust, read_part_ball_NH

# from zoom_analysis.rascas.errs import dumb_constant_mag, get_cl_err

from zoom_analysis.rascas.rascas_steps import read_params

from zoom_analysis.rascas.read_rascas import (
    cube_to_erg_s_A_cm2_as2,
    read_mock_dump,
    spec_to_erg_s_A_cm2,
    # spec_to_erg_s_A_cm2_rf,
    read_compress_mock,
    mock_spe_dump,
)

from zoom_analysis.rascas.igm import T_IGM_Inoue2014

from zoom_analysis.rascas.filts.filts import (
    read_transmissions,
)

from zoom_analysis.rascas.rascas_plots import *


def rescale_image_flux_conserving(image, scale_factor, order=2):
    # Rescale image using zoom
    rescaled_image = zoom(
        image, 1 / scale_factor, order=order
    )  # binomial interpolation

    # Conserve flux: scale intensity by (scale_factor)^2
    # rescaled_image *= scale_factor**2

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


def write_mock_imgs(
    fname, aexp, time, z, aper_sr, px_res_as, px_res_stars_as, names, imgs, st_img
):

    with h5py.File(
        fname,
        "w",
    ) as f:
        hdr = f.create_group("header")

        hdr.attrs["redshift"] = z
        hdr.attrs["aexp"] = aexp
        hdr.attrs["time_Myr"] = time
        hdr.attrs["aperture_sr"] = aper_sr
        hdr.attrs["pixel_res_as"] = px_res_as
        hdr.attrs["pixel_res_stars_as"] = px_res_stars_as
        hdr.attrs["img_shapes"] = [i.shape for i in imgs]
        hdr.attrs["band_names"] = names
        hdr.attrs["flux_units"] = "MJy.sr^-1"
        hdr.attrs["stellar_surface_density_units"] = "M_sun.as^-2"
        hdr.attrs["stellar_surface_density_help"] = (
            "computed as a 2d histogram of the stellar particles in the aperture, taking stars that are in [-0.5,0.5] around the image in the projection direction"
        )

        for i, (band_name, img) in enumerate(zip(names, imgs)):
            f.create_dataset(band_name, data=img, compression="lzf")

        f.create_dataset("stellar_surface density", data=st_img, compression="lzf")


def walk_down_dirs(root, saved_paths):

    dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

    # print(root)

    for d in dirs:

        full_d = os.path.join(root, d)
        # print(full_d)

        if d.startswith("id"):
            if not os.path.islink(full_d):
                saved_paths.append(full_d)
        else:
            saved_paths = walk_down_dirs(full_d, saved_paths)

    # print(saved_paths)

    return saved_paths


def locate_all_sims():

    sim_paths = []

    root_dirs = [
        "/data101/jlewis/sims/dust_fid/lvlmax_22",
        "/data102/jlewis/sims/lvlmax_22",
        "/data103/jlewis/sims/lvlmax_22",
    ]

    for root_dir in root_dirs:
        # print(root_dir)
        sim_paths = walk_down_dirs(root_dir, sim_paths)

    sim_paths = np.unique(sim_paths)

    # print(sim_paths)

    sim_paths = [p for p in sim_paths if "common_plots" not in p]

    return sim_paths


if __name__ == "__main__":

    

    model_name = "zoom_tgt_bc03_chabrier100_wlya_ZTEST_SMC"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_ZTEST_noDust"

    # sim_paths = locate_all_sims()

    sim_paths = [
        "/data101/jlewis/sims/NH_for_rascas",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288",
    ]
    # sim_paths = ["/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380"]
    # sim_paths = ["/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"]
    # sim_paths = ["/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289"]
    # sim_paths = ["/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890"]
    # sim_paths = ["/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130"]

    for sim_path in sim_paths:

        zed_targets = None
        snap_targets = None

        zmax = np.inf
        ndir = 12
        # ndir = 12
        rfact = 1.0
        zstt = 2.0

        # overwrite = True
        overwrite = False

        nist_pc = 3.085678e18  # cm
        dist_10pc = 10.0 * nist_pc

        filter_select = [
            "MegaCam-u_new",
            "HSC-g",
            "HSC-i",
            "HSC-r",
            "HSC-y",
            "HSC-z",
            "NIRCAM-F115",
            "NIRCAM-F150",
            "NIRCAM-F277",
            "NIRCAM-F444",
            "HST-f814_res",
            # "SC-L427",
            # "SC-L484",
            # "SC-L505",
            # "SC-L527",
            # "SC-L574",
            # "SC-L624",
            # "SC-L679",
            # "SC-L709",
            # "SC-L711",
            # "SC-L738",
            # "SC-L767",
            # "SC-L816",
            # "SC-L827",
            "VISTA-H",
            "VISTA-J",
            "VISTA-Ks",
            # "ULTRAVISTA-NB118",
            "UVISTA-NB118",
            "VISTA-Y",
        ]

        filt_dict = filt_file_to_dict()

        # print(filt_dict)

        try:
            sim = ramses_sim(
                sim_path,
                nml="cosmo.nml",
                full_check=False,
                mini_snaps=True,
                full_snaps=True,
            )
        except FileNotFoundError:
            continue

        name = sim.name

        if name == "NH_for_rascas":
            name = "NewHorizon"
            intID = name
        else:
            intID = int(name[2:].split("_")[0])

        if snap_targets is not None:
            snap_target = max(snap_targets)
        else:
            snap_target = sim.snap_numbers.max()

        if type(snap_target) in [np.ndarray, list]:

            if len(snap_target) == 0:
                continue

        # snap_target = sim.get_closest_snap(zed=zed_target)
        true_zed_snap = 1.0 / sim.get_snap_exps(snap_target)[0] - 1

        sim.init_cosmo()

        sim.get_snap_times()
        sim_aexps = sim.aexps

        _, data_snaps = sim.get_snaps(tar_snaps=True, full_snaps=True, mini_snaps=False)
        data_snap_aexps = sim.get_snap_exps(data_snaps)
        close_tol = 1e-3

        if len(data_snaps) == 0:
            continue

        assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
        assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

        avail_aexps = np.intersect1d(
            sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
        )

        if len(avail_aexps) == 0:
            continue

        avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

        if zed_targets is not None:
            snap_target_args = np.asarray(
                [
                    np.argmin(np.abs(data_snap_aexps - 1.0 / (z + 1.0)))
                    for z in zed_targets
                ]
            )
            snap_targets = data_snaps[snap_target_args]
            err = data_snap_aexps[snap_target_args] - 1.0 / (
                np.asarray(zed_targets) + 1.0
            )
            snap_targets = snap_targets[np.abs(err) < close_tol]
        elif snap_targets is None:
            snap_targets = sim.snap_numbers

        rascas_path = os.path.join(sim_path, "rascas", model_name)

        if not os.path.exists(rascas_path):
            continue


        ztests_paths = np.array([
                    f
                    for f in os.listdir(path=rascas_path)
                    if f.startswith("Z")
                ])
        
        ztests = np.asarray(
                [
                    float(f[1:].replace('p','.'))
                    for f in ztests_paths
                ]
            )

        # order = np.argsort(ztests)[::-1]
        order = np.argsort(ztests)

        ztests = ztests[order]
        ztests_paths = ztests_paths[order]

        final_gid = None

        names_all, wavs_all, trans_all = read_transmissions()

        # print([(n, w) for n, w in zip(names_all, [np.median(w)/10000.0 for w in wavs_all])])

        for iz,zmock in enumerate(ztests):


            rascas_snaps = np.sort(
                np.asarray(
                    [
                        int(f.split("_")[-1])
                        for f in os.listdir(path=os.path.join(rascas_path,ztests_paths[iz]))
                        if f.startswith("output_")
                    ]
                )
            )[::-1]



            aexp_mock = 1/(1+zmock)

            for i, snap in enumerate(rascas_snaps):
                if snap_targets is not None:
                    if snap not in snap_targets:

                        continue

                try:
                    time = sim.get_snap_times(snap)[0]
                    aexp = sim.get_snap_exps(snap)[0]
                except OSError:

                    continue


                # print(i, snap, aexp, time)
                if 1.0 / aexp - 1 > zmax:
                    print(f"skipping snap {snap:d} at z={1.0/aexp-1:.2f}")
                    continue

                # if 1.0 / aexp - 1 < true_zed_snap:
                #     # print(snap,aexp,true_zed_snap)
                #     print(f"skipping snap {snap:d} at z={1.0/aexp-1:.2f} < {true_zed_snap:.2f}")
                #     continue

                if not os.path.isfile(get_halo_assoc_file(sim_path, snap)):
                    print(f"skipping snap {snap:d} at time={time:.2f} no assoc file")
                    continue

                # find_gals
                gal_folders = os.listdir(os.path.join(rascas_path,ztests_paths[iz], f"output_{snap:05d}"))
                gids = [int(gdir.split("_")[-1]) for gdir in gal_folders]


                if 'NH' in sim.name or 'newhorizon' in sim.name.lower():

                    # hm = "TreeMakerstars2_dp_rec_dust/"
                    hm = "TREE_STARS_AdaptaHOP_dp_SCnew_gross"
                    part_fct = read_part_ball_NH

                else:

                    hm = "HaloMaker_stars2_dp_rec_dust/"
                    part_fct = read_part_ball_NCdust

                for gid in gids:

                    # print(sim_path, snap, tree_hids[tree_arg])

                    # hdict, hosted_gals = get_halo_props_snap(
                    #     sim_path, snap, hosted_gals=True
                    # )

                    # if hosted_gals == {}:
                    #     print(f"no hosted gals for snap {snap:d}")
                    #     continue
                    # print(f"found hosted gals for snap {snap:d}")

                    # if final_gid is None:
                    # final_gid = gid

                    cur_gal_props = get_gal_props_snap(sim_path, snap, gid)

                    # print(cur_gal_props["pos"], snap, gid, cur_gal_props["mass"])

                    rascas_dir = os.path.join(
                        rascas_path,ztests_paths[iz], f"output_{snap:05d}", f"gal_{gid:07d}"
                    )

                    try:
                        rascas_params = read_params(
                            os.path.join(rascas_dir, "params_rascas.cfg")
                        )
                    except FileNotFoundError:
                        print(f"no params_rascas.cfg in {rascas_dir}")
                        continue
                    print(f"found rascas dir or params_rascas.cfg in: {rascas_dir}")

                    h5mock = False
                    if os.path.exists(os.path.join(rascas_dir, "mocks.h5")):
                        h5mock = True
                    rawmock = False
                    if os.path.exists(os.path.join(rascas_dir, "mock_test.spectrum")):
                        rawmock = True

                    if not h5mock and not rawmock:
                        print(f"no mocks found in {rascas_dir}... skipping")
                        continue

                    outp = os.path.join(
                        "/data101/jlewis",
                        "mock_spe",
                        model_name,
                        f"{sim.name:s}",
                        ztests_paths[iz],
                        f"{snap:d}",
                        f"gal_{gid:07d}",
                    )

                    if not overwrite:
                        test_outf = os.path.join(outp, f"mock_spectrum_{ndir-1}.h5")
                        if os.path.exists(test_outf):
                            continue

                    #         trel_s = os.path.getmtime(test_outf)

                    #         date_made = date.fromtimestamp(trel_s)

                    #         date_test = date(2025, 10, 23)

                    #         if date_made > date_test:

                    #             print(f"Found {test_outf:s}")

                    last_time = time

                    # print(f"ok snap {snap:d}")

                    # continue
                    rgal = [cur_gal_props["rmax"] * rfact]
                    r50 = [cur_gal_props["r50"] * rfact]
                    mgal = cur_gal_props["mass"]
                    agegal = cur_gal_props["max age"]

                    # ctr_gal = [0.63813084, 0.7364341052162953, 0rgl.42615680278854773]
                    # rgal = 0.0003
                    # l = 100e3  # ckpc/h
                    [aexp] = sim.get_snap_exps(snap)
                    z = 1.0 / aexp - 1.0
                    if not hasattr(sim, "cosmo_model"):
                        sim.init_cosmo()
                    t = sim.cosmo_model.age(z).value  # Gyr
                    tmock = sim.cosmo_model.age(zmock).value  # Gyr
                    # l = (
                    #     sim.cosmo.boxlen
                    #     * sim.unit_l(sim.aexp_stt)
                    #     / (ramses_pc * 1e6)
                    #     / sim.aexp_stt
                    # )
                    l = sim.cosmo.lcMpc

                    pfs_path = os.path.join(rascas_dir, "PFSDump", f"pfsdump")

                    if not os.path.exists(outp):
                        os.makedirs(outp, exist_ok=True)

                    try:

                        if h5mock:

                            mocks_spe, mocks_cube, pfs_header = read_compress_mock(
                                os.path.join(rascas_dir, "mocks.h5")
                            )

                        else:
                            mocks_spe = read_mock_dump(
                                os.path.join(rascas_dir, "mock_test.spectrum"), ndir=ndir
                            )
                            mocks_cube = read_mock_dump(
                                os.path.join(rascas_dir, "mock_test.cube"), ndir=ndir
                            )
                    except FileNotFoundError:
                        continue

                    print('read mocks')

                    aper_code_units = mocks_cube["direction_0"]["cube_side"]

                    aper_cm = aper_code_units * l * (1e6 * ramses_pc)
                    # distl = sim.cosmo_model.luminosity_distance(z).value * (1e6 * ramses_pc)
                    # aper_as = np.arctan(aper_cm / distl) * 180.0 / np.pi * 3600.0
                    aper_ckpc = aper_cm / 3.08e18 / 1e3
                    aper_as = (
                        aper_code_units * l * 1e3
                    ) * sim.cosmo_model.arcsec_per_kpc_comoving(zmock).value

                    fstar = os.path.join(
                        sim.path, hm, f"GAL_{snap:05d}", f"gal_stars_{gid:07d}"
                    )
                    # print(fstar)
                    stars = read_gal_stars(fstar)
                    convert_star_units(stars, snap, sim)

                    print('read galaxy stars')

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
                        stars_ball = read_data_ball(
                            sim,
                            snap,
                            cur_gal_props["pos"],
                            aper_code_units,
                            int(cur_gal_props["host hid"]),
                            data_types=["stars"],
                            tgt_fields=["age", "mass", "birth_time", "metallicity", "pos"],
                            read_part_ball_fct=part_fct
                        )
                    except (FileNotFoundError, KeyError):
                        print("no stellar data")
                        continue
                    # print(stars_ball.keys())

                    if stars_ball is None:
                        print(
                            f"Empty star ball for sim {sim.name:s} in snapshot {snap:d} and galaxy {gid:d}"
                        )
                        continue

                    stmasses = sfhs.correct_mass(
                        sim,
                        stars_ball["age"],
                        stars_ball["mass"],
                        stars_ball["metallicity"],
                    )

                    sfh_ball, sfr_ball, ssfr_ball, t_sfr_ball, _ = sfhs.get_sf_stuff(
                        stars_ball, z, sim, deltaT=10.0
                    )

                    print('read star ball')

                    # break

                    gal_dict = {}
                    gal_dict["mass"] = sfh
                    gal_dict["sfr"] = sfr
                    gal_dict["ssfr"] = ssfr
                    gal_dict["mass_aper"] = sfh_ball
                    gal_dict["sfr_aper"] = sfr_ball
                    gal_dict["ssfr_aper"] = ssfr_ball
                    gal_dict["time"] = t_sfr
                    gal_dict["age"] = np.max(stars["agepart"])
                    gal_dict["age_wmstar"] = np.average(
                        stars["agepart"], weights=stars["mpart"]
                    )

                    rascas_dirs = get_directions_cart(hp.npix2nside(ndir))

                    if not hasattr(sim, "cosmo_model"):
                        sim.init_cosmo()

                    # fig, ax = plt.subplots(3, 4)

                    # ax = np.ravel(ax)

                    ang_per_ckpc = sim.cosmo_model.arcsec_per_kpc_comoving(zmock).value

                    extent_as = 1.5
                    extent_ckpc = extent_as / ang_per_ckpc

                    print(sim.name, snap, 1.0 / aexp - 1, gid)

                    for idir in range(ndir):

                        outf = os.path.join(outp, f"mock_spectrum_{idir}.h5")

                        spec = mocks_spe[f"direction_{idir:d}"]
                        # spec2 = mocks_spe[f"direction_{idir:d}"]
                        cube = mocks_cube[f"direction_{idir:d}"]

                        if cube["cube"].sum() == 0:

                            print("cube full of empty cells!!!!... skipping")
                            print(os.path.join(rascas_path,ztests_paths[iz]))
                            continue

                        else:
                            print(cube["cube"].sum() )

                        # print(cube)

                        # load spectrum
                        # aper_cm = spec["aperture"] * l
                        aper_as2 = aper_as**2
                        aper_sr = aper_as2 * (np.pi / 180 / 3600) ** 2

                        # R_obj = aper_cm * 0.5
                        # aper_sr = get_aper_sr(sim.cosmo_model.luminosity_distance, z, R_obj)

                        wavs = [1.15, 1.5, 2.77, 4.44]  # microns

                        px_sr = aper_sr / np.prod(np.shape(cube["cube"])[:2])
                        px_as2 = aper_as2 / np.prod(np.shape(cube["cube"])[:2])
                        # px_ckpc = px_as / ang_per_ckpc

                        # print("aper_sr:", aper_sr)

                        cur_dir = rascas_dirs[idir]

                        basis = basis_from_vect(cur_dir)
                        # print(cur_dir, basis)

                        rgal_ckpc = aper_code_units * sim.cosmo.lcMpc * 1e3

                        pixel_as = (
                            # 2*
                            rgal_ckpc
                            * ang_per_ckpc
                            / cube["cube"].shape[0]
                        )

                        fig, ax = plt.subplots(1, 1)

                        vmin = 1e8
                        vmax = 1e13  # msun/as^2

                        delta_res = 2
                        nbins_stmass = cube["cube"].shape[0] * delta_res
                        px_as2_stmass = (
                            px_as2 / (delta_res) ** 2
                        ) * 4  # because of rgal shape for flux vs rgal*2 shape for stars
                        px_as_stmass = np.sqrt(px_as2_stmass)

                        # print(fig,ax,sim,aexp,basis[::-1],tgt_nbins,stmasses, stars_ball["pos"], cur_gal_props["pos"], rgal)

                        try:
                            st_img = plot_stars(
                                fig,
                                ax,
                                sim,
                                aexp_mock,
                                basis,
                                nbins_stmass,  # as good as sim resolution
                                stmasses / px_as2_stmass,
                                stars_ball["pos"],
                                cur_gal_props["pos"],
                                aper_code_units * sim.cosmo.lcMpc * 1e3,
                                cb=True,
                                cmap="gray",
                                label="Stellar surface density [M$_\odot$/as$^2$]",
                                vmin=vmin,
                                vmax=vmax,
                                transpose=True,
                                lower=True,
                                # transpose=False,
                                mode="sum",  # sum or mean in cells
                                binning="cic",  # simple just use a histogram
                            )
                        except MemoryError:
                            print("memory error... skipping")
                        # st_img = plot_fields("stellar mass",fig,ax,aexp,)

                        # print(np.log10([st_img.sum(), stmasses.sum(), sfh[0]]))

                        # print(st_img.min(), st_img.max(), np.mean(st_img), np.median(st_img))

                        st_fig = os.path.join(outp, f"stars_{idir:d}.png")

                        fig.savefig(st_fig)

                        plt.close()

                        print(f"saved {st_fig}")

                        # spec_rf = spec_to_erg_s_A_cm2_rf(spec, pfs_path,z=z)
                        lmin = spec["lambda_min"]
                        lmax = spec["lambda_max"]
                        nlamb = int(spec["lambda_npix"])
                        lambda_bins_rf = np.linspace(lmin, lmax, nlamb)
                        lambda_bins = lambda_bins_rf * (1.0 + zmock)

                        # print(lmin, lmax)
                        # print(lmin * (1 + z), lmax * (1 + z))
                        # print(lambda_bins)

                        if h5mock:
                            pfs = pfs_header
                        else:
                            pfs = pfs_path

                        spec_obs = spec_to_erg_s_A_cm2(spec, pfs, zmock)

                        spec_rf = spec_to_erg_s_A_cm2(spec, pfs, 0.0)
                        # spec_rf = spec_rf / (
                        #     4 * np.pi * dist_10pc**2
                        # )  # at 10 pc from the galaxy for absolute magnitudes ?????

                        # print(lmin, lmax)
                        # print(spec["spectrum"].max(), spec["spectrum"].min())

                        # fig, ax = plt.subplots(1, 1)

                        # ax.plot(lambda_bins, spec["spectrum"])
                        # apply IGM absorption

                        spec_obs_igm = spec_obs * T_IGM_Inoue2014(lambda_bins, zmock)

                        # fig.savefig("test")

                        # print(lambda_bins[0], lambda_bins[-1])

                        if len(filter_select) == 0:
                            filter_select = names_all

                        # ditch filters outside of spectral coverage
                        names = [
                            names_all[i]
                            for i in range(len(names_all))
                            if np.all(wavs_all[i] <= lmax * (1.0 + zmock))
                            and np.all(wavs_all[i] >= lmin * (1 + zmock))
                            and names_all[i] in filter_select
                        ]

                        trans = [
                            trans_all[i]
                            for i in range(len(names_all))
                            if np.all(wavs_all[i] <= lmax * (1.0 + zmock))
                            and np.all(wavs_all[i] >= lmin * (1 + zmock))
                            and names_all[i] in filter_select
                        ]

                        wavs = [
                            wavs_all[i]
                            for i in range(len(names_all))
                            if np.all(wavs_all[i] <= lmax * (1.0 + zmock))
                            and np.all(wavs_all[i] >= lmin * (1 + zmock))
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
                        dx_cm = dx_box * sim.unit_l(sim.aexp_stt) / sim.aexp_stt #comoving
                        

                        # print(cube["cube"].max())
                        cube_to_erg_s_A_cm2_as2(cube, pfs, zmock, aper_as2)
                        # print(cube["cube"].max())

                        # print(np.max(spec["spectrum"]))

                        # print(list(zip(spec_obs_igm, spec_obs)))
                        #
                        filt_names, mag_spec, mags_no_err, mags, mag_errs = gen_spec(
                            aexp_mock, spec_obs_igm, lambda_bins, [names, wavs, trans], rf=False
                        )
                        filt_names_rf, mag_spec_rf, mags_no_err_rf, mags_rf, mag_errs_rf = (
                            gen_spec(
                                1.0, spec_obs, lambda_bins_rf, [names, wavs, trans], rf=True
                            )
                        )

                        print("spec mags:", list(zip(names, mags_no_err)))

                        # print(list(zip(spec_obs_igm, spec_rf)))
                        # print(list(zip(names, mags, mags_rf, lambda_bins)))

                        # lambda_ctr_args = np.asarray(
                        #     [np.argmin(np.abs(wav_ctr - lambda_bins)) for wav_ctr in wav_ctrs]
                        # )

                        # print(list(zip(mag_spec[lambda_ctr_args], mag_spec_rf[lambda_ctr_args])))

                        print("done integrating on spectrum")

                        _, band_ctrs, imgs = gen_imgs(
                            zmock, lambda_bins, cube, [names, wavs, trans]
                        )

                        if cube["cube"].sum() == 0:

                            print("cube full of empty cells!!!!... skipping")
                            print(os.path.join(rascas_path,ztests_paths[iz]))
                            continue

                        else:
                            print(cube["cube"].sum() )

                        print('done integrating on cube')

                        # print(cube["cube"].shape, imgs.shape)

                        # fig, ax = plt.subplots(1, 2)

                        # ax[0].imshow(
                        #     cube["cube"][:, :, -50:].sum(axis=2).T,
                        #     origin="lower",
                        #     norm=LogNorm(),
                        # )
                        # ax[1].imshow(imgs[-1, :, :].T, origin="lower", norm=LogNorm())

                        # fig.savefig("test_intregration_image")

                        new_imgs = []

                        rot_ang_deg = +20
                        rot_ang_rad = rot_ang_deg * np.pi / 180.0

                        for ifilt, filt in enumerate(filt_names):

                            # print(imgs[ifilt].shape)

                            # images all need to be padded to allow for rotation of ground based cutouts
                            new_img = np.pad(
                                imgs[ifilt],
                                pad_width=int(
                                    np.ceil(
                                        0.5
                                        * imgs[ifilt].shape[0]
                                        * 2**0.5
                                        * (
                                            +np.cos(np.pi * 0.25)
                                            - np.cos(np.pi * 0.25 + rot_ang_rad)
                                        )
                                    )
                                ),
                            )

                            grnd_kw = ['Megacam','HSC','VISTA']

                            ground_filter = np.any([kw in filt for kw in grnd_kw])

                            if ground_filter:

                                # ground based images need to be rotated
                                new_img = rotate(new_img, rot_ang_deg, reshape=False)

                            new_imgs.append(new_img)
                            # print(ifilt, filt, new_imgs[ifilt].shape)

                        imgs = np.asarray(new_imgs)

                        nplots = int(len(imgs))
                        ncols = 3
                        nrows = nplots // ncols
                        overshoot = ncols * nrows - nplots
                        if overshoot < 0:
                            nrows += 1
                            overshoot += ncols

                        # fig, ax = plt.subplots(nrows, ncols, figsize=(6 * nrows, 6 * ncols))

                        # ax = ax.flatten()

                        # for iplot in range(nplots):

                        #     ax[iplot].imshow(imgs[iplot], origin="lower", norm=LogNorm())

                        # fig.savefig("test_band_rotation")

                        # input("")

                        # continue


                        # fnus = flamb_fnu(imgs.sum(axis=(1, 2)), band_ctrs * aexp)
                        # ms = fnu_to_mAB(fnus / np.prod(imgs[0].shape))
                        # print("integrated img map:", list(zip(names, ms)))
                        Ts = np.asarray(
                            [T_IGM_Inoue2014(band_ctr, z) for band_ctr in band_ctrs]
                        )

                        imgs_MJy = [
                            flamb_fnu(img, band_ctr) * (aper_as2) / aper_sr / 1e-23 / 1e6
                            for band_ctr, img in zip(band_ctrs, imgs)
                        ]  # MJy/sr
                        imgs_MJy_wIGM = [
                            flamb_fnu(img, band_ctr)
                            * (aper_as2)
                            / aper_sr
                            / 1e-23
                            / 1e6
                            * T
                            for img, band_ctr, T in zip(imgs, band_ctrs, Ts)
                        ]

                        # mags_img_MJy = (
                        #     -2.5
                        #     * np.log10(
                        #         np.nansum(imgs_MJy, axis=(1, 2))
                        #         * Ts
                        #         * (aper_sr / np.prod(np.shape(imgs_MJy)[1:]))
                        #         * 1e6
                        #     )
                        #     + 8.9
                        # )
                        # print("integrated img mags", mags_img_MJy)

                        fnu_imgs = (
                            flamb_fnu(np.nansum(imgs, axis=(1, 2)), band_ctrs)
                            * Ts
                            * (aper_as2 / np.prod(imgs.shape[1:]))
                        )

                        mags_imgs_fnu = -2.5 * np.log10(fnu_imgs) - 48.60

                        print("integrated img mags:", mags_imgs_fnu)


                        # assert np.all(np.abs(mags_imgs_fnu-mags_no_err)<0.5), "something is wrong, spectra magntidudes don't match those integrated from images"

                        if not np.all(np.abs(mags_imgs_fnu - mags_no_err) < 1.0):
                            print(
                                "something is wrong, spectra magntidudes don't match those integrated from images"
                            )
                            print(f"idir:{idir} snap:{snap}, sim:{sim.name}")
                            print(np.abs(mags_imgs_fnu - mags_no_err))
                            print("Most likely: nearby object entering/exiting field")
                            # raise ValueError

                        img_dir = os.path.join(outp, "MJy_images")

                        if not os.path.exists(img_dir):
                            os.makedirs(img_dir, exist_ok=True)

                        fname = os.path.join(
                            img_dir,
                            f"mock_images_{sim.name:s}_{snap}_{gid}_{idir:d}_noIGM.h5",
                        )

                        # px_as2 = [aper_as2/np.prod(img.shape) for img in imgs_MJy]

                        write_mock_imgs(
                            fname,
                            aexp_mock,
                            time,
                            zmock,
                            aper_sr,
                            pixel_as,
                            px_as_stmass,
                            names,
                            imgs_MJy,
                            st_img,
                        )

                        fname = os.path.join(
                            outp,
                            "MJy_images",
                            f"mock_images_{sim.name:s}_{snap}_{gid}_{idir:d}_wIGM.h5",
                        )
                        write_mock_imgs(
                            fname,
                            aexp_mock,
                            time,
                            zmock,
                            aper_sr,
                            pixel_as,
                            px_as_stmass,
                            names,
                            imgs_MJy_wIGM,
                            st_img,
                        )

                        print("applying psf")

                        # psf = np.asarray([filt_dict[name]['psf_as']/2.355 for name,w in zip(names[:],wavs[:])])
                        psfs = [load_psf(filt_dict[name]["psf_fname"]) for name in names[:]]
                        print([(names[ipsf], psf[1]) for ipsf, psf in enumerate(psfs)])
                        psfs_rebinned = [
                            dilate(psf[0], psf[1] / pixel_as) / (psf[1] / pixel_as) ** 2
                            # dilate(psf[0], psf[1] / pixel_as)
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
                            f"mock_images_{sim.name:s}_{snap}_{gid}_{idir:d}_wIGM_wPSF.h5",
                        )
                        # imgs_MJy_wIGM_wPSF = [gaussian_filter(img, fwhm_px) for img,fwhm_px in zip(imgs_MJy_wIGM,psf_pxs)]
                        # imgs_MJy_wIGM_wPSF = [convolve(img,psf) for img,psf in zip(imgs_MJy_wIGM,psfs_rebinned)]
                        imgs_MJy_wIGM_wPSF = [
                            fftconvolve(img, psf)
                            for img, psf in zip(imgs_MJy_wIGM, psfs_rebinned)
                        ]

                        # reduce sizes - cut zero filled edges

                        for iband in range(len(names)):

                            cur_img = imgs_MJy_wIGM_wPSF[iband]

                            decal = 0

                            while np.sum(cur_img[decal:-decal, decal:-decal]) == 0:
                                decal += 1

                            imgs_MJy_wIGM_wPSF[iband] = cur_img[decal:-decal, decal:-decal]

                        # r50_as = r50[0] * l * 1e3 * ang_per_ckpc * 0.5
                        # r50_npx = r50_as / pixel_as

                        # r50_img_flux = [img[int(img.shape[0]*0.5-r50_npx):int(img.shape[0]*0.5+r50_npx),int(img.shape[1]*0.5-r50_npx):int(img.shape[1]*0.5+r50_npx)] for img in imgs_MJy_wIGM]
                        # r50_img_flux_wpsf = [img[int(img.shape[0]*0.5-r50_npx):int(img.shape[0]*0.5+r50_npx),int(img.shape[1]*0.5-r50_npx):int(img.shape[1]*0.5+r50_npx)] for img in imgs_MJy_wIGM_wPSF]

                        # r50_mags = [-2.5*np.log10(np.sum(fl)*aper_sr/np.prod(fl.shape)) + 8.9 for fl in r50_img_flux]
                        # r50_mags_wpsf = [-2.5*np.log10(np.sum(fl)*aper_sr/np.prod(fl.shape)) + 8.9 for fl in r50_img_flux_wpsf]

                        # print(r50_mags)
                        # print(r50_mags_wpsf)

                        # print([img.sum() for img in imgs_MJy_wIGM_wPSF])
                        # print([img.sum() for img in imgs_MJy_wIGM])

                        convolved_sizes = np.asarray(
                            [i.shape[0] for i in imgs_MJy_wIGM_wPSF]
                        )
                        convolved_sizes_as = convolved_sizes * pixel_as
                        # convolved_sizes_ckpc = convolved_sizes * pixel_as / ang_per_ckpc
                        # convolved_sizes_cm = convolved_sizes_ckpc * 1e3 * ramses_pc

                        # convolved_sizes_sr = np.asarray([get_aper_sr(sim.cosmo_model.luminosity_distance, z, size_cm * 0.5) for size_cm in convolved_sizes_cm])

                        convolved_sizes_as2 = convolved_sizes_as**2
                        convolved_sizes_sr = (
                            convolved_sizes_as2 * (np.pi / 3600.0 / 180.0) ** 2.0
                        )

                        # convolved_aper_sr = np.pi * (convolved_aper_as / 3600 / 180 * np.pi) ** 2

                        print("convolved with psfs")

                        # print(
                        #     "img mags, psf:", list(zip(names, [-2.5*np.log10(np.nansum(10**(np.asarray(img)/-2.5))) for img in imgs_MJy_wIGM_wPSF]))
                        # )

                        # images with just PSF are super huge !
                        # skip writting them for now
                        # write_mock_imgs(
                        #     fname,
                        #     aexp,
                        #     time,
                        #     z,
                        #     convolved_sizes_sr,
                        #     pixel_as,
                        #     px_as_stmass,
                        #     names,
                        #     imgs_MJy_wIGM_wPSF,
                        #     st_img,
                        # )

                        print("rebinning to correct resolution")

                        imgs_MJy_wIGM_wPSF_rebinned = []

                        # ress = np.asarray(
                        #     [filt_dict[name]["res_as"] for name, w in zip(names[:], wavs[:])]
                        # )
                        ress = [psf[1] for psf in psfs]
                        real_ress = []
                        # real_aper_as = []

                        for iband in range(len(names)):

                            img = imgs_MJy_wIGM_wPSF[iband]
                            tgt_res = ress[iband]

                            cur_size = np.asarray(list(img.shape))
                            cur_res_as = pixel_as

                            # print(tgt_res,cur_res_as)

                            if cur_res_as < tgt_res:

                                bin_fact = tgt_res / cur_res_as
                                tgt_size = np.int32(np.ceil(cur_size / bin_fact))

                                # print(tgt_res,cur_res_as,bin_fact)

                                # int_fact = int(np.ceil(bin_fact))

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
                                    rescale_image_flux_conserving(
                                        img, bin_fact, order=4
                                    )  # actually not conserving !!!
                                )

                                # print(img.sum(),imgs_MJy_wIGM_wPSF_rebinned[-1].sum())

                                real_ress.append(
                                    (
                                        img.shape[0]
                                        / imgs_MJy_wIGM_wPSF_rebinned[-1].shape[0]
                                    )
                                    * cur_res_as
                                )
                                # print(real_ress[-1])

                            else:

                                imgs_MJy_wIGM_wPSF_rebinned.append(img)

                                real_ress.append(cur_res_as)

                            # real_aper_as.append(real_ress[-1]*imgs_MJy_wIGM_wPSF_rebinned[-1].shape[0])

                        # print([img.shape for img in imgs_MJy_wIGM_wPSF_rebinned])
                        # print([img.shape for img in imgs_MJy_wIGM_wPSF])

                        fname = os.path.join(
                            outp,
                            "MJy_images",
                            f"mock_images_{sim.name:s}_{snap}_{gid}_{idir:d}_wIGM_wPSF_rebinned.h5",
                        )

                        write_mock_imgs(
                            fname,
                            aexp_mock,
                            time,
                            zmock,
                            convolved_sizes_sr,
                            real_ress,
                            px_as_stmass,
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
                            aexp_mock,
                            rgal,
                            l,
                            lambda_bins,
                            mag_spec,
                            mags,
                            mag_errs,
                            -2.5
                            * np.log10(
                                [
                                    img_MJy * 1e6 * T * (aper_sr / np.prod(imgs.shape))
                                    for T, img_MJy in zip(Ts, imgs_MJy)
                                ]
                            )
                            + 8.9,
                            band_ctrs,
                            [names, wavs, trans],
                            # aper_ckpc,
                            np.full(len(mags),aper_ckpc),
                            vmin=np.full_like(mag_lims, 20),
                            vmax=mag_lims,
                            img_extent_ckpc=extent_ckpc,
                        )

                        weighted_age = gal_dict["age_wmstar"]

                        ztxt = f"z={zmock:.2f}"
                        agetxt = f"age U={tmock:.2f} Gyr"
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
                            outp, f"mock_spectrum_{sim.name:s}_{snap}_{gid}_{idir:d}.png"
                        )

                        fig_spe.savefig(png_fig)

                        plt.close()

                        # fig_spe.savefig(png_fig.replace(".png", ".pdf"), format="pdf")

                        mag_imgs_wpsf = [
                            -2.5
                            * np.log10(
                                img_MJy * 1e6 * (cur_aper_sr / np.prod(img_MJy.shape))
                            )
                            + 8.9
                            for img_MJy, cur_aper_sr in zip(
                                imgs_MJy_wIGM_wPSF, convolved_sizes_sr
                            )
                        ]

                        spec_psf = [
                            -2.5 * np.log10(np.nansum(10 ** (np.asarray(img) / -2.5)))
                            for img in mag_imgs_wpsf
                        ]

                        print("img mags, psf:", list(zip(names, spec_psf)))

                        mag_imgs_wpsf_rebinned = [
                            -2.5
                            * np.log10(
                                img_MJy * 1e6 * (cur_aper_sr / np.prod(img_MJy.shape))
                            )
                            + 8.9
                            for img_MJy, cur_aper_sr in zip(
                                imgs_MJy_wIGM_wPSF_rebinned, convolved_sizes_sr
                            )
                        ]

                        # print(list(zip(names,real_ress,[cur_aper_sr / np.prod(img_MJy.shape) for img_MJy, cur_aper_sr in zip(imgs_MJy_wIGM_wPSF_rebinned, convolved_sizes_sr)])))

                        spec_psf_rebinned = [
                            -2.5 * np.log10(np.nansum(10 ** (np.asarray(img) / -2.5)))
                            for img in mag_imgs_wpsf_rebinned
                        ]

                        print(
                            "img mags, psf+rebinned:", list(zip(names, spec_psf_rebinned))
                        )

                        # # cut to right sizes
                        # mag_imgs_wpsf_rebinned_right_size = []

                        # for iband in range(len(names)):

                        #     # pixel_size_as = filt_dict[names[iband]]["res_as"]
                        #     # pixel_size_as = psfs[iband][1]
                        #     pixel_size_as = real_ress[iband]

                        #     d_A = sim.cosmo_model.angular_diameter_distance(1.0 / aexp - 1.0).value
                        #     theta = pixel_size_as
                        #     theta_radian = theta * np.pi / 180 / 3600
                        #     pixel_size_kpc = d_A * theta_radian * 1e3 / aexp

                        #     half_size = int(0.5 * mag_imgs_wpsf_rebinned[iband].shape[0])

                        #     cur_image = np.copy(mag_imgs_wpsf_rebinned[iband])

                        #     nb_pixels = int( rgal_ckpc / pixel_size_kpc)

                        #     cur_image = cur_image[
                        #         half_size - nb_pixels : half_size + nb_pixels,
                        #         half_size - nb_pixels : half_size + nb_pixels,
                        #     ]

                        #     mag_imgs_wpsf_rebinned_right_size.append(cur_image)

                        fig_spe, ax_spe = plot_spe_bands(
                            aexp_mock,
                            rgal,
                            l,
                            lambda_bins,
                            mag_spec,
                            mags,
                            mag_errs,
                            mag_imgs_wpsf_rebinned,
                            band_ctrs,
                            [names, wavs, trans],
                            # aper_ckpc,
                            np.full(len(mags),aper_ckpc),                        
                            vmax=mag_lims,
                            vmin=np.full_like(mag_lims, 20),
                            img_extent_ckpc=extent_ckpc,
                        )

                        # if you want to see the spectrum from the cube but with psf and rebinned
                        # ax_spe[1].plot(band_ctrs / 10000, spec_psf, c="k", ls="--")
                        # ax_spe[1].plot(band_ctrs / 10000, spec_psf_rebinned, c="k", ls=":")

                        weighted_age = gal_dict["age_wmstar"]

                        ztxt = f"z={zmock:.2f}"
                        agetxt = f"age U={tmock:.2f} Gyr"
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
                            outp,
                            f"mock_spectrum_wPSF_rebinned_{sim.name:s}_{snap}_{gid}_{idir:d}.png",
                        )

                        fig_spe.savefig(png_fig)

                        plt.close()

                        # fig_spe.savefig(png_fig.replace(".png", ".pdf"), format="pdf")

                        print(f"spec plot done: {png_fig}")

                        print("output magnitudes dump")

                        # rbins_as = np.logspace(-2, np.log10(rgal_ckpc * ang_per_ckpc * 2), 50)

                        # profs = np.zeros((len(names), len(rbins_as) - 1))

                        # fig, ax = plt.subplots(
                        #     len(names), 1, figsize=(6, 3 * len(names)), layout="constrained",
                        #     sharex=True,sharey=True
                        # )

                        # for img_name, cur_px_as, img in zip(
                        #     ["IGM", "IGMxPSF", "IGMxPSFxresolution"],
                        #     [pixel_as, pixel_as, real_ress],
                        #     [imgs_MJy_wIGM, imgs_MJy_wIGM_wPSF, imgs_MJy_wIGM_wPSF_rebinned],
                        # ):

                        #     for iband in range(len(names)):

                        #         img_band = img[iband]

                        #         img_size = img_band.shape[0]
                        #         img_center = int(0.5 * img_size)

                        #         Y, X = np.mgrid[0:img_size, 0:img_size]

                        #         if type(cur_px_as) is list:
                        #             band_px_as = cur_px_as[iband]
                        #         else:
                        #             band_px_as = cur_px_as

                        #         band_px_sr = band_px_as**2 * (np.pi / 180 / 3600) ** 2

                        #         for ir, r in enumerate(rbins_as[1:]):

                        #             r_out = rbins_as[ir + 1]

                        #             r_out_px = r_out / band_px_as
                        #             r_in_px = rbins_as[ir] / band_px_as

                        #             dist = np.linalg.norm([X - img_center, Y - img_center], axis=0)

                        #             # cond = (dist < r_out_px) * (dist >= r_in_px) * (img_band > 0)
                        #             cond = dist < r_out_px

                        #             profs[iband, ir] = (np.nansum(img_band[cond])) * band_px_sr
                        #             # profs[iband, ir] = np.nansum(img_band[cond]) * band_px_sr

                        #         profs = -2.5 * np.log10(profs * 1e6) + 8.9

                        #         (l,) = ax[iband].plot(
                        #             0.5 * (rbins_as[1:] + rbins_as[:-1]),
                        #             profs[iband, :],
                        #             label=img_name,
                        #         )

                        #         ax[iband].axvline(
                        #             band_px_as, ls="--", label="dx", color=l.get_color()
                        #         )

                        # ax[0].set_ylabel("Flux [MJy/sr]")
                        # for a in ax:
                        #     a.set_xlabel("Radius [as]")
                        #     a.set_xscale("log")
                        #     a.invert_yaxis()

                        # for a, band in zip(ax, names):
                        #     a.set_title(band)
                        #     a.set_ylabel("mag AB(<r)")
                        #     a.axvline(r50[0] * sim.cosmo.lcMpc * 1e3 * ang_per_ckpc, ls=":", color="k", label="r50")
                        #     a.axvline(rgal_ckpc *  ang_per_ckpc, ls="--", color="k", label="rgal")

                        # ax[0].legend()
                        # fig.savefig(os.path.join(outp,f"mock_spectrum_profiles_{sim.name}_{snap}_{gid}_{idir:d}.png"))

                        mock_spe_dump(
                            outf,
                            zmock,
                            # 2 * rgal_ckpc / ang_per_ckpc,
                            rgal_ckpc / ang_per_ckpc,
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
