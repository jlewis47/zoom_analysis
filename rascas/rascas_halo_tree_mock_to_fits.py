from matplotlib.patches import Circle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from astropy import wcs as WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from reproject import reproject_interp

# from matplotlib.colors import LogNorm
# from scipy.stats import binned_statistic_2d
import os


from zoom_analysis.rascas.rascas_steps import filt_file_to_dict, get_directions_cart
from zoom_analysis.read.read_data import read_data_ball
from zoom_analysis.visu.visu_fct import plot_stars, basis_from_vect

from matplotlib.colors import LogNorm

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

from zoom_analysis.rascas.field_fits import (
    gals_to_fit_pos,
    read_fits,
    save_fits,
    tile_to_file,
    get_hdr_res,
)
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

    hm = "HaloMaker_stars2_dp_rec_dust/"

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
    dsims = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/"
    # sim_id = "id26646"
    sim_id = "id52380"
    # sim_id = "id180130" 
    sim_path = os.path.join(dsims, sim_id)
    zed_targets = None
    snap_targets = None
    # snap_targets=[205]
    # gid=1
    # snap_targets=[206]
    # gid=2
    # snap_targets=[207]
    # snap_targets=np.arange(150,208)
    # snap_targets=np.arange(150,199)
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

    # idir = 1
    idir = 3

    overwrite = True
    debug = False

    nist_pc = 3.085678e18  # cm
    dist_10pc = 10.0 * nist_pc

    dust_model = "SMC"

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
    model_name = f"zoom_tgt_bc03_chabrier100_wlya_{dust_model}"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_SMC_draine"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_YD"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_MW"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_noDust"
    # model_name = "zoom_tgt_bc03_chabrier100"
    # model_name = "zoom_tgt_bpass_kroupa100_wlya_noDust"
    # model_name = "zoom_tgt_bpass_chabrier100"
    # model_name = "zoom_tgt_bpass_chabrier100_ignoreDust"

    psf_dict = filt_file_to_dict()

    sim = ramses_sim(sim_path, nml="cosmo.nml")

    name = sim.name
    intID = int(name[2:].split("_")[0])

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
    true_zed_snap = 1.0 / sim.get_snap_exps(snap_target)[0] - 1

    sim.init_cosmo()

    sim.get_snap_times()
    sim_aexps = sim.aexps

    # hid, halo_dict, galaxies = find_zoom_tgt_halo(sim, snap_target, debug=False)

    _, data_snaps = sim.get_snaps(tar_snaps=True, full_snaps=True, mini_snaps=False)
    data_snap_aexps = sim.get_snap_exps(data_snaps)
    close_tol = 1e-3

    assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
    assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

    avail_aexps = np.intersect1d(
        sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
    )
    avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

    if zed_targets is not None:
        snap_target_args = np.asarray(
            [np.argmin(np.abs(data_snap_aexps - 1.0 / (z + 1.0))) for z in zed_targets]
        )
        snap_targets = data_snaps[snap_target_args]
        err = data_snap_aexps[snap_target_args] - 1.0 / (np.asarray(zed_targets) + 1.0)
        snap_targets = snap_targets[np.abs(err) < close_tol]
    elif snap_targets is None:
        snap_targets = sim.snap_numbers

    print(f"Targets are {snap_targets}")

    hid_start, _, _, true_start_aexp, found = starting_hid_from_hagn(
        zstt, sim, hagn_sim, intID, avail_aexps, avail_times
    )

    true_snap = sim.get_closest_snap(aexp=true_start_aexp)

    true_zed_snap = 1.0 / true_start_aexp - 1.0

    tree_hids, tree_datas, tree_aexps = read_tree_file_rev_correct_pos(
        # tree_gids, tree_datas, tree_aexps = read_tree_file_rev(
        tree_name,
        sim,
        true_snap,
        byte_file,
        true_zed_snap,
        [hid_start],
        # [gid],
        # tree_type="halo",
        tgt_fields=["m", "x", "y", "z", "r"],
        star=False,
        verbose=False,
    )

    if not hasattr(sim, "cosmo_model"):
        sim.init_cosmo()

    tree_times = sim.cosmo_model.age(1.0 / tree_aexps - 1.0).value * 1e3  # Myr

    filt = tree_datas["x"][0] != -1

    for key in tree_datas:
        tree_datas[key] = tree_datas[key][0][filt]
    # tree_gids = tree_gids[0][filt]
    tree_hids = tree_hids[0][filt]
    # tree_gids = tree_hids[0][filt]
    tree_aexps = tree_aexps[filt]
    tree_times = tree_times[filt]

    tree_snaps = np.asarray([sim.get_closest_snap(aexp=a) for a in tree_aexps])

    gal_props_tree = get_assoc_pties_in_tree(
        sim,
        tree_aexps,
        tree_hids,
        assoc_fields=["r50", "rmax", "mass", "pos", "host hid"],
    )

    r50s = gal_props_tree["r50"]
    rmaxs = gal_props_tree["rmax"]
    masses = gal_props_tree["mass"]
    poss = gal_props_tree["pos"]
    hids = gal_props_tree["host hid"]

    smooth_gal_props = smooth_props(gal_props_tree)

    sim_aexps = sim.get_snap_exps()  # [::-1]
    sim_times = sim.get_snap_times()  # [::-1]
    sim_snaps = sim.snap_numbers  # [::-1]

    rascas_path = os.path.join(dsims, sim_id, "rascas", model_name)

    rascas_snaps = np.asarray(
        [
            int(f.split("_")[-1])
            for f in os.listdir(rascas_path)
            if f.startswith("output_")
        ]
    )
    # print(rascas_snaps)
    # print(sim_snaps)

    last_time = sim_times[-1] + 2 * deltaT

    final_gid = None

    fnames = []
    all_rgals = []
    all_rgals_px = []
    zeds = []
    snapshots = []
    galaxy_ids = []
    all_mgals = []
    all_r50s = []
    all_m_r50s = []
    all_sfrs_10 = []
    all_sfrs_10_r50 = []
    band_names = []
    band_sums_mag = []
    band_sums_mag_r50 = []
    img_size = []
    # sfr10s=[]
    # sfr100s=[]
    # sfr1000s=[]

    for i, (snap, aexp, time) in enumerate(
        zip(sim_snaps[::-1], sim_aexps[::-1], sim_times[::-1])
    ):
        if snap_targets is not None:
            if snap not in snap_targets:

                continue

        print(i, snap, aexp, time)
        if 1.0 / aexp - 1 > zmax:
            print(f"skipping snap {snap:d} at z={1.0/aexp-1:.2f}")
            continue

        # if 1.0 / aexp - 1 < true_zed_snap:
        #     # print(snap,aexp,true_zed_snap)
        #     print(f"skipping snap {snap:d} at z={1.0/aexp-1:.2f} < {true_zed_snap:.2f}")
        #     continue

        if time >= (last_time + deltaT):
            print(f"skipping snap {snap:d} at time={time:.2f} > {last_time:.2f}")
            continue

        last_time = time

        if np.min(np.abs(time - tree_times)) > deltaT:
            print(f"skipping snap {snap:d} at time={time:.2f} > {tree_times.max():.2f}")
            continue

        if not os.path.isfile(get_halo_assoc_file(sim_path, snap)):
            print(f"skipping snap {snap:d} at time={time:.2f} no assoc file")
            continue

        tree_arg = np.argmin(np.abs(time - tree_times))
        if np.abs(time - tree_times[tree_arg]) > deltaT:
            print(f"skipping snap {snap:d} at time={time:.2f} > {tree_times.max():.2f}")
            continue
        print(f"found tree data for snap {snap:d}")

        tree_arg = np.argmin(np.abs(time - tree_times))

        print(sim_path, snap, tree_hids[tree_arg])

        hdict, hosted_gals = get_halo_props_snap(
            sim_path, snap, tree_hids[tree_arg], hosted_gals=True
        )

        if hosted_gals == {}:
            print(f"no hosted gals for snap {snap:d}")
            continue
        print(f"found hosted gals for snap {snap:d}")

        aexp_arg = np.argmin(np.abs(aexp - gal_props_tree["aexps"]))

        gid = gal_props_tree["gids"][aexp_arg]

        # if final_gid is None:
        # final_gid = gid

        _, cur_gal_props = get_gal_props_snap(sim_path, snap, gid)

        # print(cur_gal_props["pos"], snap, gid, cur_gal_props["mass"])

        rascas_dir = os.path.join(rascas_path, f"output_{snap:05d}", f"gal_{gid:07d}")

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
        rgal = gal_props_tree["rmax"][aexp_arg] * rfact
        r50 = gal_props_tree["r50"][aexp_arg] * rfact
        mgal = gal_props_tree["mass"][aexp_arg]
        # sfr10 = gal_props_tree["sfr10"][aexp_arg]
        # sfr100 = gal_props_tree["sfr100"][aexp_arg]
        # sfr1000 = gal_props_tree["sfr1000"][aexp_arg]
        agegal = cur_gal_props["max age"]

        # ctr_gal = [0.63813084, 0.7364341052162953, 0rgl.42615680278854773]
        # rgal = 0.0003
        # l = 100e3  # ckpc/h
        [aexp] = sim.get_snap_exps(snap)
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

        rgal_ckpc = rgal * l * 1e3
        rgal_as = sim.cosmo_model.arcsec_per_kpc_comoving(z).value * rgal_ckpc
        r50_ckpc = r50 * l * 1e3
        r50_as = sim.cosmo_model.arcsec_per_kpc_comoving(z).value * r50_ckpc

        # pfs_path = os.path.join(rascas_dir, "PFSDump", f"pfsdump")
        #
        # names_all, wavs_all, trans_all = read_transmissions()

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

        # sfh, sfr, ssfr, t_sfr, _ = sfhs.get_sf_stuff(stars, z, sim, deltaT=10.0)

        try:
            stars_ball = read_data_ball(sim,
                                        snap,
                                        gal_props_tree["pos"][aexp_arg],
                                        rgal,
                                        int(gal_props_tree['host hid'][aexp_arg]),
                                        data_types=['stars'],
                                        tgt_fields=["age","mass", "birth_time", "metallicity", "pos"],)
        except FileNotFoundError:
            print("Couldn't find data files")
            print("skipping")
            continue
        # print(stars_ball.keys())

        stmasses = sfhs.correct_mass(
            sim, stars_ball["age"], stars_ball["mass"], stars_ball["metallicity"]
        )

        sfh_ball, sfr_ball, ssfr_ball, t_sfr_ball, _ = sfhs.get_sf_stuff(
            stars_ball, z, sim, deltaT=10.0
        )

        # break

        print(stars_ball["pos"].shape)


        dist_to_ctr = np.linalg.norm(stars_ball["pos"] - gal_props_tree["pos"][aexp_arg],axis=1)

        st_in_r50 = dist_to_ctr <= r50

        stars_ball_r50 = {k:stars_ball[k][st_in_r50] for k in stars_ball.keys()}

        sfh_ball_r50, sfr_ball_r50, ssfr_ball_r50, t_sfr_ball_r50, _ = sfhs.get_sf_stuff(
            stars_ball_r50, z, sim, deltaT=10.0
        )



        # gal_dict = {}
        # gal_dict["mass"] = sfh
        # gal_dict["sfr"] = sfr
        # gal_dict["ssfr"] = ssfr
        # gal_dict["mass_aper"] = sfh_ball
        # gal_dict["sfr_aper"] = sfr_ball
        # gal_dict["ssfr_aper"] = ssfr_ball
        # gal_dict["time"] = t_sfr
        # gal_dict["age"] = np.max(stars["agepart"])
        # gal_dict["age_wmstar"] = np.average(stars["agepart"], weights=stars["mpart"])

        # ctrd_star_pos = stars["pos"]-cur_gal_props["pos"]

        if not hasattr(sim, "cosmo_model"):
            sim.init_cosmo()

        img_dir = os.path.join(outp, "MJy_images")

        if not os.path.exists(img_dir):
            os.makedirs(img_dir, exist_ok=True)

        fname_mock_img = os.path.join(
            outp,
            "MJy_images",
            f"mock_images_{sim_id}_{snap}_{gid}_{idir:d}_wIGM_wPSF_rebinned.h5",
        )

        if os.path.exists(fname_mock_img):

            fnames.append(fname_mock_img)
            all_rgals.append(rgal_as)
            all_r50s.append(r50_as)
            zeds.append(1.0 / aexp - 1)
            all_mgals.append(mgal)
            snapshots.append(snap)
            galaxy_ids.append(gid)
            all_sfrs_10.append(sfr_ball.sum())
            all_sfrs_10_r50.append(sfr_ball_r50.sum())
            all_m_r50s.append(stmasses[st_in_r50].sum())
            # sfr10s.append(sfr10)
            # sfr100s.append(sfr100)
            # sfr1000s.append(sfr1000)

    filt_dict = {
        "f115w": "F115",
        "f150w": "F150",
        "f277w": "F277",
        "f444w": "F444",
        "HST-F814W": "f814_res",
        "HSC-r": "rHSC",
        "HSC-g": "gHSC",
        "HSC-i": "iHSC",
        "HSC-y": "yHSC",
        "HSC-z": "zHSC",
        "CFHT-u": "u_new",
    }

    all_rgals = np.asarray(all_rgals)

    tile_code, tile_shape_px, found_pos_px, all_rgals_px, found_all, wcs = (
        gals_to_fit_pos(all_rgals * 1.1, niter=len(all_rgals) * 100000)
    )

    found_pos_deg = wcs.pixel_to_world(found_pos_px[:, 0], found_pos_px[:, 1])

    # upper_y, upper_x = found_pos_px + rgals_px
    # lower_y, lower_x = found_pos_px - rgals_px

    assert found_all, "didn't find a spot for all my galaxies :'/"

    out_dir = os.path.join(
        "/data101/jlewis/zoom_flats",
        f"{sim_id:s}",
        "zoom_target",
        f"{idir:d}",
        f"{dust_model:s}",
    )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_flats_cat = os.path.join(out_dir, f"out_cat_{tile_code}.txt")
    out_flats_img_residual = os.path.join(out_dir, f"out_cat_{tile_code}_residual.png")
    out_flats_img = os.path.join(out_dir, f"out_cat_{tile_code}.png")

    fname_res, _ = tile_to_file("f444w", tile_code)

    f444_hdr, _ = read_fits(fname_res)


    for ifilt,filt in enumerate([
        "CFHT-u",
        "HSC-g",
        "HSC-i",
        "HSC-r",
        "HSC-y",
        "HSC-z",
        "HST-F814W",
        "f115w",
        "f150w",
        "f277w",
        "f444w",
    ]):
        # for ifile,filt in enumerate(["f277w"]):

        band_names.append(filt)

        fname_res, _ = tile_to_file(filt, tile_code)

        res_hdr, res_data = read_fits(fname_res)

        # res_hdu = fits.PrimaryHDU(data=res_data, header=res_hdr)

        wcs_cur_tile = WCS.WCS(res_hdr)

        # resolution_hdr_mas = res_hdr["CDELT1"]*3600*1e3

        resolution_hdr_mas = get_hdr_res(res_hdr) * 1e3

        # resolution_hdr_mas = psf_dict[filt_dict[filt]]["res_as"] * 1e3

        # res_data = array, footprint = reproject_interp(res_hdu, f444_hdr)

        res_data = res_data.T

        out_flats_fname = os.path.join(out_dir, f"out_flat_{filt:s}_{tile_code}.fits")

        filt_sum_mag = []
        filt_sum_mag_r50 = []

        for fname, rgal, rgal_as, r50_as, found_pos in zip(
            fnames[:],
            all_rgals_px[:],
            all_rgals[:],
            all_r50s[:],
            np.transpose([found_pos_deg.ra.value, found_pos_deg.dec.value])[:],
        ):

            with h5py.File(fname, "r") as src:
                try:
                    img_gal = src[filt_dict[filt]][()]
                    px_as = src["header"].attrs["pixel_res_as"]
                except KeyError:
                    print(f"error reading {filt_dict[filt]} from {fname}")
                    continue

            size = img_gal.shape

            xrgal = int(size[0] * 0.5)
            yrgal = int(size[1] * 0.5)

            # print(found_pos, rgal, xrgal, yrgal, res_data.shape)
            # print(filt, size)

            xpos_deg, ypos_deg = found_pos

            sky = SkyCoord(xpos_deg * u.degree, ypos_deg * u.degree)
            cur_found_pos_px = wcs_cur_tile.world_to_pixel(sky)

            xmin = int(cur_found_pos_px[0]) - xrgal
            ymin = int(cur_found_pos_px[1]) - yrgal

            # this way might be a pixel off but who cares for what I'm doing
            res_data[xmin : xmin + size[0], ymin : ymin + size[1]] += img_gal

            # aper_sr = np.pi * (aper_as / 3600 / 180 * np.pi) ** 2
            rgal_sr = np.pi * (2 * rgal_as / 3600 / 180 * np.pi) ** 2
            rgal_px = abs(
                int(
                    np.ceil(
                        wcs_cur_tile.wcs_world2pix(
                            xpos_deg + rgal_as / 3600.0,
                            ypos_deg,
                            0,
                        )[0]
                        - wcs_cur_tile.wcs_world2pix(xpos_deg, ypos_deg, 0)[0]
                    )
                )
            )

            r50_sr = np.pi * (2 * r50_as / 3600 / 180 * np.pi) ** 2
            r50_px = rgal_px * r50_as/rgal_as
            # aper_ratio = r50_as / aper_as
            # aper_size_px = aper_ratio * img_gal.shape[0]
            # r_select_px = int(0.5 * aper_size_px)
            half_size_px = int(0.5 * img_gal.shape[0])

            Ximg, Yimg = (
                np.mgrid[0 : img_gal.shape[0], 0 : img_gal.shape[1]] - half_size_px
            )
            select_circ = np.linalg.norm([Ximg, Yimg], axis=0) < rgal_px

            aper_mag = (
                -2.5
                * np.log10(
                    np.sum(img_gal[select_circ]) * 1e6 * (rgal_sr / (rgal_px * 2) ** 2)
                )
                + 8.90
            )

            filt_sum_mag.append(aper_mag)

            if r50_px >= 1 : 

                r50_px = int(np.ceil(r50_px))

                select_circ_r50 = np.linalg.norm([Ximg, Yimg], axis=0) < r50_px

                aper_mag_r50 = (
                    -2.5
                    * np.log10(
                        np.sum(img_gal[select_circ]) * 1e6 * (r50_sr / (r50_px * 2) ** 2)
                    )
                    + 8.90
                )

            else:

                flux_central_px = img_gal[half_size_px, half_size_px]
                aper_mag_r50 = -2.5*np.log10(flux_central_px * r50_px**2 * 1e6 * (r50_sr / (r50_px * 2) ** 2)) + 8.90

            filt_sum_mag_r50.append(aper_mag_r50)

            print(filt, aper_mag, aper_mag_r50)

            if debug:

                fig, ax = plt.subplots(1, 2)

                ax[0].imshow(img_gal.T, origin="lower", vmin=0, vmax=1.5)
                ax[1].imshow(
                    res_data[xmin : xmin + size[0], ymin : ymin + size[1]].T,
                    origin="lower",
                    vmin=0,
                    vmax=1.5,
                )

                print(
                    img_gal.max(),
                    img_gal.min(),
                    np.sum(img_gal == 0),
                    np.prod(img_gal.shape),
                )

                dfig_name = f"{filt}_test_insertion"
                print("debug: saving {dfig_name}")

                fig.savefig(dfig_name)

            print(rgal, img_gal.shape)

            # input('')

        # newHDU = fits.PrimaryHDU(data=res_data,)

        # res_data = array, footprint = reproject_interp(newHDU, res_hdr)

        band_sums_mag.append(filt_sum_mag)
        band_sums_mag_r50.append(filt_sum_mag_r50)

        # res_hdr["BAND"] = filt

        save_fits(out_flats_fname, res_hdr, res_data, overwrite=True)

    # res_data[:,-800:]=50 # top
    # res_data[4500:5500,7750:14000]=50 # in field middle
    # res_data[8750:10000,1000:5500]=50 # in field middle
    # res_data[9000:9750,-2300:]=50 # in field middle
    # res_data[13750:14000,10500:13000]=50 # in field middle

    font = {"family": "normal", "weight": "bold", "size": 50}

    matplotlib.rc("font", **font)

    fig, ax = plt.subplots(1, 1, figsize=(100, 100))

    img = ax.imshow(
        res_data.T, origin="lower", vmin=-0.015, vmax=0.015, cmap="bwr"
    )  # ,norm=LogNorm())

    plt.colorbar(img, ax=ax, label="MJ/sr")

    for found_pos, rgal_px in zip(found_pos_px, all_rgals_px):

        circ = Circle(found_pos, 150, edgecolor="k", facecolor="none", linewidth=15)
        ax.add_artist(circ)

    ax.set_xlabel("pixels")
    ax.set_ylabel("pixels")

    fig.savefig(out_flats_img_residual)

    fig, ax = plt.subplots(1, 1, figsize=(100, 100))

    img = ax.imshow(
        res_data.T, origin="lower", vmin=0, vmax=3, cmap="Greys_r"
    )  # ,norm=LogNorm())

    plt.colorbar(img, ax=ax, label="MJ/sr")

    for found_pos, rgal_px in zip(found_pos_px, all_rgals_px):

        circ = Circle(found_pos, 150, edgecolor="white", facecolor="none", linewidth=15)
        ax.add_artist(circ)

    ax.set_xlabel("pixels")
    ax.set_ylabel("pixels")

    fig.savefig(out_flats_img)

    np.savetxt(
        out_flats_cat,
        np.transpose(
            [
                np.int16(found_pos_px[:, 0]),
                np.int16(found_pos_px[:, 1]),
                all_rgals_px,
                found_pos_deg.ra.value,
                found_pos_deg.dec.value,
                np.asarray(all_rgals * 1.1) / 3600.0,
                np.asarray(all_r50s) / 3600.0,  # broken
                zeds,
                snapshots,
                galaxy_ids,
                all_mgals,
                all_m_r50s,
                all_sfrs_10,
                all_sfrs_10_r50,
                *band_sums_mag,
                *band_sums_mag_r50,
                # sfr100s,
                # sfr1000s,
            ]
        ),
        # fmt="%.6e",
        header=("x(px) y(px) rgal(px) ra dec rgal(deg) r50(deg) z snapshot galid mgal(msun) m_r50(msun) sfr_10Myr(msun/Myr) sfr_10Myr_r50(msun/Myr)"
        + " ".join([b + "_rgal(magAB)" for b in band_names]) + " ".join([b + "_r50(magAB)" for b in band_names])),
    )
