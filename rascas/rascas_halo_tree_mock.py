import numpy as np
import matplotlib.pyplot as plt

# from matplotlib.colors import LogNorm
# from scipy.stats import binned_statistic_2d
import os

from zoom_analysis.rascas.rascas_steps import get_directions_cart
from zoom_analysis.visu.visu_fct import plot_stars, basis_from_vect

from zoom_analysis.halo_maker.read_treebricks import (
    convert_star_units,
    read_zoom_brick,
    read_gal_stars,
    read_zoom_stars,
)
from zoom_analysis.stars import sfhs

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
    smooth_props
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

    hm = "HaloMaker_stars2_dp_rec_dust/"
    # dsims = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal"
    # sim_id = "id74099"
    # dsims = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/"
    # sim_id = "id242756_novrel_lowerSFE_stgNHboost_strictSF"
    # sim_id = "id242756_novrel"
    # sim_id = "id242704_novrel"
    # sim_id = "id21892_novrel"
    dsims = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/"
    # sim_id = "id26646"
    sim_id = "id52380"
    sim_path = os.path.join(dsims, sim_id)
    snap_targets=[205]
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

    overwrite = True

    nist_pc = 3.085678e18  # cm
    dist_10pc = 10.0 * nist_pc

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
    model_name = "zoom_tgt_bc03_chabrier100_wlya"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_ndust"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_YD"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_MW"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_noDust"
    # model_name = "zoom_tgt_bc03_chabrier100"
    # model_name = "zoom_tgt_bpass_kroupa100_wlya_noDust"
    # model_name = "zoom_tgt_bpass_chabrier100"
    # model_name = "zoom_tgt_bpass_chabrier100_ignoreDust"

    sim = ramses_sim(sim_path, nml="cosmo.nml")

    tree_name = os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat")
    byte_file = os.path.join(sim.path, "TreeMakerDM_dust")  # , "tree_rev_nbytes")

    # tree_name = os.path.join(sim.path, "TreeMakerTreeHalo", "tree_rev.dat")
    # byte_file = os.path.join(sim.path, "TreeMakerTreeHalo", "tree_rev_nbytes")
    # tree_name = os.path.join(sim.path, "TreeMakerTreeGal", "tree_rev.dat")
    # byte_file = os.path.join(sim.path, "TreeMakerTreeGal", "tree_rev_nbytes")

    zed_target = 2.0

    snap_target = sim.get_closest_snap(zed=zed_target)
    true_zed_snap = 1./sim.get_snap_exps(snap_target)-1

    sim.init_cosmo()

    sim.get_snap_times()


    hid, halo_dict, galaxies = find_zoom_tgt_halo(sim, snap_target, debug=False)

    tree_hids, tree_datas, tree_aexps = read_tree_file_rev_correct_pos(
        # tree_gids, tree_datas, tree_aexps = read_tree_file_rev(
        tree_name,
        sim,
        snap_target,
        byte_file,
        true_zed_snap,
        [hid],
        # [gid],
        # tree_type="halo",
        tgt_fields=["m", "x", "y", "z", "r"],
        star=False,
        debug=False,
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
        sim, tree_aexps, tree_hids, assoc_fields=["r50", "rmax", "mass", "pos", "host hid"]
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

    for i, (snap, aexp, time) in enumerate(
        zip(sim_snaps[::-1], sim_aexps[::-1], sim_times[::-1])
    ):
        if snap_targets is not None:
            if snap not in snap_targets:
                continue

        # print(i, snap, aexp, time)
        if 1.0 / aexp - 1 > zmax:
            continue

        if 1.0 / aexp - 1 < true_zed_snap:
            continue

        if time >= (last_time + deltaT):
            continue

        last_time = time

        if np.min(np.abs(time - tree_times)) > deltaT:
            continue

        if not os.path.isfile(get_halo_assoc_file(sim_path, snap)):
            continue

        tree_arg = np.argmin(np.abs(time - tree_times))
        if np.abs(time - tree_times[tree_arg]) > deltaT:
            continue
        print(f"found tree data for snap {snap:d}")

        tree_arg = np.argmin(np.abs(time - tree_times))

        print(sim_path, snap, tree_hids[tree_arg])

        hdict, hosted_gals = get_halo_props_snap(sim_path, snap, tree_hids[tree_arg])

        if hosted_gals == {}:
            continue
        print(f"found hosted gals for snap {snap:d}")

        aexp_arg = np.argmin(np.abs(aexp - gal_props_tree["aexps"]))

        gid = gal_props_tree['gids'][aexp_arg]

        # if final_gid is None:
        final_gid = gid

        _, cur_gal_props = get_gal_props_snap(sim_path, snap, gid)

        # print(cur_gal_props["pos"], snap, gid, cur_gal_props["mass"])

        rascas_dir = os.path.join(
            rascas_path, f"output_{snap:05d}", f"gal_{final_gid:07d}"
        )

        # # does the gal folder exist?
        # if not os.path.exists(rascas_dir) and os.path.exists(
        #     os.path.join(
        #         rascas_path,
        #         f"output_{snap:05d}",
        #     )
        # ):
        #     # does another gal folder exist?
        #     if np.any(
        #         [
        #             f.startswith("gal")
        #             for f in os.listdir(os.path.join(rascas_path, f"output_{snap:05d}"))
        #         ]
        #     ):
        #         # rename to what we expect
        #         gal_dir = [
        #             f
        #             for f in os.listdir(os.path.join(rascas_path, f"output_{snap:05d}"))
        #             if f.startswith("gal")
        #         ][0]
        #         os.rename(
        #             os.path.join(rascas_path, f"output_{snap:05d}", gal_dir), rascas_dir
        #         )
        #         # print(
        #         #     rascas_dir, os.path.join(rascas_path, f"output_{snap:05d}", gal_dir)
        #         # )

        try:
            rascas_params = read_params(os.path.join(rascas_dir, "params_rascas.cfg"))
        except FileNotFoundError:
            continue
        print(f"found rascas dir or params_rascas.cfg in: {rascas_dir}")

        if not os.path.exists(os.path.join(rascas_dir, "mock_test.spectrum")):
            print(f'no mock_test.spectrum in {rascas_dir}')
            continue
        print(f"found mock_test.spectrum in {rascas_dir}")

        last_time = time

        # print(f"ok snap {snap:d}")

        # continue
        rgal = [gal_props_tree["rmax"][aexp_arg] * rfact]
        mgal = gal_props_tree["mass"][aexp_arg]
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

        pfs_path = os.path.join(rascas_dir, "PFSDump", f"pfsdump")
        #
        names_all, wavs_all, trans_all = read_transmissions()

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

        aper_cm = rgal[0] * 2 * l * (1e6 * ramses_pc)
        # distl = sim.cosmo_model.luminosity_distance(z).value * (1e6 * ramses_pc)
        # aper_as = np.arctan(aper_cm / distl) * 180.0 / np.pi * 3600.0
        aper_as = (rgal[0] * 2 * l * 1e3) * sim.cosmo_model.arcsec_per_kpc_comoving(
            z
        ).value

        fstar = os.path.join(sim.path, hm, f"GAL_{snap:05d}", f"gal_stars_{gid:07d}")
        # print(fstar)
        stars = read_gal_stars(fstar)
        convert_star_units(stars, snap, sim)

        sfh, sfr, ssfr, t_sfr, _ = sfhs.get_sf_stuff(stars, z, sim, deltaT=10.0)

        # print(cur_gal_props)
        # now get in ball of size aperture like rascas does
        stars_ball = read_part_ball_NCdust(
            sim,
            snap,
            gal_props_tree["pos"][aexp_arg],
            gal_props_tree["rmax"][aexp_arg] * rfact * np.sqrt(2),
            ["mass", "birth_time", "metallicity", "pos"],
            fam=2,
        )

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
        gal_dict["age"] = np.max(stars["agepart"])
        gal_dict["age_wmstar"] = np.average(stars["agepart"], weights=stars["mpart"])

        directions = get_directions_cart(hp.npix2nside(ndir))

        # ctrd_star_pos = stars["pos"]-cur_gal_props["pos"]

        if not hasattr(sim, "cosmo_model"):
            sim.init_cosmo()

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

            print("aper_sr:", aper_sr)

            cur_dir = directions[idir]

            basis = basis_from_vect(cur_dir)
            print(cur_dir, basis)

            rgal_ckpc = 2 * rgal[0] * l * 1e3
            tgt_res = 0.5 * (l * 1e3 / 2 ** sim.namelist["amr_params"]["levelmax"])
            tgt_nbins = int(rgal_ckpc / tgt_res)

            real_res = rgal_ckpc / tgt_nbins
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

            pixel_as = tgt_res * (sim.cosmo_model.arcsec_per_kpc_comoving(z).value)

            # print(pixel_as)

            fig, ax = plt.subplots(1, 1)

            # vmin = 3 * np.min(stmasses) / pixel_as**2
            # vmax = 1e3 * np.max(stmasses) / pixel_as**2

            vmin = 1e7
            vmax = 1e14  # msun/as^2

            # basis_order = [basis[2], basis[1], basis[0]]
            basis_order = [basis[0], basis[1], basis[2]]
            # print(basis, basis_order)

            # print(fig,ax,sim,aexp,basis[::-1],tgt_nbins,stmasses, stars_ball["pos"], cur_gal_props["pos"], rgal)
            st_img = plot_stars(
                fig,
                ax,
                sim,
                aexp,
                basis_order,
                tgt_nbins,
                stmasses / pixel_as**2,
                stars_ball["pos"],
                gal_props_tree["pos"][aexp_arg],
                rgal[0] * l * 1e3,
                cb=True,
                cmap="gray",
                label="Stellar surface density [M$_\odot$/as$^2$]",
                vmin=vmin,
                vmax=vmax,
                transpose=True,
                # transpose=False,
                mode="sum",  # sum or mean in cells
                binning="simple",  # simple just use a histogram
            )

            print(np.log10([st_img.sum(), stmasses.sum(), sfh[0]]))

            print(st_img.min(), st_img.max(), np.mean(st_img), np.median(st_img))

            st_fig = os.path.join(outp, f"stars_{idir:d}.png")

            fig.savefig(st_fig)

            print(f"saved {st_fig}")

            # spec_rf = spec_to_erg_s_A_cm2_rf(spec, pfs_path,z=z)
            lmin = spec["lambda_min"]
            lmax = spec["lambda_max"]
            nlamb = int(spec["lambda_npix"])
            lambda_bins_rf = np.linspace(lmin, lmax, nlamb)
            lambda_bins = lambda_bins_rf * (1.0 + z)

            print(lmin,lmax)
            print(lmin*(1+z),lmax*(1+z))

            spec_obs = spec_to_erg_s_A_cm2(spec, pfs_path, z)

            spec_rf = spec_to_erg_s_A_cm2(spec, pfs_path, 0.0)
            spec_rf = spec_rf / (
                4 * np.pi * dist_10pc**2
            )  # at 10 pc from the galaxy for absolute magnitudes

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
                if np.all(wavs_all[i] < lmax * (1.0 + z))
                and np.all(wavs_all[i] > lmin * (1 + z))
            ]

            trans = [
                trans_all[i]
                for i in range(len(names_all))
                if np.all(wavs_all[i] < lmax * (1.0 + z))
                and np.all(wavs_all[i] > lmin * (1 + z))
            ]

            wavs = [
                wavs_all[i]
                for i in range(len(names_all))
                if np.all(wavs_all[i] < lmax * (1.0 + z))
                and np.all(wavs_all[i] > lmin * (1 + z))
            ]

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
            filt_names, mag_spec_rf, mags_no_err_rf, mags_rf, mag_errs_rf = gen_spec(
                1.0, spec_obs, lambda_bins, [names, wavs, trans], rf=True
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
                flamb_fnu(img, band_ctr * aexp) * aper_as**2 / aper_sr / 1e-23 / 1e6
                for band_ctr, img in zip(band_ctrs, imgs)
            ]  # MJy/sr
            imgs_MJy_wIGM = [
                flamb_fnu(img, band_ctr * aexp)
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
                flamb_fnu(np.nansum(imgs, axis=(1, 2)), band_ctrs * aexp)
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

            fig_spe, ax_spe = plot_spe_bands(
                aexp,
                rgal,
                l,
                lambda_bins,
                mag_spec,
                mags,
                mag_errs,
                imgs,
                band_ctrs,
                [names, wavs, trans],
                vmax=28,
                vmin=16,
            )

            weighted_age = gal_dict["age_wmstar"]

            ztxt = f"z={z:.2f}"
            agetxt = f"age U={t:.2f} Gyr"
            agegaltxt = f"age gal={weighted_age/1e3:.2f} Gyr"
            masstxt = "M$_\star$=" + f"{mgal:.2e}" + " M$_\odot$"
            txt = "\n".join([ztxt, agetxt, agegaltxt, masstxt])
            ax_spe[1].text(
                0.85,
                0.05,
                txt,
                transform=ax_spe[1].transAxes,
                fontsize=12,
                va="bottom",
                ha="left",
            )

            png_fig = os.path.join(
                outp, f"mock_spectrum_{sim_id}_{snap}_{gid}_{idir:d}.png"
            )



            fig_spe.savefig(png_fig)

            fig_spe.savefig(png_fig.replace(".png", ".pdf"), format="pdf")

            print(f"spec plot done: {png_fig}")

            print("output magnitudes dump")

            # print(band_mags, len(band_mags))
            # print(names, len(names))

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

        # # recreate wavelength vector
        # nlamb = int(cube["lambda_npix"])
        # lambda_bins = np.linspace(cube["lambda_min"], cube["lambda_max"], nlamb)

        # yrascas = np.linspace(-rgal, +rgal, cube["cube"].shape[0]) * l * 1e3
        # zrascas = np.linspace(-rgal, +rgal, cube["cube"].shape[1]) * l * 1e3

        # # bw_img = cube["cube"].sum(axis=2)

        # # bw_img = cube["cube"][np.argmax(cube["cube"].sum(axis=(0, 1))), :, :]
        # # color_img = np.array_split(cube["cube"], 3, axis=2)
        # # color_img = np.asarray([band.sum(axis=2) for band in color_img])[::-1, :,ﬁ :]
        # # color_img = np.swapaxes(color_img, 0, 2)  # / np.max(color_img)

        # color_band_ctr, color_img = convolve_nircam_cube(
        #     lambda_bins * (1.0 + z), cube["cube"], filt_type="W", tgt_bands=[277, 150, 90]
        # )

        # # color_img_mab = flamb_to_mAB(color_img.swapaxes(0, 2), color_band_ctr).swapaxes(0, 2)
        # # color_img_norm = np.asarray([band / np.max(band) for band in color_img]).T
        # # band_ints = [2, 4, 6]
        # # color_img_norm = make_lupton_rgb(
        # # color_img[2], color_img[1], color_img[0], stretch=0.5, Q=1
        # # )

        # # color_img_norm = make_lupton_rgb(
        # #     flamb_to_mAB(color_img[2], color_band_ctr[2]),
        # #     flamb_to_mAB(color_img[1], color_band_ctr[1]),
        # #     flamb_to_mAB(color_img[0], color_band_ctr[0]),
        # #     stretch=0.5,
        # #     Q=1,
        # # )

        # b = np.log10(color_img[2])
        # b[np.isfinite(b) == False] = 0
        # g = np.log10(color_img[1])
        # g[np.isfinite(g) == False] = 0
        # r = np.log10(color_img[0])
        # r[np.isfinite(r) == False] = 0

        # # b = flamb_to_mAB(color_img[2], color_band_ctr[2])
        # # g = flamb_to_mAB(color_img[1], color_band_ctr[1])
        # # r = flamb_to_mAB(color_img[0], color_band_ctr[0])
        # # color_img = color_img[band_ints, :, :]
        # # color_band_ctr = color_band_ctr[band_ints]

        # # color_img_norm = make_lupton_rgb(
        # #     color_img[0], color_img[1], color_img[2], stretch=0.9, Q=10
        # # )

        # # min_mag = np.min((np.min(r[r > -999]), np.min(g[g > -999]), np.min(b[b > -999])))
        # # max_mag = np.max((np.min(r[r < 999]), np.min(g[g < 999]), np.min(b[b < 999])))

        # min_mag = np.min([r, g, b])
        # max_mag = np.max([r, g, b])

        # # b = (b - min_mag) / (max_mag - min_mag)
        # # g = (g - min_mag) / (max_mag - min_mag)
        # # r = (r - min_mag) / (max_mag - min_mag)

        # b = (b - max_mag) / min_mag
        # g = (g - max_mag) / min_mag
        # r = (r - max_mag) / min_mag
        # # b = (b - b.max()) / b.min()
        # # g = (g - g.max()) / g.min()
        # # r = (r - r.max()) / r.min()

        # # print(color_img.shape)

        # fig_compo, ax_compo = plot_mock_composite(yrascas, zrascas, b, g, r)

        # # read photons

        # photons = read_photon_dump(os.path.join(rascas_dir, "RASCASDump", "rascas_dump.dat"))

        # # print(photons)

        # # find centre
        # # print(np.median(photons["xlast"], axis=0))

        # # print(np.max(photons["nb_abs"]))
        # # print(np.diff(photons["xlast"], axis=0))

        # fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
        # img_size = int(150)

        # ph = read_PFS_dump(os.path.join(rascas_dir, "PFSDump", f"pfsdump"))

        # pos = ph["x_em"]
        # y = np.linspace(pos[:, 1].min(), pos[:, 1].max(), img_size)
        # z = np.linspace(pos[:, 2].min(), pos[:, 2].max(), img_size)

        # pfs_img, _, _, _ = binned_statistic_2d(
        #     pos[:, 1], pos[:, 2], pos[:, 2], bins=[y, z], statistic="count"
        # )

        # img = ax.imshow(
        #     pfs_img.T,
        #     origin="lower",
        #     cmap="gray_r",
        #     norm=LogNorm(vmin=1),
        #     extent=np.asarray(
        #         [
        #             (y[0] - ctr_gal[1])[0] * l * 1e3,
        #             (y[-1] - ctr_gal[1])[0] * l * 1e3,
        #             (z[0] - ctr_gal[2])[0] * l * 1e3,
        #             (z[-1] - ctr_gal[2])[0] * l * 1e3,
        #         ]
        #     ),
        # )

        # cb = fig.colorbar(img, ax=ax)
        # cb.set_label("Number of photons")

        # # draw circle at ctr_gal, radius rgal
        # # circle = plt.Circle((ctr_gal[1] * l, ctr_gal[2] * l), rgal * l, color="r", fill=False)
        # # circle = plt.Circle((0.0, 0.0), rgal * l, color="r", fill=False)
        # # ax.add_artist(circle)

        # pos_last = photons["xlast"]

        # # print(pos_last)

        # # x = np.linspace(pos_last[:, 1].min(), pos_last[:, 1].max(), img_size)
        # # y = np.linspace(pos_last[:, 2].min(), pos_last[:, 2].max(), img_size)

        # # pfs_img_last, _, _, _ = binned_statistic_2d(
        # #     pos_last[:, 1], pos_last[:, 2], pos_last[:, 2], bins=[x, y], statistic="count"
        # # )

        # # print(pfs_img_last.max())

        # # ax.imshow(
        # #     pfs_img_last.T,
        # #     origin="lower",
        # #     cmap="gray",
        # #     norm=LogNorm(vmin=1),
        # #     extent=[x[0], x[-1], y[0], y[-1]],
        # # )

        # # print(x[0], x[-1], y[0], y[-1])
        # # print(yrascas[0], yrascas[-1], zrascas[0], zrascas[-1])

        # ax.set_xlim((y[0] - ctr_gal[1]) * l * 1e3, (y[-1] - ctr_gal[1]) * l * 1e3)
        # ax.set_ylim((z[0] - ctr_gal[2]) * l * 1e3, (z[-1] - ctr_gal[2]) * l * 1e3)

        # ax.set_xlabel(r"$y, \mathrm{ckpc/h}$")
        # ax.set_ylabel(r"$z, \mathrm{ckpc/h}$")

        # xarg, yarg = np.where(pfs_img == pfs_img.max())

        # # ax.axvline(x=x[xarg[0]] * l, c="r")
        # # ax.axhline(y=y[yarg[0]] * l, c="r")

        # # print(x[xarg[0]], y[yarg[0]])

        # fig.savefig("pfs_img.png")

        # print("photons PFS done")

        # print("done")
