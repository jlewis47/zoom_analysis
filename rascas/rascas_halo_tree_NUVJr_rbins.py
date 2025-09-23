import numpy as np
import matplotlib.pyplot as plt

# from matplotlib.colors import LogNorm
# from scipy.stats import binned_statistic_2d
import os

from scipy.stats import f

from zoom_analysis.rascas.rascas_steps import get_directions_cart
from zoom_analysis.visu.visu_fct import plot_stars, basis_from_vect

from zoom_analysis.read.read_data import read_data_ball

# from zoom_analysis.halo_maker.read_treebricks import (
#     convert_star_units,
#     read_zoom_brick,
#     read_gal_stars,
#     read_zoom_stars,
# )
from zoom_analysis.stars import sfhs

import healpy as hp

# from zoom_analysis.stars.star_reader import read_part_ball,
# from f90_tools.star_reader import read_part_ball_NCdust

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
    m_to_M
)

# from zoom_analysis.trees.tree_reader import read_tree_file_rev
from zoom_analysis.trees.tree_reader import read_tree_file_rev_correct_pos


# from matplotlib.gridspec import GridSpec

# from astropy.visualization import make_lupton_rgb

# from scipy.integrate import simps

from zoom_analysis.constants import *

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



if __name__ == "__main__":

    verbose=False

    snap_targets=None
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
    snap_targets=np.unique(np.concatenate([np.arange(180,208,5),[207]]))
    # snap_targets=[207]
    # gid=4
    # snap_targets=[206]
    # gid=2
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
    rfact = 3.0

    sfr_dt = 100.0 #Myr

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
    # model_name = "zoom_tgt_bc03_chabrier100_wlya"
    model_name = "zoom_tgt_bc03_chabrier100_wlya_SMC"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_ndust"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_SMC_draine"
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

    t0 = sim.times[np.argmin(np.abs(snap_targets.min()-sim.snap_numbers))]
    t1 = sim.times[np.argmin(np.abs(snap_targets.max()-sim.snap_numbers))]

    rbins_ckpc=np.logspace(0,2.2,7)

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

    sim_aexps = sim.get_snap_exps(snap_targets)  # [::-1]
    sim_times = sim.get_snap_times(snap_nbs=snap_targets)  # [::-1]
    # sim_snaps = sim.snap_numbers  # [::-1]

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

    for rbin_ckpc in rbins_ckpc:

        final_gid = None
        last_time = sim_times[-1] + 2 * deltaT
        print(f"rbin_ckpc: {rbin_ckpc:.1f} ckpc")

        zoom_Mstars = np.zeros((len(snap_targets)))
        zoom_SFR = np.zeros((len(snap_targets)))
        zoom_sSFR = np.zeros((len(snap_targets)))
        zoom_NUV_r = np.zeros((len(snap_targets),ndir))
        zoom_r_j = np.zeros((len(snap_targets),ndir))
        zoom_F150W_F277W = np.zeros((len(snap_targets),ndir))
        zoom_F277W_F444W = np.zeros((len(snap_targets),ndir))

        itgt=0

        for i, (snap, aexp, time) in enumerate(
            zip(snap_targets[::-1], sim_aexps[::-1], sim_times[::-1])
        ):

            # print(i, snap, aexp, time)
            if 1.0 / aexp - 1 > zmax:
                print("z too high")
                continue

            if 1.0 / aexp - 1 < true_zed_snap:
                print("z too low")
                continue

            if time >= (last_time + deltaT):
                print("time too high")
                continue

            last_time = time

            if np.min(np.abs(time - tree_times)) > deltaT:
                print("no tree data")
                continue

            if not os.path.isfile(get_halo_assoc_file(sim_path, snap)):
                print(f"no assoc file for snap {snap:d}")
                continue

            tree_arg = np.argmin(np.abs(time - tree_times))
            if np.abs(time - tree_times[tree_arg]) > deltaT:
                print(f"no tree data for snap {snap:d}")
                continue
            print(f"found tree data for snap {snap:d}")

            tree_arg = np.argmin(np.abs(time - tree_times))

            print(sim_path, snap, tree_hids[tree_arg])

            hdict, hosted_gals = get_halo_props_snap(sim_path, snap, tree_hids[tree_arg])

            if hosted_gals == {}:
                print(f"no hosted gals for snap {snap:d}")
                continue
            print(f"found hosted gals for snap {snap:d}")

            aexp_arg = np.argmin(np.abs(aexp - gal_props_tree["aexps"]))

            gid = gal_props_tree['gids'][aexp_arg]

            # if final_gid is None:
            # final_gid = gid
            # gid=4

            _, cur_gal_props = get_gal_props_snap(sim_path, snap, gid)

            # print(cur_gal_props["pos"], snap, gid, cur_gal_props["mass"])

            rascas_dir = os.path.join(
                rascas_path, f"output_{snap:05d}", f"gal_{gid:07d}"
            )

            try:
                rascas_params = read_params(os.path.join(rascas_dir, "params_rascas.cfg"))
            except FileNotFoundError:
                continue
            if verbose:
                print(f"found rascas dir or params_rascas.cfg in: {rascas_dir}")

            if not os.path.exists(os.path.join(rascas_dir, "mock_test.spectrum")):
                if verbose:
                    print(f'no mock_test.spectrum in {rascas_dir}')
                continue
            # if verbose:
            print(f"found mock_test.spectrum in {rascas_dir}")

            last_time = time

            # print(f"ok snap {snap:d}")

            tgt_pos = cur_gal_props["pos"]

            # continue
            # rgal = gal_props_tree["rmax"][aexp_arg] * rfact
            rgal = smooth_gal_props["r50"][aexp_arg] * rfact
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
            mocks_spe_rf = read_mock_dump(
                os.path.join(rascas_dir, "mock_test.spectrum"), ndir=ndir
            )
            mocks_cube = read_mock_dump(
                os.path.join(rascas_dir, "mock_test.cube"), ndir=ndir
            )
            mocks_cube_rf = read_mock_dump(
                os.path.join(rascas_dir, "mock_test.cube"), ndir=ndir
            )


            rbin = rbin_ckpc / (sim.cosmo.lcMpc * 1e3)  # code

            # aper_cm = rgal * 2 * l * (1e6 * ramses_pc)
            aper_cm = rbin_ckpc * (1e3 * ramses_pc)
            # distl = sim.cosmo_model.luminosity_distance(z).value * (1e6 * ramses_pc)
            # aper_as = np.arctan(aper_cm / distl) * 180.0 / np.pi * 3600.0
            aper_as = (rbin * 2 * l * 1e3) * sim.cosmo_model.arcsec_per_kpc_comoving(
                z
            ).value

            fstar = os.path.join(sim.path, hm, f"GAL_{snap:05d}", f"gal_stars_{gid:07d}")
            # print(fstar)
            # stars = read_gal_stars(fstar)
            # convert_star_units(stars, snap, sim)

            stars_ball = read_data_ball(sim, 
                                        snap, 
                                        gal_props_tree["pos"][aexp_arg],
                                        rbin, 
                                        int(gal_props_tree['host hid'][aexp_arg]),
                                        data_types=["stars"], 
                                        tgt_fields=['age','mass','metallicity','pos'])

            if stars_ball is None:
                print(f'{rbin_ckpc:.2f}, no stars')
                continue

            # sfh, sfr, ssfr, t_sfr, _ = sfhs.get_sf_stuff(stars, z, sim, deltaT=10.0)

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

            # print(stars_ball.keys())

            stmasses = sfhs.correct_mass(
                sim, stars_ball["age"], stars_ball["mass"], stars_ball["metallicity"]
            )

            if stmasses.sum() == 0:
                print("no stars in ball")
                continue

            # sfh_ball, sfr_ball, ssfr_ball, t_sfr_ball, _ = sfhs.get_sf_stuff(
            #     stars_ball, z, sim, deltaT=sfr_dt
            # )

            # print(stars_ball['age'].min(),stars_ball['age'].max())

            gal_mass = stmasses.sum()
            gal_sfr = stmasses[stars_ball['age']<sfr_dt].sum()/1e6/sfr_dt #sfr_dt myr
            gal_ssfr = gal_sfr / gal_mass

            zoom_Mstars[itgt] = gal_mass
            zoom_SFR[itgt] = gal_sfr
            zoom_sSFR[itgt] = gal_ssfr


            assert zoom_Mstars[itgt]>0, 'no stellar mass'
                

            directions = get_directions_cart(hp.npix2nside(ndir))

            # ctrd_star_pos = stars["pos"]-cur_gal_props["pos"]

            if not hasattr(sim, "cosmo_model"):
                sim.init_cosmo()

            for idir in range(ndir):

                outf = os.path.join(outp, f"mock_spectrum_{idir}.h5")

                if not overwrite and os.path.exists(outf):
                    continue

                spec = mocks_spe[f"direction_{idir:d}"]
                spec_rf = mocks_spe_rf[f"direction_{idir:d}"]
                cube = mocks_cube[f"direction_{idir:d}"]
                cube_rf = mocks_cube_rf[f"direction_{idir:d}"]

                # load spectrum
                # aper_cm = spec["aperture"] * l
                aper_sr = np.pi * (aper_as / 3600 / 180 * np.pi) ** 2

                print("aper_sr:", aper_sr)

                cur_dir = directions[idir]

                basis = basis_from_vect(cur_dir)
                # print(cur_dir, basis)

                # rgal_ckpc = 2 * rgal * l * 1e3

                tgt_res = 0.5 * (l * 1e3 / 2 ** sim.namelist["amr_params"]["levelmax"])
                tgt_nbins = int(rbin_ckpc / tgt_res)

                real_res = rbin_ckpc / tgt_nbins
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

                # fig, ax = plt.subplots(1, 1)

                # vmin = 3 * np.min(stmasses) / pixel_as**2
                # vmax = 1e3 * np.max(stmasses) / pixel_as**2

                # basis_order = [basis[2], basis[1], basis[0]]
                basis_order = [basis[0], basis[1], basis[2]]
                # print(basis, basis_order)

                # spec_rf = spec_to_erg_s_A_cm2_rf(spec, pfs_path,z=z)
                lmin = spec["lambda_min"]
                lmax = spec["lambda_max"]
                nlamb = int(spec["lambda_npix"])
                lambda_bins_rf = np.linspace(lmin, lmax, nlamb)
                lambda_bins = lambda_bins_rf * (1.0 + z)

                # print(lmin,lmax)
                # print(lmin*(1+z),lmax*(1+z))

                spec_obs = spec_to_erg_s_A_cm2(spec, pfs_path, z)
                spec_rf = spec_to_erg_s_A_cm2(spec, pfs_path, 0)

                # spec_rf = spec_to_erg_s_A_cm2(spec, pfs_path, 0.0)
                # spec_rf = spec_rf / (
                #     4 * np.pi * dist_10pc**2
                # )  # at 10 pc from the galaxy for absolute magnitudes

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

                filt_names, mag_spec, mags_no_err, mags, mag_errs = gen_spec(
                    aexp, spec_obs_igm, lambda_bins, [names, wavs, trans], rf=False
                )
                filt_names, mag_spec_rf, mags_no_err_rf, mags_rf, mag_errs_rf = gen_spec(
                    1.0, spec_rf, lambda_bins_rf, [names, wavs, trans], rf=True
                )

                dx_box = cube["cube_side"] / cube["cube"].shape[0]
                dx_cm = dx_box * sim.unit_l(sim.aexp_stt) / sim.aexp_stt

                aper_px = aper_as / pixel_as

                #place rf sources at 10 pc
                r_pc = rbin_ckpc*1e3*aexp #physical parsecs
                aper_10pc_rad = np.arctan(10./r_pc)
                aper_10pc_as = 180/np.pi * aper_10pc_rad * 3600
                aper_10pc_px = aper_10pc_as/pixel_as                

                # print(cube["cube"].max())
                cube_to_erg_s_A_cm2_as2(cube, pfs_path, z, aper_as) 
                cube_to_erg_s_A_cm2_as2(cube_rf, pfs_path, 0.0, aper_10pc_as) 

                _, band_ctrs, imgs = gen_imgs(z, lambda_bins, cube, [names, wavs, trans], rpix = aper_px)           
                _, band_ctrs_rf, imgs_rf = gen_imgs(0.0, lambda_bins_rf, cube_rf, [names, wavs, trans], rpix = aper_px)           


                Ts = np.asarray([T_IGM_Inoue2014(band_ctr, zs=z) for band_ctr in band_ctrs])
                
                fnu_imgs = (
                    flamb_fnu(np.nansum(imgs, axis=(1, 2)), band_ctrs)
                    * Ts
                    * (aper_as**2 / np.prod(imgs[0].shape))
                    )
                
                fnu_imgs_rf = (
                    flamb_fnu(np.nansum(imgs_rf, axis=(1, 2)), band_ctrs)
                    # * Ts
                    * (aper_10pc_as**2 / np.prod(imgs[0].shape))
                )

                mags_imgs_fnu = -2.5 * np.log10(fnu_imgs) - 48.60
                mags_imgs_fnu_rf = -2.5 * np.log10(fnu_imgs_rf) - 48.60

                print("spec mags:", list(zip(names, mags_no_err)))
                print("img mags:",list(zip(names,mags_imgs_fnu)))

                print("rf spec mags:", list(zip(names, mags_no_err_rf)))
                print("rf img mags:",list(zip(names,mags_imgs_fnu_rf)))

                #convert to absolute

                # Mags_imgs = m_to_M(mags_imgs_fnu, 1.0 / aexp - 1)

                # Mag_spec = m_to_M(mag_spec, 1./aexp-1)
                # Mags_no_err = m_to_M(mag_spec, 1./aexp-1)
                # Mags = m_to_M(mags, 1./aexp-1)
                # Mag_errs = m_to_M(mag_errs, 1./aexp-1)

                nuv_arg = np.where(np.asarray(filt_names) == "NUV")[0][0]
                j_arg = np.where(np.asarray(filt_names) == "J")[0][0]
                r_arg = np.where(np.asarray(filt_names) == "rHSC")[0][0]
                f150_arg = np.where(np.asarray(filt_names) == "F150")[0][0]
                f277_arg = np.where(np.asarray(filt_names) == "F277")[0][0]
                f444_arg = np.where(np.asarray(filt_names) == "F444")[0][0]

                zoom_NUV_r[itgt, idir] = mags_imgs_fnu_rf[nuv_arg] - mags_imgs_fnu_rf[r_arg]
                zoom_r_j[itgt, idir] = mags_imgs_fnu_rf[r_arg] - mags_imgs_fnu_rf[j_arg]
                zoom_F150W_F277W[itgt, idir] = mags_imgs_fnu[f150_arg] - mags_imgs_fnu[f277_arg]
                zoom_F277W_F444W[itgt, idir] = mags_imgs_fnu[f277_arg] - mags_imgs_fnu[f444_arg]

                # zoom_NUV_r[i, idir] = 


                # print(list(zip(spec_obs_igm, spec_rf)))
                # print(list(zip(names, mags, mags_rf, lambda_bins)))

                # lambda_ctr_args = np.asarray(
                #     [np.argmin(np.abs(wav_ctr - lambda_bins)) for wav_ctr in wav_ctrs]
                # )

                # print(list(zip(mag_spec[lambda_ctr_args], mag_spec_rf[lambda_ctr_args])))

                print("done integrating on spectrum")

            itgt+=1

            fig,ax = plt.subplots(2,2,figsize=(10,10),sharex=False,sharey=False)

            non_nul = zoom_Mstars>0


            print(zoom_Mstars)
            print(non_nul)


            time_colors = plt.cm.viridis((np.log10(sim_times)-np.log10(t0))/(np.log10(t1)-np.log10(t0)))

            ax[0,0].scatter(zoom_Mstars[non_nul],zoom_SFR[non_nul],c=time_colors[non_nul],s=25)
            ax[0,0].set_xlabel('Mstars')
            ax[0,0].set_ylabel('SFR')
            ax[0,0].set_yscale('log')
            ax[0,0].set_xscale('log')
            ax[0,0].plot([1e7,1e12],[1e-4,1e1],color='k',ls='--',label="sSFR=1e10 yr^-1")
            ax[0,0].plot([1e7,1e12],[1e-3,1e2],color='k',ls=':',label="sSFR=1e11 yr^-1")

            med_zoom_r_j = np.nanmedian(zoom_r_j[non_nul],axis=1)
            med_zoom_NUV_r = np.nanmedian(zoom_NUV_r[non_nul],axis=1)
            yerr = np.nanpercentile(zoom_NUV_r[non_nul],[10,90],axis=1)
            # dw = med_zoom_NUV_r-yerr[0]
            # up = yerr[1]-med_zoom_NUV_r
            # up=np.nanmax(zoom_NUV_r[non_nul],axis=1)
            # dw=np.nanmin(zoom_NUV_r[non_nul],axis=1)
            # yerr = np.nanstd(zoom_NUV_r[non_nul],axis=1)
            dw = yerr[0]
            up = yerr[1]

            # for i_dot in range(len(med_zoom_r_j)):
            #     ax[1,0].errorbar(med_zoom_r_j[i_dot],med_zoom_NUV_r[i_dot],yerr=yerr[i_dot],fmt='o',markersize=5,color=time_colors[non_nul][i_dot])

            ax[1,0].scatter(med_zoom_r_j,med_zoom_NUV_r,c=time_colors[non_nul],s=25)
            # ax

            for i_dot in range(len(med_zoom_r_j)):

                ax[1,0].plot([med_zoom_r_j[i_dot]]*2,[up[i_dot],dw[i_dot]],c=time_colors[non_nul][i_dot])

                # ax[1,0].errorbar(med_zoom_r_j[i_dot],med_zoom_NUV_r[i_dot],c=time_colors[non_nul][i_dot],yerr=(dw[i_dot],up[i_dot]),markersize=25)

            ax[1,0].set_xlabel('r-J')
            ax[1,0].set_ylabel('NUV-r')

            ax[1,0].plot([-0.2,0.8],[3.05,3.05],c='k',ls='--')
            ax[1,0].plot([0.8,1.6],[3.05,6.0],c='k',ls='--')

            ax[0,1].scatter(np.nanmedian(zoom_F277W_F444W[non_nul],axis=1),np.nanmedian(zoom_F150W_F277W[non_nul],axis=1),c=time_colors[non_nul],s=25)
            ax[0,1].set_xlabel('F277W-F444W')
            ax[0,1].set_ylabel('F150W-F277W')

            ax[1,1].axis('off')

            xrange = zoom_F277W_F444W[non_nul].min(),zoom_F277W_F444W[non_nul].max()
            yrange = zoom_F150W_F277W[non_nul].min(),zoom_F150W_F277W[non_nul].max()
            xplot = np.linspace(*xrange)
            wedge_plot1 = 1.5 + 6.25 * xplot
            wedge_plot2 = 1.5 - 0.5 * xplot
            wedge_plot3 = 2.8 * xplot

            meet_12 = 0
            meet_23 = -1.5/(-0.5-2.8)

            arg_12 = np.argmin(np.abs(xplot-meet_12))
            arg_23 = np.argmin(np.abs(xplot-meet_23))
            

            ax[0,1].plot(xplot[arg_12:arg_23], 
                        wedge_plot1[arg_12:arg_23], color='k', ls='--')
            ax[0,1].plot(xplot[arg_12:arg_23],
                            wedge_plot2[arg_12:arg_23], color='k', ls='--')
            ax[0,1].plot(xplot[arg_23:],
                            wedge_plot3[arg_23:], color='k', ls='--')

            cb = plt.colorbar(ax[0,0].collections[0], ax=ax[1,1], orientation='vertical',label='time [Gyr]')
            cb_ticks = cb.get_ticks()
            cb_tick_labels = cb.ax.get_xticklabels()
            cb_tick_labels_Gyr = [sim_times[np.argmin(np.abs(l-time_colors))]for l in cb_tick_labels]
            cb.set_ticklabels(cb_tick_labels_Gyr)

            # handles = ax[0,1].
            # ax[1,1].legend(handles,a
            # loc='center',fontsize=10)
            

            rbin_str = ("%.1f"%rbin_ckpc).replace('.','p')
            fig_name=f'rascas_halo_tree_NUVJr_rbin_{rbin_str:s}_ckpc.png'
            print(f"writing {fig_name}")
            fig.savefig(fig_name)