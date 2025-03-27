import numpy as np
import matplotlib.pyplot as plt

# from matplotlib.colors import LogNorm
# from scipy.stats import binned_statistic_2d
import os

from zoom_analysis.halo_maker.read_treebricks import (
    convert_star_units,
    # read_zoom_brick,
    read_gal_stars,
    # read_zoom_stars,
)
from zoom_analysis.stars import sfhs
from zoom_analysis.stars.star_reader import read_part_ball

from gremlin.read_sim_params import ramses_sim

from zoom_analysis.halo_maker.assoc_fcts import (
    # get_gal_assoc_file,
    # get_halo_props_snap,
    get_gal_props_snap,
    # find_zoom_tgt_halo,
    # get_halo_assoc_file,
)

# from zoom_analysis.trees.tree_reader import read_tree_file_rev


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

    hm = "HaloMaker_stars2_dp_rec_dust/"
    # dsims = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal"
    # sim_id = "id74099"
    dsims = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/"
    sim_id = "id242756_novrel"
    # sim_id = "id242704_novrel"
    # sim_id = "id21892_novrel"
    sim_path = os.path.join(dsims, sim_id)

    ndir = 12
    # ndir = 12
    rfact = 1.1

    mlim = 1e11
    fpure = 0.9999

    tree_type = f"gal_tree_massive_{mlim:.2e}_purity_{fpure:.4f}"

    overwrite = False

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
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_noDust"
    # model_name = "zoom_tgt_bc03_chabrier100"
    # model_name = "zoom_tgt_bpass_kroupa100_wlya_noDust"
    # model_name = "zoom_tgt_bpass_chabrier100"
    # model_name = "zoom_tgt_bpass_chabrier100_ignoreDust"

    sim = ramses_sim(sim_path, nml="cosmo.nml")
    intID = int(sim.name.split("id")[1].split("_")[0])

    if not hasattr(sim, "cosmo_model"):
        sim.init_cosmo()

    sim_aexps = sim.get_snap_exps()  # [::-1]
    sim_times = sim.get_snap_times()  # [::-1]
    sim_snaps = sim.snap_numbers  # [::-1]

    rascas_path = os.path.join(dsims, sim_id, "rascas", model_name)
    mock_path = os.path.join("/data101/jlewis", "mock_spe", model_name, sim_id)

    tree_snaps_gids_fname = f"{tree_type:s}{intID:d}.txt"
    rascas_entries = np.loadtxt(
        os.path.join(rascas_path, tree_snaps_gids_fname), dtype=np.int32
    )

    # print(rascas_entries)

    rascas_snaps = rascas_entries[:, 0]
    rascas_gids = rascas_entries[:, 1:]

    # print(rascas_snaps, rascas_gids)
    # print(sim_snaps)

    final_gid = None

    for i, (snap, aexp, time) in enumerate(
        zip(sim_snaps[::-1], sim_aexps[::-1], sim_times[::-1])
    ):
        # print(snap, rascas_snaps)

        wh_snap = snap == rascas_snaps
        if not np.any(wh_snap):
            continue

        snap_gids = rascas_gids[wh_snap][0]

        # print(snap_gids)

        print(f"snap {snap:d} has {len(snap_gids):d} galaxies with rascas outputs")

        for gid in snap_gids:

            # print(gid)

            print(f"making mocks for snap {snap:d} and gal {gid:d}")

            if gid == -1:
                continue

            _, cur_gal_props = get_gal_props_snap(sim_path, snap, gid)

            rascas_dir = os.path.join(
                rascas_path, f"output_{snap:05d}", f"gal_{gid:07d}"
            )

            try:
                rascas_params = read_params(
                    os.path.join(rascas_dir, "params_rascas.cfg")
                )
            except FileNotFoundError:
                print(f"couldn't find rascas dir or params_rascas.cfg in: {rascas_dir}")
                continue

            if not os.path.exists(os.path.join(rascas_dir, "mock_test.spectrum")):
                continue

            last_time = time

            # print(f"ok snap {snap:d}")

            # continue
            rgal = [cur_gal_props["rmax"] * rfact]
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
                mock_path,
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

            aper_cm = rgal[0] * 2 * l
            distl = sim.cosmo_model.luminosity_distance(z).value
            aper_as = np.tan(aper_cm / distl) * 180.0 / np.pi * 3600.0

            fstar = os.path.join(
                sim.path, hm, f"GAL_{snap:05d}", f"gal_stars_{gid:07d}"
            )
            # print(fstar)

            stars = read_gal_stars(fstar)
            convert_star_units(stars, snap, sim)

            sfh, sfr, ssfr, t_sfr, _ = sfhs.get_sf_stuff(stars, z, sim, deltaT=10.0)

            # now get in ball of size aperture like rascas does
            stars_ball = read_part_ball(
                sim,
                snap,
                cur_gal_props["pos"],
                cur_gal_props["rmax"] * rfact,
                ["mass", "birth_time", "metallicity"],
                fam=2,
            )

            sfh_ball, sfr_ball, ssfr_ball, t_sfr_ball, _ = sfhs.get_sf_stuff(
                stars_ball, z, sim, deltaT=10.0
            )

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

            for idir in range(ndir):

                outf = os.path.join(outp, f"mock_spectrum_{idir}.h5")

                if not overwrite and os.path.exists(outf):
                    continue

                spec = mocks_spe[f"direction_{idir:d}"]
                cube = mocks_cube[f"direction_{idir:d}"]

                # load spectrum
                aper_cm = spec["aperture"] * l * aexp * 1e6 * ramses_pc

                lmin = spec["lambda_min"]
                lmax = spec["lambda_max"]
                nlamb = int(spec["lambda_npix"])
                lambda_bins_rf = np.linspace(lmin, lmax, nlamb)
                lambda_bins = lambda_bins_rf * (1.0 + z)

                spec_obs = spec_to_erg_s_A_cm2(spec, pfs_path, z)

                spec_rf = spec_to_erg_s_A_cm2(spec, pfs_path, 0.0)
                spec_rf = spec_rf / (
                    4 * np.pi * dist_10pc**2
                )  # at 10 pc from the galaxy for absolute magnitudes

                # apply IGM absorption
                spec_obs_igm = spec_obs * T_IGM_Inoue2014(lambda_bins, z)

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

                wav_ctrs = np.asarray([wav[int(0.5 * len(wav))] for wav in wavs])

                wav_order = np.argsort(wav_ctrs)

                wavs = [wavs[order] for order in wav_order]
                names = [names[order] for order in wav_order]
                trans = [trans[order] for order in wav_order]

                # load cube
                dx_box = cube["cube_side"] / cube["cube"].shape[0]
                dx_cm = dx_box * sim.unit_l(sim.aexp_stt) / sim.aexp_stt
                cube_to_erg_s_A_cm2_as2(cube, pfs_path, z, dx_cm)

                filt_names, mag_spec, mags, mag_errs = gen_spec(
                    aexp, spec_obs_igm, lambda_bins, [names, wavs, trans], rf=False
                )
                filt_names, mag_spec_rf, mags_rf, mag_errs_rf = gen_spec(
                    1.0, spec_obs, lambda_bins, [names, wavs, trans], rf=True
                )

                print("done integrating on spectrum")

                _, band_ctrs, imgs = gen_imgs(
                    z, lambda_bins, cube, [names, wavs, trans]
                )

                print("done integrating on cube")

                with h5py.File(
                    os.path.join(
                        outp, f"mock_images_{sim_id}_{snap}_{gid}_{idir:d}.h5"
                    ),
                    "w",
                ) as f:
                    for i, (band_name, img) in enumerate(zip(names, imgs)):
                        f.create_dataset(band_name, data=img, compression="lzf")

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

                fig_spe.savefig(
                    os.path.join(
                        outp, f"mock_spectrum_{sim_id}_{snap}_{gid}_{idir:d}.png"
                    )
                )
                fig_spe.savefig(
                    os.path.join(
                        outp, f"mock_spectrum_{sim_id}_{snap}_{gid}_{idir:d}.pdf"
                    )
                )

                print("spec plot done")

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

    os.system(
        f"cp {os.path.join(rascas_path, tree_snaps_gids_fname):s} {os.path.join(mock_path, tree_snaps_gids_fname):s}"
    )
