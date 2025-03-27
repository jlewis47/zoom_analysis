import numpy as np
import matplotlib.pyplot as plt

# from matplotlib.colors import LogNorm
# from scipy.stats import binned_statistic_2d
import os

from zoom_analysis.halo_maker.read_treebricks import (
    convert_star_units,
    read_zoom_brick,
    read_gal_stars,
)
from zoom_analysis.stars import sfhs

from gremlin.read_sim_params import ramses_sim

from zoom_analysis.halo_maker.assoc_fcts import (
    get_halo_props_snap,
    get_gal_props_snap,
    find_zoom_tgt_halo,
    get_halo_assoc_file,
)

from zoom_analysis.trees.tree_reader import read_tree_file_rev


# from matplotlib.gridspec import GridSpec

# from astropy.visualization import make_lupton_rgb

# from scipy.integrate import simps

from zoom_analysis.constants import *

from zoom_analysis.rascas.errs import dumb_constant_mag, get_cl_err

from zoom_analysis.rascas.rascas_steps import read_params
from zoom_analysis.rascas.read_rascas import (
    cube_to_erg_s_A_cm2_as2,
    read_mock_dump,
    spec_to_erg_s_A_cm2,
    spec_to_erg_s_A_cm2_rf,
)

from zoom_analysis.rascas.igm import T_IGM_Inoue2014

from zoom_analysis.rascas.filts.filts import (
    read_transmissions,
)

from zoom_analysis.rascas.rascas_plots import *


if __name__ == "__main__":

    hm = "HaloMaker_stars2_dp_rec_dust/"
    dsims = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal"
    sim_id = "id74099"
    # dsims = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/"
    # sim_id = "id242756_novrel"
    sim_path = os.path.join(dsims, sim_id)
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
    hid = 203
    # zed_target = 4.1
    zed_target = 2
    # zmax = 6.0
    zmax = 4.0
    deltaT = 50  # Myr
    ndir = 1
    # ndir = 12
    rfact = 1.1

    # overwrite = True

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
    # model_name = "zoom_tgt_bpass_chabrier100"
    # model_name = "zoom_tgt_bpass_chabrier100_ignoreDust"

    sim = ramses_sim(sim_path, nml="cosmo.nml")

    tree_name = os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat")
    byte_file = os.path.join(sim.path, "TreeMakerDM_dust")  # , "tree_rev_nbytes")

    # tree_name = os.path.join(sim.path, "TreeMakerTreeHalo", "tree_rev.dat")
    # byte_file = os.path.join(sim.path, "TreeMakerTreeHalo", "tree_rev_nbytes")
    # tree_name = os.path.join(sim.path, "TreeMakerTreeGal", "tree_rev.dat")
    # byte_file = os.path.join(sim.path, "TreeMakerTreeGal", "tree_rev_nbytes")

    # hid, halo_dict, galaxies = find_zoom_tgt_halo(sim, zed_target, debug=False)

    tree_hids, tree_datas, tree_aexps = read_tree_file_rev(
        # tree_gids, tree_datas, tree_aexps = read_tree_file_rev(
        tree_name,
        byte_file,
        zed_target,
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
    tree_gids = tree_hids[0][filt]
    tree_aexps = tree_aexps[filt]
    tree_times = tree_times[filt]

    tree_snaps = np.asarray([sim.get_closest_snap(aexp=a) for a in tree_aexps])

    sim_aexps = sim.get_snap_exps()  # [::-1]
    sim_times = sim.get_snap_times()  # [::-1]
    sim_snaps = sim.snap_numbers  # [::-1]

    # print(rascas_snaps)
    # print(sim_snaps)

    last_time = sim_times[-1] + 2 * deltaT

    final_gid = None

    for i, (snap, aexp, time) in enumerate(
        zip(sim_snaps[::-1], sim_aexps[::-1], sim_times[::-1])
    ):

        fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

        ax_out = axs[0]
        ax_flux = axs[1]
        ax_mag = axs[2]

        # for i, snap in enumerate(rascas_snaps):

        # aexp = sim.get_snap_exps(snap)[0]
        # time = sim.get_snap_times(snap)[0]

        # if snap != 278:
        #     continue

        if 1.0 / aexp - 1 > zmax:
            print(f"skip z {snap:d} {1./aexp-1.:.2f} {time:.2f}")
            continue

        if time >= (last_time + deltaT):
            print(f"skip time {snap:d} {1./aexp-1.:.2f} {time:.2f}")
            continue

        if not os.path.exists(get_halo_assoc_file(sim_path, snap)):
            print(f"no halo assoc file for snap {snap:d}")
            continue

        if np.min(np.abs(time - tree_times)) > deltaT:
            continue

        # detect_model_names
        not_allowed = [
            "zoom_tgt_bpass_chabrier100",
            "zoom_tgt_bpass_chabrier100_new",
            "zoom_tgt_bpass_chabrier100_ignoreDust",
            "zoom_tgt_ignore_Dust",
        ]
        model_names = [
            d
            for d in os.listdir(os.path.join(dsims, sim_id, "rascas"))
            # if d.startswith("zoom") and d not in not_allowed
            if os.path.isdir(os.path.join(dsims, sim_id, "rascas", d))
        ]

        for model_name in model_names:

            spec_obs = None

            rascas_path = os.path.join(dsims, sim_id, "rascas", model_name)
            rascas_snaps = np.asarray(
                [
                    int(f.split("_")[-1])
                    for f in os.listdir(rascas_path)
                    if f.startswith("output_")
                ]
            )

            tree_arg = np.argmin(np.abs(time - tree_times))

            print(sim_path, snap, tree_hids[0][tree_arg])

            hdict, hosted_gals = get_halo_props_snap(
                sim_path, snap, tree_hids[0][tree_arg]
            )

            if hosted_gals == {}:
                continue

            gid = int(hosted_gals["gids"][np.argmax(hosted_gals["mass"])])

            # if final_gid is None:
            final_gid = gid

            _, cur_gal_props = get_gal_props_snap(sim_path, snap, gid)

            # print(cur_gal_props["pos"], snap, gid, cur_gal_props["mass"])

            rascas_dir = os.path.join(
                rascas_path, f"output_{snap:05d}", f"gal_{final_gid:07d}"
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
                "/data101/jlewis",
                "mock_spe",
                "debug",
                sim_id,
                f"{snap:d}",
                f"gal_{gid:07d}",
            )

            if not os.path.exists(outp):
                os.makedirs(outp, exist_ok=True)

            mocks_spe = read_mock_dump(
                os.path.join(rascas_dir, "mock_test.spectrum"), ndir=ndir
            )

            aper_cm = rgal[0] * 2 * l
            distl = sim.cosmo_model.luminosity_distance(z).value
            aper_as = np.tan(aper_cm / distl) * 180.0 / np.pi * 3600.0
            #
            # fstar = os.path.join(sim.path, hm, f"GAL_{snap:05d}", f"gal_stars_{gid:07d}")
            # print(fstar)
            # stars = read_gal_stars(fstar)
            # convert_star_units(stars, snap, sim)

            # sfh, sfr, ssfr, t_sfr, _ = sfhs.get_sf_stuff(stars, z, sim, deltaT=10.0)

            idir = 0

            outf = os.path.join(outp, f"mock_spectrum_{idir}.h5")

            spec = mocks_spe[f"direction_{idir:d}"]
            # spec2 = mocks_spe[f"direction_{idir:d}"]
            # cube = mocks_cube[f"direction_{idir:d}"]

            # load spectrum
            aper_cm = spec["aperture"] * l * aexp * 1e6 * ramses_pc

            # spec_rf = spec_to_erg_s_A_cm2_rf(spec, pfs_path,z=z)
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

            # dx_box = cube["cube_side"] / cube["cube"].shape[0]
            # dx_cm = dx_box * sim.unit_l(sim.aexp_stt) / sim.aexp_stt
            # cube_to_erg_s_A_cm2_as2(cube, pfs_path, z, dx_cm)

            # print(np.max(spec["spectrum"]))

            # print(list(zip(spec_obs_igm, spec_obs)))
            #
            filt_names, mag_spec, mags, mag_errs = gen_spec(
                aexp, spec_obs_igm, lambda_bins, [names, wavs, trans], rf=False
            )
            filt_names, mag_spec_rf, mags_rf, mag_errs_rf = gen_spec(
                1.0, spec_obs, lambda_bins, [names, wavs, trans], rf=True
            )

            # print(list(zip(spec_obs_igm, spec_rf)))
            # print(list(zip(names, mags, mags_rf, lambda_bins)))

            # lambda_ctr_args = np.asarray(
            #     [np.argmin(np.abs(wav_ctr - lambda_bins)) for wav_ctr in wav_ctrs]
            # )

            # print(list(zip(mag_spec[lambda_ctr_args], mag_spec_rf[lambda_ctr_args])))

            print("done integrating on spectrum")

            (l,) = ax_flux.plot(lambda_bins / 1e4, spec_obs, label=model_name)
            ax_flux.plot(lambda_bins / 1e4, spec_obs_igm, ls="--", c=l.get_color())
            # ax_spe.plot(lambda_bins,spec_rf,ls='-.',c=l.get_color())

            ax_out.plot(
                lambda_bins / 1e4, spec["spectrum"], c=l.get_color(), label=model_name
            )

            ax_mag.plot(
                lambda_bins[mag_spec > 0] / 1e4,
                mag_spec[mag_spec > 0],
                c=l.get_color(),
                label=model_name,
            )
            # ax_mag.errorbar(
            #     wav_ctrs / 1e4, mags, yerr=mag_errs, fmt="o", c=l.get_color()
            # )

        for ax in [ax_flux, ax_out, ax_mag]:

            ax.grid()
            ax.set_xlabel(r"$\lambda$, $\mu$m")
            ax.set_xscale("log")
        for ax in [ax_flux, ax_out]:
            ax.set_yscale("log")

        ax_out.legend(framealpha=0.0)
        ax_out.set_ylim(1, 1e4)
        ax_flux.set_ylim(1e-20, 5e-17)
        ax_mag.set_ylim(17.5, 32.5)

        ax_mag.invert_yaxis()

        ax_flux.set_ylabel(r"$F_{\lambda}$, erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$")
        ax_mag.set_ylabel(r"$m_{AB}$")
        ax_out.set_ylabel(r"flux, rascas units")

        # ztxt = f"z={z:.2f}"
        # agetxt = f"age U={t:.2f} Gyr"
        # agegaltxt = f"age gal={agegal:.2f} Gyr"
        # masstxt = "M$_\star$=" + f"{mgal:.2e}" + " M$_\odot$"
        # txt = "\n".join([ztxt, agetxt, agegaltxt, masstxt])
        # ax_spe.text(
        #     0.85,
        #     0.05,
        #     txt,
        #     transform=ax_spe[1].transAxes,
        #     fontsize=12,
        #     va="bottom",
        #     ha="left",
        # )

        if np.all(spec_obs == None):
            continue

        fig.savefig(
            os.path.join(outp, f"debug_spectrum_{sim_id}_{snap}_{gid}_{idir:d}.png")
        )
        fig.savefig(
            os.path.join(outp, f"debug_spectrum_{sim_id}_{snap}_{gid}_{idir:d}.pdf")
        )

        print(
            "wrote: ",
            os.path.join(outp, f"debug_spectrum_{sim_id}_{snap}_{gid}_{idir:d}.png"),
        )

    # print(band_mags, len(band_mags))
    # print(names, len(names))

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
