from ast import arg
import numpy as np
import os

# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
import h5py
from scipy.spatial import cKDTree

from zoom_analysis.halo_maker.read_treebricks import (
    # convert_brick_units,
    get_tgt_partIDs,
    read_zoom_brick,
    read_zoom_stars,
)
from zoom_analysis.stars.sfhs import correct_mass
from assoc_fcts import find_star_ctr_period, get_r50, get_r90, get_reff

from gremlin.read_sim_params import ramses_sim

import argparse

# import mpi4py.MPI as MPI


def get_star_props(
    sim: ramses_sim,
    rmaxs,
    r50s,
    r90s,
    reffs,
    max_age,
    mstar_age,
    sfr10,
    sfr100,
    sfr1000,
    igal,
    stars,
):
    stars_pos, star_ctr, star_ext = find_star_ctr_period(stars["pos"])
    ages = stars["agepart"]
    mstars = stars["mpart"]

    rmaxs[igal] = np.max(
        star_ext
    )  # np.max(np.linalg.norm(stars_pos - star_ctr[None, :], axis=1))

    # print(sim.namelist)
    dx = 1.0 / 2 ** sim.namelist["amr_params"]["levelmax"]  #

    r50s[igal] = get_r50(stars_pos, mstars, star_ctr, dx)
    r90s[igal] = get_r90(stars_pos, mstars, star_ctr, dx)
    reffs[igal] = get_reff(stars_pos, mstars, star_ctr, dx)

    max_age[igal] = np.max(ages)
    mstar_age[igal] = np.average(ages, weights=mstars)

    young = ages <= 1000.0
    ages = ages[young]
    mstars = mstars[young]
    Zs = stars["Zpart"][young]

    correctd_mstars = correct_mass(sim, ages, mstars, Zs)  # this is very slow
    # print(np.log10(correctd_mstars.sum()), r50s[igal],r90s[igal], reffs[igal])

    sfr10[igal] = correctd_mstars[ages <= 10].sum() / 10.0
    sfr100[igal] = correctd_mstars[ages <= 100].sum() / 100.0
    sfr1000[igal] = correctd_mstars.sum() / 1000.0
    return ages, mstars


def write_star_dsets(
    fgal,
    dict_name,
    gal_mass,
    rmaxs,
    r50s,
    r90s,
    reffs,
    max_age,
    mstar_age,
    sfr10,
    sfr100,
    sfr1000,
):
    grp = fgal.create_group(dict_name)

    grp.create_dataset("mass", data=gal_mass, compression="lzf", dtype=np.float32)
    grp.create_dataset("rmax", data=rmaxs, compression="lzf", dtype=np.float32)
    grp.create_dataset("r50", data=r50s, compression="lzf", dtype=np.float32)
    grp.create_dataset("r90", data=r90s, compression="lzf", dtype=np.float32)
    grp.create_dataset("reff", data=reffs, compression="lzf", dtype=np.float32)
    grp.create_dataset("max age", data=max_age, compression="lzf", dtype=np.float32)
    grp.create_dataset("mstar age", data=mstar_age, compression="lzf", dtype=np.float32)
    grp.create_dataset("sfr10", data=sfr10, compression="lzf", dtype=np.float32)
    grp.create_dataset("sfr100", data=sfr100, compression="lzf", dtype=np.float32)
    grp.create_dataset("sfr1000", data=sfr1000, compression="lzf", dtype=np.float32)


"""
hmsmr has some high halo mass outliers...
these are small non central galaxies in the haloes

make a CENTRAL flag for the most massive halo galaxy THEN rerun all

"""


def read_contam_hm(fname):

    ids, npart_contam, npart = np.genfromtxt(fname, unpack=True)

    return ids, npart_contam, npart


gm_name = "stars2_dp_rec_dust"
hm_name = "DM_dust"


def associate(sim_dirs, snaps=[], verbose=False, overwrite=False):

    for sim_dir in sim_dirs:

        print(sim_dir)
        sim = ramses_sim(sim_dir, nml="cosmo.nml")

        # contam_thresh = 99.9

        output_nbs = sim.snap_numbers
        if len(output_nbs) == 0:
            continue
        nsnaps = len(output_nbs)
        aexps = sim.get_snap_exps(output_nbs)
        # aexps = sim.aexps

        outdir = os.path.join(sim_dir, "association")
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # comm = MPI.COMM_WORLD
        # rank = comm.Get_rank()
        # size = comm.Get_size()

        # size = 1
        # rank = 0

        # can split but later ones are fuller... longer to process
        # best to create rank targets using [::size] and then split
        # outputs_rank = output_nbs[rank::size]
        # aexps_rank = aexps[rank::size]

        if len(snaps) == 0:
            snaps = output_nbs

        # print(snaps)

        # for snap, aexp in zip(outputs_rank, aexps_rank):
        for snap, aexp in zip(output_nbs[::-1], aexps[::-1]):
            # for snap, aexp in zip(output_nbs, aexps):

            # if snap != 294:
            #     continue

            # print(snap)

            if snap not in snaps:
                continue

            fname_halo = os.path.join(outdir, f"assoc_{snap:03d}_halo_lookup.h5")
            fname_gal = os.path.join(outdir, f"assoc_{snap:03d}_gal_lookup.h5")

            # print(fname_halo, fname_gal)

            if not overwrite and (
                os.path.isfile(fname_halo) and os.path.isfile(fname_gal)
            ):
                continue

            # print(f"rank {rank} is handling snap {snap} at aexp {aexp}")
            print(f"Handling snap {snap} at aexp {aexp}")

            dm_brick = read_zoom_brick(
                snap, sim, "HaloMaker_" + hm_name, galaxy=False, star=False
            )

            if dm_brick == 0:
                print(f"no dm brick found for snap {snap}")
                continue

            halo_pos = np.asarray(list(dm_brick["positions"].values()))
            hids = dm_brick["hosting info"]["hid"]
            halo_lvl = dm_brick["hosting info"]["hlvl"]
            halo_rvir = dm_brick["virial properties"]["rvir"]
            halo_mvir = dm_brick["virial properties"]["mvir"]

            gm_brick = read_zoom_brick(snap, sim, "HaloMaker_" + gm_name)

            if gm_brick == 0:
                print(f"no gm brick found for snap {snap}")
                continue

            gpos = np.asarray(list(gm_brick["positions"].values()))
            gal_tree = cKDTree(gpos.T, boxsize=1 + 1e-6)

            gids = gm_brick["hosting info"]["hid"]
            g_host_gids = gm_brick["hosting info"]["hosth"]
            g_sub_gids = gm_brick["hosting info"]["hostsub"]
            g_nb_subs = gm_brick["hosting info"]["nbsub"]
            gal_mass = gm_brick["hosting info"]["hmass"]

            most_massive_hid = np.empty_like(gids)
            most_massive_hmass = np.empty_like(gal_mass, dtype=np.float32)
            most_massive_purity = np.empty_like(gal_mass, dtype=np.float32)
            most_massive_central = np.empty_like(gal_mass, dtype=bool)
            most_massive_hmass[:] = -np.inf

            # read contamination
            contam_fname = os.path.join(
                sim_dir, "HaloMaker_" + hm_name, f"contam_halos{snap:03d}"
            )
            contam_hid, contam_npart, npart = read_contam_hm(contam_fname)

            fpure = 1.0 - contam_npart / npart

            # check contam hid order and halo list order are the same
            union_hids, arg1, arg2 = np.intersect1d(
                hids, contam_hid, return_indices=True
            )

            fpure_all = np.ones(len(halo_mvir), dtype=np.float32)
            fpure_all[arg1] = fpure[arg2]

            fhalo = h5py.File(fname_halo, "w")

            for hnb, (hid, hpos, hr, hm) in enumerate(
                zip(hids, halo_pos.T, halo_rvir, halo_mvir)
            ):

                if verbose:
                    print(f"Halo {hid} at {hpos} with rvir {hr}")

                # find galaxies within rvir
                gal_inds = gal_tree.query_ball_point(hpos, r=hr)

                if len(gal_inds) > 0:
                    halo_gal_masses = gal_mass[gal_inds]
                    central_ind = gal_inds[np.argmax(halo_gal_masses)]

                for ind in gal_inds:

                    if gal_mass[ind] > most_massive_hmass[ind]:
                        most_massive_hmass[ind] = hm
                        most_massive_hid[ind] = hid
                        most_massive_purity[ind] = fpure_all[hnb]
                        if ind == central_ind:
                            most_massive_central[ind] = True
                        else:
                            most_massive_central[ind] = False

            fhalo.create_dataset(
                "fpure", data=fpure_all, compression="lzf", dtype=np.float32
            )
            fhalo.create_dataset("hid", data=hids, compression="lzf")
            fhalo.create_dataset(
                "mvir", data=halo_mvir, compression="lzf", dtype=np.float32
            )

            for hnb, (hid, hpos, hr, hm) in enumerate(
                zip(hids, halo_pos.T, halo_rvir, halo_mvir)
            ):

                h_dset = fhalo.create_group(f"halo_{hid:07d}")

                h_dset.create_dataset("pos", data=hpos)
                h_dset.create_dataset("rvir", data=hr)

                gal_inds = np.where(most_massive_hid == hid)[0]
                if verbose:
                    print(f"Found {len(gal_inds)} galaxies within rvir of halo {hid}")

                Ngals = len(gal_inds)
                h_dset.create_dataset("Ngals", data=Ngals)

                if Ngals > 0:

                    g_dset = h_dset.create_group("galaxies")

                    g_dset.create_dataset("gids", data=gids[gal_inds], dtype=np.float32)
                    g_dset.create_dataset(
                        "mass", data=gal_mass[gal_inds], dtype=np.float32
                    )
                    g_dset.create_dataset("pos", data=gpos[:, gal_inds])

            fhalo.close()

            fgal = h5py.File(fname_gal, "w")

            rmaxs = np.empty_like(gal_mass, dtype=np.float32)
            r50s = np.empty_like(gal_mass, dtype=np.float32)
            r90s = np.empty_like(gal_mass, dtype=np.float32)
            reffs = np.empty_like(gal_mass, dtype=np.float32)
            max_age = np.empty_like(gal_mass, dtype=np.float32)
            mstar_age = np.empty_like(gal_mass, dtype=np.float32)
            sfr10 = np.empty_like(gal_mass, dtype=np.float32)
            sfr100 = np.empty_like(gal_mass, dtype=np.float32)
            sfr1000 = np.empty_like(gal_mass, dtype=np.float32)

            gal_mass_main = np.empty_like(gal_mass, dtype=np.float32)
            rmaxs_main = np.empty_like(gal_mass, dtype=np.float32)
            r50s_main = np.empty_like(gal_mass, dtype=np.float32)
            r90s_main = np.empty_like(gal_mass, dtype=np.float32)
            reffs_main = np.empty_like(gal_mass, dtype=np.float32)
            max_age_main = np.empty_like(gal_mass, dtype=np.float32)
            mstar_age_main = np.empty_like(gal_mass, dtype=np.float32)
            sfr10_main = np.empty_like(gal_mass, dtype=np.float32)
            sfr100_main = np.empty_like(gal_mass, dtype=np.float32)
            sfr1000_main = np.empty_like(gal_mass, dtype=np.float32)

            for igal in range(len(gids)):

                if verbose:
                    print(f"Processing gal {igal} of {len(gids)}")

                # get stars
                stars = read_zoom_stars(
                    sim,
                    snap,
                    gids[igal],
                    hm="HaloMaker_" + gm_name,
                    tgt_fields=["pos", "mpart", "agepart", "Zpart", "IDs"],
                )

                ages, mstars = get_star_props(
                    sim,
                    rmaxs,
                    r50s,
                    r90s,
                    reffs,
                    max_age,
                    mstar_age,
                    sfr10,
                    sfr100,
                    sfr1000,
                    igal,
                    stars,
                )

                # get sub galaxies
                arg_sub_gals = np.where(g_host_gids == gids[igal])[0]

                if len(arg_sub_gals) > 0:  # has subs

                    # fname_brick = os.path.join(
                    #     sim.path, "HaloMaker_" + gm_name, f"tree_bricks{snap:03d}"
                    # )

                    # main_IDs = get_tgt_partIDs(
                    #     fname_brick, gids[igal], star=True, galaxy=True
                    # )

                    # get sub_stars
                    sub_stIDs = []
                    for sub_gid in gids[arg_sub_gals]:

                        if sub_gid == gids[igal]:
                            continue

                        # sub_stIDs.extend(
                        # get_tgt_partIDs(fname_brick, sub_gid, star=True, galaxy=True)
                        # )

                        sub_stIDs.extend(
                            read_zoom_stars(
                                sim,
                                snap,
                                sub_gid,
                                hm="HaloMaker_" + gm_name,
                                tgt_fields=["IDs"],
                            )["IDs"]
                        )

                    sub_stIDs = np.array(sub_stIDs, dtype=np.int32)
                    all_IDs = stars["IDs"]

                    if verbose:
                        print(f"Gal {igal} has {len(all_IDs)} stars")
                        print(f"Gal {igal} has {len(sub_stIDs)} substars")
                        # print(f"Gal {igal} has {main_only_bool.sum()} main stars")
                        # print(sub_stIDs)

                    main_only_bool = np.in1d(all_IDs, sub_stIDs) == False

                    if len(sub_stIDs) == 0:
                        assert main_only_bool.sum() == len(
                            all_IDs
                        ), "no substars but not all main stars"

                    if len(sub_stIDs) == 0:

                        gal_mass_main[igal] = gal_mass[igal]
                        rmaxs_main[igal] = rmaxs[igal]
                        r50s_main[igal] = r50s[igal]
                        r90s_main[igal] = r90s[igal]
                        reffs_main[igal] = reffs[igal]
                        max_age_main[igal] = max_age[igal]
                        mstar_age_main[igal] = mstar_age[igal]
                        sfr10_main[igal] = sfr10[igal]
                        sfr100_main[igal] = sfr100[igal]
                        sfr1000_main[igal] = sfr1000[igal]

                        continue

                    main_stars = {}

                    for key in stars.keys():
                        cur_d = stars[key]
                        if type(cur_d) in [np.ndarray, list]:
                            main_stars[key] = cur_d[main_only_bool]
                        else:
                            main_stars[key] = cur_d

                    main_ages, main_mstars = get_star_props(
                        sim,
                        rmaxs_main,
                        r50s_main,
                        r90s_main,
                        reffs_main,
                        max_age_main,
                        mstar_age_main,
                        sfr10_main,
                        sfr100_main,
                        sfr1000_main,
                        igal,
                        main_stars,
                    )

                    gal_mass_main[igal] = main_mstars.sum()

                if verbose:
                    print(f"Gal {igal} has {len(ages)} young stars")
                    print(f"stellar mass, sfr10, sfr100, sfr1000")
                    print(
                        f"{mstars.sum():.1e}, {sfr10[igal]:.1e}, {sfr100[igal]:.1e}, {sfr1000[igal]:.1e}"
                    )
                    print("purity is ", most_massive_purity[igal])

            fgal.create_dataset("gids", data=gids, compression="lzf")
            fgal.create_dataset("pos", data=gpos, compression="lzf")
            fgal.create_dataset("host hid", data=most_massive_hid, compression="lzf")
            fgal.create_dataset("central", data=most_massive_central, compression="lzf")
            fgal.create_dataset("host gid", data=g_host_gids, compression="lzf")
            fgal.create_dataset(
                "host mass",
                data=most_massive_hmass,
                compression="lzf",
                dtype=np.float32,
            )
            fgal.create_dataset(
                "host purity",
                data=most_massive_purity,
                compression="lzf",
                dtype=np.float32,
            )
            write_star_dsets(
                fgal,
                "all_stars",
                gal_mass,
                rmaxs,
                r50s,
                r90s,
                reffs,
                max_age,
                mstar_age,
                sfr10,
                sfr100,
                sfr1000,
            )

            write_star_dsets(
                fgal,
                "main_stars",
                gal_mass_main,
                rmaxs_main,
                r50s_main,
                r90s_main,
                reffs_main,
                max_age_main,
                mstar_age_main,
                sfr10_main,
                sfr100_main,
                sfr1000_main,
            )

            fgal.close()

        # comm.Barrier()

        print("done")


if __name__ == "__main__":

    sim_dirs = (
        [
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_evenlesscoarse",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_accBoost",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_gravSink_SE",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_DynBondiGravSinkMass_SE",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_fullresIC",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_kicks",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowMseed",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_lowerSFE",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_chabrier",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id74099",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_evenlesscoarse",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_coarse",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_leastcoarse",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_leastcoarse",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_leastcoarse",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_boostgrowth",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
            # "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242704",
            # "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242756",
            # "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242704_lower_nstar",
            # "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242756_lower_nstar",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_noAGN",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE_stgNHboost_stricterSF",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE_highAGNtherm",
            # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130_nh2",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_high_SN",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_high_thermal_eff",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_smooth_ref",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_fullresIC",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646",
            # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
            # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
            # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id180130",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondsi",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
            # "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242704_lower_nstar",
            # "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242756_lower_nstar",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id112288",
            # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288",
            # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288_novrel_lowerSFE_stgNHboost_strictSF/",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_lowerSFE_NHboost",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictBH",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_smallICs",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_stricterSF",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_stricterSF_radioHeavy",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_medSFE_stgNHboost_stricterSF",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_highSFE_stgNHboost_strictestSF",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_stricterSF_SEdd2",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictestSF_lowSNe",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_XtremeSF_lowSNe",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictestSF_lowSNe",
            # "/automnt/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_XtremeSF_lowSNe",
            # "/automnt/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_MegaSF_lowSNe",
            # "/automnt/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_SuperMegaSF_midSNe",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_XtremeLowSFE_stgNHboost_strictSF",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_SuperLowSFE_stgNHboost_strictSF",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_fullresIC",
            "/data103/jlewis/sims/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF/",
            "/data102/jlewis/sims/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF_high_sconsta\
nt",
            "/data102/jlewis/sims/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF_Vhigh_sconst\
ant",
            "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF_VVhigh_sconst\
ant",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag",
            # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh",
            # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2",
            # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_noHydroDragB4z4merge",
            # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
            # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE",
            # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
            # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id180130_model5",
            # "/data103/jlewis/sims/lvlmax_22/mh1e12/id180130",
            # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e11/id292074",
            # "/data103/jlewis/sims/lvlmax_22/mh1e12/id147479",
            # "/data103/jlewis/sims/lvlmax_22/mh1e12/id138140",
            # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
            # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
            # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id147479",
            # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id138140",
            # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_noAGN",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_boostgrowth",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi/id180130_meanBondi_higher_nmax",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi/id180130_meanBondi_Sconstant",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nrg_SN",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_pdrag",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highSFE_lowSN",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE_lowNsink",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE_lowSN",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost_lowerSFE",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_minMassDynGrav",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_low_dboost",
        ]
        + [
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_low_dboost",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_lowerSFE",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_lowerSFE_lowNsink",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_minMassDynGrav",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_minMassDynHydroGravBHL",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_fullresIC",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_high_nsink",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highSFE_lowSN",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE_lowSN",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_lowerSFE",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_noAGN",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE_highAGNtherm",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowSFE",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_boostgrowth",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_high_SN",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_high_thermal_eff",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowMseed",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_pdrag",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_vlowSFE",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nrg_SN",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_smooth_ref",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd",
            # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd_high_fstar",
            # "/data103/jlewis/sims/lvlmax_21/mh1e12/id180130_256",
            # "/data102/jlewis/sims/lvlmax_21/mh1e12/id180130_model6_eps0p05",
            # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_drag",
            # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_DynBondiGravSinkMass",
            # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_lowerSFE",
            # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
            # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_lowerSFE_drag",
            # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth",
            # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_early_refine",
            # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_boostNH",
            # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_boostNH_resimBoostFriction",
            # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_frcAccrt",
            # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_lowerSFE_drag",
        ]
    )

    Argparse = argparse.ArgumentParser()

    Argparse.add_argument("--sim_dirs", nargs="+", default=sim_dirs)
    Argparse.add_argument("--snaps", nargs="+", default=[], type=int)
    Argparse.add_argument("--verbose", action="store_true", default=False)
    Argparse.add_argument("--overwrite", action="store_true", default=False)

    associate(**vars(Argparse.parse_args()))
