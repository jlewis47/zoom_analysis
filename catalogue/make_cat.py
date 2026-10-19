###
# TODO: rewrite in simpler manner: arrays for all stats? then shared via mpi4py
# TODO: reformat writing format to be faster to parse
###


from f90_tools.star_reader import (
    read_part_ball_NCdust,
    read_part_ball_NH,
    read_part_ball_YDBondi,
    read_part_ball_hagn,
)
from h5tools.h5py_tools import update_h5
from hagn.catalogues import make_super_cat
from hagn.utils import get_nh_sim
from zoom_analysis.catalogue.cat_tools import args2catname, radius2str
from zoom_analysis.halo_maker.assoc_fcts import (
    get_gal_props_snap,
    get_halo_props_snap,
    get_assoc_pties_in_tree,
    smooth_props,
    has_assoc,
)
from zoom_analysis.stars.dynamics import extract_nh_kinematics
from zoom_analysis.read.read_data import read_data_ball
from zoom_analysis.sinks.sink_reader import (
    read_sink_bin,
    read_sink_bin_NH,
    convert_sink_units,
    snap_to_coarse_step,
)
from zoom_analysis.trees.tree_reader import read_tree_file_rev_correct_pos
from zoom_analysis.constants import mass_H_cgs, ramses_pc, Msun_cgs

from gremlin.read_sim_params import ramses_sim

import numpy as np
import h5py
import argparse
import os
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from matplotlib.patches import Ellipse


def get_fdepl(
    density, tgt_volumes, gas_data, MgoverSil, FeoverSil, SioverSil, OoverSil, arg=None
):

    if np.all(arg == None):
        arg = np.full(len(density), True)

    mO = np.sum(density[arg] * tgt_volumes[arg] * gas_data["chem_O"][arg])
    mFe = np.sum(density[arg] * tgt_volumes[arg] * gas_data["chem_Fe"][arg])
    mMg = np.sum(density[arg] * tgt_volumes[arg] * gas_data["chem_Mg"][arg])
    mC = np.sum(density[arg] * tgt_volumes[arg] * gas_data["chem_C"][arg])
    mSi = np.sum(density[arg] * tgt_volumes[arg] * gas_data["chem_Si"][arg])
    # mN = np.sum(density * tgt_volumes * gas_data["chem_N"])
    # mS = np.sum(density * tgt_volumes * gas_data["chem_S"])

    mdC = np.sum(
        (gas_data["dust_bin01"][arg] + gas_data["dust_bin02"][arg])
        * tgt_volumes[arg]
        * density[arg]
    )
    mdS = np.sum(
        (gas_data["dust_bin03"][arg] + gas_data["dust_bin04"][arg])
        * tgt_volumes[arg]
        * density[arg]
    )
    # md = mdC + mdS

    mdMg = mdS * MgoverSil
    mdFe = mdS * FeoverSil
    mdSi = mdS * SioverSil
    mdO = mdS * OoverSil

    fdMg = 1 - mdMg / mMg
    fdFe = 1 - mdFe / mFe
    fdO = 1 - mdO / mO
    fdSi = 1 - mdSi / mSi
    fdC = 1 - mdC / mC

    return fdMg, fdFe, fdO, fdSi, fdC


def gas_stats(
    fname, tgt_f, tgt_pos, aexp, gas_data, sim, thresh_dense=0.1, overwrite=False
):

    MgoverSil = 0.141
    FeoverSil = 0.324
    SioverSil = 0.163
    OoverSil = 0.372

    density = gas_data["density"]  # g/cc

    if len(density) == 0:
        return 0

    temp = gas_data["temperature"]  # T[K]/mu
    metals = gas_data["metallicity"]

    has_dust = "dust_bin01" in gas_data.keys()

    if has_dust:
        dust_bin1 = gas_data["dust_bin01"]  # g/cc
        dust_bin2 = gas_data["dust_bin02"]  # g/cc
        dust_bin3 = gas_data["dust_bin03"] / SioverSil  # g/cc
        dust_bin4 = gas_data["dust_bin04"] / SioverSil  # g/cc
    vels = (
        np.transpose(
            [gas_data["velocity_x"], gas_data["velocity_y"], gas_data["velocity_z"]]
        )
        / 1e5
    )  # km/s
    pos = np.transpose([gas_data["x"], gas_data["y"], gas_data["z"]])

    no_bulk_vels = vels - np.average(vels, axis=0, weights=density)
    rad_vels = np.sum(no_bulk_vels * (pos - tgt_pos), axis=1)
    out_vels = np.sign(rad_vels) > 0

    rads = np.linalg.norm((pos-tgt_pos),axis=1)
    max_rad = rads.max()
    #I only have leaf cells so take the closest every time
    ok_cells = np.abs(rads-max_rad)<(1.0/ 2 ** gas_data["ilevel"])

    surfaces = (
        (sim.cosmo.lcMpc * ramses_pc * 1e6 * aexp) / 2 ** gas_data["ilevel"]
    ) ** 2  # cc
    vols = surfaces ** (3/2)

    print(rad_vels,rad_vels.shape)

    if np.any((out_vels == False)*ok_cells):
        vels_in = np.sum(
            rad_vels[(out_vels == False)*ok_cells], axis=1
        )
        tot_inflow = np.sum(vels_in)
        tot_inflow_mass = np.sum(vels_in*density[(out_vels == False)*ok_cells]*surfaces[(out_vels == False)*ok_cells])
    if np.any(out_vels*ok_cells):
        vels_out = np.sum(rad_vels[out_vels*ok_cells], axis=1)
        # tot_outflow = np.sum(vels_out)
        tot_outflow_mass = np.sum(vels_out*density[out_vels*ok_cells]*surfaces[out_vels*ok_cells])



    # if out_vels.sum() > 0:
    #     mean_outflow_wMass = np.average(vels_out, weights=density[out_vels])
    # else:
    #     mean_outflow_wMass = 0
    # if (out_vels == False).sum() > 0:
    #     mean_inflow_wMass = np.average(vels_in, weights=density[out_vels == False])
    # else:
    #     mean_inflow_wMass = 0



    dense_gas = (density / (mass_H_cgs)) > thresh_dense  # H/cc

    if np.any(dense_gas):

        mass_dense = np.sum(density[dense_gas] * vols[dense_gas]) / Msun_cgs  # Msun
        temp_mean_dense = np.average(temp[dense_gas], weights=density[dense_gas])  # T
        metals_mean_dense = np.average(
            metals[dense_gas], weights=density[dense_gas]
        )  # Z
        metals_mass_dense = (
            np.sum(metals[dense_gas] * density[dense_gas] * vols[dense_gas]) / Msun_cgs
        )  # Msun
        if has_dust:
            dust_bin1_mean_dense = np.average(
                dust_bin1[dense_gas], weights=density[dense_gas]
            )  # g/cc
            dust_bin1_mass_dense = (
                np.sum(dust_bin1[dense_gas] * density[dense_gas] * vols[dense_gas])
                / Msun_cgs
            )  # Msun

            dust_bin4_mean_dense = np.average(
                dust_bin4[dense_gas], weights=density[dense_gas]
            )  # g/cc
            dust_bin4_mass_dense = (
                np.sum(dust_bin4[dense_gas] * density[dense_gas] * vols[dense_gas])
                / Msun_cgs
            )  # Msun

            dust_bin2_mean_dense = np.average(
                dust_bin2[dense_gas], weights=density[dense_gas]
            )  # g/cc
            dust_bin2_mass_dense = (
                np.sum(dust_bin2[dense_gas] * density[dense_gas] * vols[dense_gas])
                / Msun_cgs
            )  # Msun

            dust_bin3_mean_dense = np.average(
                dust_bin3[dense_gas], weights=density[dense_gas]
            )  # g/cc
            dust_bin3_mass_dense = (
                np.sum(dust_bin3[dense_gas] * density[dense_gas] * vols[dense_gas])
                / Msun_cgs
            )  # Msun

            dtm_dens = (
                dust_bin1_mass_dense
                + dust_bin2_mass_dense
                + dust_bin3_mass_dense
                + dust_bin4_mass_dense
            ) / (metals_mass_dense 
                + dust_bin1_mass_dense
                + dust_bin2_mass_dense
                + dust_bin3_mass_dense
                + dust_bin4_mass_dense)

            (
                fdMg_dense,
                fdFe_dense,
                fdO_dense,
                fdSi_dense,
                fdC_dense,
            ) = get_fdepl(
                density,
                vols,
                gas_data,
                MgoverSil,
                FeoverSil,
                SioverSil,
                OoverSil,
                dense_gas,
            )

        # dens_no_bulk_vels = vels[dense_gas]  - np.average(vels[dense_gas], axis=0,weights=density)
        # dens_vel_out = np.sum(dens_no_bulk_vels * (pos[dense_gas]-tgt_pos), axis=1)
        # dens_tot_outflow = np.sum(dens_vel_out)
        
        dense_outflows = dense_gas*out_vels*ok_cells
        dense_inflows = dense_gas*(out_vels == False)*ok_cells
        dens_tot_outflow = np.sum(vels_out[dense_outflows])
        if (dense_gas * out_vels).sum() > 0:
            dens_mean_outflow_wMass = np.average(
                vels_out[dense_gas[out_vels*ok_cells]], weights=density[dense_gas * out_vels]
            )
        else:
            dens_mean_outflow_wMass = 0
        dens_tot_inflow = np.sum(vels_in[dense_inflows])
        if (dense_gas * (out_vels == False)).sum() > 0:
            dens_mean_inflow_wMass = np.average(
                vels_in[dense_gas[out_vels == False]],
                weights=density[dense_gas * (out_vels == False)],
            )
        else:
            dens_mean_inflow_wMass = 0

        vol_dense = np.sum(vols[dense_gas])  # cc
        # dense_gas_f = tgt_f.create_group(f"Hpcc>{thresh_dense:0.1f}_gas")

        print(mass_dense)

        dens_gas_f = "/".join([tgt_f, f"Hpcc>{thresh_dense:0.1f}_gas"])
        # update_h5(fname, "/".join(dens_gas_f, "Mgas_msun", data=mass_dense, dtype=np.float32)
        update_h5(
            fname,
            "/".join([dens_gas_f, "Mgas_msun"]),
            data=mass_dense,
            dtype=np.float32,
            overwrite=overwrite,
        )
        # update_h5(fname, "/".join(dens_gas_f, "T_mean_K", data=temp_mean_dense, dtype=np.float32)
        update_h5(
            fname,
            "/".join([dens_gas_f, "T_mean_K"]),
            data=temp_mean_dense,
            dtype=np.float32,
            overwrite=overwrite,
        )
        # update_h5(fname, "/".join(dens_gas_f, "Z_mean", data=metals_mean_dense, dtype=np.float32)
        update_h5(
            fname,
            "/".join([dens_gas_f, "Z_mean"]),
            data=metals_mean_dense,
            dtype=np.float32,
            overwrite=overwrite,
        )
        update_h5(
            fname,
            "/".join([dens_gas_f, "Mmetals_msun"]),
            data=metals_mass_dense,
            dtype=np.float32,
            overwrite=overwrite,
        )

        if has_dust:
            update_h5(
                fname,
                "/".join([dens_gas_f, "dust_bin1_mean"]),
                data=dust_bin1_mean_dense,
                dtype=np.float32,
                overwrite=overwrite,
            )
            update_h5(
                fname,
                "/".join([dens_gas_f, "dust_bin1_mass_msun"]),
                data=dust_bin1_mass_dense,
                dtype=np.float32,
                overwrite=overwrite,
            )
            update_h5(
                fname,
                "/".join([dens_gas_f, "dust_bin2_mean"]),
                data=dust_bin2_mean_dense,
                dtype=np.float32,
                overwrite=overwrite,
            )
            update_h5(
                fname,
                "/".join([dens_gas_f, "dust_bin2_mass_msun"]),
                data=dust_bin2_mass_dense,
                dtype=np.float32,
                overwrite=overwrite,
            )
            update_h5(
                fname,
                "/".join([dens_gas_f, "dust_bin3_mean"]),
                data=dust_bin3_mean_dense,
                dtype=np.float32,
                overwrite=overwrite,
            )
            update_h5(
                fname,
                "/".join([dens_gas_f, "dust_bin3_mass_msun"]),
                data=dust_bin3_mass_dense,
                dtype=np.float32,
                overwrite=overwrite,
            )
            update_h5(
                fname,
                "/".join([dens_gas_f, "dust_bin4_mean"]),
                data=dust_bin4_mean_dense,
                dtype=np.float32,
                overwrite=overwrite,
            )
            update_h5(
                fname,
                "/".join([dens_gas_f, "dust_bin4_mass_msun"]),
                data=dust_bin4_mass_dense,
                dtype=np.float32,
                overwrite=overwrite,
            )
            update_h5(
                fname,
                "/".join([dens_gas_f, "dtm"]),
                data=dtm_dens,
                overwrite=overwrite,
                dtype=np.float32,
            )
            update_h5(
                fname,
                "/".join([dens_gas_f, "fdepl_Mg"]),
                data=fdMg_dense,
                overwrite=overwrite,
                dtype=np.float32,
            )
            update_h5(
                fname,
                "/".join([dens_gas_f, "fdepl_Fe"]),
                data=fdFe_dense,
                overwrite=overwrite,
                dtype=np.float32,
            )
            update_h5(
                fname,
                "/".join([dens_gas_f, "fdepl_O"]),
                data=fdO_dense,
                overwrite=overwrite,
                dtype=np.float32,
            )
            update_h5(
                fname,
                "/".join([dens_gas_f, "fdepl_Si"]),
                data=fdSi_dense,
                overwrite=overwrite,
                dtype=np.float32,
            )
            update_h5(
                fname,
                "/".join([dens_gas_f, "fdepl_C"]),
                data=fdC_dense,
                overwrite=overwrite,
                dtype=np.float32,
            )
        update_h5(
            fname,
            "/".join([dens_gas_f, "vol_cc"]),
            data=vol_dense,
            overwrite=overwrite,
            dtype=np.float64,
        )

        update_h5(
            fname,
            "/".join([dens_gas_f, "tot_outflow_kms"]),
            data=dens_tot_outflow,
            dtype=np.float32,
            overwrite=overwrite,
        )
        update_h5(
            fname,
            "/".join([dens_gas_f, "tot_inflow_kms"]),
            data=dens_tot_inflow,
            dtype=np.float32,
            overwrite=overwrite,
        )
        update_h5(
            fname,
            "/".join([dens_gas_f, "mean_outflow_wMass_kms"]),
            data=dens_mean_outflow_wMass,
            dtype=np.float32,
            overwrite=overwrite,
        )
        update_h5(
            fname,
            "/".join([dens_gas_f, "mean_inflow_wMass_kms"]),
            data=dens_mean_inflow_wMass,
            dtype=np.float32,
            overwrite=overwrite,
        )

    mass = np.sum(density * vols) / Msun_cgs  # Msun

    temp_mean = np.average(temp, weights=density)  # T

    metals_mean = np.average(metals, weights=density)  # Z
    metals_mass = np.sum(metals * density * vols) / Msun_cgs  # Msun

    if has_dust:

        dust_bin1_mean = np.average(dust_bin1, weights=density)  # g/cc
        dust_bin1_mass = np.sum(dust_bin1 * density * vols) / Msun_cgs  # Msun

        dust_bin2_mean = np.average(dust_bin2, weights=density)  # g/cc
        dust_bin2_mass = np.sum(dust_bin2 * density * vols) / Msun_cgs  # Msun

        dust_bin3_mean = np.average(dust_bin3, weights=density)  # g/cc
        dust_bin3_mass = np.sum(dust_bin3 * density * vols) / Msun_cgs  # Msun

        dust_bin4_mean = np.average(dust_bin4, weights=density)  # g/cc
        dust_bin4_mass = np.sum(dust_bin4 * density * vols) / Msun_cgs  # Msun

        dtm = (dust_bin1_mass + dust_bin2_mass + dust_bin3_mass + dust_bin4_mass) / (
            metals_mass
        )

        fdMg, fdFe, fdO, fdSi, fdC = get_fdepl(
            density, vols, gas_data, MgoverSil, FeoverSil, SioverSil, OoverSil
        )

    vol_tot = np.sum(vols)  # cc

    # gas = tgt_f.create_group("gas")
    gas_f = "/".join([tgt_f, f"gas"])
    update_h5(
        fname,
        "/".join([gas_f, "Mgas_tot_msun"]),
        data=mass,
        overwrite=overwrite,
        dtype=np.float32,
    )

    update_h5(
        fname,
        "/".join([gas_f, "T_mean_K"]),
        data=temp_mean,
        overwrite=overwrite,
        dtype=np.float32,
    )

    update_h5(
        fname,
        "/".join([gas_f, "Z_mean"]),
        data=metals_mean,
        overwrite=overwrite,
        dtype=np.float32,
    )
    update_h5(
        fname,
        "/".join([gas_f, "Mmetals_msun"]),
        data=metals_mass,
        overwrite=overwrite,
        dtype=np.float32,
    )

    if has_dust:

        update_h5(
            fname,
            "/".join([gas_f, "dust_bin1_mean"]),
            data=dust_bin1_mean,
            dtype=np.float32,
            overwrite=overwrite,
        )
        update_h5(
            fname,
            "/".join([gas_f, "dust_bin1_mass_msun"]),
            data=dust_bin1_mass,
            dtype=np.float32,
            overwrite=overwrite,
        )

        update_h5(
            fname,
            "/".join([gas_f, "dust_bin2_mean"]),
            data=dust_bin2_mean,
            dtype=np.float32,
            overwrite=overwrite,
        )
        update_h5(
            fname,
            "/".join([gas_f, "dust_bin2_mass_msun"]),
            data=dust_bin2_mass,
            dtype=np.float32,
        )

        update_h5(
            fname,
            "/".join([gas_f, "dust_bin3_mean"]),
            data=dust_bin3_mean,
            dtype=np.float32,
            overwrite=overwrite,
        )
        update_h5(
            fname,
            "/".join([gas_f, "dust_bin3_mass_msun"]),
            data=dust_bin3_mass,
            dtype=np.float32,
            overwrite=overwrite,
        )

        update_h5(
            fname,
            "/".join([gas_f, "dust_bin4_mean"]),
            data=dust_bin4_mean,
            dtype=np.float32,
            overwrite=overwrite,
        )
        update_h5(
            fname,
            "/".join([gas_f, "dust_bin4_mass_msun"]),
            data=dust_bin4_mass,
            dtype=np.float32,
            overwrite=overwrite,
        )

        update_h5(
            fname,
            "/".join([gas_f, "dtm"]),
            data=dtm,
            overwrite=overwrite,
            dtype=np.float32,
        )

        update_h5(
            fname,
            "/".join([gas_f, "fdepl_Mg"]),
            data=fdMg,
            overwrite=overwrite,
            dtype=np.float32,
        )
        update_h5(
            fname,
            "/".join([gas_f, "fdepl_Fe"]),
            data=fdFe,
            overwrite=overwrite,
            dtype=np.float32,
        )
        update_h5(
            fname,
            "/".join([gas_f, "fdepl_O"]),
            data=fdO,
            overwrite=overwrite,
            dtype=np.float32,
        )
        update_h5(
            fname,
            "/".join([gas_f, "fdepl_Si"]),
            data=fdSi,
            overwrite=overwrite,
            dtype=np.float32,
        )
        update_h5(
            fname,
            "/".join([gas_f, "fdepl_C"]),
            data=fdC,
            overwrite=overwrite,
            dtype=np.float32,
        )

    update_h5(
        fname,
        "/".join([gas_f, "vol_tot_cc"]),
        data=vol_tot,
        overwrite=overwrite,
        dtype=np.float64,
    )

    update_h5(
        fname,
        "/".join([gas_f, "tot_outflow_rate_Msuns"]),
        data=tot_outflow_mass,
        overwrite=overwrite,
        dtype=np.float32,
    )
    update_h5(
        fname,
        "/".join([gas_f, "tot_inflow_rate_Msuns"]),
        data=tot_inflow_mass,
        overwrite=overwrite,
        dtype=np.float32,
    )

    update_h5(
        fname,
        "/".join([gas_f, "mean_outflow_wMass_kms"]),
        data=mean_outflow_wMass,
        dtype=np.float32,
        overwrite=overwrite,
    )
    update_h5(
        fname,
        "/".join([gas_f, "mean_inflow_wMass_kms"]),
        data=mean_inflow_wMass,
        dtype=np.float32,
        overwrite=overwrite,
    )


def star_stats(fname, tgt_f, star_data, pos_gal, overwrite=False):

    # compute mass, metals, age, sfrs
    # do basic kinematics
    # fit ellipsoid to stars

    if "agepart" in star_data.keys():
        ages = star_data["agepart"]
    elif "age" in star_data.keys():
        ages = star_data["age"]
    else:
        print("couldn't find ages for stars")
        raise IndexError

    if len(ages) == 0:
        return 0

    if "mpart" in star_data.keys():
        mass = star_data["mpart"]
    elif "mass" in star_data.keys():
        mass = star_data["mass"]
    else:
        print("couldn't find mass for stars")
        raise IndexError

    if "Zpart" in star_data.keys():
        Zpart = star_data["Zpart"]
    elif "metallicity" in star_data.keys():
        Zpart = star_data["metallicity"]
    else:
        print("couldn't find Z for stars")
        raise IndexError

    metals = np.average(Zpart, weights=mass)
    metals_med = np.median(Zpart)
    metals_max = np.max(Zpart)
    metals_min = np.min(Zpart)

    age = np.average(ages, weights=mass)
    age_med = np.median(ages)
    age_max = np.max(ages)
    age_min = np.min(ages)

    sfr10 = np.sum(mass[ages < 10]) / 10e6
    sfr100 = np.sum(mass[ages < 100]) / 100e6
    sfr500 = np.sum(mass[ages < 500]) / 500e6
    sfr1000 = np.sum(mass[ages < 1000]) / 1000e6

    rot_data, kin_sep_data = extract_nh_kinematics(
        mass, star_data["pos"], star_data["vel"], pos_gal, debug=False
    )

    vrot = rot_data["Vrot"]  # km/s
    disp = rot_data["disp"]  # km/s

    # print(rot_data.keys(), rot_data["Vrot"], rot_data["disp"])

    fdisk = kin_sep_data["fdisk"]  # bulge/disk ala Angles-Alcazar+13
    fbulge = kin_sep_data["fbulge"]
    mdisk = kin_sep_data["Mdisk"]

    Mintertia = np.dot(star_data["pos"].T, star_data["pos"]) 
    eigvals, eigvecs = np.linalg.eig(Mintertia)

    order = np.argsort(eigvals)

    axis3, axis2, axis1 = eigvecs[:,order].T

    axis1 /= np.linalg.norm(axis1)
    axis2 /= np.linalg.norm(axis2)
    axis3 /= np.linalg.norm(axis3)

    ctrd_star_pos = star_data["pos"] - pos_gal

    # new_pos = ctrd_star_pos
    new_pos = np.transpose([np.dot(ctrd_star_pos, eigvect) for eigvect in [axis1,axis2,axis3]])

    new_pos = new_pos - np.mean(new_pos, axis=0)
    # measure full extent... but it truncated distribution then just get shape of truncation
    # a = abs(np.max(new_pos[:, 0]) - np.min(new_pos[:, 0]))
    # b = abs(np.max(new_pos[:, 1]) - np.min(new_pos[:, 1]))
    # c = abs(np.max(new_pos[:, 2]) - np.min(new_pos[:, 2]))
    a,b,c=np.sqrt(np.average((new_pos[:,:])**2,axis=0,weights=mass))
    #measure distribution's shape, but biased depending on threshold?
    # a = np.percentile(new_pos[:, 0], 95)
    # b = np.percentile(new_pos[:, 1], 95)
    # c = np.percentile(new_pos[:, 2], 95)

    # print(np.max(np.linalg.norm(new_pos, axis=1)))

    # #TESTING 
    # print(a, b, c)

    # xbins = np.linspace(new_pos[:, 0].min(), new_pos[:, 0].max(), 50)
    # ybins = np.linspace(new_pos[:, 1].min(), new_pos[:, 1].max(), 50)
    # zbins = np.linspace(new_pos[:, 2].min(), new_pos[:, 2].max(), 50)

    # # print(xbins, ybins, zbins)

    # fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))
    # ellxy = Ellipse((0, 0), a, b, facecolor="none", edgecolor="k")
    # ax[0].add_artist(ellxy)
    # imgxy = binned_statistic_2d(
    #     new_pos[:, 0], new_pos[:, 1], new_pos[:, 1], "count", bins=[xbins, ybins]
    # ).statistic

    # ax[0].imshow(
    #     imgxy.T,
    #     origin="lower",
    #     extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
    #     vmin=1,
    #     vmax=int(0.005 * new_pos.shape[0]),
    # )
    # # ax[0].scatter(new_pos[:, 0], new_pos[:, 1])
    # # ax[0].scatter((ctrd_star_pos)[:, 0], (ctrd_star_pos)[:, 1])

    # ellyz = Ellipse((0, 0), b, c, facecolor="none", edgecolor="k")
    # ax[1].add_artist(ellyz)
    # imgyz = binned_statistic_2d(
    #     new_pos[:, 1], new_pos[:, 2], new_pos[:, 2], "count", bins=[xbins, ybins]
    # ).statistic
    # ax[1].imshow(
    #     imgyz.T,
    #     origin="lower",
    #     extent=[ybins.min(), ybins.max(), zbins.min(), zbins.max()],
    #     vmin=1,
    #     vmax=int(0.005 * new_pos.shape[0]),
    # )
    # # ax[1].scatter(new_pos[:, 1], new_pos[:, 2])
    # # ax[1].scatter((ctrd_star_pos)[:, 1], (ctrd_star_pos)[:, 2])

    # fig.savefig("pca_test")

    # raise KeyError #stupid way to exit after test plot

    # stars = tgt_f.create_group("stars")
    stars_f = os.path.join(tgt_f, "stars")
    update_h5(
        fname,
        "/".join([stars_f, "Mstar_msun"]),
        data=np.sum(mass),
        dtype=np.float32,
        overwrite=overwrite,
    )

    update_h5(
        fname,
        "/".join([stars_f, "Zstar"]),
        data=metals,
        dtype=np.float32,
        overwrite=overwrite,
    )
    update_h5(
        fname,
        "/".join([stars_f, "Zstar_median"]),
        data=metals_med,
        dtype=np.float32,
        overwrite=overwrite,
    )
    update_h5(
        fname,
        "/".join([stars_f, "Zstar_max"]),
        data=metals_max,
        dtype=np.float32,
        overwrite=overwrite,
    )
    update_h5(
        fname,
        "/".join([stars_f, "Zstar_min"]),
        data=metals_min,
        dtype=np.float32,
        overwrite=overwrite,
    )

    update_h5(
        fname,
        "/".join([stars_f, "age_Myr"]),
        data=age,
        dtype=np.float32,
        overwrite=overwrite,
    )
    update_h5(
        fname,
        "/".join([stars_f, "age_median_Myr"]),
        data=age_med,
        dtype=np.float32,
        overwrite=overwrite,
    )
    update_h5(
        fname,
        "/".join([stars_f, "age_max_Myr"]),
        data=age_max,
        dtype=np.float32,
        overwrite=overwrite,
    )
    update_h5(
        fname,
        "/".join([stars_f, "age_min_Myr"]),
        data=age_min,
        dtype=np.float32,
        overwrite=overwrite,
    )

    update_h5(
        fname,
        "/".join([stars_f, "sfr10_Msun_per_yr"]),
        data=sfr10,
        dtype=np.float32,
        overwrite=overwrite,
    )
    update_h5(
        fname,
        "/".join([stars_f, "sfr100_Msun_per_yr"]),
        data=sfr100,
        dtype=np.float32,
        overwrite=overwrite,
    )
    update_h5(
        fname,
        "/".join([stars_f, "sfr500_Msun_per_yr"]),
        data=sfr500,
        dtype=np.float32,
        overwrite=overwrite,
    )
    update_h5(
        fname,
        "/".join([stars_f, "sfr1000_Msun_per_yr"]),
        data=sfr1000,
        dtype=np.float32,
        overwrite=overwrite,
    )

    update_h5(
        fname,
        "/".join([stars_f, "vrot_kms"]),
        data=vrot,
        dtype=np.float32,
        overwrite=overwrite,
    )
    update_h5(
        fname,
        "/".join([stars_f, "vdisp_kms"]),
        data=disp,
        dtype=np.float32,
        overwrite=overwrite,
    )

    update_h5(
        fname,
        "/".join([stars_f, "fdisk"]),
        data=fdisk,
        dtype=np.float32,
        overwrite=overwrite,
    )
    update_h5(
        fname,
        "/".join([stars_f, "fbulge"]),
        data=fbulge,
        dtype=np.float32,
        overwrite=overwrite,
    )
    update_h5(
        fname,
        "/".join([stars_f, "mdisk_msun"]),
        data=mdisk,
        dtype=np.float32,
        overwrite=overwrite,
    )

    update_h5(
        fname, "/".join([stars_f, "a"]), data=a, dtype=np.float32, overwrite=overwrite
    )
    update_h5(
        fname, "/".join([stars_f, "b"]), data=b, dtype=np.float32, overwrite=overwrite
    )
    update_h5(
        fname, "/".join([stars_f, "c"]), data=c, dtype=np.float32, overwrite=overwrite
    )


def dm_stats(fname, tgt_f, dm_data, pos_gal, overwrite=False):

    # compute mass, metals, age, sfrs
    # do basic kinematics
    # fit ellipsoid to stars
    masses = dm_data["mass"]
    mass = masses.sum()

    if len(masses) == 0:
        return 0

    rot_data, kin_sep_data = extract_nh_kinematics(
        masses, dm_data["pos"], dm_data["vel"], pos_gal
    )

    vrot = rot_data["Vrot"]  # km/s
    disp = rot_data["disp"]  # km/s

    # print(rot_data.keys(), rot_data["Vrot"], rot_data["disp"])

    fdisk = kin_sep_data["fdisk"]  # bulge/disk ala Angles-Alcazar+13
    fbulge = kin_sep_data["fbulge"]
    mdisk = kin_sep_data["Mdisk"]
    Mintertia = np.dot(dm_data["pos"].T, dm_data["pos"]) 
    eigvals, eigvecs = np.linalg.eig(Mintertia)

    order = np.argsort(eigvals)

    axis3, axis2, axis1 = eigvecs[:,order].T

    axis1 /= np.linalg.norm(axis1)
    axis2 /= np.linalg.norm(axis2)
    axis3 /= np.linalg.norm(axis3)

    ctrd_dm_pos = dm_data["pos"] - pos_gal

    # new_pos = ctrd_dm_pos
    new_pos = np.transpose([np.dot(ctrd_dm_pos, eigvect) for eigvect in [axis1,axis2,axis3]])

    new_pos = new_pos - np.mean(new_pos, axis=0)
    # measure full extent... but it truncated distribution then just get shape of truncation
    # a = abs(np.max(new_pos[:, 0]) - np.min(new_pos[:, 0]))
    # b = abs(np.max(new_pos[:, 1]) - np.min(new_pos[:, 1]))
    # c = abs(np.max(new_pos[:, 2]) - np.min(new_pos[:, 2]))
    a,b,c=np.sqrt(np.average((new_pos[:,:])**2,axis=0,weights=masses))

    # dm = tgt_f.create_group("dm")
    dm_f = os.path.join(tgt_f, "dm")
    update_h5(
        fname,
        "/".join([dm_f, "Mstar_msun"]),
        data=mass,
        dtype=np.float32,
        overwrite=overwrite,
    )

    update_h5(
        fname,
        "/".join([dm_f, "vrot_kms"]),
        data=vrot,
        dtype=np.float32,
        overwrite=overwrite,
    )
    update_h5(
        fname,
        "/".join([dm_f, "vdisp_kms"]),
        data=disp,
        dtype=np.float32,
        overwrite=overwrite,
    )

    update_h5(
        fname,
        "/".join([dm_f, "fdisk"]),
        data=fdisk,
        dtype=np.float32,
        overwrite=overwrite,
    )
    update_h5(
        fname,
        "/".join([dm_f, "fbulge"]),
        data=fbulge,
        dtype=np.float32,
        overwrite=overwrite,
    )
    update_h5(
        fname,
        "/".join([dm_f, "mdisk_msun"]),
        data=mdisk,
        dtype=np.float32,
        overwrite=overwrite,
    )

    update_h5(
        fname, "/".join([dm_f, "a"]), data=a, dtype=np.float32, overwrite=overwrite
    )
    update_h5(
        fname, "/".join([dm_f, "b"]), data=b, dtype=np.float32, overwrite=overwrite
    )
    update_h5(
        fname, "/".join([dm_f, "c"]), data=c, dtype=np.float32, overwrite=overwrite
    )


def sink_stats(
    fname,
    tgt_f,
    sim,
    snap,
    tgt_pos,
    tgt_rad,
    overwrite=False,
    sink_read_fct=read_sink_bin,
):

    coarse_step, found = snap_to_coarse_step(snap, sim)

    if found == False:
        return 0

    sink_file = os.path.join(sim.sink_path, f"sink_{coarse_step:05d}.dat")

    sinks = sink_read_fct(sink_file)

    convert_sink_units(sinks, sim.get_snap_exps(snap), sim)

    # print(sinks.keys())

    sink_pos = sinks["position"]
    sink_mass = sinks["mass"]

    sink_pos_tree = KDTree(sink_pos, boxsize=1.0 + 1e-6)

    in_search = sink_pos_tree.query_ball_point(tgt_pos, tgt_rad)

    if len(in_search) > 1:

        sink_f = "/".join([tgt_f, "sinks"])

        cat_sids = sinks["identity"][in_search]

        # sink_grp = tgt_file.create_group("sinks")

        update_h5(
            fname,
            "/".join([sink_f, "host_ids"]),
            data=cat_sids,
            dtype=np.int32,
            compression="lzf",
            overwrite=overwrite,
        )

        max_mass_arg = np.argmax(sink_mass[in_search])

        scalar_keys = ["aexp", "ndim", "nsink", "unit_l", "unit_d", "unit_t"]

        max_sink_f = "/".join([sink_f, "max_sink"])
        for key in sinks.keys():
            if key not in scalar_keys:
                update_h5(
                    fname,
                    "/".join([max_sink_f, key]),
                    data=np.max(sinks[key][in_search]),
                    overwrite=overwrite,
                )

        if len(cat_sids) > 1:

            mean_sink_f = "/".join([sink_f, "mean_sink"])
            for key in sinks.keys():
                if key not in ["x", "y", "z", "pos", "position"] + scalar_keys:
                    update_h5(
                        fname,
                        "/".join([mean_sink_f, key]),
                        data=np.mean(sinks[key][in_search]),
                        overwrite=overwrite,
                    )

            median_sink_f = "/".join([sink_f, "median_sink"])
            for key in sinks.keys():
                if key not in ["x", "y", "z", "pos", "position"] + scalar_keys:
                    update_h5(
                        fname,
                        "/".join([median_sink_f, key]),
                        data=np.median(sinks[key][in_search]),
                        overwrite=overwrite,
                    )

            tot_sink_f = "/".join([sink_f, "tot_sink"])
            for key in sinks.keys():
                if key not in ["x", "y", "z", "pos", "position"] + scalar_keys:
                    update_h5(
                        fname,
                        "/".join([tot_sink_f, key]),
                        data=np.sum(sinks[key][in_search]),
                        overwrite=overwrite,
                    )


def make_cat(
    sim: ramses_sim,
    snap,
    cat_type,
    rtype,
    rfact,
    fpure_thresh=0.9999,
    central=False,
    overwrite=False,
    ignore_exist=True,
    id=None,
    use_mpi4py=False,
    ball_reader_fct=read_part_ball_NCdust,
    sink_reader_fct=read_sink_bin,
    target_fields=["stars", "gas", "dm"],
):

    stars = False
    gas = False
    dm = False

    if "stars" in target_fields:
        stars = True
    if "gas" in target_fields:
        gas = True
    if "dm" in target_fields:
        dm = True

    if id is not None:
        if type(id) not in [list, np.ndarray]:
            id = [id]

    if not has_assoc(sim.path, snap):
        print(f"Snap {snap:d} has no assoc file and maybe no detected galaxies")
        return 0

    if not hasattr(sim, "cosmo_model"):
        sim.init_cosmo()

    simdir = sim.path

    fdir = os.path.join(simdir, "catalogues")

    ok_rank = True
    if use_mpi4py:
        ok_rank = rank==0
    
    if not os.path.exists(fdir) and ok_rank:
        os.makedirs(fdir, exist_ok=True)
    if use_mpi4py:
        comm.barrier()

    snap_aexp = sim.get_snap_exps(snap)[0]

    cat_name = args2catname(snap, cat_type, rtype, rfact=rfact, subs=central == False)

    fname = os.path.join(fdir, cat_name)

    print(f"Working on snapshot: {snap}")

    if use_mpi4py and rank == 0:
        if mode == "w":
            h5py.File(fname, "w")
            mode = "a"

            comm.barrier()

    try:
        galaxies = get_gal_props_snap(simdir, snap)
    except (KeyError, FileNotFoundError):
        print(f"Couldn't read galaxy properties for snap {snap}")
        return 0

    # with h5py.File(fname, mode) as tgt_f:
    central_pure = (galaxies["host purity"] > fpure_thresh)
    if central:
        central_pure *= (galaxies["central"])
        
    gids = galaxies["gids"][central_pure]
    host_hids = galaxies["host hid"][central_pure]
    # print(galaxies.keys())

    halos = get_halo_props_snap(simdir, snap)

    hids = halos["hid"]
    # pure_with_gals = (halos["fpure"]>fpure_thresh)*(halos['Ngals']>0)
    # hids = halos['hid'][pure_with_gals]
    # print(halos.keys())

    # get radii

    # sim_tree_hids, sim_tree_datas, sim_tree_aexps = read_tree_file_rev_correct_pos(
    #     os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat"),
    #     sim,
    #     snap,
    #     os.path.join(sim.path, "TreeMakerDM_dust"),
    #     zstart=1./snap_aexp-1,
    #     tgt_ids=[hid],
    #     star=False,
    #     fpure_min=fpure_thresh,
    #     verbose=False
    # )

    # snap_arg = np.argmin(np.abs(sim_tree_aexps-snap_aexp))

    # tree_pties = get_assoc_pties_in_tree(sim, sim_tree_aexps, sim_tree_hids[0], assoc_fields=[rtype])
    # smooth_pties = smooth_props(tree_pties)

    # if len(smooth_pties[rtype])==0:
    #     continue
    # # else:
    # #     print(smooth_pties)
    # #     input('')

    # tgt_r[i] = smooth_pties[rtype][0]

    # print(tgt_r[i])

    match rtype:
        case "rmax":
            tgt_r = galaxies["rmax"][central_pure]
        case "r50":
            tgt_r = galaxies["r50"][central_pure]
        case "r90":
            tgt_r = galaxies["r90"][central_pure]
        case "reff":
            tgt_r = galaxies["reff"][central_pure]
        case "rvir":
            tgt_r = np.zeros(len(host_hids))
            for i, hid in enumerate(host_hids):
                hid_key = f"halo_{hid:07d}"
                if hid_key in halos.keys():
                    tgt_r[i] = halos[hid_key]["rvir"] * rfact
        case "ckpc":
            ckpc_code = 1.0 / (sim.cosmo.lcMpc * 1e3)
            tgt_r = np.full(
                len(galaxies["reff"][central_pure]),
                rfact*ckpc_code
            )
        case "as":
            ckpc_code = 1.0 / (sim.cosmo.lcMpc * 1e3)
            cur_z = sim.get_snap_exps(snap)[0]
            as_ckpc = sim.cosmo_model.arcsec_per_kpc_comoving(cur_z).value
            as_code = as_ckpc / ckpc_code
            tgt_r = np.full(len(galaxies["reff"][central_pure]), rfact/as_code)

    tgt_r *= rfact

    if cat_type == "galaxy":
        print("Making galaxy catalogue")
        tgt_pos = galaxies["pos"].T[central_pure]
        # tgt_r = galaxies["r50"] * rfact
        mass_obj = galaxies["mass"][central_pure]
        ids = gids
        hids = galaxies["host hid"][central_pure]
        fpure = galaxies["host purity"][central_pure]

        # pure = (fpure > fpure_thresh) * (fpure <= 1.0)  # weird values

        # tgt_pos = tgt_pos[central_pure]
        # tgt_r = tgt_r[central_pure]
        # ids = ids[central_pure]
        # gids = gids[central_pure]
        # hids = hids[central_pure]
        # mass_obj = mass_obj[central_pure]
        # fpure = fpure[central_pure]

        update_h5(fname, "galaxy_ids", gids, overwrite=overwrite, dtype=np.int32)
        update_h5(fname, "pos", tgt_pos, overwrite=overwrite, dtype=np.float64)
        update_h5(
            fname, f"{rtype}x{rfact:.1f}", tgt_r, overwrite=overwrite, dtype=np.float64
        )
        update_h5(fname, "host ids", hids, overwrite=overwrite, dtype=np.int32)
        update_h5(
            fname,
            "host mass",
            galaxies["host mass"][central_pure],
            overwrite=overwrite,
            dtype=np.float32,
        )
        update_h5(fname, "host purity", fpure, overwrite=overwrite, dtype=np.float32)

    elif cat_type == "halo":
        print("Making halo catalogue")

        all_hids = halos["hid"]
        all_hid_args = np.searchsorted(host_hids, all_hids)
        tgt_pos = np.zeros((len(hids), 3))
        # tgt_r = np.zeros(len(hids))
        # tgt_pos = halos["pos"]
        # tgt_r = halos["rvir"] * rfact
        for i, hid in enumerate(host_hids):
            hid_key = f"halo_{hid:07d}"
            if hid_key in halos.keys():
                tgt_pos[i] = halos[hid_key]["pos"]
                # tgt_r[i] = halos[hid_key]["rvir"] * rfact
        mass_obj = halos["mvir"][all_hid_args]
        ids = hids

        # pure = halos["fpure"] > fpure_thresh
        # tgt_pos = tgt_pos[pure]
        # tgt_r = tgt_r[pure]
        # ids = ids[pure]
        # hids = hids[pure]
        # mass_obj = mass_obj[pure]

        update_h5(fname, "halo_ids", hids, overwrite=overwrite, dtype=np.int32)
        update_h5(fname, "pos", tgt_pos, overwrite=overwrite, dtype=np.float64)
        update_h5(
            fname, f"{rtype}x{rfact:.1f}", tgt_r, overwrite=overwrite, dtype=np.float64
        )
    else:
        raise ValueError("Invalid catalogue type")

    if use_mpi4py:

        # divide up galaxies or halos so that each rank has a subset
        # of equal mass (-> roughly equal number of particles and cells)
        tot_mass = mass_obj.sum()

        mass_cdf = np.cumsum(mass_obj) / tot_mass

        # need to do it for all ranks and make sure I don't do the same one several times !!!

        isplit = rank / size
        isplitp1 = (rank + 1) / size

        arg_low = np.argmin(np.abs(mass_cdf - isplit))
        arg_high = np.argmin(np.abs(mass_cdf - isplitp1)) + 1

        print(f"Rank {rank} working on objects {arg_low} to {arg_high}")
        print(len(tgt_pos), arg_high - arg_low)

        tgt_pos = tgt_pos[arg_low:arg_high]
        tgt_r = tgt_r[arg_low:arg_high]
        ids = ids[arg_low:arg_high]
        hids = hids[arg_low:arg_high]

        # old way: equal number per rank
        # tgt_pos = np.array_split(tgt_pos, size)[rank]
        # tgt_r = np.array_split(tgt_r, size)[rank]
        # ids = np.array_split(ids, size)[rank]
        

    for i_obj, (pos, r, obj_id) in enumerate(zip(tgt_pos, tgt_r, ids)):

        if r == 0:
            print('zero size galaxy')
            continue

        if id is not None:
            if obj_id not in id:
                continue
            else:
                print(f"found {cat_type} {id}")

        if use_mpi4py:

            print(
                f"Rank {rank} working on object {i_obj}/{len(tgt_r)}, ID:{ids[i_obj]}"
            )
        else:
            print(f"Working on object {i_obj}/{len(tgt_r)}, ID:{ids[i_obj]}")

        if cat_type == "galaxy":
            gid = ids[i_obj]
        else:
            gid = None

        tgt_fields = []
        data_types = []

        if stars:
            tgt_fields += [
                "mpart",
                "agepart",
                "birth_time",
                "age",
                "Zpart",
                "metallicity",
                "vel",
                "pos",
                "mass",
            ]
            data_types.append("stars")
        if gas:
            tgt_fields += [
                "density",
                "ilevel",
                "temperature",
                "metallicity",
                "chem_O",
                "chem_Fe",
                "chem_Mg",
                "chem_C",
                "chem_Si",
                "chem_N",
                "chem_S",
                "dust_bin01",
                "dust_bin02",
                "dust_bin03",
                "dust_bin04",
                "velocity_x",
                "velocity_y",
                "velocity_z",
            ]
            data_types.append("gas")
        if dm:
            tgt_fields += ["mpart", "vel", "pos", "mass"]
            data_types.append("dm")
        tgt_fields = list(set(tgt_fields))

        file_grp = f"{cat_type}_{ids[i_obj]:07d}"

        with h5py.File(fname, "r") as src:

            grp_exists = file_grp in src
            data_present = False
            if grp_exists:
                found_keys = [
                    int(
                        ty in src[file_grp]
                        and type(src[file_grp][ty][dt][()])
                        in [int, float, np.int32, np.float32, np.int64, np.float64]
                    )
                    for ty in data_types
                    if ty in src[file_grp].keys()
                    for dt in src[file_grp][ty].keys()
                ]
                # found_keys=[]
                # for ty in data_types:

                #     for dt in src[file_grp][ty].keys()
                data_present = np.sum(found_keys) == len(found_keys)

        if (grp_exists and data_present) and ignore_exist:
            print("found existing data, skipping...")
            continue

        # try:
        data = read_data_ball(
            sim,
            snap,
            tgt_pos=pos,
            tgt_r=r,
            host_halo=hids[i_obj],
            gid=gid,
            data_types=data_types,
            tgt_fields=tgt_fields,
            read_part_ball_fct=ball_reader_fct,
        )
        # except (AssertionError, KeyError, FileNotFoundError):
        # print(f"problem with object")
        # continue

        # compute gas stuff
        if gas:

            if "gas" in data.keys():
                gas_data = data["gas"]
            else:
                gas_data = data

            aexp = sim.get_snap_exps(snap)
            gas_stats(fname, file_grp, pos, aexp, gas_data, sim, overwrite=overwrite)

        # compute dm stuff
        if dm:

            if "dm" in data.keys():
                dm_data = data["dm"]
            else:
                dm_data = data

            dm_stats(fname, file_grp, dm_data, pos, overwrite=overwrite)

        # compute star stuff
        if stars:

            if "stars" in data.keys():
                stars_data = data["stars"]
            else:
                stars_data = data

            star_stats(fname, file_grp, stars_data, pos, overwrite=overwrite)

        try:
            # compute sink stuff
            sink_stats(
                fname,
                file_grp,
                sim,
                snap,
                pos,
                r,
                overwrite=overwrite,
                sink_read_fct=sink_reader_fct,
            )
        except AssertionError:
            continue

    if not use_mpi4py:
        print("Finished snapshot")
    else:
        print(f"Rank {rank} finished snapshot")
        comm.barrier()

    print(f"Writing to file: {fname}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "simdir", type=str, nargs="+", help="path(s) to the simulation(s)"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["galaxy", "halo"],
        help="type of catalogue to build, either galaxy or halo based",
        default="galaxy",
    )
    parser.add_argument(
        "--rtype",
        type=str,
        help="choice of radii for property extraction",
        choices=["r50", "reff", "r90", "rvir", "rmax","as","ckpc"],
        default="r50",
    )
    parser.add_argument(
        "--rfact",
        type=float,
        help="rfact x rtype is the radius within which properties are to be measured",
        default=1.0,
    )
    parser.add_argument(
        "--fpure",
        type=float,
        help="purity floor for halos",
        default=0.9999,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing catalogues",
        default=False,
    )
    parser.add_argument(
        "--update_existing",
        action="store_true",
        help="update existing entries, but don't overwrite (superceeded by --overwrite flag)",
        default=False,
    )
    parser.add_argument(
        "--mpi4py",
        action="store_true",
        help="use mpi4py to parallelize catalogue creation over multiple galaxies/halos",
        default=False,
    )
    parser.add_argument(
        "--central",
        action="store_true",
        help="only make cat of centrals",
        default=False,
    )
    parser.add_argument(
        "--snap",
        type=int,
        nargs="+",
        help="snapshot number(s) to be read",
        default=None,
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        help="ids of targets (galaxies or halos) to process",
        default=None,
    )
    parser.add_argument(
        "--tgt_fields",
        type=str,
        nargs="+",
        help="fields to load and compute stats for, can be gas, stars, dm",
        default=["gas","stars","dm"],
    )
    parser.add_argument(
        "--sim_type",
        type=str,
        help="reference to simulation code version, used to pick reading functions",
        default="yomp_dust",
        choices=["yomp_dust", "NH", "HAGN"],
    )

    args = parser.parse_args()

    (
        simdirs,
        cat_type,
        rfact,
        rtype,
        fpure_thresh,
        overwrite,
        use_mpi4py,
        snap,
        central,
        sim_type,
    ) = (
        args.simdir,
        args.type,
        args.rfact,
        args.rtype,
        args.fpure,
        args.overwrite,
        args.mpi4py,
        args.snap,
        args.central,
        args.sim_type,
    )

    if use_mpi4py:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
        size = 1

    for simdir in simdirs:

        sim = ramses_sim(simdir, nml="cosmo.nml")

        print("Reading data from simulation: ", simdir)
        print(f"{len(sim.snap_numbers)} snapshots found")

        if snap is not None:
            snap_iter = snap
        else:
            snap_iter = sim.snap_numbers

        aexps = sim.get_snap_exps()
        times = sim.get_snap_times()
        zeds = 1.0 / aexps - 1.0

        zstt = zeds.min()
        snap_stt = sim.snap_numbers.max()

        # pass sim name argument
        # pick read part and read sink fcts
        sim_fct_lookup = {
            "NH": {
                "part_ball_fct": read_part_ball_NH,
                "sink_read_fct": read_sink_bin_NH,
            },
            "yomp_dust": {
                "part_ball_fct": read_part_ball_NCdust,
                "sink_read_fct": read_sink_bin,
            },
            "HAGN": {
                "part_ball_fct": read_part_ball_hagn,
                "sink_read_fct": read_sink_bin,
            },
        }

        for snap in snap_iter[::-1]:

            make_cat(
                sim,
                snap,
                cat_type,
                rtype,
                rfact,
                fpure_thresh=fpure_thresh,
                central=central,
                overwrite=overwrite,
                ignore_exist=args.update_existing==False,
                ball_reader_fct=sim_fct_lookup[sim_type]["part_ball_fct"],
                sink_reader_fct=sim_fct_lookup[sim_type]["sink_read_fct"],
                target_fields=args.tgt_fields,
                id = args.ids
            )
