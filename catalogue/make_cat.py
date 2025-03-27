###
# TODO: rewrite in simpler manner: arrays for all stats? then shared via mpi4py
###


from zoom_analysis.halo_maker.assoc_fcts import (
    get_gal_props_snap,
    get_halo_props_snap,
    get_assoc_pties_in_tree,
    smooth_props,
)
from zoom_analysis.stars.dynamics import extract_nh_kinematics
from zoom_analysis.read.read_data import read_data_ball
from zoom_analysis.sinks.sink_reader import (
    read_sink_bin,
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

parser = argparse.ArgumentParser()

parser.add_argument("simdir", type=str, nargs="+", help="path(s) to the simulation(s)")
parser.add_argument(
    "--type",
    type=str,
    choices=["galaxy", "halo"],
    help="type of catalogue to build, either galaxy or halo based",
    default="galaxy",
)
parser.add_argument(
    "--rball",
    type=float,
    help="rballxr50 is the radius within which properties are to be measured",
    default=2.0,
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
    "--mpi4py",
    action="store_true",
    help="use mpi4py to parallelize catalogue creation over multiple galaxies/halos",
    default=False,
)
parser.add_argument(
    "--snap", type=int, nargs="+", help="snapshot number(s) to be read", default=None
)


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


def gas_stats(tgt_f, tgt_pos, aexp, gas_data, sim, thresh_dense=0.1):

    MgoverSil = 0.141
    FeoverSil = 0.324
    SioverSil = 0.163
    OoverSil = 0.372

    density = gas_data["density"]  # g/cc

    temp = gas_data["temperature"]  # T[K]/mu
    metals = gas_data["metallicity"]
    dust_bin1 = gas_data["dust_bin01"]  # g/cc
    dust_bin2 = gas_data["dust_bin02"]  # g/cc
    dust_bin3 = gas_data["dust_bin03"] / SioverSil  # g/cc
    dust_bin4 = gas_data["dust_bin04"] / SioverSil  # g/cc
    vels = np.transpose([gas_data["velocity_x"], gas_data["velocity_y"], gas_data["velocity_z"]])/1e5 #km/s
    pos = np.transpose([gas_data["x"], gas_data["y"], gas_data["z"]])

    no_bulk_vels = vels  - np.average(vels, axis=0,weights=density)
    vel_out = np.sum(no_bulk_vels * (pos-tgt_pos), axis=1)
    tot_outflow = np.sum(vel_out)
    mean_outflow_wMass = np.average(vel_out,weights=density)


    vols = (
        (sim.cosmo.lcMpc * ramses_pc * 1e6 * 1e2 * aexp) / 2 ** gas_data["ilevel"]
    ) ** 3  # cc

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
        ) / (metals_mass_dense)

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

        dens_no_bulk_vels = vels[dense_gas]  - np.average(vels[dense_gas], axis=0,weights=density)
        dens_vel_out = np.sum(no_bulk_vels[dense_gas] * (pos[dense_gas]-tgt_pos), axis=1)
        dens_tot_outflow = np.sum(vel_out[dense_gas])
        dens_mean_outflow_wMass = np.average(vel_out[dense_gas],weights=density[dense_gas])

        vol_dense = np.sum(vols[dense_gas])  # cc
        dense_gas_f = tgt_f.create_group(f"Hpcc>{thresh_dense:0.1f}_gas")
        dense_gas_f.create_dataset("Mgas_msun", data=mass_dense, dtype=np.float32)
        dense_gas_f.create_dataset("T_mean_K", data=temp_mean_dense, dtype=np.float32)
        dense_gas_f.create_dataset("Z_mean", data=metals_mean_dense, dtype=np.float32)
        dense_gas_f.create_dataset(
            "Mmetals_msun", data=metals_mass_dense, dtype=np.float32
        )
        dense_gas_f.create_dataset(
            "dust_bin1_mean", data=dust_bin1_mean_dense, dtype=np.float32
        )
        dense_gas_f.create_dataset(
            "dust_bin1_mass_msun", data=dust_bin1_mass_dense, dtype=np.float32
        )
        dense_gas_f.create_dataset(
            "dust_bin2_mean", data=dust_bin2_mean_dense, dtype=np.float32
        )
        dense_gas_f.create_dataset(
            "dust_bin2_mass_msun", data=dust_bin2_mass_dense, dtype=np.float32
        )
        dense_gas_f.create_dataset(
            "dust_bin3_mean", data=dust_bin3_mean_dense, dtype=np.float32
        )
        dense_gas_f.create_dataset(
            "dust_bin3_mass_msun", data=dust_bin3_mass_dense, dtype=np.float32
        )
        dense_gas_f.create_dataset(
            "dust_bin4_mean", data=dust_bin4_mean_dense, dtype=np.float32
        )
        dense_gas_f.create_dataset(
            "dust_bin4_mass_msun", data=dust_bin4_mass_dense, dtype=np.float32
        )
        dense_gas_f.create_dataset("dtm", data=dtm_dens, dtype=np.float32)
        dense_gas_f.create_dataset("fdepl_Mg", data=fdMg_dense, dtype=np.float32)
        dense_gas_f.create_dataset("fdepl_Fe", data=fdFe_dense, dtype=np.float32)
        dense_gas_f.create_dataset("fdepl_O", data=fdO_dense, dtype=np.float32)
        dense_gas_f.create_dataset("fdepl_Si", data=fdSi_dense, dtype=np.float32)
        dense_gas_f.create_dataset("fdepl_C", data=fdC_dense, dtype=np.float32)
        dense_gas_f.create_dataset("vol_cc", data=vol_dense, dtype=np.float64)

        dense_gas_f.create_dataset("tot_outflow_kms", data=tot_outflow, dtype=np.float32)
        dense_gas_f.create_dataset("mean_outflow_wMass_kms", data=mean_outflow_wMass, dtype=np.float32)

    mass = np.sum(density * vols) / Msun_cgs  # Msun

    temp_mean = np.average(temp, weights=density)  # T

    metals_mean = np.average(metals, weights=density)  # Z
    metals_mass = np.sum(metals * density * vols) / Msun_cgs  # Msun

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

    gas = tgt_f.create_group("gas")
    gas.create_dataset("Mgas_tot_msun", data=mass, dtype=np.float32)

    gas.create_dataset("T_mean_K", data=temp_mean, dtype=np.float32)

    gas.create_dataset("Z_mean", data=metals_mean, dtype=np.float32)
    gas.create_dataset("Mmetals_msun", data=metals_mass, dtype=np.float32)

    gas.create_dataset("dust_bin1_mean", data=dust_bin1_mean, dtype=np.float32)
    gas.create_dataset("dust_bin1_mass_msun", data=dust_bin1_mass, dtype=np.float32)

    gas.create_dataset("dust_bin2_mean", data=dust_bin2_mean, dtype=np.float32)
    gas.create_dataset("dust_bin2_mass_msun", data=dust_bin2_mass, dtype=np.float32)

    gas.create_dataset("dust_bin3_mean", data=dust_bin3_mean, dtype=np.float32)
    gas.create_dataset("dust_bin3_mass_msun", data=dust_bin3_mass, dtype=np.float32)

    gas.create_dataset("dust_bin4_mean", data=dust_bin4_mean, dtype=np.float32)
    gas.create_dataset("dust_bin4_mass_msun", data=dust_bin4_mass, dtype=np.float32)

    gas.create_dataset("dtm", data=dtm, dtype=np.float32)

    gas.create_dataset("fdepl_Mg", data=fdMg, dtype=np.float32)
    gas.create_dataset("fdepl_Fe", data=fdFe, dtype=np.float32)
    gas.create_dataset("fdepl_O", data=fdO, dtype=np.float32)
    gas.create_dataset("fdepl_Si", data=fdSi, dtype=np.float32)
    gas.create_dataset("fdepl_C", data=fdC, dtype=np.float32)

    gas.create_dataset("vol_tot_cc", data=vol_tot, dtype=np.float64)

    gas.create_dataset("tot_outflow_kms", data=tot_outflow, dtype=np.float32)
    gas.create_dataset("mean_outflow_wMass_kms", data=mean_outflow_wMass, dtype=np.float32)


def star_stats(tgt_f, star_data, pos_gal):

    # compute mass, metals, age, sfrs
    # do basic kinematics
    # fit ellipsoid to stars

    ages = star_data["agepart"]

    mass = np.sum(star_data["mpart"])

    metals = np.average(star_data["Zpart"], weights=star_data["mpart"])
    metals_med = np.median(star_data["Zpart"])
    metals_max = np.max(star_data["Zpart"])
    metals_min = np.min(star_data["Zpart"])

    age = np.average(ages, weights=star_data["mpart"])
    age_med = np.median(ages)
    age_max = np.max(ages)
    age_min = np.min(ages)

    sfr10 = np.sum(star_data["mpart"][ages < 10]) / 10e6
    sfr100 = np.sum(star_data["mpart"][ages < 100]) / 100e6
    sfr500 = np.sum(star_data["mpart"][ages < 500]) / 500e6
    sfr1000 = np.sum(star_data["mpart"][ages < 1000]) / 1000e6

    rot_data, kin_sep_data = extract_nh_kinematics(
        star_data["mpart"], star_data["pos"], star_data["vel"], pos_gal, debug=False
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
    axis3, axis2, axis1 = eigvecs[:, order].T

    axis1 /= np.linalg.norm(axis1)
    axis2 /= np.linalg.norm(axis2)
    axis3 /= np.linalg.norm(axis3)

    new_pos = np.dot(star_data["pos"], eigvecs)
    new_pos = new_pos - np.mean(new_pos, axis=0)
    a = abs(np.max(new_pos[:, 0]) - np.min(new_pos[:, 0]))
    b = abs(np.max(new_pos[:, 1]) - np.min(new_pos[:, 1]))
    c = abs(np.max(new_pos[:, 2]) - np.min(new_pos[:, 2]))

    stars = tgt_f.create_group("stars")
    stars.create_dataset("Mstar_msun", data=mass, dtype=np.float32)

    stars.create_dataset("Zstar", data=metals, dtype=np.float32)
    stars.create_dataset("Zstar_median", data=metals_med, dtype=np.float32)
    stars.create_dataset("Zstar_max", data=metals_max, dtype=np.float32)
    stars.create_dataset("Zstar_min", data=metals_min, dtype=np.float32)

    stars.create_dataset("age_Myr", data=age, dtype=np.float32)
    stars.create_dataset("age_median_Myr", data=age_med, dtype=np.float32)
    stars.create_dataset("age_max_Myr", data=age_max, dtype=np.float32)
    stars.create_dataset("age_min_Myr", data=age_min, dtype=np.float32)

    stars.create_dataset("sfr10_Msun_per_yr", data=sfr10, dtype=np.float32)
    stars.create_dataset("sfr100_Msun_per_yr", data=sfr100, dtype=np.float32)
    stars.create_dataset("sfr500_Msun_per_yr", data=sfr500, dtype=np.float32)
    stars.create_dataset("sfr1000_Msun_per_yr", data=sfr1000, dtype=np.float32)

    stars.create_dataset("vrot_kms", data=vrot, dtype=np.float32)
    stars.create_dataset("vdisp_kms", data=disp, dtype=np.float32)

    stars.create_dataset("fdisk", data=fdisk, dtype=np.float32)
    stars.create_dataset("fbulge", data=fbulge, dtype=np.float32)
    stars.create_dataset("mdisk_msun", data=mdisk, dtype=np.float32)

    stars.create_dataset("a", data=a, dtype=np.float32)
    stars.create_dataset("b", data=b, dtype=np.float32)
    stars.create_dataset("c", data=c, dtype=np.float32)


def dm_stats(tgt_f, dm_data, pos_gal):

    # compute mass, metals, age, sfrs
    # do basic kinematics
    # fit ellipsoid to stars

    mass = np.sum(dm_data["mass"])

    rot_data, kin_sep_data = extract_nh_kinematics(
        dm_data["mass"], dm_data["pos"], dm_data["vel"], pos_gal
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
    axis3, axis2, axis1 = eigvecs[:, order].T

    axis1 /= np.linalg.norm(axis1)
    axis2 /= np.linalg.norm(axis2)
    axis3 /= np.linalg.norm(axis3)

    new_pos = np.dot(dm_data["pos"], eigvecs)
    # new_pos = new_pos - np.mean(new_pos, axis=0)

    a = abs(np.max(new_pos[:, 0]) - np.min(new_pos[:, 0]))
    b = abs(np.max(new_pos[:, 1]) - np.min(new_pos[:, 1]))
    c = abs(np.max(new_pos[:, 2]) - np.min(new_pos[:, 2]))

    dm = tgt_f.create_group("dm")
    dm.create_dataset("Mstar_msun", data=mass, dtype=np.float32)

    dm.create_dataset("vrot_kms", data=vrot, dtype=np.float32)
    dm.create_dataset("vdisp_kms", data=disp, dtype=np.float32)

    dm.create_dataset("fdisk", data=fdisk, dtype=np.float32)
    dm.create_dataset("fbulge", data=fbulge, dtype=np.float32)
    dm.create_dataset("mdisk_msun", data=mdisk, dtype=np.float32)

    dm.create_dataset("a", data=a, dtype=np.float32)
    dm.create_dataset("b", data=b, dtype=np.float32)
    dm.create_dataset("c", data=c, dtype=np.float32)


def sink_stats(sim, snap, tgt_pos, tgt_rad, tgt_file):

    coarse_step = snap_to_coarse_step(snap, sim)
    sink_file = os.path.join(sim.sink_path, f"sink_{coarse_step:05d}.dat")

    sinks = read_sink_bin(sink_file)

    convert_sink_units(sinks, sim.get_snap_exps(snap), sim)

    # print(sinks.keys())

    sink_pos = sinks["position"]
    sink_mass = sinks["mass"]

    sink_pos_tree = KDTree(sink_pos, boxsize=1.0 + 1e-6)

    in_search = sink_pos_tree.query_ball_point(tgt_pos, tgt_rad)

    if len(in_search) > 1:

        cat_sids = sinks["identity"][in_search]

        sink_grp = tgt_file.create_group("sinks")

        sink_grp.create_dataset(
            "host_ids", data=cat_sids, dtype=np.int32, compression="lzf"
        )

        max_mass_arg = np.argmax(sink_mass[in_search])

        scalar_keys = ["aexp", "ndim", "nsink", "unit_l", "unit_d", "unit_t"]

        max_mass_grp = sink_grp.create_group("max_mass_sink")
        for key in sinks.keys():
            if key not in scalar_keys:
                max_mass_grp.create_dataset(
                    key, data=sinks[key][in_search][max_mass_arg]
                )

        if len(cat_sids) > 1:

            mean_grp = sink_grp.create_group("mean_sink")
            for key in sinks.keys():
                if key not in ["x", "y", "z", "pos", "position"] + scalar_keys:
                    mean_grp.create_dataset(key, data=np.mean(sinks[key][in_search]))

            median_grp = sink_grp.create_group("median_sink")
            for key in sinks.keys():
                if key not in ["x", "y", "z", "pos", "position"] + scalar_keys:
                    median_grp.create_dataset(
                        key, data=np.median(sinks[key][in_search])
                    )

            tot_grp = sink_grp.create_group("total_sink")
            for key in sinks.keys():
                if key not in ["x", "y", "z", "pos", "position"] + scalar_keys:
                    tot_grp.create_dataset(key, data=np.sum(sinks[key][in_search]))


args = parser.parse_args()

simdirs, cat_type, rfact, fpure_thresh, overwrite, use_mpi4py, snap = (
    args.simdir,
    args.type,
    args.rball,
    args.fpure,
    args.overwrite,
    args.mpi4py,
    args.snap,
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

    sim = ramses_sim(simdir)

    print("Reading data from simulation: ", simdir)
    print(f"{len(sim.snap_numbers)} snapshots found")

    if snap is not None:
        snap_iter = snap
    else:
        snap_iter = sim.snap_numbers

    aexps = sim.get_snap_exps()
    zeds = 1.0 / aexps - 1

    hids = get_halo_props_snap(simdir, sim.snap_numbers[-1])["hid"]

    sim_tree_hids, sim_tree_datas, sim_tree_aexps = read_tree_file_rev_correct_pos(
        os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat"),
        sim,
        sim.snap_numbers[-1],
        os.path.join(sim.path, "TreeMakerDM_dust"),
        zstart=zeds.min(),
        tgt_ids=hids,
        star=False,
    )

    tree_pties = get_assoc_pties_in_tree(sim, sim_tree_aexps, sim_tree_hids)
    smooth_pties = smooth_props(tree_pties)

    for snap in snap_iter:

        fdir = os.path.join(simdir, "catalogues")
        if not os.path.exists(fdir) and rank == 0:
            os.makedirs(fdir, exist_ok=True)
        if use_mpi4py:
            comm.barrier()

        rtype = None
        if cat_type == "galaxy":
            rtype = "r50"
        elif cat_type == "halo":
            rtype = "rvir"

        rfact_str = ("%.1f" % rfact).replace(".", "p")

        fname = os.path.join(fdir, f"{cat_type}_{rfact_str}X{rtype}_{snap}.hdf5")

        if overwrite or not os.path.exists(fname):
            mode = "w"
        else:
            mode = "a"

        print(f"Working on snapshot: {snap}")

        if use_mpi4py and rank == 0:
            if mode == "w":
                h5py.File(fname, "w")
                mode = "a"

                comm.barrier()

        with h5py.File(fname, mode) as tgt_f:

            galaxies = get_gal_props_snap(simdir, snap)
            gids = galaxies["gids"]
            # print(galaxies.keys())

            halos = get_halo_props_snap(simdir, snap)
            # print(halos.keys())

            if cat_type == "galaxy":
                print("Making galaxy catalogue")
                tgt_pos = galaxies["pos"].T
                tgt_r = galaxies["r50"] * rfact
                mass_obj = galaxies["mass"]
                ids = gids
                hids = galaxies["host hid"]
                fpure = galaxies["host purity"]

                pure = (fpure > fpure_thresh) * (fpure <= 1.0)  # weird values

                tgt_pos = tgt_pos[pure]
                tgt_r = tgt_r[pure]
                ids = ids[pure]
                gids = gids[pure]
                hids = hids[pure]
                mass_obj = mass_obj[pure]
                fpure = fpure[pure]

                tgt_f.create_dataset(
                    "galaxy ids", data=gids, dtype=np.int32, compression="lzf"
                )
                tgt_f.create_dataset(
                    "pos", data=tgt_pos, dtype=np.float64, compression="lzf"
                )
                tgt_f.create_dataset(
                    "r50", data=tgt_r, dtype=np.float64, compression="lzf"
                )
                tgt_f.create_dataset(
                    "host ids",
                    data=galaxies["host hid"],
                    dtype=np.int32,
                    compression="lzf",
                )
                tgt_f.create_dataset(
                    "host mass",
                    data=galaxies["host mass"],
                    dtype=np.float32,
                    compression="lzf",
                )
                tgt_f.create_dataset(
                    "host purity",
                    data=galaxies["host purity"],
                    dtype=np.float32,
                    compression="lzf",
                )

            elif cat_type == "halo":
                print("Making halo catalogue")

                hids = halos["hid"]
                tgt_pos = np.zeros((len(hids), 3))
                tgt_r = np.zeros(len(hids))
                # tgt_pos = halos["pos"]
                # tgt_r = halos["rvir"] * rfact
                for i, hid in enumerate(hids):
                    hid_key = f"halo_{hid:07d}"
                    tgt_pos[i] = halos[hid_key]["pos"]
                    tgt_r[i] = halos[hid_key]["rvir"] * rfact
                mass_obj = halos["mvir"]
                ids = hids

                pure = halos["fpure"] > fpure_thresh
                tgt_pos = tgt_pos[pure]
                tgt_r = tgt_r[pure]
                ids = ids[pure]
                hids = hids[pure]
                mass_obj = mass_obj[pure]

                tgt_f.create_dataset(
                    "halo ids", data=hids, dtype=np.int32, compression="lzf"
                )
                tgt_f.create_dataset(
                    "pos", data=tgt_pos, dtype=np.float64, compression="lzf"
                )
                tgt_f.create_dataset(
                    "rvir", data=tgt_r, dtype=np.float64, compression="lzf"
                )
            else:
                raise ValueError("Invalid catalogue type")

            if use_mpi4py:

                # divide up galaxies or halos so that each rank has a subset
                # of equal mass (-> rouhly equal number of particles and cells)
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

                print(len(tgt_pos))

                # old way: equal number per rank
                # tgt_pos = np.array_split(tgt_pos, size)[rank]
                # tgt_r = np.array_split(tgt_r, size)[rank]
                # ids = np.array_split(ids, size)[rank]

            for i_obj, (pos, r) in enumerate(zip(tgt_pos, tgt_r)):

                if use_mpi4py:

                    print(f"Rank {rank} working on object {i_obj}, ID:{ids[i_obj]}")
                else:
                    print(f"Working on object {i_obj}, ID:{ids[i_obj]}")

                if cat_type == "galaxy":
                    gid = ids[i_obj]
                else:
                    gid = None

                try:
                    data = read_data_ball(
                        sim,
                        snap,
                        tgt_pos=pos,
                        tgt_r=r,
                        host_halo=hids[i_obj],
                        gid=gid,
                        data_types=["gas", "stars", "dm"],
                        tgt_fields=[
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
                            "mpart",
                            "agepart",
                            "Zpart",
                            "vel",
                            "pos",
                            "mass",
                        ],
                    )
                except AssertionError:
                    continue

                file_grp = tgt_f.create_group(f"{cat_type}_{ids[i_obj]:07d}")

                # compute gas stuff
                if data["gas"] is not None:
                    aexp = sim.get_snap_exps(snap)
                    gas_stats(file_grp, tgt_pos, aexp, data["gas"], sim)

                # compute dm stuff
                if data["dm"] is not None:
                    dm_stats(file_grp, data["dm"], pos)

                # compute star stuff
                if data["stars"] is not None:
                    star_stats(file_grp, data["stars"], pos)

                try:
                    # compute sink stuff
                    sink_stats(sim, snap, pos, r, file_grp)
                except AssertionError:
                    continue

            if not use_mpi4py:
                print("Finished snapshot")
            else:
                print(f"Rank {rank} finished snapshot")
                comm.barrier()

            print(f"Writing to file: {fname}")
