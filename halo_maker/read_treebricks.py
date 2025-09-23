import numpy as np
import os

from f90_tools.IO import read_record, read_tgt_fields, skip_record

# from astropy.cosmology import FlatLambdaCDM

from gremlin.read_sim_params import ramses_sim

from .friedman import ct_init_cosmo

from ..constants import *


def read_brickfile(fname, star=None, galaxy=True):
    """
    Positions origin is at the center of the box
    Masses (h%m,h%datas%mvir) are in units of 10^11 Msol, and
    Lengths (h%p%x,h%p%y,h%p%z,h%r,h%datas%rvir) are in units of Mpc
    Velocities (h%v%x,h%v%y,h%v%z,h%datas%cvel) are in km/s
    Energies (h%ek,h%ep,h%et) are in
    Temperatures (h%datas%tvir) are in K
    Angular Momentum (h%L%x,h%L%y,h%L%z) are in
    Other quantities are dimensionless (h%my_number,h%my_timestep,h%spin)"""

    if star == None:
        if "star" in fname:
            star = True
        else:
            star = False

    # print(star)

    if galaxy:
        dt = np.float64
    else:
        dt = np.float32

    with open(fname, "rb") as src:
        nbodies = read_record(src, 1, np.int32)
        # print(nbodies)
        massp = read_record(src, 1, dt)
        aexp = read_record(src, 1, dt)
        omega_t = read_record(src, 1, dt)
        age_univ = read_record(src, 1, dt)
        nh, nsub = read_record(src, 2, np.int32)
        nb_structs = nh + nsub
        # print(nbodies, massp, aexp, omega_t, age_univ, nh, nsub, nb_structs)

        # Create empty arrays to store the quantities
        hid = np.empty(nb_structs, dtype=np.int32)
        tstep = np.empty(nb_structs, dtype=np.float32)
        hlvl = np.empty(nb_structs, dtype=np.int32)
        hosth = np.empty(nb_structs, dtype=np.int32)
        hostsub = np.empty(nb_structs, dtype=np.int32)
        nbsub = np.empty(nb_structs, dtype=np.int32)
        nextsub = np.empty(nb_structs, dtype=np.int32)
        hmass = np.empty(nb_structs, dtype=np.float32)
        pos = np.empty((nb_structs, 3), dtype=np.float64)
        vel = np.empty((nb_structs, 3), dtype=np.float32)
        AngMom = np.empty((nb_structs, 3), dtype=np.float32)
        ellipse = np.empty((nb_structs, 4), dtype=np.float32)
        Ek = np.empty(nb_structs, dtype=np.float32)
        Ep = np.empty(nb_structs, dtype=np.float32)
        Et = np.empty(nb_structs, dtype=np.float32)
        spin = np.empty(nb_structs, dtype=np.float32)
        if galaxy:
            sigma = np.empty(nb_structs, dtype=np.float32)
            sigma_bulge = np.empty(nb_structs, dtype=np.float32)
            m_bulge = np.empty(nb_structs, dtype=np.float32)
        rvir = np.empty(nb_structs, dtype=np.float32)
        mvir = np.empty(nb_structs, dtype=np.float32)
        tvir = np.empty(nb_structs, dtype=np.float32)
        cvel = np.empty(nb_structs, dtype=np.float32)
        rho0 = np.empty(nb_structs, dtype=np.float32)
        r_c = np.empty(nb_structs, dtype=np.float32)

        if star:
            nbin = np.empty(nb_structs, dtype=np.int32)
            rr = np.empty((nb_structs, 100), dtype=dt)
            rho = np.empty((nb_structs, 100), dtype=dt)

        for istrct in range(nb_structs):
            num_parts = read_record(src, 1, np.int32)

            # Discard particle IDs
            # print(num_parts)
            # read_record(src, num_parts, np.int32)
            skip_record(src, 1)  # , debug=True)

            hid[istrct] = read_record(src, 1, np.int32)
            tstep[istrct] = read_record(src, 1, np.float32)
            (
                hlvl[istrct],
                hosth[istrct],
                hostsub[istrct],
                nbsub[istrct],
                nextsub[istrct],
            ) = read_record(src, 5, np.int32)
            hmass[istrct] = read_record(src, 1, dt)
            pos[istrct] = read_record(src, 3, dt)
            vel[istrct] = read_record(src, 3, dt)
            AngMom[istrct] = read_record(src, 3, dt)
            ellipse[istrct] = read_record(src, 4, dt)
            Ek[istrct], Ep[istrct], Et[istrct] = read_record(src, 3, dt)
            spin[istrct] = read_record(src, 1, dt)
            if galaxy:
                sigma[istrct], sigma_bulge[istrct], m_bulge[istrct] = read_record(
                    src, 3, dt
                )  # skip this for DM  # skip
            rvir[istrct], mvir[istrct], tvir[istrct], cvel[istrct] = read_record(
                src, 4, dt
            )
            # cvel[istrct] = read_record(src, 1, dt)
            rho0[istrct], r_c[istrct] = read_record(src, 2, dt)

            if star:
                nbin[istrct] = read_record(src, 1, np.int32)
                rr[istrct, :] = read_record(src, 100, dt)
                rho[istrct, :] = read_record(src, 100, dt)

        cosmo = {
            "aexp": aexp,
            "omega_t": omega_t,
            "age_univ": age_univ,
        }

        hosting_info = {
            "nh": nh,
            "nsub": nsub,
            "hid": hid,
            "tstep": tstep,
            "hlvl": hlvl,
            "hosth": hosth,
            "hostsub": hostsub,
            "nbsub": nbsub,
            "nextsub": nextsub,
            "hmass": hmass,
        }
        positions = {"x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2]}
        velocities = {"vx": vel[:, 0], "vy": vel[:, 1], "vz": vel[:, 2]}
        ellipsis_fit = {
            "r": ellipse[:, 0],
            "a": ellipse[:, 1],
            "b": ellipse[:, 2],
            "c": ellipse[:, 3],
        }
        angular_momentum = {"Lx": AngMom[:, 0], "Ly": AngMom[:, 1], "Lz": AngMom[:, 2]}
        energies = {"Ek": Ek, "Ep": Ep, "Et": Et}
        virial_properties = {
            "spin": spin,
            "rvir": rvir,
            "mvir": mvir,
            "tvir": tvir,
            "cvel": cvel,
        }
        profile_fits = {"rho0": rho0, "r_c": r_c}

        treebrick = {
            "cosmology": cosmo,
            "hosting info": hosting_info,
            "positions": positions,
            "velocities": velocities,
            "angular momentum": angular_momentum,
            "smallest ellipse": ellipsis_fit,
            "energies": energies,
            "virial properties": virial_properties,
            "profile fits": profile_fits,
        }

        return treebrick


def get_tgt_partIDs(fname, tgt_hid, star=None, galaxy=True):
    if star == None:
        if "star" in fname:
            star = True
        else:
            star = False

    # print(star)

    if galaxy:
        dt = np.float64
    else:
        dt = np.float32

    with open(fname, "rb") as src:
        nbodies = read_record(src, 1, np.int32)
        # print(nbodies)
        massp = read_record(src, 1, dt)
        aexp = read_record(src, 1, dt)
        omega_t = read_record(src, 1, dt)
        age_univ = read_record(src, 1, dt)
        nh, nsub = read_record(src, 2, np.int32)
        nb_structs = nh + nsub
        # print(nbodies, massp, aexp, omega_t, age_univ, nh, nsub, nb_structs

        for istrct in range(nb_structs):
            num_parts = read_record(src, 1, np.int32)

            # print(num_parts)

            # Discard particle IDs
            read_IDs = read_record(src, num_parts, np.int32)

            assert (
                len(read_IDs) == num_parts
            ), "Number of IDs read does not match number of particles"

            hid = read_record(src, 1, np.int32)
            skip_record(src, 9)
            if galaxy:
                skip_record(src, 1)  # skip this for DM  # skip

            skip_record(src, 2)

            if star:
                skip_record(src, 3)

            if tgt_hid == hid:
                part_IDs = read_IDs

        return part_IDs


def read_zoom_brick(snap, sim, hm, sim_path=None, **kwargs):

    if sim_path == None:
        sim_path = sim.path

    # get treebrick_file
    fname_brick = os.path.join(sim_path, hm, f"tree_bricks{snap:03d}")
    print(f"Reading tree_bricks file: {fname_brick}")
    if not os.path.exists(fname_brick):
        return 0

    bricks = read_brickfile(fname_brick, **kwargs)
    convert_brick_units(bricks, sim)

    return bricks


def convert_brick_units(treebrick, sim: ramses_sim):

    # print(treebrick)
    # print(treebrick["cosmology"])
    aexp = treebrick["cosmology"]["aexp"]
    h = sim.cosmo["H0"] / 100.0
    box_len = sim.unit_l(aexp) / (ramses_pc * 1e6)  # / h  # * aexp  # proper Mpc

    # print(
    # box_len,
    # sim.cosmo["unit_l"],
    # aexp,
    # )

    # box units
    treebrick["positions"]["x"] /= box_len
    treebrick["positions"]["y"] /= box_len
    treebrick["positions"]["z"] /= box_len
    treebrick["virial properties"]["rvir"] /= box_len
    treebrick["smallest ellipse"]["r"] /= box_len

    # origin 0,0,0
    treebrick["positions"]["x"] += 0.5
    treebrick["positions"]["y"] += 0.5
    treebrick["positions"]["z"] += 0.5

    print(np.min(treebrick["positions"]["x"]),np.max(treebrick["positions"]["x"]))
    # convert masses to Msol
    treebrick["virial properties"]["mvir"] *= 1e11
    treebrick["hosting info"]["hmass"] *= 1e11


def get_halos(treebrick):
    """
    filter out subhalos (type=1) and background particles (type=2)
    """
    hosting_info = treebrick["hosting info"]
    whs = hosting_info["hlvl"] == 0
    new_treebrick = {}
    for key in treebrick.keys():
        for subkey in treebrick[key].keys():

            new_treebrick[key][subkey] = treebrick[key][subkey][whs]

    return new_treebrick


def get_halo_subs(hid, hosting_info):
    """
    return list of subhalo IDs for a given halo ID
    """
    return np.where(hosting_info["hosth"] == hid)[0]


def get_halos_properties(ids, treebrick):
    hosting_info = treebrick["hosting info"]

    # Get the indices of the halos in the treebrick
    indices = np.where(np.in1d(hosting_info["hid"], ids))[0][0]

    # make new treebrick dict with only requested haloes
    new_treebrick = {}
    for key in treebrick.keys():
        new_treebrick[key] = {}
        for subkey in treebrick[key].keys():
            if not type(treebrick[key][subkey]) in [
                np.float32,
                np.float64,
                np.int32,
                np.int64,
            ]:
                new_treebrick[key][subkey] = treebrick[key][subkey][indices]
            else:
                new_treebrick[key][subkey] = treebrick[key][subkey]

    return new_treebrick


def read_gal_stars(fname, tgt_fields=None):
    with open(fname, "rb") as src:
        hid = read_record(src, 1, np.int32)
        # print(hid)
        lvl = read_record(src, 1, np.int32)

        mass = read_record(src, 1, np.float64)
        x, y, z = read_record(src, 3, np.float64)
        # print(x, y, z)
        vx, vy, vz = read_record(src, 3, np.float64)
        Lx, Ly, Lz = read_record(src, 3, np.float64)
        nb_parts = read_record(src, 1, np.int32)

        fields = np.asarray(["pos", "vel", "mpart", "IDs", "agepart", "Zpart"])
        dims = [3, 3, 1, 1, 1, 1]
        dtypes_read = ["f8", "f8", "f8", "i4", "f8", "f8"]
        # dtypes_store = ["f8", "f8", "f4", "i4", "f4", "f4"]
        # idxs = np.arange(len(fields))

        if tgt_fields is None:
            tgt_fields = fields

        data = {}

        read_tgt_fields(
            data, tgt_fields, list(zip(fields, dims, dtypes_read)), src, nb_parts
        )

        data["hid"] = hid
        data["lvl"] = lvl
        data["mass"] = mass

        data["x"] = x
        data["y"] = y
        data["z"] = z

        data["vx"] = vx
        data["vy"] = vy
        data["vz"] = vz

        data["Lx"] = Lx
        data["Ly"] = Ly
        data["Lz"] = Lz

        data["nb_parts"] = nb_parts

        return data


def read_zoom_stars(sim, snap, gid, hm="HaloMaker_stars2_dp_rec_dust", **kwargs):

    fstar = os.path.join(sim.path, hm, f"GAL_{snap:05d}", f"gal_stars_{gid:07d}")
    stars = read_gal_stars(fstar, **kwargs)
    convert_star_units(stars, snap, sim)

    # Joe 29/10/24 - what does this do ?

    # if "family" in stars:
    #     stars_only = stars["family"] == 2
    # else:
    # if "agepart" in stars:
    #     stars_only = stars["agepart"] >= 0
    # elif "age" in stars:
    #     stars_only = stars["age"] >= 0
    # else:
    #     raise ValueError("No age or agepart in star data")

    # for k in stars.keys():
    #     if not isinstance(stars[k], np.ndarray) and not isinstance(stars[k], list):
    #         # if k=='nb_parts':
    #         continue
    #     if len(stars[k]) == len(stars_only):
    #         stars[k] = stars[k][stars_only]

    return stars


def ct_prop2time(tau, h0):
    return tau / (h0 / ramses_pc * 1e1) / (365.25 * 24.0 * 3600.0)


def ct_conf2time(tau, tau_frw, t_frw):
    # return look-back time in yr

    i = np.digitize(tau, tau_frw)

    return t_frw[i] * (tau - tau_frw[i - 1]) / (tau_frw[i] - tau_frw[i - 1]) + t_frw[
        i - 1
    ] * (tau - tau_frw[i]) / (tau_frw[i - 1] - tau_frw[i])


def ct_aexp2time(aexp, aexp_frw, t_frw):
    i = np.digitize(aexp, aexp_frw)

    return t_frw[i] * (aexp - aexp_frw[i - 1]) / (
        aexp_frw[i] - aexp_frw[i - 1]
    ) + t_frw[i - 1] * (aexp - aexp_frw[i]) / (aexp_frw[i - 1] - aexp_frw[i])


def convert_star_time(star_birthtime, sim: ramses_sim, aexp, cosmo_fname=None):

    # print(cosmo_fname)

    if cosmo_fname is None:
        if os.access(sim.path, os.W_OK):
            cosmo_fname = os.path.join(sim.path, "friedman.txt")
        else:
            fried_dir = os.path.join("/data101/jlewis", "friedman", sim.name)
            if not os.path.exists(fried_dir):
                os.makedirs(fried_dir, exist_ok=True)
            cosmo_fname = os.path.join(fried_dir, "friedman.txt")

    # setup friedman stuff
    aexp_frw, hexp_frw, tau_frw, t_frw = ct_init_cosmo(
        cosmo_fname,
        sim.cosmo["Omega_m"],
        sim.cosmo["Omega_l"],
        sim.cosmo["Omega_k"],
        sim.cosmo["H0"],
    )

    time_simu = ct_aexp2time(aexp, aexp_frw, t_frw)  # yr
    # print(time_simu)

    star_age = (time_simu - ct_conf2time(star_birthtime, tau_frw, t_frw)) * 1e-6  # Myr

    return star_age


def convert_star_units(star_dict, snap, sim: ramses_sim, cosmo_fname=None):
    aexp = sim.get_snap_exps([snap])

    # Myr = 3600 * 365 * 24 * 1e6  # s
    # Myr = 3.15576000e13  # s

    box_len = sim.unit_l(aexp) / (ramses_pc * 1e6)  # / h / aexp  # * aexp  # proper Mpc

    if "x" in star_dict:
        star_dict["x"] /= box_len
        star_dict["y"] /= box_len
        star_dict["z"] /= box_len

        star_dict["x"] += 0.5
        star_dict["y"] += 0.5
        star_dict["z"] += 0.5

    if "xpart" in star_dict:
        star_dict["xpart"] /= box_len
        star_dict["ypart"] /= box_len
        star_dict["zpart"] /= box_len

        star_dict["xpart"] += 0.5
        star_dict["ypart"] += 0.5
        star_dict["zpart"] += 0.5

    else:
        pos_keys = ["pos", "positions", "position"]
        for k in pos_keys:
            if k in star_dict:
                star_dict[k] /= box_len
                star_dict[k] += 0.5

    if "mass" in star_dict:
        star_dict["mass"] *= 1e11
    if "mpart" in star_dict:
        star_dict["mpart"] *= 1e11
    if "Zpart" in star_dict:
        star_dict["Zpart"] /= 0.02
    if "metallicity" in star_dict:
        star_dict["metallicity"] /= 0.02

    # ages to Myr
    if "agepart" in star_dict:
        star_dict["agepart"] = convert_star_time(
            star_dict["agepart"], sim, aexp, cosmo_fname
        )
    if "birth_time" in star_dict:
        star_dict["age"] = convert_star_time(
            star_dict["birth_time"], sim, aexp, cosmo_fname
        )


def get_gal_stars(
    snap, id, sim: ramses_sim, hmaker="HaloMaker_stars2_dp_rec_dust", fields=None
):
    path = sim.path

    dname = os.path.join(path, hmaker)

    fname = os.path.join(dname, f"GAL_{snap:05d}", f"gal_stars_{id:07d}")

    gal_data = read_gal_stars(fname, tgt_fields=fields)

    convert_star_units(gal_data, snap, sim)

    return gal_data
