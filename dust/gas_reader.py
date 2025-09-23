import numpy as np
from gremlin.read_sim_params import ramses_sim
from zoom_analysis.constants import Bmann_cgs, ramses_pc
from zoom_analysis.visu.read_amr2cell import read_amrcells


def gas_pos_rad(sim: ramses_sim, snap, field, pos, rad, verbose=False):
    """
    sim: ramses_sim object from GREMLIN
    snap is the snapshot number
    field is a list of strings or integers of the fields to extract
    pos is the position of the center of the ball (code units)
    extract gas cell info from cells within rad of pos (code units)
    rad is either a scalar or a 2,3 array containing [min, max]
    """

    # print(snap, sim.get_snaps(full_snaps=True, mini_snaps=False))

    # check is full snap not mini
    assert (
        snap in sim.get_snaps(full_snaps=True, mini_snaps=False)[1]
    ), "snap not in full snap list, only mini snap data around galaxies and halos available"

    # print(field)

    if type(field) == str:
        field = [field]
    elif field is None:
        field = np.arange(1, len(sim.hydro) + 1, dtype=int)

    radial = False
    try:
        len(rad)
    except TypeError:
        radial = True

    if radial:
        mins = pos - rad
        maxs = pos + rad
    else:
        mins = rad[0]
        maxs = rad[1]

    idx = []
    for f in field:
        if type(f) == str:

            if f == "temperature":
                idx.extend(
                    [
                        i
                        for i in range(1, len(sim.hydro) + 1)
                        if sim.hydro[str(i)] in ["density", "pressure"]
                    ]
                )

            elif f == "metallicity":
                idx.extend(
                    [
                        i
                        for i in range(1, len(sim.hydro) + 1)
                        if sim.hydro[str(i)] in ["density", "metallicity"]
                    ]
                )

            elif f in ["DTM", "DTMC", "DTMSi", "DTMCs", "DTMCl", "DTMSis", "DTMSil"]:
                idx.extend(
                    [
                        i
                        for i in range(1, len(sim.hydro) + 1)
                        if sim.hydro[str(i)]
                        in [
                            "density",
                            "metallicity",
                            "dust_bin01",
                            "dust_bin02",
                            "dust_bin03",
                            "dust_bin04",
                        ]
                    ]
                )
            elif f == "velocity":
                idx.extend(
                    [
                        i
                        for i in range(1, len(sim.hydro) + 1)
                        if sim.hydro[str(i)]
                        in [
                            "density",
                            "velocity_x",
                            "velocity_y",
                            "velocity_z",
                        ]
                    ]
                )
            elif f == "mach":
                idx.extend(
                    [
                        i
                        for i in range(1, len(sim.hydro) + 1)
                        if sim.hydro[str(i)]
                        in [
                            "density",
                            "pressure",
                            "velocity_x",
                            "velocity_y",
                            "velocity_z",
                        ]
                    ]
                )
                field.append("temperature")

            elif f == "alpha_vir":
                idx.extend(
                    [
                        i
                        for i in range(1, len(sim.hydro) + 1)
                        if sim.hydro[str(i)]
                        in [
                            "density",
                            "velocity_x",
                            "velocity_y",
                            "velocity_z",
                        ]
                    ]
                )

            else:

                # for i in range(1, len(sim.hydro) + 1):
                # print(sim.hydro[str(i)], f, sim.hydro[str(i)] == f)
                idx.extend(
                    [i for i in range(1, len(sim.hydro) + 1) if sim.hydro[str(i)] == f]
                )
        elif type(f) in [int, np.int64, np.int32]:
            idx.append(f)

    idx = np.unique(idx)

    if verbose:
        print(f"Reading indexes are:{idx}")

    cells = read_amrcells(sim, snap, np.ravel([mins, maxs]), idx, debug=verbose)

    if "temperature" in field:
        cells["temperature"] = []
    if "DTM" in field:
        cells["DTM"] = []
    if "DTMC" in field:
        cells["DTMC"] = []
    if "DTMSi" in field:
        cells["DTMSi"] = []
    if "DTMCs" in field:
        cells["DTMCs"] = []
    if "DTMCl" in field:
        cells["DTMCl"] = []
    if "DTMSis" in field:
        cells["DTMSis"] = []
    if "DTMSil" in field:
        cells["DTMSil"] = []
    if "mach" in field:
        cells["mach"] = []
    if "alpha_vir" in field:
        cells["alpha_vir"] = []

    return cells


# use mini_snaps_only


# use full snaps and mini snaps


def get_neighbor_inds(cells):

    ilvls = cells["ilevel"]
    x = cells["x"]
    y = cells["y"]
    z = cells["z"]
    neighbor_inds = np.zeros((len(x), 6), dtype=int)
    # left, right, bottom, top, front, back

    for i in range(len(x)):

        if np.any(np.isnan(neighbor_inds[i])):
            dx = 1.0 / 2 ** ilvls[i]
            ilvl = ilvls[i]
            this_lvl = np.where(ilvls == ilvl)[0]

            left = this_lvl[np.argmin(x[this_lvl] - x[i])]
            ok_left = np.abs(x[i] - x[this_lvl][left]) < dx

            # print(x[this_lvl], dx, x[i], x[i] - dx, x[this_lvl][left], x[i]-x[this_lvl][left], ok_left)

            right = this_lvl[np.argmin(x[this_lvl] - x[i])]
            ok_right = np.abs(x[i] - x[this_lvl][right]) < dx

            bottom = this_lvl[np.argmin(y[this_lvl] - y[i])]
            ok_bottom = np.abs(y[i] - y[this_lvl][bottom]) < dx

            top = this_lvl[np.argmin(y[this_lvl] - y[i])]
            ok_top = np.abs(y[i] - y[this_lvl][top]) < dx

            front = this_lvl[np.argmin(z[this_lvl] - z[i])]
            ok_front = np.abs(z[i] - z[this_lvl][front]) < dx

            back = this_lvl[np.argmin(z[this_lvl] - z[i])]
            ok_back = np.abs(z[i] - z[this_lvl][back]) < dx

            # print(f"i: {i}, left: {left}, right: {right}, bottom: {bottom}, top: {top}, front: {front}, back: {back}")

            neighbor_inds[i] = [
                this_lvl[left] * ok_left,
                this_lvl[right] * ok_right,
                this_lvl[bottom] * ok_bottom,
                this_lvl[top] * ok_top,
                this_lvl[front] * ok_front,
                this_lvl[back] * ok_back,
            ]

            if ok_left:
                neighbor_inds[left, 1] = i
            if ok_right:
                neighbor_inds[right, 0] = i
            if ok_bottom:
                neighbor_inds[bottom, 3] = i
            if ok_top:
                neighbor_inds[top, 2] = i
            if ok_front:
                neighbor_inds[front, 5] = i
            if ok_back:
                neighbor_inds[back, 4] = i

    return neighbor_inds


def code_to_cgs(sim, aexp, amrdata):

    out_data = {}

    field_list = list(amrdata.keys())

    if "mach" in field_list or "alpha_vir" in field_list:
        Gfact_cgs, dx_cm, sigma_v2, snd_speed = prepare_mach_alpha(
            sim, aexp, amrdata, field_list
        )

    if "temperature" in field_list:
        arg_temp = field_list.index("temperature")
        a = field_list[arg_temp]
        b = field_list[0]

        field_list[arg_temp] = b
        field_list[0] = a

    print(amrdata.keys())

    pressure_key = "pressure"
    if not "pressure" in amrdata.keys() and "thermal_pressure" in amrdata.keys():
        pressure_key = "thermal_pressure"

    for field in field_list:

        if field == "temperature":
            unit_T = sim.unit_T(aexp)
            # print(unit_T)
            # print((amrdata[pressure_key] / amrdata["density"]) * unit_T)
            out_data[field] = (amrdata[pressure_key] / amrdata["density"]) * unit_T
        elif field in ["DTM", "DTMC", "DTMSi", "DTMCs", "DTMCl", "DTMSis", "DTMSil"]:
            dust_tot = (
                amrdata["dust_bin01"]
                + amrdata["dust_bin02"]
                + amrdata["dust_bin03"]
                + amrdata["dust_bin04"]
            )
            if field == "DTM":
                out_data[field] = dust_tot / (amrdata["metallicity"] + dust_tot)
            elif field == "DTMC":
                out_data[field] = (amrdata["dust_bin01"] + amrdata["dust_bin02"]) / (
                    amrdata["metallicity"] + dust_tot
                )
            elif field == "DTMCs":
                out_data[field] = (amrdata["dust_bin01"]) / (
                    amrdata["metallicity"] + dust_tot
                )
            elif field == "DTMCl":
                out_data[field] = (amrdata["dust_bin02"]) / (
                    amrdata["metallicity"] + dust_tot
                )
            elif field == "DTMSi":
                out_data[field] = (amrdata["dust_bin03"] + amrdata["dust_bin04"]) / (
                    amrdata["metallicity"] + dust_tot
                )
            elif field == "DTMSis":
                out_data[field] = (amrdata["dust_bin03"]) / (
                    amrdata["metallicity"] + dust_tot
                )
            elif field == "DTMSil":
                out_data[field] = (amrdata["dust_bin04"]) / (
                    amrdata["metallicity"] + dust_tot
                )

        elif field in [
            "density",
            # "dust_bin01",
            # "dust_bin02",
            # "dust_bin03",
            # "dust_bin04",
        ]:
            out_data[field] = amrdata[field] * sim.unit_d(aexp)
        elif field in ["velocity_x", "velocity_y", "velocity_z"]:
            out_data[field] = amrdata[field] * sim.unit_v(aexp)
        elif field in ["mach"]:
            out_data[field] = get_mach(sigma_v2, snd_speed)
        elif field in ["alpha_vir"]:
            out_data[field] = get_alpha_vir(
                sim, aexp, amrdata, Gfact_cgs, dx_cm, sigma_v2, snd_speed
            )

        else:
            out_data[field] = amrdata[field]
        # elif field == "pressure":
        #     unit_m =
        #     out_data[field] = amrdata[pressure_key] * sim.unit_ [into g.cm^-1.s^-2]

    return out_data


def get_mach(sigma_v2, snd_speed):
    return np.sqrt(sigma_v2) / snd_speed


def get_alpha_vir(sim, aexp, amrdata, Gfact_cgs, dx_cm, sigma_v2, snd_speed):
    return (
        5.0
        * (sigma_v2 + snd_speed**2)
        * sim.unit_v(aexp) ** 2
        / (amrdata["density"] * sim.unit_d(aexp) * Gfact_cgs * np.pi * dx_cm**2)
    )


def prepare_mach_alpha(sim, aexp, amrdata):
    Gfact_cgs = 6.67e-11 * 1e6 / 1e3  # cm^3 / g / s^2

    dx_cm = sim.unit_l(aexp) * ramses_pc / 2 ** amrdata["ilevel"]  # cm

    neighbor_inds = get_neighbor_inds(amrdata)

    ok_left = (neighbor_inds[:, 0] != 0) * (neighbor_inds[:, 1] != 0)
    ok_bottom = (neighbor_inds[:, 2] != 0) * (neighbor_inds[:, 3] != 0)
    ok_front = (neighbor_inds[:, 4] != 0) * (neighbor_inds[:, 5] != 0)

    d = amrdata["density"]

    d_left = np.where(neighbor_inds[:, 0] == 0, 0, d[neighbor_inds[:, 0]])
    d_right = np.where(neighbor_inds[:, 1] == 0, d, d[neighbor_inds[:, 1]])

    d_bottom = np.where(neighbor_inds[:, 2] == 0, 0, d[neighbor_inds[:, 2]])
    d_top = np.where(neighbor_inds[:, 3] == 0, d, d[neighbor_inds[:, 3]])

    d_front = np.where(neighbor_inds[:, 4] == 0, 0, d[neighbor_inds[:, 4]])
    d_back = np.where(neighbor_inds[:, 5] == 0, d, d[neighbor_inds[:, 5]])

    print(d, d_left, d_right, d_bottom, d_top, d_front, d_back)

    vel_x = amrdata["velocity_x"]
    vel_y = amrdata["velocity_y"]
    vel_z = amrdata["velocity_z"]

    vel_x_left = np.where(neighbor_inds[:, 0] == 0, 0, vel_x[neighbor_inds[:, 0]])
    vel_x_right = np.where(neighbor_inds[:, 1] == 0, vel_x, vel_x[neighbor_inds[:, 1]])
    vel_x_bottom = np.where(neighbor_inds[:, 2] == 0, 0, vel_x[neighbor_inds[:, 2]])
    vel_x_top = np.where(neighbor_inds[:, 3] == 0, vel_x, vel_x[neighbor_inds[:, 3]])
    vel_x_front = np.where(neighbor_inds[:, 4] == 0, 0, vel_x[neighbor_inds[:, 4]])
    vel_x_back = np.where(neighbor_inds[:, 5] == 0, vel_x, vel_x[neighbor_inds[:, 5]])

    vel_y_left = np.where(neighbor_inds[:, 0] == 0, 0, vel_y[neighbor_inds[:, 0]])
    vel_y_right = np.where(neighbor_inds[:, 1] == 0, vel_y, vel_y[neighbor_inds[:, 1]])
    vel_y_bottom = np.where(neighbor_inds[:, 2] == 0, 0, vel_y[neighbor_inds[:, 2]])
    vel_y_top = np.where(neighbor_inds[:, 3] == 0, vel_y, vel_y[neighbor_inds[:, 3]])
    vel_y_front = np.where(neighbor_inds[:, 4] == 0, 0, vel_y[neighbor_inds[:, 4]])
    vel_y_back = np.where(neighbor_inds[:, 5] == 0, vel_y, vel_y[neighbor_inds[:, 5]])

    vel_z_left = np.where(neighbor_inds[:, 0] == 0, 0, vel_z[neighbor_inds[:, 0]])
    vel_z_right = np.where(neighbor_inds[:, 1] == 0, vel_z, vel_z[neighbor_inds[:, 1]])
    vel_z_bottom = np.where(neighbor_inds[:, 2] == 0, 0, vel_z[neighbor_inds[:, 2]])
    vel_z_top = np.where(neighbor_inds[:, 3] == 0, vel_z, vel_z[neighbor_inds[:, 3]])
    vel_z_front = np.where(neighbor_inds[:, 4] == 0, 0, vel_z[neighbor_inds[:, 4]])
    vel_z_back = np.where(neighbor_inds[:, 5] == 0, vel_z, vel_z[neighbor_inds[:, 5]])

    print(
        vel_x, vel_x_left, vel_x_right, vel_x_bottom, vel_x_top, vel_x_front, vel_x_back
    )

    # velocity gradient tensor terms **2

    # we average the velocity of the cell and its neighbors to get velocities at interfaces
    # weighting by densities

    vx_x = ok_left * ((d * vel_x + d_left * vel_x_left) / (d + d_left)) - (
        d * vel_x + d_right * vel_x_right
    ) / (d + d_right)
    vx_y = ok_bottom * ((d * vel_x + d_bottom * vel_x_bottom)) / (d + d_bottom) - (
        d * vel_x + d_top * vel_x_top
    ) / (d + d_top)
    vx_z = ok_front * ((d * vel_x + d_front * vel_x_front) / (d + d_front)) - (
        d * vel_x + d_back * vel_x_back
    ) / (d + d_back)

    vy_x = ok_left * ((d * vel_y + d_left * vel_y_left) / (d + d_left)) - (
        d * vel_y + d_right * vel_y_right
    ) / (d + d_right)
    vy_y = ok_bottom * ((d * vel_y + d_bottom * vel_y_bottom) / (d + d_bottom)) - (
        d * vel_y + d_top * vel_y_top
    ) / (d + d_top)
    vy_z = ok_front * ((d * vel_y + d_front * vel_y_front) / (d + d_front)) - (
        d * vel_y + d_back * vel_y_back
    ) / (d + d_back)

    vz_x = ok_left * ((d * vel_z + d_left * vel_z_left) / (d + d_left)) - (
        d * vel_z + d_right * vel_z_right
    ) / (d + d_right)
    vz_y = ok_bottom * ((d * vel_z + d_bottom * vel_z_bottom) / (d + d_bottom)) - (
        d * vel_z + d_top * vel_z_top
    ) / (d + d_top)
    vz_z = ok_front * ((d * vel_z + d_front * vel_z_front) / (d + d_front)) - (
        d * vel_z + d_back * vel_z_back
    ) / (d + d_back)

    print(vx_x, vx_y, vx_z, vy_x, vy_y, vy_z, vz_x, vz_y, vz_z)

    sigma_v2 = (
        vx_x**2
        + vx_y**2
        + vx_z**2
        + vy_x**2
        + vy_y**2
        + vy_z**2
        + vz_x**2
        + vz_y**2
        + vz_z**2
    )

    gamma = sim.namelist["hydro_params"]["gamma"]
    snd_speed = np.sqrt(amrdata["pressure"] / d * gamma)

    return Gfact_cgs, dx_cm, sigma_v2, snd_speed
