import numpy as np
from gremlin.read_sim_params import ramses_sim
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

            elif f == "DTM":
                idx.extend(
                    [
                        i
                        for i in range(1, len(sim.hydro) + 1)
                        if sim.hydro[str(i)]
                        in [
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

    return cells


# use mini_snaps_only


# use full snaps and mini snaps


def code_to_cgs(sim, aexp, amrdata):

    out_data = {}

    for field in amrdata.keys():

        if field == "temperature":
            unit_T = sim.unit_T(aexp)
            # print(unit_T)
            # print((amrdata["pressure"] / amrdata["density"]) * unit_T)
            out_data[field] = (amrdata["pressure"] / amrdata["density"]) * unit_T
        elif field == "DTM":
            dust_tot = (
                amrdata["dust_bin01"]
                + amrdata["dust_bin02"]
                + amrdata["dust_bin03"]
                + amrdata["dust_bin04"]
            )
            out_data[field] = dust_tot / (amrdata["metallicity"] + dust_tot)
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
        else:
            out_data[field] = amrdata[field]
        # elif field == "pressure":
        #     unit_m =
        #     out_data[field] = amrdata[field] * sim.unit_ [into g.cm^-1.s^-2]

    return out_data
