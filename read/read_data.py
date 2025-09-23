from f90_tools.star_reader import read_part_ball_NCdust
from zoom_analysis.dust.gas_reader import gas_pos_rad, code_to_cgs
from compress_zoom.read_compressd import read_compressed_target, check_for_compressd
from gremlin.read_sim_params import ramses_sim
from zoom_analysis.halo_maker.assoc_fcts import get_halo_props_snap
from zoom_analysis.stars.star_reader import convert_star_time
from scipy.spatial import cKDTree
import numpy as np
from zoom_analysis.dust.gas_reader import prepare_mach_alpha, get_alpha_vir, get_mach

# from zoom_analysis.zoom_helpers import decentre_coordinates


def read_data_ball(
    sim: ramses_sim,
    snap,
    tgt_pos,
    tgt_r,
    host_halo=None,
    gid=None,
    data_types=[],
    tgt_fields=None,
    minmax=None,
):

    if tgt_fields == None:
        tgt_fields = []

    possible_data_types = ["stars", "gas", "dm"]
    if data_types == [] or data_types == None:
        data_types = possible_data_types

    todo = False
    for dt in data_types:
        if not dt in possible_data_types:
            print(f"Data type {dt} not recognised")
        else:
            todo = True

    assert todo, "No valid data types..."

    datas = {}

    oob = False
    if host_halo != None:
        halo_dict = get_halo_props_snap(
            sim.path, snap, hids=host_halo, hosted_gals=False
        )
        hkey = f"halo_{host_halo:07d}"
        # htag = f"halo_{host_halo:07d}"
        # rvir = halo_dict[htag]["rvir"]
        # hpos = halo_dict[htag]["pos"]
        rvir = halo_dict[hkey]["rvir"]
        hpos = halo_dict[hkey]["pos"]
        # print(hpos, tg[hkey]t_pos)
        # hpos = decentre_coordinates(halo_dict["pos"], sim.path)
        # tgt_pos = decentre_coordinates(tgt_pos, sim.path)
        # print(hpos, tgt_pos)

        # out of bounds if tgt_r + (tgt_pos - hpos) > rvir*2 or - tgt_r + (tgt_pos - hpos) < rvir*2

        oob = np.any((tgt_pos + tgt_r) > (hpos + 2 * rvir)) or np.any(
            (-tgt_r + tgt_pos) < (-rvir * 2 + hpos)
        )

        if oob:
            print("oob")
            print(tgt_pos + tgt_r, hpos + 2 * rvir)
            print(-tgt_r + tgt_pos, -rvir * 2 + hpos)

            print(oob, tgt_r, 2 * rvir)

        oob = False

        # oob = np.any(np.abs(tgt_r + (tgt_pos - hpos)) > rvir * 2)
        # print(oob, tgt_r, rvir)
        # print(np.abs(tgt_r + (tgt_pos - hpos)))

    else:
        oob = True
        # pass

    # print(host_halo, oob)
    # print(tgt_pos, hpos)
    # print(np.abs(tgt_r + (tgt_pos - hpos)))
    # print(check_for_compressd(sim, snap, host_halo), host_halo, oob)

    if (not check_for_compressd(sim, snap, host_halo)) or oob or (host_halo == None):
        # if (
        #     load_amr
        #     and snap in sim.get_snaps(full_snaps=True, mini_snaps=False)[1]
        # ):
        print("No compressed file")
        if oob:
            print("out of bounds")
        if host_halo == None:
            print("undefined halo target")

        if "gas" in data_types:
            if minmax is not None:
                limits = minmax
            else:
                limits = tgt_r

            code_cells = gas_pos_rad(
                # sim, snap, [1, 6, 16, 17, 18, 19], ctr_stars, extent_stars
                sim,
                snap,
                tgt_fields,
                tgt_pos,
                limits,
            )

            datas["gas"] = code_to_cgs(sim, snap, code_cells)

            print("Read gas")

        if "stars" in data_types:

            if "age" in tgt_fields:
                tgt_fields[tgt_fields.index("age")] = "birth_time"

            datas["stars"] = read_part_ball_NCdust(
                sim,
                snap,
                tgt_pos,
                tgt_r,
                tgt_fields=tgt_fields,
                fam=2,
            )

            if "age" in tgt_fields and "birth_time" in datas["stars"]:
                datas["stars"]["age"] = convert_star_time(
                    datas["stars"]["birth_time"], sim, sim.get_snap_exps(snap)
                )
            print("Read stars")

        if "dm" in data_types:

            datas["dm"] = read_part_ball_NCdust(
                sim,
                snap,
                tgt_pos,
                tgt_r,
                tgt_fields=tgt_fields,
                fam=1,
            )
            print("Read dm")

    else:
        # print("reading compressed file")

        if type(tgt_fields) == str:
            tgt_fields = [tgt_fields]

        if "gas" in data_types:
            if not "x" in tgt_fields:
                tgt_fields.extend(["x", "y", "z"])
            if not "density" in tgt_fields:
                tgt_fields.append("density")
            if not "ilevel" in tgt_fields:
                tgt_fields.extend(["ilevel"])
            if "velocity" in tgt_fields:
                tgt_fields.extend(["velocity_x", "velocity_y", "velocity_z"])
            if "mach" in tgt_fields or "alpha_vir" in tgt_fields:
                tgt_fields.extend(
                    [
                        "velocity_x",
                        "velocity_y",
                        "velocity_z",
                        "temperature",
                        "density",
                        "ilevel",
                    ]
                )
            if (
                "DTM" in tgt_fields
                or "DTMC" in tgt_fields
                or "DTMCs" in tgt_fields
                or "DTMCl" in tgt_fields
                or "DTMSi" in tgt_fields
                or "DTMSis" in tgt_fields
                or "DTMSil" in tgt_fields
            ):
                tgt_fields.extend(
                    [
                        "density",
                        "metallicity",
                        "dust_bin01",
                        "dust_bin02",
                        "dust_bin03",
                        "dust_bin04",
                    ]
                )
        if "stars" in data_types or "dm" in data_types:
            if not "pos" in tgt_fields:
                tgt_fields.append("pos")

        tgt_fields = np.unique(tgt_fields).tolist()

        read_datas = read_compressed_target(
            sim,
            snap,
            hid=host_halo,
            gid=gid,
            data_type=data_types,
            target_fields=tgt_fields,
            # pos=tgt_pos,
            # rad=tgt_r,
        )

        assert read_datas != None, "No data read"

        if "stars" in read_datas:
            stars = read_datas["stars"]
            datas["stars"] = stars
            # # keep only stars within tgt_r of tgt_pos
            st_tree = cKDTree(stars["pos"], boxsize=1 + 1e-6)
            st_pos_args = st_tree.query_ball_point(tgt_pos, tgt_r)

            if len(st_pos_args) > 0:
                for field in stars:
                    if stars[field].shape != ():
                        stars[field] = stars[field][st_pos_args]
                datas["stars"] = stars
            else:
                print("No stars")
                datas["stars"] = None

        if "dm" in read_datas:
            dm = read_datas["dm"]
            datas["dm"] = dm
            # # keep only dm within tgt_r of tgt_pos
            dm_tree = cKDTree(dm["pos"], boxsize=1 + 1e-6)
            dm_pos_args = dm_tree.query_ball_point(tgt_pos, tgt_r)

            if len(dm_pos_args) > 0:
                for field in dm:
                    dm[field] = dm[field][dm_pos_args]

                datas["dm"] = dm
            else:
                print("No dm")
                datas["dm"] = None

        if "gas" in read_datas:
            cells = read_datas["gas"]

            if "mach" in tgt_fields or "alpha_vir" in tgt_fields:
                aexp = sim.get_snap_exps(snap)[0]
                cells["pressure"] = (
                    cells["temperature"] / sim.unit_T(aexp) * cells["density"]
                )

                Gfact_cgs, dx_cm, sigma_v2, snd_speed = prepare_mach_alpha(
                    sim, aexp, cells
                )

                print("Sound speed", snd_speed.min(), snd_speed.max(), snd_speed.mean())
                print("Sigma V2", sigma_v2.min(), sigma_v2.max(), sigma_v2.mean())

                if "mach" in tgt_fields:
                    cells["mach"] = get_mach(sigma_v2, snd_speed)
                    print(
                        "Mach number",
                        cells["mach"].min(),
                        cells["mach"].max(),
                        cells["mach"].mean(),
                    )
                if "alpha_vir" in tgt_fields:
                    cells["alpha_vir"] = get_alpha_vir(
                        sim, aexp, cells, Gfact_cgs, dx_cm, sigma_v2, snd_speed
                    )
                    print(
                        "Alpha virial",
                        cells["alpha_vir"].min(),
                        cells["alpha_vir"].max(),
                        cells["alpha_vir"].mean(),
                    )

            if "DTM" in tgt_fields:
                cells["DTM"] = (
                    cells["dust_bin01"]
                    + cells["dust_bin02"]
                    + cells["dust_bin03"]
                    + cells["dust_bin04"]
                ) / (
                    cells["dust_bin01"]
                    + cells["dust_bin02"]
                    + cells["dust_bin03"]
                    + cells["dust_bin04"]
                    + cells["metallicity"]
                )

            if "DTMC" in tgt_fields:
                cells["DTMC"] = (cells["dust_bin01"] + cells["dust_bin02"]) / (
                    cells["dust_bin01"]
                    + cells["dust_bin02"]
                    + cells["dust_bin03"]
                    + cells["dust_bin04"]
                    + cells["metallicity"]
                )

            if "DTMCs" in tgt_fields:
                cells["DTMC"] = (cells["dust_bin01"]) / (
                    cells["dust_bin01"]
                    + cells["dust_bin02"]
                    + cells["dust_bin03"]
                    + cells["dust_bin04"]
                    + cells["metallicity"]
                )

            if "DTMCl" in tgt_fields:
                cells["DTMCl"] = (+cells["dust_bin02"]) / (
                    cells["dust_bin01"]
                    + cells["dust_bin02"]
                    + cells["dust_bin03"]
                    + cells["dust_bin04"]
                    + cells["metallicity"]
                )

            if "DTMSi" in tgt_fields:
                cells["DTMSi"] = (cells["dust_bin03"] + cells["dust_bin04"]) / (
                    cells["dust_bin01"]
                    + cells["dust_bin02"]
                    + cells["dust_bin03"]
                    + cells["dust_bin04"]
                    + cells["metallicity"]
                )

            if "DTMSis" in tgt_fields:
                cells["DTMSis"] = (cells["dust_bin03"]) / (
                    cells["dust_bin01"]
                    + cells["dust_bin02"]
                    + cells["dust_bin03"]
                    + cells["dust_bin04"]
                    + cells["metallicity"]
                )

            if "DTMSil" in tgt_fields:
                cells["DTMSil"] = (cells["dust_bin04"]) / (
                    cells["dust_bin01"]
                    + cells["dust_bin02"]
                    + cells["dust_bin03"]
                    + cells["dust_bin04"]
                    + cells["metallicity"]
                )

            datas["gas"] = cells
            # keep only cells within tgt_r of tgt_pos
            cell_tree = cKDTree(
                np.transpose([cells["x"], cells["y"], cells["z"]]), boxsize=1 + 1e-6
            )
            cell_pos_args = cell_tree.query_ball_point(tgt_pos, tgt_r)

            for field in cells:
                cells[field] = np.float64(cells[field][cell_pos_args])

            if (
                type(cells["density"]) in [np.float64, np.float32]
                or len(cells["density"]) == 0
            ):
                print("No cells")
                datas["gas"] = None
            else:
                datas["gas"] = cells
            #     print(cells)
            #     datas["gas"] = code_to_cgs(sim, sim.get_snap_exps(snap)[0], cells)
            #     print(datas["gas"])

    if len(datas.keys()) == 1:
        datas = datas[list(datas.keys())[0]]

    return datas
