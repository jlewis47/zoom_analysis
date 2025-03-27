from f90_tools.star_reader import read_part_ball_NCdust
from zoom_analysis.dust.gas_reader import gas_pos_rad, code_to_cgs
from compress_zoom.read_compressd import read_compressed_target, check_for_compressd
from gremlin.read_sim_params import ramses_sim
from zoom_analysis.halo_maker.assoc_fcts import get_halo_props_snap
from zoom_analysis.stars.star_reader import convert_star_time
from scipy.spatial import cKDTree
import numpy as np

from zoom_analysis.zoom_helpers import decentre_coordinates


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
        halo_dict, _ = get_halo_props_snap(sim.path, snap, hid=host_halo)
        rvir = halo_dict["rvir"]
        hpos = halo_dict["pos"]
        # print(hpos, tgt_pos)
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

        oob=False

        # oob = np.any(np.abs(tgt_r + (tgt_pos - hpos)) > rvir * 2)
        # print(oob, tgt_r, rvir)
        # print(np.abs(tgt_r + (tgt_pos - hpos)))

    else:
        oob = True

    # print(host_halo, oob)
    # print(tgt_pos, hpos)
    # print(np.abs(tgt_r + (tgt_pos - hpos)))
    # print(check_for_compressd(sim, snap, host_halo), host_halo)

    if (not check_for_compressd(sim, snap, host_halo)) or oob or (host_halo == None):
        # if (
        #     load_amr
        #     and snap in sim.get_snaps(full_snaps=True, mini_snaps=False)[1]
        # ):
        print("No compressed file")

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
            if 'velocity' in tgt_fields:
                tgt_fields.extend(['velocity_x', 'velocity_y', 'velocity_z'])
        if "stars" in data_types or "dm" in data_types:
            if not "pos" in tgt_fields:
                tgt_fields.append("pos")

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
