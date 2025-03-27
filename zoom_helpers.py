import os
import numpy as np
from hagn.tree_reader import interpolate_tree_position, read_tree_rev
from .halo_maker.assoc_fcts import find_zoom_tgt_halo
from gremlin.read_sim_params import ramses_sim


def get_old_ctr(sim_path):

    if "old_ctr.txt" in os.listdir(sim_path):

        old_ctr = np.genfromtxt(os.path.join(sim_path, "old_ctr.txt"), delimiter=",")
    else:
        old_ctr = -1

    return old_ctr


def decentre_coordinates(coords, sim_path):

    old_ctr = get_old_ctr(sim_path)
    if np.any(old_ctr == -1):
        return coords
    else:
        # print(old_ctr)
        new_coords = np.copy(coords) - old_ctr + 0.5  # new centre is 0.5,0.5,0.5
        if np.any(new_coords < 0):
            new_coords[new_coords < 0] += 1
        elif np.any(new_coords > 1):
            new_coords[new_coords > 1] -= 1

        return new_coords

def recentre_coordinates(coords, sim_path,sim_ctr):

    old_ctr = get_old_ctr(sim_path)
    if np.any(old_ctr == -1):
        delta = sim_ctr-0.5
        new_coords = coords - delta    
    else:
        # print(old_ctr)
        new_coords = np.copy(coords) + old_ctr - 0.5  # new centre is 0.5,0.5,0.5
    if np.any(new_coords < 0):
        new_coords[new_coords < 0] += 1
    elif np.any(new_coords > 1):
        new_coords[new_coords > 1] -= 1

    return new_coords


# def centre_coordinates(coords, sim_ctr):

#     # ctr = np.mean(coords, axis=0)
#     delta = sim_ctr-0.5
#     new_coords = coords - delta

#     if np.any(new_coords < 0):
#         new_coords[new_coords < 0] += 1
#     elif np.any(new_coords > 1):
#         new_coords[new_coords > 1] -= 1

#     return new_coords


def find_starting_position(
    sim,
    avail_aexps,
    hagn_tree_aexps,
    hagn_tree_datas,
    hagn_tree_times,
    avail_times,
    delta_t=5,
):
    decal = -1
    found = False

    l_hagn = sim.cosmo.lcMpc

    # print(avail_aexps)
    # print(hagn_tree_aexps)

    while not found and decal > -(len(avail_aexps) - 1):

        hagn_tree_arg = np.argmin(np.abs(hagn_tree_aexps - avail_aexps[decal]))
        hagn_mass_sim_snap = hagn_tree_datas["m"][hagn_tree_arg]

        # print(np.min(np.abs(hagn_tree_times - avail_times[decal])))

        # print(avail_aexps[decal],avail_times[decal])

        hagn_ctr, hagn_rvir = interpolate_tree_position(
            avail_times[decal],
            hagn_tree_times,
            hagn_tree_datas,
            l_hagn * avail_aexps[decal],
            delta_t=delta_t,
        )

        if hagn_ctr is None:
            print("hagn interpolation failed")
            decal -= 1
            continue

        hagn_ctr = decentre_coordinates(hagn_ctr, sim.path)

        # print(hagn_ctr, hagn_rvir * 2.5)

        try:

            # print(3.0 * hagn_rvir * sim.cosmo.lcMpc, "cMpc")
            hid_start, halo_dict, hosted_gals = find_zoom_tgt_halo(
                sim,
                sim.get_closest_snap(aexp=avail_aexps[decal]),
                tgt_mass=hagn_mass_sim_snap,
                tgt_ctr=hagn_ctr,
                tgt_rad=hagn_rvir * 3.0,
                debug=True,
            )
            found = True
        except (FileNotFoundError, ValueError):
            print("no halo found")
            decal -= 1
            continue

        print(f"using sim snapshot z={1./avail_aexps[decal]-1}")

    start_aexp = avail_aexps[decal]

    if found:
        return hid_start, halo_dict, hosted_gals, found, start_aexp
    else:
        return None, None, None, found, None


def starting_hid_from_hagn(
    zstt, sim, hagn_sim, intID, avail_aexps, avail_times, ztgt=None
):
    """
    look for intID in hagn tree at zstt, find starting hid in sim halo tree at ztgt
    """

    if ztgt is None:
        # ztgt = zstt
        ztgt = 1.0 / avail_aexps[-1] - 1.0

    hagn_sim.init_cosmo()

    # print(zstt, intID, avail_aexps, avail_times)

    # get hagn tree
    hagn_tree_hids, hagn_tree_datas, hagn_tree_aexps = read_tree_rev(
        zstt,
        [intID],
        tree_type="halo",
        # [gid],
        # tree_type="gal",
        target_fields=["m", "x", "y", "z", "r"],
        sim="hagn",
    )

    # print(hagn_tree_datas["m"])

    # print(hagn_tree_datas)

    # print(list(zip(1.0 / hagn_tree_aexps - 1.0, np.log10(hagn_tree_datas["m"][0]))))

    for d in hagn_tree_datas:
        hagn_tree_datas[d] = hagn_tree_datas[d][0]

    hagn_tree_times = hagn_sim.cosmo_model.age(1.0 / hagn_tree_aexps - 1.0).value * 1e3

    lim_zeds = avail_aexps <= (1 / (ztgt + 1.0))

    hid_start, halo_dict, gal_dict, found, start_aexp = find_starting_position(
        sim,
        avail_aexps[lim_zeds],
        hagn_tree_aexps,
        hagn_tree_datas,
        hagn_tree_times,
        avail_times[lim_zeds],
    )

    return hid_start, halo_dict, gal_dict, start_aexp, found


def project_direction(axis1, axis2):
    """
    find rotation matrix so that R(axis2) = axis1

    """

    if np.all(axis1 != axis2):
        axis_rot = np.cross(axis1, axis2)
        axis_rot /= np.linalg.norm(axis_rot)

        c = np.dot(axis2, axis1)  # np.cos(ang_rot)
        s = np.linalg.norm(np.cross(axis2, [axis1]))  # np.sin(ang_rot)
        t = 1 - c

        ax, ay, az = axis_rot

        R = (
            np.array(
                [
                    ax**2 * t + c,
                    ax * ay * t - az * s,
                    ax * az * t + ay * s,
                    ay * ax * t + az * s,
                    ay**2 * t + c,
                    ay * az * t - ax * s,
                    az * ax * t - ay * s,
                    az * ay * t + ax * s,
                    az**2 * t + c,
                ]
            )
            .reshape((3, 3))
            .T
        )
    else:
        R = np.eye(3)

    return R


def check_if_in_zoom(pos, sim):

    sph = "rzoom" in sim.namelist["refine_params"]

    pos_zoom = np.transpose(
        [
            sim.namelist["refine_params"]["xzoom"],
            sim.namelist["refine_params"]["yzoom"],
            sim.namelist["refine_params"]["zzoom"],
        ]
    )

    pos_zoom = decentre_coordinates(pos_zoom, sim.path)

    if sph:

        rzoom = sim.namelist["refine_params"]["rzoom"]

        return np.linalg.norm(pos - pos_zoom, axis=1) < rzoom

    else:  # ellipsoid

        azoom = sim.namelist["refine_params"]["azoom"]
        bzoom = sim.namelist["refine_params"]["bzoom"]
        czoom = sim.namelist["refine_params"]["czoom"]

        pos = pos - pos_zoom
        r_ell = np.array(
            [
                pos[:, 0] ** 2 / azoom**2,
                pos[:, 1] ** 2 / bzoom**2,
                pos[:, 2] ** 2 / czoom**2,
            ]
        ).T

        return r_ell.sum(axis=1) < 1

def check_in_all_sims_vol(pos_to_check,cur_sim,sim_dirs):

    vol_args=None

    min_vol = np.inf

    # print(cur_sim.name)

    for isim, test_sim_dir  in enumerate(sim_dirs):
        test_sim = ramses_sim(test_sim_dir, nml="cosmo.nml")

        vol,check_vol = test_sim.get_volume()

        if vol < min_vol:
            min_vol = vol

        if hasattr(cur_sim,"zoom_ctr"):
            cur_sim_ctr = cur_sim.zoom_ctr
        else:
            test_old_ctr = get_old_ctr(test_sim.path)
            cur_sim_ctr = test_old_ctr


        # print(cur_sim_ctr,test_sim.zoom_ctr)

        if np.all(test_sim.zoom_ctr==[0.5,0.5,0.5]):
            if np.all(cur_sim_ctr==[0.5,0.5,0.5]):
                coords_for_test=np.copy(pos_to_check)
                # print('one')
                # print(coords_for_test.mean(axis=0),pos_to_check.mean(axis=0))
            else:
                coords_for_test = recentre_coordinates(pos_to_check,cur_sim.path,cur_sim_ctr)
                # print('two')
                # print(coords_for_test.mean(axis=0),pos_to_check.mean(axis=0))
        else:
            if np.all(cur_sim_ctr==[0.5,0.5,0.5]):
                
                coords_for_test = recentre_coordinates(pos_to_check,cur_sim.path,cur_sim_ctr)
                # print('thee')
                # print(cur_sim_ctr,test_sim.zoom_ctr)
                # print(recentre_coordinates(cur_sim_ctr,cur_sim.path,cur_sim_ctr),test_sim.zoom_ctr)
                # print(coords_for_test.mean(axis=0),pos_to_check.mean(axis=0))
            else:
                coords_for_test=np.copy(pos_to_check)
                # print('four')
                # print(coords_for_test.mean(axis=0),pos_to_check.mean(axis=0))

        if vol_args is None:
            vol_args = check_vol(coords_for_test)
        else:
            vol_args*=check_vol(coords_for_test)

        # print(test_sim.name,vol_args.sum())

    return vol_args,min_vol