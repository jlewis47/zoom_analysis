import numpy as np
import h5py
import os
from gremlin.read_sim_params import ramses_sim
from zoom_analysis.halo_maker.assoc_fcts import get_gal_props_snap
from zoom_analysis.halo_maker.read_treebricks import read_zoom_stars
from zoom_analysis.trees.tree_reader import read_tree_file_rev_correct_pos


def get_snap_gal_insitu(sim, snap_num, gal_id, hm="HaloMaker_stars2_dp_rec_dust/"):

    if type(sim) is str:
        sim = ramses_sim(sim)
    elif not isinstance(sim, ramses_sim):
        raise ValueError("sim must be a path string or a ramses_sim object")

    snap_zed = 1.0 / sim.get_snap_exps(snap_num)[0] - 1.0
    snap_time = sim.get_snap_times(snap_num)[0]  # in Gyr

    # gprops = get_gal_props_snap(sim.path, snap_num)

    # hid = gprops["hid"]

    tree_name = os.path.join(sim.path, "TreeMakerstars2_dp_rec_dust", "tree_rev.dat")
    byte_file = os.path.join(sim.path, "TreeMakerstars2_dp_rec_dust")
    # tree_name = os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat")
    # byte_file = os.path.join(sim.path, "TreeMakerDM_dust")

    tree_gids, tree_datas, tree_aexps = read_tree_file_rev_correct_pos(
        tree_name, sim, snap_num, byte_file, snap_zed, tgt_ids=[gal_id]
    )
    # tree_hids, tree_datas, tree_aexps = read_tree_file_rev_correct_pos(
    #     tree_name, sim, snap_num, byte_file, snap_zed, tgt_ids=[hid]
    # )
    tree_zeds = 1.0 / tree_aexps - 1.0
    tree_times = sim.cosmo_model.age(1.0 / tree_zeds - 1.0)  # in Gyr

    start_stars = read_zoom_stars(sim, snap_num, gal_id, hm=hm)
    n_start_stars = len(start_stars["IDs"])
    start_star_ids = start_stars["IDs"]
    start_star_ages = start_stars["agepart"]  # units?

    print(start_star_ages.max(), start_star_ages.min(), start_star_ages.mean())

    mexs = np.zeros(len(tree_gids), dtype=np.float32)
    ms = np.zeros(len(tree_gids), dtype=np.float32)
    mins = np.zeros(len(tree_gids), dtype=np.float32)

    ms[0] = np.sum(start_stars["masspart"])

    tree_snaps = sim.get_closest_snap(aexp=tree_aexps)

    time_next_step = snap_time
    next_stars = start_stars

    # go UP the tree

    for i in range(1, len(tree_gids)):
        cur_snap = tree_snaps[i]
        cur_aexp = tree_aexps[i]
        cur_time = tree_times[i]
        dt = abs(time_next_step - cur_time)

        cur_gid = tree_gids[i]

        cur_tree_stars = read_zoom_stars(sim, cur_snap, cur_gid, hm=hm)

        ms[i] = np.sum(cur_tree_stars["masspart"])

        cur_tree_star_ids = cur_tree_stars["IDs"]
        # ncur_tree_stars = len(cur_tree_star_ids)
        # matched, arg_cur, arg_next = np.intersect1d(
        #     cur_tree_star_ids, next_stars["id"], return_indices=True
        # )

        # ex-situ stars are those that are not in the main branch
        # so they are the stars not found in the previous step AND formed before the previous step
        unmatched = np.setdiff1d(cur_tree_star_ids, next_stars["IDs"])

        # yound stars are those formed since the last step
        young = cur_tree_stars["agepart"] < time_next_step - cur_time

        # ex situ is unmatched and old

        mexs[i - 1] = np.sum(next_stars["masspart"][unmatched == young])
        mins[i - 1] = ms[i - 1] - mexs[i - 1]

        # old = next_stars["agepart"][arg_next] >= time_next_step - cur_time
        # old_arg = np.where(old)[0]

        # ex_situ = matched[old_arg]

        time_next_step = cur_time
        next_stars = cur_tree_stars

    return tree_gids, ms, mexs, mins


def get_sub_insitu(sim, snap_num, gid, hm="HaloMaker_stars2_dp_rec_dust/"):

    if type(sim) is str:
        sim = ramses_sim(sim)
    elif not isinstance(sim, ramses_sim):
        raise ValueError("sim must be a path string or a ramses_sim object")

    snap_zed = 1.0 / sim.get_snap_exps(snap_num)[0] - 1.0
    snap_time = sim.get_snap_times(snap_num)[0]  # in Gyr

    tree_name = os.path.join(sim.path, "TreeMakerstars2_dp_rec_dust", "tree_rev.dat")
    byte_file = os.path.join(sim.path, "TreeMakerstars2_dp_rec_dust")
    # tree_name = os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat")
    # byte_file = os.path.join(sim.path, "TreeMakerDM_dust")

    gid_props = get_gal_props_snap(sim.path, snap_num, gid)

    gid_stars = read_zoom_stars(sim, snap_num, gid, hm=hm)
    gid_star_ids = gid_stars["IDs"]
    m_gid = gid_stars["mpart"].sum()

    central_gid = gid_props["host gid"]

    tree_gids_central, tree_datas_central, tree_aexps_central = (
        read_tree_file_rev_correct_pos(
            tree_name,
            sim,
            snap_num,
            byte_file,
            snap_zed,
            tgt_ids=[central_gid],
            tgt_fields=["x", "y", "z"],
        )
    )
    tree_zeds = 1.0 / tree_aexps_central - 1.0
    tree_times = sim.cosmo_model.age(1.0 / tree_zeds - 1.0)  # in Gyr

    central_tree_snaps = sim.get_closest_snap(aexp=tree_aexps_central)

    next_stars = read_zoom_stars(sim, snap_num, central_gid, hm=hm)

    frac_not_found = 0
    fexitu = 0  # if no tree steps then no ancestor and all insitu

    for i in range(1, len(tree_gids_central)):

        cur_snap = central_tree_snaps[i]
        cur_gid = tree_gids_central[i]
        cur_time = tree_times[i]

        cur_tree_stars = read_zoom_stars(sim, cur_snap, cur_gid, hm=hm)
        cur_tree_IDs = cur_tree_stars["IDs"]

        # matched, start_arg, cur_arg = np.intersect1d(
        #     gid_stars["mpart"], cur_tree_IDs, return_indices=True
        # )

        gid_stars_exist = gid_stars["agepart"] > (snap_time - cur_time)
        frac_exist = gid_stars["mpart"][gid_stars_exist].sum() / m_gid

        if frac_exist == 0:  #
            fexitu = frac_not_found
            break

        not_found = np.in1d(gid_star_ids, cur_tree_IDs) == False

        m_not_found = gid_stars["mpart"][not_found]
        frac_not_found = m_not_found / m_gid

        next_stars = cur_tree_stars

    return fexitu
