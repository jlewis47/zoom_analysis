from ast import main, type_ignore
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os

from .read_treebricks import convert_brick_units, read_brickfile

from gremlin.read_sim_params import ramses_sim


def find_zoom_tgt(treebricks, x, y, z, rmax):
    # Find the target

    # print(treebricks["positions"])
    tree_x, tree_y, tree_z, rvir = (
        treebricks["positions"]["x"],
        treebricks["positions"]["y"],
        treebricks["positions"]["z"],
        treebricks["virial properties"]["rvir"],
    )

    # print(np.min(tree_x), np.max(tree_x))
    # print(np.min(tree_y), np.max(tree_y))
    # print(np.min(tree_z), np.max(tree_z))

    tree_pos = np.asarray([tree_x, tree_y, tree_z]).T
    tree = cKDTree(tree_pos, boxsize=1.0 + 1e-6)

    # print(x, y, z, rmax)

    # Find the target's progenitor
    args_in_ball = tree.query_ball_point([x, y, z], r=rmax)

    # print(args_in_ball)
    if len(args_in_ball) == 0:
        print("No progenitor found... trying next snapshot")
        return -1, [x, y, z]

    # Find the main branch
    ball_ids = treebricks["hosting info"]["hid"][args_in_ball]
    ball_masses = treebricks["hosting info"]["hmass"][args_in_ball]

    main_id = ball_ids[np.argmax(ball_masses)]
    wh_main = treebricks["hosting info"]["hid"] == main_id
    coords = np.r_[tree_pos[wh_main][0], rvir[wh_main]]

    # print(coords, treebricks["hosting info"]["hmass"][wh_main])

    return main_id, coords


def get_snap_main(mids, aexps, sim: ramses_sim, zoom_coords, snap_nb, det_type="dm"):
    if det_type == "dm":
        halomaker_sort = "HaloMaker_DM"
    elif det_type == "star":
        halomaker_sort = "HaloMaker_star2"
    else:
        raise ValueError("det_type must be either 'dm' or 'star'")

    fname = os.path.join(sim.path, halomaker_sort, f"tree_bricks{snap_nb:03d}")

    treebricks = read_brickfile(fname)

    # print(treebricks["positions"]["x"])

    convert_brick_units(treebricks, sim)

    # print(treebricks["positions"]["x"])

    # first unfilled index in id list
    last_id = np.min(np.where(mids == 0)[0])
    # last_id = snap_nb - 1

    aexps[last_id] = treebricks["cosmology"]["aexp"]
    mids[last_id], new_coords = find_zoom_tgt(
        treebricks, zoom_coords[0], zoom_coords[1], zoom_coords[2], zoom_coords[3]
    )

    # return np.r_[new_coords, zoom_coords[3]]
    return new_coords


def chain_main_branch(sim_path, snap_max=-1, det_type="dm"):
    sim = ramses_sim(sim_path)

    snaps = sim.snap_numbers.copy()
    snaps = snaps[snaps > 0]  # skip first step which isn't run but pre-processing

    if snap_max != -1:
        snaps = [sn for sn in snaps if sn <= snap_max]

    # start from end
    snaps = snaps[::-1]

    xzoom = sim.namelist["refine_params"]["xzoom"]
    yzoom = sim.namelist["refine_params"]["yzoom"]
    zzoom = sim.namelist["refine_params"]["zzoom"]
    rzoom = sim.namelist["refine_params"]["rzoom"]

    print("Found zoom parameters:")
    print(f"xzoom = {xzoom}")
    print(f"yzoom = {yzoom}")
    print(f"zzoom = {zzoom}")
    print(f"rzoom = {rzoom}")

    zoom_coords = [xzoom, yzoom, zzoom, rzoom * 0.5]
    new_coords = np.copy(zoom_coords)

    mids = np.zeros(len(snaps), dtype="i4")
    aexps = np.zeros(len(snaps), dtype="f4")

    for isnap, snap_nb in enumerate(snaps):
        print(f"Finding main branch for snap {snap_nb}...")
        try:
            new_coords = get_snap_main(
                mids, aexps, sim, new_coords, snap_nb, det_type=det_type
            )
        except FileNotFoundError:
            print(f"File not found for snap {snap_nb}...")

    # print(list(zip(snaps, aexps, mids)))

    # remove empty entries
    not_empty = (mids != 0) * (mids != -1)
    snaps = snaps[not_empty]
    aexps = aexps[not_empty]
    mids = mids[not_empty]

    return snaps, aexps, mids


def get_main_props(sim: ramses_sim, snaps, mids, brick_type="DM", pties=None):
    if pties == None:
        # no specified props so we take a generic set
        pties = [
            "mvir",
            "rvir",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "aexp",
            "agesim",
            "Lx",
            "Ly",
            "Lz",
            "a",
            "b",
            "c",
            "r",
        ]

    props = {}
    for prop in pties:
        props[prop] = np.zeros(len(snaps), dtype="f4")

    for isnap, snap_nb in enumerate(snaps):
        print(f"Fetching properties for snap {snap_nb}...")
        fname = f"tree_bricks{snap_nb:03d}"
        treebricks = read_brickfile(
            os.path.join(sim.path, "HaloMaker_" + brick_type, fname)
        )

        convert_brick_units(treebricks, sim)

        categories = list(treebricks.keys())

        file_ids = treebricks["hosting info"]["hid"]
        file_arg = np.where(file_ids == mids[isnap])[0][0]

        for cat in categories:
            cat_props = np.asarray(list(treebricks[cat].keys()))
            present_cats = np.in1d(cat_props, pties)

            for prop in cat_props[present_cats]:
                if type(treebricks[cat][prop]) == np.ndarray:
                    props[prop][isnap] = treebricks[cat][prop][file_arg]
                else:
                    props[prop][isnap] = treebricks[cat][prop]

    return props
