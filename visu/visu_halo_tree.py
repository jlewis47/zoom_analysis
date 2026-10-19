import numpy as np
from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from matplotlib import patheffects as pe
import os

import argparse

from hagn.utils import get_hagn_sim

from zoom_analysis.constants import *
from zoom_analysis.zoom_helpers import starting_hid_from_hagn

from zoom_analysis.halo_maker.assoc_fcts import (
    find_zoom_tgt_gal,
    get_central_gal_for_hid,
    get_gal_props_snap,
    get_halo_props_snap,
    get_halo_assoc_file,
    get_assoc_pties_in_tree,
    smooth_props,
)


# from zoom_analysis.visu.visu_fct_bckp import (
from zoom_analysis.visu.visu_fct import (
    # make_amr_img,
    # make_amr_img_smooth,
    # make_yt_img,
    basis_from_vect,
    plot_zoom_BHs,
    plot_zoom_gals,
    plot_zoom_halos,
    # plot_stars,
    plot_fields,
)

from zoom_analysis.trees.tree_reader import read_tree_file_rev_correct_pos


vmin = vmax = None
cmap = None
cb = True
mode = "sum"
marker_color = "white"

fields_config = {
    "density": {
        "cmap": "magma",
        "vmin": 1e-26,
        "vmax": 1e-20,
    },
    "velocity": {
        "cmap": "bwr",
        "subtract_mean": True,
        "mode": "mean",  # alternative: "max"
        "vmin": -1e3,
        "vmax": 1e3,
    },
    "temperature": {
        "cmap": "plasma",
        "vmin": 1e3,
        "vmax": 1e7,
    },
    "metallicity": {
        "cmap": "YlOrRd",
        "vmin": 1e-5,
        "vmax": 1e-1,
        "mode": "mean",
    },
    "dust_bin01": {
        "vmin": 1e-8,
        "vmax": 1e-3,
    },
    "dust_bin02": {
        "vmin": 1e-8,
        "vmax": 1e-3,
    },
    "dust_bin03": {
        "vmin": 1e-8,
        "vmax": 1e-3,
    },
    "dust_bin04": {
        "vmin": 1e-8,
        "vmax": 1e-3,
    },
    "DTM": {
        "cmap": "YlGnBu",
        "vmin": 1e-4,
        "vmax": 0.5,
        "mode": "mean",
    },
    "DTMCs": {
        "cmap": "YlGnBu",
        "vmin": 1e-4,
        "vmax": 0.5,
        "mode": "mean",
    },
    "DTMCl": {
        "cmap": "YlGnBu",
        "vmin": 1e-4,
        "vmax": 0.5,
        "mode": "mean",
    },
    "DTMSil": {
        "cmap": "YlGnBu",
        "vmin": 1e-4,
        "vmax": 0.5,
        "mode": "mean",
    },
    "DTMSis": {
        "cmap": "YlGnBu",
        "vmin": 1e-4,
        "vmax": 0.5,
        "mode": "mean",
    },
    "stellar mass": {
        "cmap": "gray",  # alternative: "viridis"
        "vmin": 6e4,
        "vmax": 1e9,
        "marker_color": "r",
    },
    "stellar age": {
        "cmap": "Spectral_r",
        "vmin": 1e1,   # Myr
        "vmax": 1e3,
        "mode": "mean",
    },
    "SFR1": {
        "cmap": "hot",
        "vmin": 1,
        "vmax": 5e2,
        "mode": "mean",
    },
    "SFR10": {
        "cmap": "hot",
        "vmin": 1,
        "vmax": 5e2,
        "mode": "mean",
    },
    "SFR100": {
        "cmap": "hot",
        "vmin": 1,
        "vmax": 5e2,
        "mode": "mean",
    },
    "SFR300": {
        "cmap": "hot",
        "vmin": 1,
        "vmax": 5e2,
        "mode": "mean",
    },
    "SFR500": {
        "cmap": "hot",
        "vmin": 1,
        "vmax": 5e2,
        "mode": "mean",
    },
    "SFR1000": {
        "cmap": "hot",
        "vmin": 1,
        "vmax": 5e2,
        "mode": "mean",
    },
    "dm mass": {
        "cmap": "viridis",
    },
    "pressure": {
        "vmin": 10,   # k
        "vmax": 1e5,
    },
}


arg_parse = argparse.ArgumentParser()

arg_parse.add_argument('sim',type=str,help="path to ramses simulation")
arg_parse.add_argument('field',type=str,choices=fields_config.keys())
arg_parse.add_argument('aperture',type=float,help="aperture size in ckpc, default use r50",default=-1)

args = arg_parse.parse_args()

sim_dir = args.sim
field = args.field

sim = ramses_sim(sim_dir, nml="cosmo.nml")

name = sim.name
intID = int(name.split("id")[-1].split("_")[0])

zoom_ctr = sim.zoom_ctr

if np.all(zoom_ctr == [0.5, 0.5, 0.5]):
    centre = True
    # zero_point = [0.5, 0.5, 0.5]
else:
    centre = False
    # zero_point = [0, 0, 0]

if "refine_params" in sim.namelist:
    if "rzoom" in sim.namelist["refine_params"]:
        zoom_r = sim.namelist["refine_params"]["rzoom"]
    else:
        zoom_r = sim.namelist["refine_params"]["azoom"]

else:

    pass
#
# zoom_r =
# zoom_ctr = []
#
snaps = sim.snap_numbers
aexps = sim.get_snap_exps()
times = sim.get_snap_times()

zstt = 2.0
tgt_zed = 2.5
max_zed = 8.0

delta_t = 4  # Myr
every_snap = True  # try and interpolate between tree nodes if not found

rad_fact = 5.0  # fraction of radius to use as plot window
use_r50 = True
fixed_r_ckpc = args.aperture

plot_win_str = str(rad_fact).replace(".", "p")
if fixed_r_ckpc > 0:
    plot_win_str = f"{fixed_r_ckpc}ckpc"


overwrite = True
# overwrite = False
clean = False  # no markers for halos/bhs
# clean = True  # no markers for halos/bhs
annotate = False
gal_markers = True
halo_markers = True
subtract_mean = False
transpose = True
lower = True

only_main_stars = True

if only_main_stars:
    main_st_str = "mainStars"
else:
    main_st_str = ""

pdf = False
# zdist = 50  # ckpc
# hm = "HaloMaker_DM_dust"

hagn_sim = get_hagn_sim()


# for field in density temperature metallicity DTM DTMCs DTMCl DTMSis DTMSil "stellar mass" SFR100; do pyjob_long visu_halo_tree.py /data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380 $field; done


field_setup = fields_config[field]

if "cmap" in field_setup:
    cmap = field_setup["cmap"]

if "vmin" in field_setup:
    vmin = field_setup["vmin"]

if "vmax" in field_setup:
    vmax = field_setup["vmax"]

if "mode" in field_setup:
    mode = field_setup["mode"]

if "subtract_mean" in field_setup:
    subtract_mean = field_setup["subtract_mean"]

if "marker_color" in field_setup:
    marker_color = field_setup["marker_color"]

sim = ramses_sim(sim_dir, nml="cosmo.nml")
sim.init_cosmo()
l = sim.unit_l(sim.aexp_stt) / (ramses_pc * 1e6) / sim.aexp_stt


sim_aexps = sim.get_snap_exps()  # [::-1]
sim_times = sim.get_snap_times()  # [::-1]
sim_snaps = sim.snap_numbers  # [::-1]

assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

avail_aexps = np.intersect1d(
    sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
)

# print(avail_aexps)

tgt_zed = max(tgt_zed, 1.0 / avail_aexps[-1] - 1.0)

print(tgt_zed)

avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

print(intID)

hid_start, halos, galaxies, start_aexp, found = starting_hid_from_hagn(
    zstt, sim, hagn_sim, intID, avail_aexps, avail_times, ztgt=tgt_zed
)

start_zed = 1.0 / start_aexp - 1.0
start_snap = sim.get_closest_snap(aexp=start_aexp)

# print(hid_start, halos, galaxies, start_zed)

# print(hid_start, start_zed)
# print(halos)
# print(galaxies)
# gid = galaxies["gids"][galaxies["mass"].argmax()]
# print(start_aexp)

# gid, gal_props = find_zoom_tgt_gal(sim, tgt_zed, pure_thresh=0.9999, debug=False)

# r50 = gal_props["r50"]


# print(tree_aexps)


# print(l)

# directions = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
# directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
# directions = [
#     [0.52704628, -0.52704628, -0.66666667],
#     [-0.70710678, -0.70710678, 0.0],
#     [-0.47140452, 0.47140452, -0.74535599],
# ]
rascas_dir = [0,0,1]  # dirnb 0

# rascas_dir = [0.52704628, -0.52704628, -0.66666667] # - dir nb 1
# rascas_dir = [0.52704628, 0.52704628, 0.66666667]  # dirnb 0
# rascas_dir = [-0.527046, 0.527046, 0.666667]  # dirnb 1
# rascas_dir = [0.52704628, -0.52704628, 0.66666667]  # dirnb 3
# rascas_dir= [0.527046, 0.527046, 0.666667]
directions = list(basis_from_vect(rascas_dir))
# directions[0] = -directions[0]  # flip to align with rascas
# directions = tuple(directions)
# directions = [rascas_dir]


tree_name = os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat")
byte_file = os.path.join(sim.path, "TreeMakerDM_dust")

tree_hids, tree_datas, tree_aexps = read_tree_file_rev_correct_pos(
    tree_name,
    sim,
    start_snap,
    byte_file,
    start_zed,
    [hid_start],
    # tree_type="halo",
    tgt_fields=["m", "x", "y", "z", "r"],
    verbose=False,
    star=False,
)
tree_times = sim.cosmo_model.age(1.0 / tree_aexps - 1.0).value * 1e3  # Myr

# print(list(zip(tree_hids, tree_datas["m"])))
# print(tree_hids)

filt = tree_datas["x"][0] != -1

for key in tree_datas:
    tree_datas[key] = tree_datas[key][0][filt]
tree_hids = tree_hids[0][filt]
tree_aexps = tree_aexps[filt]
tree_times = tree_times[filt]
# print(tree_aexps)
# print(tree_times, times)
#

# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()

# print(rank)t
# rank = 0
#
# rank_snaps = np.array_split(snaps, size)[rank]
# rank_aexps = np.array_split(aexps, size)[rank]
#
# print(rank, rank_snaps)

# print(rank_snaps[100])

# print(tree_datas["r"])


rgal = "rgal"
if fixed_r_ckpc > 0:
    rgal = ""

outdir = os.path.join(
    sim_dir, "maps_own_tree", "halo", f"{plot_win_str}{rgal:s}", f"{main_st_str:s}"
)

if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

option_str = ""
if annotate and not clean:
    option_str += "_annotate"
elif clean:
    option_str += "_clean"

gal_props_tree = get_assoc_pties_in_tree(
    sim, tree_aexps, tree_hids, assoc_fields=["r50", "rmax", "mass", "pos", "host hid"]
)

r50s = gal_props_tree["r50"]
rmaxs = gal_props_tree["rmax"]
masses = gal_props_tree["mass"]
poss = gal_props_tree["pos"]
hids = gal_props_tree["host hid"]

smooth_gal_props = smooth_props(gal_props_tree)

# for snap, aexp, time in zip(snaps[:], aexps[:], times[:]):
for snap, aexp, time in zip(snaps[::-1], aexps[::-1], times[::-1]):

    # if snap != 143:
    #     continue

    if not os.path.exists(get_halo_assoc_file(sim.path, snap)):
        continue

    zed = 1.0 / aexp - 1.0
    if zed > max_zed:
        continue

    # if snap != 185:
    #     continue

    if fixed_r_ckpc <= 0:
        if use_r50:
            rstr = "r50"
        else:
            rstr = "rmax"
    else:
        rstr = ""

    fout = os.path.join(
        outdir,
        f"{field.replace(' ','_')}_{snap:04d}_{plot_win_str}{rstr}{option_str:s}.png",
    )

    if os.path.isfile(fout) and not overwrite:
        continue

    l_pMpc = l * aexp

    cur_hid = tree_hids[np.argmin(np.abs(tree_aexps - aexp))]

    cur_gid, hosted_gals = get_central_gal_for_hid(
        sim, cur_hid, snap, main_stars=only_main_stars
    )

    if cur_gid is None:
        continue

    # print(cur_hid, cur_gid, tgt_pos, "%e" % hosted_gals["mass"].max(), tgt_rad)

    # print(snap, cur_gid)
    # _, cur_gal_props = get_gal_props_snap(
    #     sim_dir, snap, cur_gid, main_stars=only_main_stars
    # )

    aexp_arg = np.argmin(np.abs(aexp - gal_props_tree["aexps"]))

    cur_r50 = smooth_gal_props["r50"][aexp_arg]

    if not use_r50:
        cur_rad = smooth_gal_props["rmax"][aexp_arg]
    else:
        cur_rad = cur_r50

    cur_mass = masses[aexp_arg]
    # cur_pos = poss[aexp_arg]
    cur_pos = gal_props_tree["pos"][aexp_arg]
    tgt_pos = cur_pos  # centre on the galaxy we found...
    # tgt_pos = cur_posfg

    # print(cur_hid, cur_gid, cur_pos, tgt_pos, "%e" % cur_mass, cur_rmax)

    # tgt_pos -= zero_point
    # do edge reflections
    # tgt_pos[tgt_pos < 0] += 1
    # tgt_pos[tgt_pos > 1] -= 1
    # print(tgt_pos)

    # print(cur_rmax, "%.1e" % cur_mass)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")

    # data_path = os.path.join(sim.path, "amr2cell", f"output_{snap:05d}/out_amr2cell")

    # if not os.path.exists(data_path):
    #     continue

    # print(snap)
    # print(tgt_pos, tgt_rad)
    # print(1.0 / aexp - 1, 1.0 / tree_aexps[tree_arg] - 1, tree_arg)
    # if snap != 308:
    #     continue

    # print(snap, tgt_pos, tgt_rad, zdist, hagn_l_pMpc)

    # rad_tgt = cur_rmax * rad_fact

    rad_tgt = cur_rad * rad_fact

    if fixed_r_ckpc > 0:
        rad_tgt = fixed_r_ckpc / (sim.cosmo.lcMpc * 1e3)

    zdist = rad_tgt / 1 * sim.cosmo.lcMpc * 1e3

    # print(tgt_pos, tgt_rad, rad_tgt, zdist)
    # print(tgt_pos - tgt_rad * 0.15, tgt_pos + tgt_rad * 0.15)

    # print(rad_tgt * 5, rad_tgt * 5 * sim.cosmo.lcMpc * 1e3)

    args = {
        "cb": cb,
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "mode": mode,
        "transpose": transpose,
        "lower": lower,
        "color": marker_color,
        "hid": int(hids[aexp_arg]),
        "subtract_mean": subtract_mean,
    }

    try:
        img = plot_fields(
            field, fig, ax, aexp, directions, tgt_pos, rad_tgt, sim, **args
        )
        ax.text(
            0.05,
            0.9,
            "z = %.2f" % zed,
            color="white",
            transform=ax.transAxes,
            path_effects=[pe.withStroke(linewidth=1, foreground="black")],
            ha="left",
            size=20,
            zorder=999,
        )
    except FileNotFoundError:
        continue

    if not clean:

        circ = Circle(
            (
                (zoom_ctr[0] - tgt_pos[0]) * sim.cosmo.lcMpc * 1e3,
                (zoom_ctr[1] - tgt_pos[1]) * sim.cosmo.lcMpc * 1e3,
            ),
            zoom_r * sim.cosmo.lcMpc * 1e3,
            fill=False,
            edgecolor=marker_color,
            lw=2,
            zorder=999,
        )
        circ = Circle(
            (
                (0) * sim.cosmo.lcMpc * 1e3,
                (0) * sim.cosmo.lcMpc * 1e3,
            ),
            cur_r50 * sim.cosmo.lcMpc * 1e3,
            fill=False,
            edgecolor=marker_color,
            lw=2,
            ls=":",
            zorder=999,
        )

        ax.add_patch(circ)

        # plot zoom galaxies
        if gal_markers:
            plot_zoom_gals(
                ax,
                snap,
                sim,
                tgt_pos,
                rad_tgt,
                zdist,
                hm="HaloMaker_stars2_dp_rec_dust",
                gal_markers=gal_markers,
                annotate=annotate,
                direction=directions[0],
                # transpose=True,
                **args,
            )

        # plot zoom hals
        if halo_markers:
            plot_zoom_halos(
                ax,
                snap,
                sim,
                tgt_pos,
                rad_tgt,
                zdist,
                hm="HaloMaker_DM_dust",
                gal_markers=gal_markers,
                annotate=annotate,
                direction=directions[0],
                # transpose=True,
                **args,
            )

        # plot zoom BHs
        try:
            plot_zoom_BHs(
                ax, snap, sim, tgt_pos, rad_tgt, zdist, direction=directions[0], **args
            )
        except (ValueError, AssertionError):
            pass

    print(f"writing {fout}")

    ax.grid(False)

    fig.savefig(
        fout,
        dpi=300,
        format="png",
    )
    if pdf:
        fig.savefig(
            fout.replace(".png", ".pdf"),
            dpi=300,
            format="pdf",
        )

    plt.close()

    # break
