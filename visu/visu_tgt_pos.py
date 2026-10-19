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
        "vmin": 1e-28,
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
    "ilevel": {
        "mode":"max",
        "weights":None,
        "log":False
    },
}


arg_parse = argparse.ArgumentParser()

arg_parse.add_argument('sim',type=str,help="path to ramses simulation")
arg_parse.add_argument('field',type=str,choices=fields_config.keys())
arg_parse.add_argument('ctr',type=float,nargs=3,help='position in code units (0<x<1)')
arg_parse.add_argument('snap',type=int,help='target snap number')
arg_parse.add_argument('aperture',type=float,help="aperture size in ckpc, default use r50",default=-1)
arg_parse.add_argument('--zdist',type=float,help="depth size in ckpc",default=-1)

args = arg_parse.parse_args()

sim_dir = args.sim
ctr = args.ctr
field = args.field
tgt_snap=args.snap
zdist = args.zdist

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
tgt_zed = 2.0
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


# for field in density temperature metallicity DTM DTMCs DTMCl DTMSis DTMSil "stellar mass" SFR100; do pyjob_long visu_halo_tree.py /data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380 $field; done
field_setup = fields_config[field]

sim = ramses_sim(sim_dir, nml="cosmo.nml")
sim.init_cosmo()
l = sim.unit_l(sim.aexp_stt) / (ramses_pc * 1e6) / sim.aexp_stt


sim_aexps = sim.get_snap_exps()  # [::-1]
sim_times = sim.get_snap_times()  # [::-1]
sim_snaps = sim.snap_numbers  # [::-1]

assert tgt_snap in sim_snaps, f"didn't find requested snap ({tgt_snap:d}) in list of snaps"


# rascas_dir = [0.52704628, -0.52704628, -0.66666667] # - dir nb 1
# rascas_dir = [0.52704628, 0.52704628, 0.66666667]  # dirnb 0
# rascas_dir = [-0.527046, 0.527046, 0.666667]  # dirnb 1
# rascas_dir = [0.52704628, -0.52704628, 0.66666667]  # dirnb 3
# rascas_dir= [0.527046, 0.527046, 0.666667]
# rascas_dir=[0,0,1]
# rascas_dir=[1,0,0]
# directions = list(basis_from_vect(rascas_dir))
directions=[[1, 0, 0],[1, 0, 0],[1, 0, 0]]
# directions[0] = -directions[0]  # flip to align with rascas
# directions = tuple(directions)
# directions = [rascas_dir]



outdir = os.path.join(
    sim_dir, "maps_own_tree", "pos", f"{plot_win_str}", f"{main_st_str:s}"
)

if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

option_str = ""
if annotate and not clean:
    option_str += "_annotate"
elif clean:
    option_str += "_clean"

aexp = sim.get_closest_snap(tgt_snap)
zed = 1.0 / aexp - 1.0

rstr=""

fout = os.path.join(
    outdir,
    f"{field.replace(' ','_')}_{tgt_snap:04d}_{plot_win_str}{rstr}{option_str:s}.png",
)

l_pMpc = l * aexp


fig, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")



rad_tgt = fixed_r_ckpc / (sim.cosmo.lcMpc * 1e3)
if args.zdist==-1:
    zdist = rad_tgt
else: 
    zdist /= (sim.cosmo.lcMpc * 1e3)

args = {
    "cb": cb,
    "cmap": cmap,
    "vmin": vmin,
    "vmax": vmax,
    "mode": mode,
    "transpose": transpose,
    "lower": lower,
    "color": marker_color,
    # "hid": int(hids[aexp_arg]),
    "subtract_mean": subtract_mean,
}

for arg in field_setup:
    args[arg] = field_setup[arg]

print(ctr)
print(fixed_r_ckpc, rad_tgt)

img = plot_fields(
    field, fig, ax, aexp, directions, ctr, rad_tgt, sim, **args
)

# ax.text(
#     0.05,
#     0.9,
#     "z = %.2f" % zed,
#     color="white",
#     transform=ax.transAxes,
#     path_effects=[pe.withStroke(linewidth=1, foreground="black")],
#     ha="left",
#     size=20,
#     zorder=999,
# )

if not clean:

    circ = Circle(
        (
            np.dot(directions[1],(zoom_ctr - ctr)) * sim.cosmo.lcMpc * 1e3,
            np.dot(directions[2],(zoom_ctr - ctr)) * sim.cosmo.lcMpc * 1e3,
        ),
        zoom_r * sim.cosmo.lcMpc * 1e3,
        fill=False,
        edgecolor=marker_color,
        lw=2,
        zorder=999,
    )

    ax.add_patch(circ)

    # plot zoom galaxies
    if gal_markers:
        plot_zoom_gals(
            ax,
            tgt_snap,
            sim,
            ctr,
            rad_tgt,
            zdist,
            hm="HaloMaker_stars2_dp_rec_dust",
            gal_markers=gal_markers,
            annotate=annotate,
            direction=directions[0],
            # transpose=transpose,
            # **args,
        )

    # plot zoom hals
    if halo_markers:
        plot_zoom_halos(
            ax,
            tgt_snap,
            sim,
            ctr,
            rad_tgt,
            zdist,
            hm="HaloMaker_DM_dust",
            gal_markers=gal_markers,
            annotate=annotate,
            direction=directions[0],
            # transpose=transpose,
            # **args,
        )

    # plot zoom BHs
    try:
        plot_zoom_BHs(
            ax, tgt_snap, sim, ctr, rad_tgt, zdist, direction=directions[0], **args
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
