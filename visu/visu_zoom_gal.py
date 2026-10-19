import numpy as np
from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt

# from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Ellipse
from zoom_analysis.constants import *
from zoom_analysis.halo_maker.assoc_fcts import find_zoom_tgt_halo

# from mpi4py import MPI

# import matplotlib.patheffects as pe
import os

# from scipy.spatial import KDTree, cKDTree
# from scipy.stats import binned_statistic_2d

# from zoom_analysis.sinks.sink_reader import (
#     read_sink_bin,
#     snap_to_coarse_step,
#     convert_sink_units,
# )


# from zoom_analysis.trees.tree_reader import read_tree_rev

from zoom_analysis.rascas.rascas_mock import read_zoom_brick
from zoom_analysis.visu.visu_fct import (
    # make_amr_img,
    make_amr_img_smooth,
    # make_yt_img,
    plot_zoom_BHs,
    plot_zoom_gals,
)

from hagn.tree_reader import read_tree_rev
# from hagn.io import read_hagn_snap_brickfile, read_hagn_sink_bin
from hagn.utils import get_hagn_sim

import mpi4py.MPI as MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# # sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479_old"
# # sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_old"
# # sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479"
# # sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704"
# # sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_eagn_T0p15"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099_inter"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id180130_model5"
# sim_dir =     "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id180130_256"
sim_dir =     "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id180130_model6_eps0p05"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_evenlesscoarse"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_leastcoarse"

# hid = 74099
# hid = 242704
# hid = 147479

# hids = [287012, 1589, 194228, 13310, 37686, 68373, 33051, 292074, 242704]

# hids = quenched_gals = np.genfromtxt(
#     f"ssfrs_pick_quenched_197_{sfr100:s}.txt",
#     names=True,
#     delimiter=",",
# )['hids']
# tgt_z = 2.0
tgt_z = 6.58

# halos with galaxies that are not quite quenched at z=2 but are for 1Gyr by z=1.6
# hids = [242756, 37686, 68373, 22851, 13310, 142760, 237150]
# gids = [2]
# gids = [1]
sim = ramses_sim(sim_dir, nml="cosmo.nml")
snap_z = sim.get_closest_snap(zed=tgt_z)
tgt_hid, halo_dict, hosted_gals = find_zoom_tgt_halo(sim, snap_z)

gids = [int(hosted_gals["gids"][np.argmax(hosted_gals["mass"])])]
# print(gids)

# tgt_snap = 294
tgt_snap = sim.get_closest_snap(zed=tgt_z)
#
# zoom_r =
# zoom_ctr = []
#
snaps = sim.snap_numbers
aexps = sim.get_snap_exps(param_save=False)
zeds = 1.0 / aexps - 1.0
times = sim.get_snap_times(param_save=False)


# tgt_zed = 2
# zmax = 3
# tgt_snap = sim.get_closest_snap(zed=tgt_zed)

wh = snaps == tgt_snap
overwrite = True
rad_fact = 0.35  # fraction of radius to use as plot window

clean = True  # no markers for halos/bhs
annotate = False
gal_markers = True

# field = "metal density"
field = "density"
# field = "temperature"
# vmin = 10  # k
# vmax = 1e5
# field = "density"
# vmin = None
# vmax = None

vmin=1e-26
vmax=1e-21
# tree_hids, tree_datas, tree_aexps = read_tree_rev(
#     tree_path, tgt_zed, tree_type="halo", target_fields=["m", "x", "y", "z", "r"]
# )

sim_dir = sim.path
# for now follow hagn halo


snap = snaps[wh][0]
aexp = aexps[wh][0]
time = times[wh][0]

print(snap, aexp, time)

hm = "HaloMaker_stars2_dp_rec_dust/"
brick = read_zoom_brick(snap, sim, hm)

brick_gids = brick["hosting info"]["hid"]


option_str = ""
if annotate and not clean:
    option_str += "_annotate"
elif clean:
    option_str += "_clean"

for gid in gids:

    wh_gid = brick_gids == gid
    rgal = brick["smallest ellipse"]["r"][wh_gid][0]
    # rgal = brick["virial properties"]["rvir"][wh_gid]
    gal_pos = np.asarray(
        [
            brick["positions"]["x"][wh_gid][0],
            brick["positions"]["y"][wh_gid][0],
            brick["positions"]["z"][wh_gid][0],
        ]
    )

    print(gal_pos, rgal)

    outdir = os.path.join(sim_dir, "maps", "gals", f"{snap:d}", f"{gid:d}")

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    plot_win_str = str(rad_fact).replace(".", "p")
    fname = f"{field[:4]}_{snap}_{plot_win_str}rvir{option_str:s}.png"
    outf = os.path.join(outdir, fname)

    if os.path.isfile(outf) and not overwrite:
        continue

    print(f"rank {rank} is handling snap {snap}")

    l_pMpc = sim.cosmo.lcMpc * aexp

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    zdist = rad_fact * rgal * sim.cosmo.lcMpc * 1e3

    print(snap, gal_pos, rad_fact, zdist, l_pMpc)
    # print(tgt_pos - rad_fact * 0.15, tgt_pos + rad_fact * 0.15)

    rad_tgt = rad_fact * rgal

    make_amr_img_smooth(
        fig,
        ax,
        field,
        snap,
        sim,
        gal_pos,
        rad_tgt,
        # zdist=-1,
        zdist=zdist,
        debug=False,
        vmin=vmin,
        vmax=vmax,
        hid=tgt_hid,
        # vmax=1e-22,
    )

    if not clean:

        point_in_pxs = fig.dpi / 72.0
        sm_cell_size_kpc = sim.cosmo.lcMpc * 1e3 / 2**sim.levelmax

        # plot hagn galaxies
        plot_zoom_gals(
            ax,
            snap,
            sim,
            gal_pos,
            rad_tgt,
            zdist,
            hm=hm,
            brick_fct=read_zoom_brick,
            annotate=annotate,
        )

        # plot hagn BHs
        plot_zoom_BHs(ax, snap, sim, gal_pos, rad_tgt, zdist)

        zoom_ctr = sim.zoom_ctr

        if "azoom" in sim.namelist["refine_params"]:

            zoom_r = sim.namelist["refine_params"]["azoom"]
            # zoom_b = sim.namelist['refine_params']['bmax']
            # zoom_c = sim.namelist['refine_params']['cmax']

        #     circ = Ellipse(
        #         (
        #             (zoom_ctr[0] - gal_pos[0]) * sim.cosmo.lcMpc * 1e3,
        #             (zoom_ctr[1] - gal_pos[1]) * sim.cosmo.lcMpc * 1e3,
        #         ),
        #         zoom_r * sim.cosmo.lcMpc * 1e3,
        #         fill=False,
        #         edgecolor="r",
        #         lw=2,
        #         zorder=999,
        # )
        elif "rzoom" in sim.namelist["refine_params"]:

            zoom_r = sim.namelist["refine_params"]["rzoom"]

        # print(zoom_ctr, gal_pos)

        ax.scatter(
            (zoom_ctr[0] - gal_pos[0]) * sim.cosmo.lcMpc * 1e3,
            (zoom_ctr[1] - gal_pos[1]) * sim.cosmo.lcMpc * 1e3,
            s=200,
            c="r",
            marker="+",
            label="Zoom center",
            zorder=999,
            lw=1,
        )

        circ = Circle(
            (
                (zoom_ctr[0] - gal_pos[0]) * sim.cosmo.lcMpc * 1e3,
                (zoom_ctr[1] - gal_pos[1]) * sim.cosmo.lcMpc * 1e3,
            ),
            zoom_r * sim.cosmo.lcMpc * 1e3,
            fill=False,
            edgecolor="r",
            lw=2,
            zorder=999,
        )

    print(f"writing file {outf:s}")
    fig.savefig(
        outf,
        dpi=300,
        format="png",
    )
    fig.savefig(
        outf.replace(".png", ".pdf"),
        dpi=300,
        format="pdf",
    )

    plt.close()

    # break
