from turtle import fillcolor
from f90nml import patch
from matplotlib.font_manager import font_family_aliases
import numpy as np
from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Ellipse, Rectangle
from zoom_analysis.constants import *

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

from zoom_analysis.halo_maker.assoc_fcts import find_zoom_tgt_gal, get_gal_props_snap
from zoom_analysis.visu.visu_fct import (
    # make_amr_img,
    make_amr_img_smooth,
    plot_stars,
    # make_yt_img,
    plot_zoom_BHs,
    plot_zoom_gals,
    plot_zoom_halos,
    plot_trail,
    plot_fields,
)

from zoom_analysis.trees.tree_reader import read_tree_file_rev


# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479_old"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_old"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_eagn_T0p15"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099_inter"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH"
sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2"

sim = ramses_sim(sim_dir, nml="cosmo.nml")

zoom_ctr = sim.zoom_ctr


if "refine_params" in sim.namelist:
    if "rzoom" in sim.namelist["refine_params"]:
        circ = True
        zoom_r = sim.namelist["refine_params"]["rzoom"]
    else:
        zoom_a = sim.namelist["refine_params"]["azoom"]
        zoom_b = sim.namelist["refine_params"]["bzoom"]
        zoom_c = sim.namelist["refine_params"]["czoom"]
        zoom_r = np.max([zoom_a, zoom_b, zoom_c])
        circ = False

    if "zoom_shape" in sim.namelist["refine_params"]:
        shape = sim.namelist["refine_params"]["zoom_shape"]
    else:
        shape = "circle"


# print(zoom_ctr, zoom_r, zoom_a, zoom_b, zoom_c)

snaps = sim.snap_numbers
aexps = sim.get_snap_exps()
times = sim.get_snap_times()

tgt_zed = 2.0
max_zed = 8.0

delta_t = 5  # Myr
every_snap = True  # try and interpolate between tree nodes if not found

rad_fact = 1.0  # fraction of radius to use as plot window
plot_win_str = str(rad_fact).replace(".", "p")

overwrite = False
clean = False  # no markers for halos/bhs
annotate = False
gal_markers = True
halo_markers = False
pdf = False
zdist = 100  # ckpc
hm = "HaloMaker_stars2_dp_rec_dust/"

# field = "density"
# vmin = 1e-29
# vmax = 1e-26
op = np.sum
log_color = True

field = "stellar mass"
cmap = "gray"
# cmap = "viridis"
vmin = 6e4
vmax = 1e9
marker_color = "r"

dv1 = [0, 0, 1]
dv2 = [0, 1, 0]
dv3 = [1, 0, 0]


# vmin = None
# vmax = None

# field = "temperature"
# vmin = 1e3
# vmax = 1e8

# field = "pressure"
# vmin = 10  # k
# vmax = 1e5
# tree_hids, tree_datas, tree_aexps = read_tree_rev(
#     tree_path, tgt_zed, tree_type="halo", target_fields=["m", "x", "y", "z", "r"]
# )

sim = ramses_sim(sim_dir, nml="cosmo.nml")
l = sim.unit_l(sim.aexp_stt) / (ramses_pc * 1e6) / sim.aexp_stt


# print(l)

sim.init_cosmo()


outdir = os.path.join(
    sim_dir,
    "maps",
    "big_pictures",
)

if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

option_str = ""
if annotate and not clean:
    option_str += "_annotate"
elif clean:
    option_str += "_clean"

for snap, aexp, time in zip(snaps, aexps, times):

    zed = 1.0 / aexp - 1.0
    if zed > max_zed:
        continue

    fout = os.path.join(
        outdir,
        f"{field[:4]}_{snap}.png",
    )
    if os.path.isfile(fout) and not overwrite:
        continue

    l_pMpc = l * aexp

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    tgt_pos = zoom_ctr
    tgt_rad = zoom_r

    # print(tgt_pos, tgt_rad)

    if circ:
        mins = maxs = None
    else:
        mins = (
            np.array([tgt_pos[0] - zoom_a, tgt_pos[1] - zoom_b, tgt_pos[2] - zoom_c])
            # * sim.cosmo.lcMpc
            # * 1e3
        )
        maxs = (
            np.array([tgt_pos[0] + zoom_a, tgt_pos[1] + zoom_b, tgt_pos[2] - zoom_c])
            # * sim.cosmo.lcMpc
            # * 1e3
        )

    # print(tgt_pos, tgt_rad, zdist, mins, maxs)

    # print(zoom_r * sim.cosmo.lcMpc * 1e3)
    # zdist = zoom_r * sim.cosmo.lcMpc * 1e3

    # make_amr_img_smooth(
    #     fig,
    #     ax,
    #     snap,
    #     sim,
    #     tgt_pos,
    #     tgt_rad,
    #     zdist=zdist,
    #     field=field,
    #     debug=True,
    #     vmin=vmin,
    #     vmax=vmax,
    #     # mins=mins,
    #     # maxs=maxs,
    #     cb=True,
    # )

    img = plot_fields(
        field,
        fig,
        ax,
        aexp,
        [dv1, dv2, dv3],
        tgt_pos,
        tgt_rad,
        sim,
        cb=True,
        vmin=vmin,
        vmax=vmax,
        transpose=False,
        cmap="magma",
        log=log_color,
        op=op,
    )

    if not clean:

        ax.scatter(
            0, 0, s=200, c="r", marker="+", label="Zoom center", zorder=999, lw=1
        )

        if shape == "circle":

            zoom_limit = Circle(
                (
                    (zoom_ctr[0] - tgt_pos[0]) * sim.cosmo.lcMpc * 1e3,
                    (zoom_ctr[1] - tgt_pos[1]) * sim.cosmo.lcMpc * 1e3,
                ),
                zoom_r * sim.cosmo.lcMpc * 1e3,
                fill=False,
                edgecolor="r",
                lw=2,
                zorder=999,
            )

        elif shape == "ellipsoid":

            zoom_limit = Ellipse(
                (
                    (zoom_ctr[0] - tgt_pos[0]) * sim.cosmo.lcMpc * 1e3,
                    (zoom_ctr[1] - tgt_pos[1]) * sim.cosmo.lcMpc * 1e3,
                ),
                2 * zoom_a * sim.cosmo.lcMpc * 1e3,
                2 * zoom_b * sim.cosmo.lcMpc * 1e3,
                fill=False,
                edgecolor="r",
                lw=2,
                zorder=999,
            )

        elif shape == "rectangle":

            zoom_limit = Rectangle(
                (
                    (zoom_ctr[0] - tgt_pos[0]) * sim.cosmo.lcMpc * 1e3
                    - zoom_a * sim.cosmo.lcMpc * 1e3,
                    (zoom_ctr[1] - tgt_pos[1]) * sim.cosmo.lcMpc * 1e3
                    - zoom_b * sim.cosmo.lcMpc * 1e3,
                ),
                2 * zoom_a * sim.cosmo.lcMpc * 1e3,
                2 * zoom_b * sim.cosmo.lcMpc * 1e3,
                fill=False,
                edgecolor="r",
                lw=2,
                zorder=999,
            )

        ax.add_patch(zoom_limit)

        # plot zoom galaxies
        plot_zoom_gals(
            ax,
            snap,
            sim,
            tgt_pos,
            tgt_rad,
            zdist,
            hm,
            gal_markers=gal_markers,
            annotate=annotate,
        )

        # plot zoom galaxies

        # plot zoom BHs
        try:
            plot_zoom_BHs(ax, snap, sim, tgt_pos, tgt_rad, zdist)
        except ValueError:
            pass

    print(f"writing {fout}")

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
