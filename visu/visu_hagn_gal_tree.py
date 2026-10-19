from turtle import fillcolor
from f90nml import patch
import numpy as np
from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from zoom_analysis.constants import *
from zoom_analysis.zoom_helpers import decentre_coordinates

from hagn.catalogues import get_halos_cat, make_super_cat, get_cat_hids

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

from zoom_analysis.visu.visu_fct import (
    # make_amr_img,
    make_amr_img_smooth,
    # make_yt_img,
    plot_zoom_BHs,
    plot_zoom_gals,
)

from zoom_analysis.constants import ramses_pc

from hagn.tree_reader import read_tree_rev
from hagn.utils import get_hagn_sim


# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479_old"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_old"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_eagn_T0p15"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099_inter"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_evenlesscoarse"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_leastcoarse"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_leastcoarse"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890"
sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18292"

sim = ramses_sim(sim_dir, nml="cosmo.nml")

zoom_ctr = sim.zoom_ctr
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


tgt_zed = 2
hagn_sim = get_hagn_sim()
tgt_snap = hagn_sim.get_closest_snap(zed=tgt_zed)
# tgt_snap = 197  # make automatic by loading hagn_sim

delta_t = 5  # Myr
every_snap = True  # try and interpolate between tree nodes if not found
rad_fact = 5.0  # fraction of radius to use as plot window

# zdist = 100  # ckpc
hm = "HaloMaker_stars2_dp_rec_dust/"

field = "density"
# field = "temperature"
# vmin = 10  # k
# vmax = 1e5
# field = "density"
vmin = None
vmax = None
# tree_hids, tree_datas, tree_aexps = read_tree_rev(
#     tree_path, tgt_zed, tree_type="halo", target_fields=["m", "x", "y", "z", "r"]
# )

super_cat = make_super_cat(tgt_snap, outf="/data101/jlewis/hagn/super_cats")

# for now follow hagn halo
hid = int(sim_dir.split("/")[-1].split("_")[0][2:])
gal_cat = get_cat_hids(super_cat, [hid])
print(gal_cat)

gid = gal_cat["gid"]


tree_hids, tree_datas, tree_aexps = read_tree_rev(
    tgt_zed, gid, tree_type="gal", target_fields=["m", "x", "y", "z", "r"], sim="hagn"
)

# print(tree_datas[])

# print(tree_aexps)

filt = tree_datas["x"][0] != -1

for key in tree_datas:
    tree_datas[key] = tree_datas[key][0][filt]
tree_hids = tree_hids[0][filt]
tree_aexps = tree_aexps[filt]
# print(tree_aexps)


hagn_l = hagn_sim.unit_l(hagn_sim.aexp_stt) / (ramses_pc * 1e6) / hagn_sim.aexp_stt


hagn_sim.init_cosmo()
tree_times = hagn_sim.cosmo_model.age(1.0 / tree_aexps - 1.0).value * 1e3  # Myr

# print(tree_times, times)
#

# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()

# print(rank)
# rank = 0
#
# rank_snaps = np.array_split(snaps, size)[rank]
# rank_aexps = np.array_split(aexps, size)[rank]
#
# print(rank, rank_snaps)

# print(rank_snaps[100])

# print(tree_datas["r"])

outdir = os.path.join(sim_dir, "gal_tree", "maps")
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)


for snap, aexp, time in zip(snaps[::-1], aexps[::-1], times[::-1]):

    hagn_l_pMpc = hagn_l * aexp

    # if not np.min(np.abs(1.0 / aexp - 1.0 / tree_aexps)) < 0.1:

    if (
        np.all(np.abs(time - tree_times) > delta_t) and every_snap
    ):  # Myr #didn't find a close enough place... try interpolating
        # between nearest tree nodes

        arg = np.argsort(np.abs(time - tree_times))
        # print(list(zip(arg, tree_times[arg])))

        arg_p = arg[0]
        if tree_times[arg_p] < time:
            arg_p1 = arg[0] - 1
        else:
            arg_p = arg_p + 1
            arg_p1 = arg[0] - 1

        if arg_p >= len(tree_times):
            continue
        if arg_p1 < 0:
            continue

        # print(tree_times[arg_p])
        # print(tree_times[arg_p1])
        assert time < tree_times[arg_p1] and time > tree_times[arg_p]

        tgt_pos_p = np.asarray(
            [
                tree_datas["x"][arg_p],
                tree_datas["y"][arg_p],
                tree_datas["z"][arg_p],
            ]
        )
        tgt_rad_p = tree_datas["r"][arg_p] / hagn_l_pMpc

        tgt_pos_p1 = np.asarray(
            [
                tree_datas["x"][arg_p1],
                tree_datas["y"][arg_p1],
                tree_datas["z"][arg_p1],
            ]
        )
        tgt_rad_p1 = tree_datas["r"][arg_p1] / hagn_l_pMpc

        tgt_pos = tgt_pos_p + (tgt_pos_p1 - tgt_pos_p) * (time - tree_times[arg_p]) / (
            tree_times[arg_p1] - tree_times[arg_p]
        )

        tgt_rad = tgt_rad_p + (tgt_rad_p1 - tgt_rad_p) * (time - tree_times[arg_p]) / (
            tree_times[arg_p1] - tree_times[arg_p]
        )

    else:  # found a close enough tree node... just read it
        tree_arg = np.argmin(np.abs(time - tree_times))

        # print(tree_arg, tree_datas["x"])

        tgt_pos = np.asarray(
            [
                tree_datas["x"][tree_arg],
                tree_datas["y"][tree_arg],
                tree_datas["z"][tree_arg],
            ]
        )
        tgt_rad = tree_datas["r"][tree_arg] / hagn_l_pMpc

    tgt_pos += 0.5 * hagn_l_pMpc
    tgt_pos /= hagn_l_pMpc  # in code units or /comoving box size

    print(tgt_pos)

    tgt_pos = decentre_coordinates(tgt_pos, sim.path)
    print(tgt_pos)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # data_path = os.path.join(sim.path, "amr2cell", f"output_{snap:05d}/out_amr2cell")

    # if not os.path.exists(data_path):
    #     continue

    # print(snap)
    # print(tgt_pos, tgt_rad)
    # print(1.0 / aexp - 1, 1.0 / tree_aexps[tree_arg] - 1, tree_arg)
    # if snap != 308:
    #     continue

    # print(tgt_pos - tgt_rad * 0.15, tgt_pos + tgt_rad * 0.15)

    rad_tgt = tgt_rad * rad_fact
    zdist = rad_tgt / 1 * sim.cosmo.lcMpc * 1e3
    print(snap, tgt_pos, tgt_rad, zdist, hagn_l_pMpc)

    make_amr_img_smooth(
        fig,
        ax,
        snap,
        sim,
        tgt_pos,
        rad_tgt,
        # zdist=-1,
        zdist=zdist,
        field=field,
        debug=False,
        vmin=vmin,
        vmax=vmax,
        # vmax=1e-22,
        cb=True,
    )
    # make_amr_img(
    #     ax,
    #     snap,
    #     sim,
    #     tgt_pos,
    #     tgt_rad * 0.15,
    #     # zdist=-1,
    #     zdist=zdist,
    #     field=field,
    #     debug=True,
    #     vmin=vmin,
    #     vmax=vmax,
    #     # vmax=1e-22,
    # )
    # make_amr_img_parts(
    #     ax,
    #     snap,
    #     sim,
    #     tgt_pos,
    #     tgt_rad * 0.15,
    #     # zdist=-1,
    #     zdist=zdist,
    #     field=field,
    #     debug=True,
    #     vmin=vmin,
    #     # vmax=1e-22,
    # )

    # print(snap)

    # tgt_pos_kpc = tgt_pos * hagn_l_pMpc * 1e3
    # tgt_rad_kpc = tgt_rad * hagn_l_pMpc * 1e3

    # make_yt_img(
    #     fig,
    #     ax,
    #     snap,
    #     sim,
    #     tgt_pos_kpc,
    #     tgt_rad_kpc,
    #     "x",
    #     zdist,
    #     hfields=[
    #         "density",
    #     ],
    # )

    ax.scatter(
        0, 0, s=200, c="r", marker="+", label="HAGN Halo center", zorder=999, lw=1
    )

    circ = Circle(
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

    ax.add_patch(circ)

    # plot zoom galaxies
    plot_zoom_gals(ax, snap, sim, tgt_pos, tgt_rad, zdist, hm)

    # plot zoom galaxies

    # plot zoom BHs
    plot_zoom_BHs(ax, snap, sim, tgt_pos, tgt_rad, zdist)

    plot_win_str = str(rad_fact).replace(".", "p")

    fig.savefig(
        os.path.join(outdir, f"{field[:4]}_{snap}_{plot_win_str}rgal.png"),
        dpi=300,
        format="png",
    )
    fig.savefig(
        os.path.join(outdir, f"{field[:4]}_{snap}_{plot_win_str}rgal.pdf"),
        dpi=300,
        format="pdf",
    )

    plt.close()

    # break
