from turtle import fillcolor
from f90nml import patch
import numpy as np
from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
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

from zoom_analysis.visu.visu_fct import (
    # make_amr_img,
    make_amr_img_smooth,
    # make_yt_img,
    plot_zoom_BHs,
    plot_zoom_gals,
)

from hagn.tree_reader import read_tree_rev
from hagn.IO import read_hagn_snap_brickfile, read_hagn_sink_bin
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
# # sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099_inter"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099"

# hid = 74099
# hid = 242704
# hid = 147479

# hids = [287012, 1589, 194228, 13310, 37686, 68373, 33051, 292074, 242704]

# hids = quenched_gals = np.genfromtxt(
#     f"ssfrs_pick_quenched_197_{sfr100:s}.txt",
#     names=True,
#     delimiter=",",
# )['hids']


# halos with galaxies that are not quite quenched at z=2 but are for 1Gyr by z=1.6
# hids = [242756, 37686, 68373, 22851, 13310, 142760, 237150]
# hids = [242756]  # , 180130, 242704, 21892]
hids = [242704]

sim = get_hagn_sim()

#
# zoom_r =
# zoom_ctr = []
#
snaps = sim.snap_numbers
aexps = sim.get_snap_exps(param_save=False)
zeds = 1.0 / aexps - 1.0
times = sim.get_snap_times(param_save=False)


tgt_zed = 2
zmax = 4.0
tgt_snap = sim.get_closest_snap(zed=tgt_zed)
actual_tgt_zed = zeds[snaps == tgt_snap]
overwrite = True
delta_t = 5  # Myr
every_snap = True  # try and interpolate between tree nodes if not found
rad_fact = 0.15  # fraction of radius to use as plot window

# vmin = vmax = None

# field = "density"
# field = "temperature"
# vmin = 1e3  # k
# vmax = 1e8
field = "density"
vmin = 1e-26
vmax = 1e-21
# tree_hids, tree_datas, tree_aexps = read_tree_rev(
#     tree_path, tgt_zed, tree_type="halo", target_fields=["m", "x", "y", "z", "r"]
# )

sim_dir = sim.path
# for now follow hagn halo

zok = (zeds <= zmax) * (zeds >= actual_tgt_zed)

snaps = snaps[zok]
aexps = aexps[zok]
times = times[zok]

for hid in hids:

    tree_hids, tree_datas, tree_aexps = read_tree_rev(
        tgt_zed, [hid], tree_type="halo", target_fields=["m", "x", "y", "z", "r"]
    )

    # print(tree_aexps)

    filt = tree_datas["x"][0] != -1

    for key in tree_datas:
        tree_datas[key] = tree_datas[key][0][filt]
    tree_hids = tree_hids[0][filt]
    tree_aexps = tree_aexps[filt]
    # print(tree_aexps)

    hagn_l = sim.unit_l(sim.aexp_stt) / (ramses_pc * 1e6) / sim.aexp_stt

    tree_times = sim.cosmo_model.age(1.0 / tree_aexps - 1.0).value * 1e3  # Myr

    outdir = os.path.join("/data101/jlewis/hagn", "maps", "halos", f"{hid}")
    if not os.path.exists(outdir) and rank == 0:
        os.makedirs(outdir, exist_ok=True)

    comm.Barrier()

    rank_snaps = np.array_split(snaps, size)[rank]
    rank_aexps = np.array_split(aexps, size)[rank]
    rank_times = np.array_split(times, size)[rank]

    plot_win_str = str(rad_fact).replace(".", "p")

    for snap, aexp, time in zip(rank_snaps, rank_aexps, rank_times):

        outf = os.path.join(outdir, f"{field[:4]}_{snap}_{plot_win_str}rvir.png")

        if os.path.isfile(outf) and not overwrite:
            continue

        print(f"rank {rank} is handling snap {snap}")

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

            tgt_pos = tgt_pos_p + (tgt_pos_p1 - tgt_pos_p) * (
                time - tree_times[arg_p]
            ) / (tree_times[arg_p1] - tree_times[arg_p])

            tgt_rad = tgt_rad_p + (tgt_rad_p1 - tgt_rad_p) * (
                time - tree_times[arg_p]
            ) / (tree_times[arg_p1] - tree_times[arg_p])

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

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # data_path = os.path.join(sim.path, "amr2cell", f"output_{snap:05d}/out_amr2cell")

        # if not os.path.exists(data_path):
        #     continue

        # print(snap)
        # print(tgt_pos, tgt_rad)
        # print(1.0 / aexp - 1, 1.0 / tree_aexps[tree_arg] - 1, tree_arg)
        # if snap != 308:
        #     continue

        zdist = tgt_rad * 0.1 / 1 * sim.cosmo.lcMpc * 1e3

        # print(snap, tgt_pos, tgt_rad, zdist, hagn_l_pMpc)
        # print(tgt_pos - tgt_rad * 0.15, tgt_pos + tgt_rad * 0.15)

        rad_tgt = tgt_rad * rad_fact

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
            debug=True,
            vmin=vmin,
            vmax=vmax,
            cb=True,
            # vmax=1e-22,
        )

        # plot hagn galaxies
        plot_zoom_gals(
            ax,
            snap,
            sim,
            tgt_pos,
            rad_tgt,
            zdist,
            brick_fct=read_hagn_snap_brickfile,
        )

        # plot hagn BHs
        plot_zoom_BHs(
            ax, snap, sim, tgt_pos, rad_tgt, zdist, sink_read_fct=read_hagn_sink_bin
        )

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

        print("wrote", outf)

        plt.close()

    # break
