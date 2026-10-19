import numpy as np
from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import os

from hagn.utils import get_hagn_sim

from zoom_analysis.constants import *
from zoom_analysis.zoom_helpers import starting_hid_from_hagn

from zoom_analysis.halo_maker.assoc_fcts import find_zoom_tgt_gal, get_gal_props_snap
from zoom_analysis.visu.visu_fct import (
    # make_amr_img,
    make_amr_img_smooth,
    # make_yt_img,
    plot_zoom_BHs,
    plot_zoom_gals,
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
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh"
sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"

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
max_zed = 6.0

delta_t = 5  # Myr
every_snap = True  # try and interpolate between tree nodes if not found

rad_fact = 1.0  # fraction of radius to use as plot window
plot_win_str = str(rad_fact).replace(".", "p")

overwrite = False
clean = False  # no markers for halos/bhs
annotate = False
gal_markers = True
pdf = False
zdist = 50  # ckpc
hm = "HaloMaker_stars2_dp_rec_dust/"

hagn_sim = get_hagn_sim()

fix_visu = False

field = "density"
vmin = 1e-26
vmax = 1e-21

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
avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3


hid_start, halos, galaxies, start_aexp = starting_hid_from_hagn(
    zstt, sim, hagn_sim, intID, avail_aexps, avail_times
)
gid = galaxies["gids"][galaxies["mass"].argmax()]
print(start_aexp)
start_zed = 1.0 / start_aexp - 1.0

# gid, gal_props = find_zoom_tgt_gal(sim, tgt_zed, pure_thresh=0.9999, debug=False)

# r50 = gal_props["r50"]


# print(tree_aexps)


# print(l)


tree_name = os.path.join(sim.path, "TreeMakerstars2_dp_rec_dust", "tree_rev.dat")
byte_file = os.path.join(sim.path, "TreeMakerstars2_dp_rec_dust")

tree_gids, tree_datas, tree_aexps = read_tree_file_rev(
    tree_name,
    byte_file,
    start_zed,
    [gid],
    # tree_type="halo",
    tgt_fields=["m", "x", "y", "z", "r"],
    debug=False,
)
tree_times = sim.cosmo_model.age(1.0 / tree_aexps - 1.0).value * 1e3  # Myr

filt = tree_datas["x"][0] != -1

for key in tree_datas:
    tree_datas[key] = tree_datas[key][0][filt]
tree_gids = tree_gids[0][filt]
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

outdir = os.path.join(
    sim_dir,
    "maps_own_tree",
    "gal",
    f"{plot_win_str}rgal",
)
if fix_visu:
    outdir += "_fixed_centre"
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
        f"{field[:4]}_{snap}_{plot_win_str}rmax{option_str:s}.png",
    )

    if os.path.isfile(fout) and not overwrite:
        continue

    l_pMpc = l * aexp

    if not fix_visu:

        # if not np.min(np.abs(1.0 / aexp - 1.0 / tree_aexps)) < 0.1:
        if (
            np.all(np.abs(time - tree_times) > delta_t) and every_snap
        ):  # Myr #didn't find a close enough place... try interpolating
            # between nearest tree nodes

            arg = np.argsort(np.abs(time - tree_times))
            tree_arg = arg
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
            tgt_rad_p = tree_datas["r"][arg_p] / l_pMpc

            tgt_pos_p1 = np.asarray(
                [
                    tree_datas["x"][arg_p1],
                    tree_datas["y"][arg_p1],
                    tree_datas["z"][arg_p1],
                ]
            )
            tgt_rad_p1 = tree_datas["r"][arg_p1] / l_pMpc

            tgt_pos = tgt_pos_p + (tgt_pos_p1 - tgt_pos_p) * (
                time - tree_times[arg_p]
            ) / (tree_times[arg_p1] - tree_times[arg_p])

            tgt_rad = tgt_rad_p + (tgt_rad_p1 - tgt_rad_p) * (
                time - tree_times[arg_p]
            ) / (tree_times[arg_p1] - tree_times[arg_p])
            tgt_rad = tgt_rad

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
            tgt_rad = tree_datas["r"][tree_arg] / l_pMpc

    else:

        # use tree to get centre and radius at last matching tree node

        # print(times.max())
        tree_arg = np.argmin(np.abs(times[-1] - tree_times))

        tgt_pos = np.asarray(
            [
                tree_datas["x"][tree_arg],
                tree_datas["y"][tree_arg],
                tree_datas["z"][tree_arg],
            ]
        )

        tgt_rad = tree_datas["r"][tree_arg] / l_pMpc

    # print(tgt_pos)
    tgt_pos += 0.5 * l_pMpc
    # print(tgt_pos)
    tgt_pos /= l_pMpc  # in code units or /comoving box size
    # print(tgt_pos)
    tgt_pos[tgt_pos < 0] += 1
    tgt_pos[tgt_pos > 1] -= 1

    # print(tgt_pos)

    # get radius of galaxy
    cur_gid = tree_gids[tree_arg]
    # print(snap, cur_gid)
    _, cur_gal_props = get_gal_props_snap(sim_dir, snap, gid)

    # cur_r50 = cur_gal_props["r50"]
    cur_rmax = cur_gal_props["rmax"]
    cur_mass = cur_gal_props["mass"]

    # tgt_pos -= zero_point
    # do edge reflections
    # tgt_pos[tgt_pos < 0] += 1
    # tgt_pos[tgt_pos > 1] -= 1
    # print(tgt_pos)

    print(cur_rmax, "%.1e" % cur_mass)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # data_path = os.path.join(sim.path, "amr2cell", f"output_{snap:05d}/out_amr2cell")

    # if not os.path.exists(data_path):
    #     continue

    # print(snap)
    # print(tgt_pos, tgt_rad)
    # print(1.0 / aexp - 1, 1.0 / tree_aexps[tree_arg] - 1, tree_arg)
    # if snap != 308:
    #     continue

    # print(snap, tgt_pos, tgt_rad, zdist, hagn_l_pMpc)

    rad_tgt = cur_rmax * rad_fact
    # rad_tgt = cur_r50 * rad_fact
    zdist = rad_tgt / 1 * sim.cosmo.lcMpc * 1e3

    # print(tgt_pos, tgt_rad, rad_tgt, zdist)
    # print(tgt_pos - tgt_rad * 0.15, tgt_pos + tgt_rad * 0.15)

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

    if not clean:

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

    # break
