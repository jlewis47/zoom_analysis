import numpy as np
from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from zoom_analysis.constants import *
from zoom_analysis.halo_maker.assoc_fcts import find_snaps_with_halos
from zoom_analysis.zoom_helpers import decentre_coordinates

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
    plot_fields,
    plot_zoom_halos,
)

from hagn.tree_reader import read_tree_rev, interpolate_tree_position
from hagn.utils import get_hagn_sim


# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479_old"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_old"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_leastcoarse"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_high_nssink"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_eagn_T0p15"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099_inter"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id74099"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id74099"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_like_SH"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_chabrier"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_evenlesscoarse"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_leastcoarse"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_leastcoarse"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242756"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242704"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"
sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE"


sim = ramses_sim(sim_dir, nml="cosmo.nml")

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
# snaps = sim.snap_numbers
snaps = find_snaps_with_halos(sim.snap_numbers, sim.path)
aexps = sim.get_snap_exps(snaps)
times = sim.get_snap_times(snaps)

tgt_zed = 2

delta_t = 5  # Myr
every_snap = True  # try and interpolate between tree nodes if not found

# rad_fact = 3.0  # fraction of radius to use as plot window
rad_fact = 1.0  # fraction of radius to use as plot window
# rad_fact = 0.1  # fraction of radius to use as plot window
# rad_fact = 0.25  # fraction of radius to use as plot window
# rad_fact = 0.35  # fraction of radius to use as plot window
# rad_fact = 0.5  # fraction of radius to use as plot window
# rad_fact = 4  # fraction of radius to use as plot window
plot_win_str = str(rad_fact).replace(".", "p")

overwrite = True
clean = False  # no markers for halos/bhs
annotate = True
gal_markers = True
halo_markers = True
pdf = False
transpose=True
cb=True
log=True

# zdist = 50  # ckpc if -! use same distance as radial plot window
zdist = -1  # ckpc if -! use same distance as radial plot window
hm = "HaloMaker_stars2_dp_rec_dust/"
hm_dm = "HaloMaker_DM_dust/"

# directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
# directions = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
directions = [[0, 1, 0],  [1, 0, 0],[0, 0, 1]]

mode="sum"

# fix_visu = False

# field = "density"
# vmin = 1e-27
# vmax = 1e-22
# cmap = "magma"

field = "dm mass"
vmin = 1e5
vmax = 4e9
cmap = "viridis"

# field = "stellar mass"
# vmin = 2e4
# vmax = 8e6
# cmap = "grey"

# field = "temperature"
# vmin = 1e3
# vmax = 1e8

# field = "metallicity"
# vmax = 1.0

# field = "dust_bin01"
# field = "dust_bin02"
# field = "dust_bin03"
# field = "dust_bin04"
# vmin = 1e-8
# vmax = 1e-4

# field = "pressure"
# vmin = 10  # k
# vmax = 1e5
# tree_hids, tree_datas, tree_aexps = read_tree_rev(
#     tree_path, tgt_zed, tree_type="halo", target_fields=["m", "x", "y", "z", "r"]
# )

args = {
    "cb": cb,
    "log": True,
    "cmap": cmap,
    "vmin": vmin,
    "vmax": vmax,
    "mode": mode,
    "transpose": False,
}


# for now follow hagn halo
hid = int(sim_dir.split("/")[-1].split("_")[0][2:])
tree_hids, tree_datas, tree_aexps = read_tree_rev(
    tgt_zed, [hid], tree_type="halo", target_fields=["m", "x", "y", "z", "r"]
)

print(hid, tgt_zed)

print(list(zip(tree_hids, tree_aexps)))


# print(tree_aexps)

filt = tree_datas["x"][0] != -1

for key in tree_datas:
    tree_datas[key] = tree_datas[key][0][filt]
tree_hids = tree_hids[0][filt]
tree_aexps = tree_aexps[filt]
# print(tree_aexps)

hagn_sim = get_hagn_sim()
hagn_l = hagn_sim.unit_l(hagn_sim.aexp_stt) / (ramses_pc * 1e6) / hagn_sim.aexp_stt

# print(hagn_l)

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

outdir = os.path.join(
    sim_dir,
    "maps",
    f"{plot_win_str}rvir",
)

os.makedirs(outdir, exist_ok=True)

option_str = ""
if annotate and not clean:
    option_str += "_annotate"
elif clean:
    option_str += "_clean"

for snap, aexp, time in zip(snaps[::-1], aexps[::-1], times[::-1]):

    fout = os.path.join(
        outdir,
        f"{field}_{snap}_{plot_win_str}rvir{option_str:s}.png",
    )
    if os.path.isfile(fout) and not overwrite:
        continue


    hagn_l_pMpc = hagn_l * aexp

    tgt_pos, tgt_rad = interpolate_tree_position(
        time,
        tree_times,
        tree_datas,
        hagn_l_pMpc,
        delta_t=delta_t,
        every_snap=every_snap,
    )

    if np.any(tgt_pos == None):
        print("interpolation failed")
        continue

    print(snap, tgt_pos, tgt_rad)
    #
    tgt_pos = decentre_coordinates(tgt_pos, sim.path)

    print(tgt_pos, tgt_rad)


    fig, ax = plt.subplots(1, 1, figsize=(10, 10))



    rad_tgt = tgt_rad * rad_fact
    if zdist in [-1,None]:
        # print(rad_tgt)
        zdist = rad_tgt * sim.cosmo.lcMpc * 1e3



    img = plot_fields(
        field,
        fig,
        ax,
        aexp,
        directions,
        tgt_pos,
        rad_tgt,
        sim,
        **args,
    )

    if np.all(img == 0):
        continue

    if not clean:

        ax.scatter(
            0, 0, s=200, c="r", marker="+", label="HAGN Halo center", zorder=999, lw=1
        )

        # circ = Circle(
        #     (
        #         (zoom_ctr[0] - tgt_pos[0]) * sim.cosmo.lcMpc * 1e3,
        #         (zoom_ctr[1] - tgt_pos[1]) * sim.cosmo.lcMpc * 1e3,
        #     ),
        #     zoom_r * sim.cosmo.lcMpc * 1e3,
        #     fill=False,
        #     edgecolor="r",
        #     lw=2,
        #     zorder=999,
        # )

        # ax.add_patch(circ)

        # plot zoom galaxies
        if gal_markers:
            plot_zoom_gals(
                ax,
                snap,
                sim,
                tgt_pos,
                tgt_rad,
                zdist,
                hm=hm,
                # gal_markers=gal_markers,
                # annotate=annotate,
                annotate=False,
                direction=directions[0],
                **args,
            )


        if halo_markers:
            # plot zoom halos
            plot_zoom_halos(
                ax,
                snap,
                sim,
                tgt_pos,
                tgt_rad,
                zdist,
                hm=hm_dm,
                annotate=annotate,
                direction=directions[0],
                **args,
            )

        # plot zoom BHs
        try:
            bh_in_frame = plot_zoom_BHs(
                ax, snap, sim, tgt_pos, tgt_rad, zdist, directions=directions[0],**args,
            )
            if bh_in_frame:
                print(f"found BHs in frame")
        except ValueError:
            pass

    fout = fout.replace(" ", "_")
    print(f"wrote {fout:s}")

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
