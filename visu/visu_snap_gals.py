from turtle import fillcolor
from f90nml import patch
from matplotlib.font_manager import font_family_aliases
import numpy as np
from f90_tools.star_reader import read_tgt_fields
from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from zoom_analysis.constants import *
from zoom_analysis.zoom_helpers import decentre_coordinates

from zoom_analysis.halo_maker.assoc_fcts import (
    get_assoc_pties_in_tree,
    get_gal_props_snap,
    smooth_props,
)

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
    # make_amr_img_smooth,
    # make_yt_img,
    plot_zoom_BHs,
    plot_zoom_gals,
    plot_zoom_halos,
    plot_fields,
)

from zoom_analysis.trees.tree_reader import (
    read_tree_file_rev_correct_pos as read_tree_file_rev,
)


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
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_like_SH"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_chabrier"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_evenlesscoarse"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_leastcoarse"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_leastcoarse"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242756"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictBH"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_stricterSF"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id180130_model5"
# sim_dir = (
#     "/data103/jlewis/sims/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF/"
# )
# sim_dir = "/data102/jlewis/sims/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF_high_sconstant/"
# sim_dir = "/data102/jlewis/sims/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF_Vhigh_sconstant/"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF_VVhigh_sconstant/"
sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289"

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
snaps = sim.snap_numbers
aexps = sim.get_snap_exps()
times = sim.get_snap_times()

tgt_zed = None
# tgt_zed = 3.36
# tgt_zed = 4.6
# tgt_zed = 5.12
tgt_zed = 2.0
# tgt_zed = 3.0
# tgt_zed = 7.3  # snap 109
# tgt_zed = 5.37
# tgt_zed = 6.08  # snap 85
# tgt_zed = 2.5  #
# tgt_zed = 3.2  #

rad_fact = 0.15  # fraction of radius to use as plot window
# rad_fact = 0.25  # fraction of radius to use as plot window
# rad_fact = 0.35  # fraction of radius to use as plot window
# rad_fact = 5.0  # fraction of radius to use as plot window
# rad_fact = 1.0  # fraction of radius to use as plot window
# rad_fact = 2.0  # fraction of radius to use as plot window
# rad_fact = 10.0  # fraction of radius to use as plot window
plot_win_str = str(rad_fact).replace(".", "p")

if tgt_zed is None:
    tgt_zed = 1.0 / sim.get_snap_exps().max() - 1

tgt_fpure = 0.9999
mlim = 5e9
# mlim = 8e8

# overwrite = False
overwrite = True
clean = False  # no markers for halos/bhs
annotate = False
gal_markers = True
pdf = False
zdist = -1  # ckpc if -! use same distance as radial plot window
hm = "HaloMaker_stars2_dp_rec_dust/"
hm_dm = "HaloMaker_DM_dust/"

vmin = None
vmax = None
log = True
mode = "mean"
mthd = np.sum
field = "density"
vmin = 1e-26
vmax = 1e-20

# log = False
# field = "DTM"
# # vmin = 0.0
# # vmax = 0.15
# vmin = 1e-3
# vmax = 1.0
# mthd = np.sum

# log = False
# field = "DTMC"
# vmin = 1e-3
# vmax = 1.0
# mthd = np.sum
# # log = False
# field = "DTMCs"
# vmin = 1e-3
# vmax = 1.0
# mthd = np.sum
# # log = False
# field = "DTMCl"
# vmin = 1e-3
# vmax = 1.0
# mthd = np.sum

## log = False
# field = "DTMSi"
# vmin = 1e-3
# vmax = 1.0
# mthd = np.sum
# log = False
# field = "DTMSis"
# vmin = 1e-3
# vmax = 1.0
# mthd = np.sum
# log = False
# field = "DTMSil"
# vmin = 1e-3
# vmax = 1.0
# mthd = np.sum


# field = "temperature"
# vmin = 1e2
# vmax = 1e7
# mthd = np.sum

# field = "metallicity"
# vmax = 4 * 2e-2
# vmin = 1e-4
# mthd = np.sum

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

# direction = [0, 1, 0]
# direction = [0, 0, 1]
dv1 = [1, 0, 0]
dv2 = [0, 1, 0]
dv3 = [0, 0, 1]
# direction = [1, 0, 0]

outdir = os.path.join(
    sim_dir,
    "maps",
    "snap_gals",
    f"{plot_win_str}rgal",
)

if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

option_str = ""
if annotate and not clean:
    option_str += "_annotate"
elif clean:
    option_str += "_clean"


zeds = 1.0 / aexps - 1.0
tgt_arg = np.argmin(np.abs(zeds - tgt_zed))
tgt_snap = snaps[tgt_arg]

tree_name = os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat")
byte_file = os.path.join(sim.path, "TreeMakerDM_dust")

tgt_aexp = sim.get_snap_exps(tgt_snap)[0]
tgt_zed = 1.0 / tgt_aexp - 1.0

if np.abs(zeds[tgt_arg] - tgt_zed) < 0.1:

    # load the assoc file
    gal_props = get_gal_props_snap(sim.path, tgt_snap)

    print(np.sum((gal_props["mass"] > mlim) * (gal_props["host purity"] > tgt_fpure)))

    # order all keys by descending halo mass
    sort_arg = np.argsort(gal_props["mass"])[::-1]

    for k in gal_props.keys():
        if len(gal_props[k]) == 3:
            gal_props[k] = gal_props[k][:, sort_arg]
        else:
            gal_props[k] = gal_props[k][sort_arg]

    for igal, (gid, tgt_pos, tgt_rad, fpure) in enumerate(
        zip(
            gal_props["gids"],
            gal_props["pos"].T,
            gal_props["rmax"],
            gal_props["host purity"],
        )
    ):

        # print(gal_props["mass"][igal], mlim)

        if fpure < tgt_fpure or gal_props["mass"][igal] < mlim:
            # print(f"skipping {gid:d}")
            # print(fpure, gal_props["mass"][igal])
            continue

        # print(gid, fpure, "%.1e" % gal_props["mass"][igal])

        fout = os.path.join(
            outdir,
            f"{gid:d}_{field}_{tgt_snap}_{plot_win_str}rvir{option_str:s}.png",
        )

        if os.path.exists(fout) and not overwrite:
            print(f"skipping {fout:s} because it exists")
            continue

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # sim_tree_hids, tree_datas, sim_tree_aexps = read_tree_file_rev(
        #     tree_name,
        #     sim,
        #     tgt_snap,
        #     byte_file,
        #     zeds[tgt_arg],
        #     [gal_props["host hid"][igal]],
        #     # tree_type="halo",
        #     tgt_fields=["m", "x", "y", "z", "r"],
        #     # debug=False,
        #     star=False,
        # )

        # sim_tree_hids = sim_tree_hids[0]

        # gal_props_tree = get_assoc_pties_in_tree(sim, sim_tree_aexps, sim_tree_hids)

        # smooth_gal_props_tree = smooth_props(gal_props_tree)

        # tree_arg = np.argmin(np.abs((1.0 / sim_tree_aexps - 1) - zeds[tgt_arg]))

        rad_tgt = tgt_rad * rad_fact
        # rad_tgt = smooth_gal_props_tree["r50"][tree_arg] * rad_fact
        zdist = rad_tgt * 5 / 1 * sim.cosmo.lcMpc * 1e3

        img = plot_fields(
            field,
            fig,
            ax,
            # sim_tree_aexps[tree_arg],
            tgt_aexp,
            [dv1, dv2, dv3],
            tgt_pos,
            rad_tgt,
            sim,
            # hid=sim_tree_hids[tree_arg],
            hid=gal_props["host hid"][igal],
            cb=True,
            vmin=vmin,
            vmax=vmax,
            transpose=False,
            cmap="magma",
            log=log,
            # op=np.sum,
            op=mthd,
            mode=mode,
        )

        if not clean:

            circ = Circle(
                (
                    (zoom_ctr[0] - tgt_pos[0]) * sim.cosmo.lcMpc * 1e3,
                    (zoom_ctr[1] - tgt_pos[1]) * sim.cosmo.lcMpc * 1e3,
                ),
                zoom_r * sim.cosmo.lcMpc * 1e3,
                fill=False,
                edgecolor="white",
                lw=2,
                zorder=999,
            )

            ax.add_patch(circ)

            # plot zoom galaxies
            plot_zoom_gals(
                ax,
                tgt_snap,
                sim,
                tgt_pos,
                tgt_rad,
                zdist,
                hm=hm,
                gal_markers=gal_markers,
                annotate=annotate,
                direction=dv3,
            )

            # plot zoom halos
            plot_zoom_halos(
                ax,
                tgt_snap,
                sim,
                tgt_pos,
                tgt_rad,
                zdist,
                hm=hm_dm,
                annotate=annotate,
            )

            # plot zoom BHs
            try:
                bh_in_frame = plot_zoom_BHs(
                    ax, tgt_snap, sim, tgt_pos, tgt_rad, zdist, direction=dv3
                )
                if bh_in_frame:
                    print(f"found BHs in frame")
            except ValueError:
                pass

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


else:

    print("no close enough snap to tgt_zed")
