import numpy as np
from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Circle

# from matplotlib.colors import LogNorm
from zoom_analysis.constants import *

# from zoom_analysis.zoom_helpers import decentre_coordinates

from zoom_analysis.halo_maker.assoc_fcts import (
    get_gal_props_snap,
    get_halo_props_snap,
    get_assoc_pties_in_tree,
    smooth_props,
)

from zoom_analysis.trees.tree_reader import (
    read_tree_file_rev_correct_pos as read_tree_file_rev,
)

from zoom_analysis.sinks.sink_reader import read_sink_bin
from f90_tools.star_reader import read_part_ball_NCdust, read_part_ball_hagn
from zoom_analysis.halo_maker.read_treebricks import read_zoom_brick
from zoom_analysis.stars.sfhs import correct_mass

from hagn.IO import read_hagn_snap_brickfile, read_hagn_sink_bin
from hagn.utils import get_hagn_sim

# from mpi4py import MPI

# import matplotlib.patheffects as pe


# from scipy.spatial import KDTree, cKDTree
# from scipy.stats import binned_statistic_2d

# from zoom_analysis.sinks.sink_reader import (
#     read_sink_bin,
#     snap_to_coarse_step,
#     convert_sink_units,
# )


# from zoom_analysis.trees.tree_reader import read_tree_rev

from zoom_analysis.visu.visu_fct import (
    # plot_stars,
    # make_amr_img,
    # make_amr_img_smooth,
    # make_yt_img,
    plot_zoom_BHs,
    plot_zoom_gals,
    plot_zoom_halos,
    plot_fields,
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
sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"
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
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE"
hagn = False

if not hagn:
    sim = ramses_sim(sim_dir, nml="cosmo.nml")
    bh_fct = read_sink_bin
    gal_fct = read_zoom_brick
    read_star_fct = read_part_ball_NCdust
    hm = "HaloMaker_stars2_dp_rec_dust/"

else:
    # for getting hagn plots... need to lookup massive hagn sink id. Can use one of the codes in ../sinks
    sim = get_hagn_sim()
    bh_fct = read_hagn_sink_bin
    gal_fct = read_hagn_snap_brickfile
    read_star_fct = read_part_ball_hagn
    hm = None
    intID = 242756
    sim_dir = os.path.join("/data101/jlewis/hagn", "maps", "halos", f"{intID}")


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

tgt_zed = 2.0
# tgt_zed = 3.0
# tgt_zed = 7.3  # snap 109
# tgt_zed = 5.37
# tgt_zed = 6.08  # snap 85
# tgt_zed = 2.5  #
# tgt_zed = 3.2  #
# tgt_zed = 4.3  #

# rad_fact = 0.15  # fraction of radius to use as plot window
# rad_fact = 0.25  # fraction of radius to use as plot window
# rad_fact = 0.35  # fraction of radius to use as plot window
# rad_fact = 1.0  # fraction of radius to use as plot window
rad_fact = 5.0  # fraction of radius to use as plot window
# rad_fact = 1.0  # fraction of radius to use as plot window
plot_win_str = str(rad_fact).replace(".", "p")

tgt_fpure = 0.9999
mlim = 1e8

overwrite = True
clean = False  # no markers for halos/bhs
annotate = True
gal_markers = True
halo_markers = True
pdf = False
zdist = -1  # ckpc if -! use same distance as radial plot window
hm = "HaloMaker_stars2_dp_rec_dust/"
hm_dm = "HaloMaker_DM_dust/"

vmin = None
vmax = None

nstar_bins = 1500

dv1 = [0, 0, 1]
dv2 = [0, 1, 0]
dv3 = [1, 0, 0]

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
#
# tgt_snap = 197
# tgt_arg = np.argmin(np.abs(snaps - tgt_snap))
# tgt_zed = zeds[tgt_arg]

aexp = aexps[tgt_arg]
zed = zeds[tgt_arg]


print(tgt_snap, tgt_arg, aexp, zed)
print(snaps[tgt_arg])

tree_name = os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat")
byte_file = os.path.join(sim.path, "TreeMakerDM_dust")

if np.abs(zed - tgt_zed) < 0.1:

    # load the assoc file
    gal_props = get_gal_props_snap(sim.path, tgt_snap)

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
            # print(f"skipping {gid:d} due to purity or mass")
            continue

        # print(gid, fpure, "%.1e" % gal_props["mass"][igal], tgt_rad * 140 * 1e3)

        # print(gal_props.keys())

        fout = os.path.join(
            outdir,
            f"{gid:d}_mstar_{tgt_snap}_{plot_win_str}rvir{option_str:s}.png",
        )

        if os.path.exists(fout) and not overwrite:
            print(f"skipping {fout:s}")
            continue

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        sim_tree_hids, tree_datas, sim_tree_aexps = read_tree_file_rev(
            tree_name,
            sim,
            tgt_snap,
            byte_file,
            zeds[tgt_arg],
            [gal_props["host hid"][igal]],
            # tree_type="halo",
            tgt_fields=["m", "x", "y", "z", "r"],
            debug=False,
            star=False,
        )

        sim_tree_hids = sim_tree_hids[0]

        gal_props_tree = get_assoc_pties_in_tree(sim, sim_tree_aexps, sim_tree_hids)

        smooth_gal_props_tree = smooth_props(gal_props_tree)

        tree_arg = np.argmin(np.abs((1.0 / sim_tree_aexps - 1) - zeds[tgt_arg]))

        # rad_tgt = tgt_rad * rad_fact
        rad_tgt = smooth_gal_props_tree["r50"][tree_arg] * rad_fact

        zdist = rad_tgt * 5

        host_hid = gal_props["host hid"][igal]
        # htgt = 789
        hprops, hgals = get_halo_props_snap(sim.path, tgt_snap, host_hid)
        # tgt_props, _ = get_halo_props_snap(sim.path, tgt_snap, htgt)

        print("host hid is", host_hid)

        hpos = hprops["pos"]
        rvir = hprops["rvir"]

        # tgt_hpos = tgt_props["pos"]
        # tgt_rvir = tgt_props["rvir"]

        print(f"host halo {host_hid:d} at {hpos*sim.cosmo.lcMpc} with rvir {rvir:.2e}")
        # print(
        #     f"target halo {htgt:d} at {tgt_hpos*sim.cosmo.lcMpc} with rvir {tgt_rvir:.2e}"
        # )

        # print(tgt_pos)
        # print(hpos)

        # print(tgt_pos, rad_tgt)

        zoom_stars = read_star_fct(
            sim,
            tgt_snap,
            tgt_pos,
            rad_tgt,
            ["birth_time", "metallicity", "mass", "pos"],
            fam=2,
        )

        ages = zoom_stars["age"]
        Zs = zoom_stars["metallicity"]
        masses = zoom_stars["mass"]

        masses = correct_mass(sim, ages, masses, Zs)
        stpos = zoom_stars["pos"]

        # xbins = np.linspace(stpos[:, 1].min(), stpos[:, 1].max(), nstar_bins)
        # ybins = np.linspace(stpos[:, 2].min(), stpos[:, 2].max(), nstar_bins)

        # ctr_st = np.mean(stpos, axis=0)
        # rad_gal = np.max(np.linalg.norm(np.asarray(stpos) - ctr_st, axis=1))

        xbins = np.linspace(
            tgt_pos[1] - rad_tgt,
            tgt_pos[1] + rad_tgt,
            nstar_bins,
        )
        ybins = np.linspace(
            tgt_pos[2] - rad_tgt,
            tgt_pos[2] + rad_tgt,
            nstar_bins,
        )

        # print(masses.sum(), len(masses))

        # smallest possible value is 5 stellar particle masses
        if len(masses) > 0:
            vmin_stmass = masses.min() * 5

            # plot_stars(
            #     fig,
            #     ax,
            #     sim,
            #     aexp,
            #     [dv1, dv2, dv3],
            #     nstar_bins,
            #     masses,
            #     stpos,
            #     tgt_pos,
            #     rad_tgt,
            #     cb=True,
            #     vmin=4e4,
            #     vmax=1e7,
            #     log=True,
            # )
            stimg = plot_fields(
                "stellar mass",
                fig,
                ax,
                aexp,
                [dv1, dv2, dv3],
                tgt_pos,
                rad_tgt,
                sim,
                cb=True,
                vmin=1e5,
                # vmax=stvmax,
                transpose=True,
                cmap="grey",
                log=True,
            )

        if not clean:

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

        zdist = rad_tgt

        # plot zoom galaxies
        if gal_markers:
            plot_zoom_gals(
                ax,
                tgt_snap,
                sim,
                tgt_pos,
                tgt_rad,
                zdist * sim.cosmo.lcMpc * 1e3,
                hm=hm,
                gal_markers=gal_markers,
                annotate=annotate,
                color="r",
            )

        if halo_markers:
            plot_zoom_halos(
                ax,
                tgt_snap,
                sim,
                tgt_pos,
                tgt_rad,
                zdist * sim.cosmo.lcMpc * 1e3,
                hm=hm_dm,
                halo_markers=halo_markers,
                annotate=annotate,
                color="r",
            )

        # ax.scatter(
        #     (hpos[0] - tgt_pos[0]) * (sim.cosmo.lcMpc * 1e3),
        #     (hpos[1] - tgt_pos[1]) * (sim.cosmo.lcMpc * 1e3),
        #     s=100,
        #     c="r",
        #     marker="x",
        #     zorder=1000,
        # )

        # # plot virial circle
        # circ_vir = Circle(
        #     (
        #         (hpos[0] - tgt_pos[0]) * (sim.cosmo.lcMpc * 1e3),
        #         (hpos[1] - tgt_pos[1]) * (sim.cosmo.lcMpc * 1e3),
        #     ),
        #     rvir * (sim.cosmo.lcMpc * 1e3),
        #     fill=False,
        #     edgecolor="r",
        #     lw=2,
        #     zorder=999,
        # )
        # ax.add_patch(circ_vir)

        # plot zoom galaxies

        # plot zoom BHs
        try:
            bh_in_frame = plot_zoom_BHs(
                ax,
                tgt_snap,
                sim,
                tgt_pos,
                tgt_rad,
                zdist * sim.cosmo.lcMpc * 1e3,
                annotate=annotate,
                color="r",
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
