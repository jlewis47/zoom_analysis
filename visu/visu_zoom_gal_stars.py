import numpy as np
from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt

# from matplotlib.colors import LogNorm

# from matplotlib.patches import Circle
from zoom_analysis.constants import *
from zoom_analysis.halo_maker.assoc_fcts import get_gal_props_snap

from zoom_analysis.visu.visu_fct import plot_stars

# from mpi4py import MPI

# import matplotlib.patheffects as pe
import os

# from scipy.stats import binned_statistic_2d

# from scipy.spatial.transform import Rotation as R

# from scipy.spatial import KDTree, cKDTree
# from scipy.stats import binned_statistic_2d

# from zoom_analysis.sinks.sink_reader import (
#     read_sink_bin,
#     snap_to_coarse_step,
#     convert_sink_units,
# )


# from zoom_analysis.trees.tree_reader import read_tree_rev

from zoom_analysis.rascas.filts.filts import binned_statistic_2d
from zoom_analysis.rascas.rascas_mock import (
    convert_star_units,
    read_gal_stars,
    read_zoom_brick,
)
from zoom_analysis.visu.visu_fct import (
    # make_yt_img,
    plot_zoom_BHs,
    plot_zoom_gals,
    basis_from_vect,
)

import healpy as hp

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
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel"
sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"

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
# gids = [24, 74, 92, 99, 102, 125, 33, 43, 49, 59, 65, 68, 94, 144, 230, 392, 365, 20]
gids = [29]

nbins = 256
# dir = [1, 0, 0]

n_hp_dirs = 1
hp_dir_nb = 0

npix = hp.nside2npix(n_hp_dirs)
pix = np.arange(npix)
xdir, ydir, zdir = hp.pix2vec(n_hp_dirs, pix)
dv1 = np.array([xdir[hp_dir_nb], ydir[hp_dir_nb], zdir[hp_dir_nb]])

dv1, dv2, dv3 = basis_from_vect(dv1)

# get rotation to allign to basis supported by direction vector
# rot = R.align_vectors([dv1, dv2, dv3], [[0, 0, 1], [0, 1, 0], [1, 0, 0]])[0]
# so direction vector is z or x3 and we plot using x1,x2


print(f"{n_hp_dirs:d} directions, ... looking in direction {hp_dir_nb:d}")
print(f"corresponding direction is {dv1}")

sim = ramses_sim(sim_dir, nml="cosmo.nml")


planx_bins = np.linspace(0, 1, nbins)
plany_bins = np.linspace(0, 1, nbins)
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
# tgt_snap = 205
# tgt_snap = 237
# wh = snaps == tgt_snap
overwrite = False


# vmin = 1
# vmax = None

sim_dir = sim.path
# for now follow hagn halo

print(snaps, wh)

snap = snaps[wh][0]
aexp = aexps[wh][0]
time = times[wh][0]

# print(snap, aexp, time)

hm = "HaloMaker_stars2_dp_rec_dust/"
# brick = read_zoom_brick(snap, sim, hm)

# brick_gids = brick["hosting info"]["hid"]


for gid in gids:

    # wh_gid = brick_gids == gid
    # rgal = brick["smallest ellipse"]["r"][wh_gid][0]
    # # rgal = brick["virial properties"]["rvir"][wh_gid]
    # gal_pos = np.asarray(
    #     [
    #         brick["positions"]["x"][wh_gid][0],
    #         brick["positions"]["y"][wh_gid][0],
    #         brick["positions"]["z"][wh_gid][0],
    #     ]
    # )

    _, gal_props = get_gal_props_snap(sim.path, snap, gid)

    # print(gal_props)

    rgal = gal_props["rmax"]
    gal_pos = gal_props["pos"]

    # print(gal_pos, rgal)

    outdir = os.path.join(sim_dir, "maps", "gals", f"{snap:d}", f"{gid:d}")

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    outf = os.path.join(outdir, f"stars_{snap}_{gid:d}_{hp_dir_nb}.png")

    # if os.path.isfile(outf) and not overwrite:
    #     continue

    print(f"rank {rank} is handling snap {snap}")

    l_pMpc = sim.cosmo.lcMpc * aexp

    fstar = os.path.join(sim.path, hm, f"GAL_{snap:05d}", f"gal_stars_{gid:07d}")
    stars = read_gal_stars(fstar)
    convert_star_units(stars, snap, sim)

    stpos = stars["pos"]
    stage = stars["agepart"]
    stmass = stars["mpart"]

    print("avg stellar age: ", np.average(stage, weights=stmass))
    print("youngest stellar age: ", np.min(stage))

    # print(f"10Myr sfr: {}, 100Myr sfr:{}, ", np.sum(stmass[stage < 10]) / 1e7)
    # print("100Myr sfr: ", np.sum(stmass[stage < 100]) / 1e8)
    # print("1Gyr sfr: ", np.sum(stmass[stage < 1000]) / 1e9)
    print(
        f"10Myr sfr: {np.sum(stmass[stage < 10]) / 1e7:.1e}, 100Myr sfr:{np.sum(stmass[stage < 100])/1e8:.1e},1Gyr sfr:{np.sum(stmass[stage < 1000])/1e9:.1e}"
    )
    print(
        f"10Myr ssfr: {np.sum(stmass[stage < 10]) / 1e7 / stmass.sum():.1e}, 100Myr ssfr:{np.sum(stmass[stage < 100])/1e8 / stmass.sum():.1e},1Gyr ssfr:{np.sum(stmass[stage < 1000])/1e9 / stmass.sum():.1e}"
    )

    print(f"Galaxy has mass: {stmass.sum():.1e} Msun")

    stctr = np.mean(stpos, axis=0)

    ctr_st_pos = stpos - stctr
    # rotate pos into frame where direction vector is basis vector of rotated basis

    rad_tgt = np.max(np.linalg.norm(ctr_st_pos, axis=1))
    # zdist = abs(stpos[2].max() - stpos[2].min()) * sim.cosmo.lcMpc * 1e3
    # dx = 1.0 / 2**sim.levelmax

    fig, ax = plt.subplots(111)

    plot_stars(
        fig,
        ax,
        [dv1, dv2, dv3],
        planx_bins,
        plany_bins,
        l_pMpc,
        stmass,
        ctr_st_pos,
        rad_tgt,
    )

    print(f"Saving to {outf}")

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
