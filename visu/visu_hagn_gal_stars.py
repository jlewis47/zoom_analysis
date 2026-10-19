from astropy.units.core import sanitize_scale
import numpy as np
from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

# from matplotlib.patches import Circle
from zoom_analysis.constants import *

import os

from scipy.stats import binned_statistic_2d

from scipy.spatial.transform import Rotation as R

from zoom_analysis.visu.visu_fct import (
    # make_yt_img,
    plot_zoom_BHs,
    plot_zoom_gals,
    basis_from_vect,
)

from hagn.utils import get_hagn_sim
from hagn.tree_reader import read_tree_rev
from hagn.association import gid_to_stars
from hagn.catalogues import make_super_cat

from zoom_analysis.stars.sfhs import correct_mass

import healpy as hp

import mpi4py.MPI as MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# hid0 = 242756
# hid0 = 242704
hid0 = 180310
# hid0 = 21892
ztgt = 2.0

qty = "mass"
vmin = 1
vmax = 1e8

# qty = "sfr1000"
# vmin = 1e-3
# vmax = 1e0

hagn_sim = get_hagn_sim()
hagn_snap = hagn_sim.get_closest_snap(zed=ztgt)

super_cat = make_super_cat(hagn_snap, "hagn", outf="/data101/jlewis/hagn/super_cats")

gid0 = super_cat["gid"][super_cat["hid"] == hid0]

nbins = 32

# overwrite = True
overwrite = False

rfact = 1.5


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


planx_bins = np.linspace(0, 1, nbins)
plany_bins = np.linspace(0, 1, nbins)
#
# zoom_r =
# zoom_ctr = []
#

hagn_tree_hids, hagn_tree_datas, hagn_tree_aexps = read_tree_rev(
    ztgt,
    [gid0],
    tree_type="gal",
    target_fields=["m", "x", "y", "z", "r", "m_father", "nb_father"],
)

good_entries = hagn_tree_hids[0] != -1

hagn_tree_hids = [hagn_tree_hids[0][good_entries]]
hagn_tree_datas = {k: [v[0][good_entries]] for k, v in hagn_tree_datas.items()}
hagn_tree_aexps = hagn_tree_aexps[good_entries]

hagn_sim.init_cosmo()
hagn_tree_times = hagn_sim.cosmo_model.age(1.0 / (hagn_tree_aexps + 1.0))

l_hagn = hagn_sim.unit_l(hagn_sim.aexp_stt) / (ramses_pc * 1e6) / hagn_sim.aexp_stt

hagn_snaps = hagn_sim.snap_numbers
hagn_aexps = hagn_sim.get_snap_exps(param_save=False)

hagn_times = hagn_sim.cosmo_model.age(1.0 / hagn_aexps - 1.0)

delta_aexp = 1e-2


for aexp, snap, time in zip(hagn_aexps[::-1], hagn_snaps[::-1], hagn_times[::-1]):

    aexp_dist = np.abs(hagn_tree_aexps - aexp) / aexp

    if aexp_dist.min() > delta_aexp:
        continue

    zed = 1.0 / aexp - 1.0

    tree_arg = np.argmin(aexp_dist)

    # hagn_main_branch_id = hagn_tree_hids[np.argmin(np.abs(hagn_tree_aexps - cur_aexp))]
    # hagn_ctr = np.asarray(
    #     [
    #         hagn_tree_datas["x"][0][tree_arg],
    #         hagn_tree_datas["y"][0][tree_arg],
    #         hagn_tree_datas["z"][0][tree_arg],
    #     ]
    # )

    # hagn_ctr += 0.5 * (l_hagn * aexp)
    # hagn_ctr /= l_hagn * aexp

    # hagn_rvir = hagn_tree_datas["r"][0][tree_arg] / (l_hagn * aexp)  # * 10

    gid = hagn_tree_hids[0][tree_arg]

    print(snap, aexp, time, gid)

    # cur_super_cat = make_super_cat(snap, "hagn", outf="/data101/jlewis/hagn/super_cats")

    # rgal = cur_super_cat["rgal"][cur_super_cat["gid"] == gid][0]

    # print(gal_pos, rgal)

    outdir = os.path.join(
        "data101/jlewis/hagn/maps",
        "gal",
        str(hid0),
    )

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    rfact_str = str(rfact).replace(".", "p")

    # if os.path.isfile(outf) and not overwrite:
    #     continue

    print(f"rank {rank} is handling snap {snap}")

    l_pMpc = hagn_sim.cosmo.lcMpc * aexp

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    stars = gid_to_stars(
        gid, snap, hagn_sim, fields=["pos", "mass", "birth_time", "metallicity"]
    )

    stpos = stars["pos"]
    stmasses = stars["mass"]
    stages = stars["age"]
    stZs = stars["metallicity"]

    correct_mass(hagn_sim, stages, stmasses, stZs)

    if qty == "mass":

        plot_qty = stmasses
        cb_label = "Stellar Mass [Msun]"

    elif qty == "sfr1000":

        sfr = stmasses * np.int8(stages < 1000.0) / 1e9  # Msun/yr

        plot_qty = sfr
        cb_label = "SFR [Msun/yr]"

    outf = os.path.join(outdir, f"stars_{snap}_{qty}_{rfact_str}rmax.png")

    # stmass = stars["agepart"]

    print(f"Galaxy has mass: {stmasses.sum():.1e} Msun")

    stctr = np.mean(stpos, axis=0)

    ctr_st_pos = stpos - stctr
    # rotate pos into frame where direction vector is basis vector of rotated basis

    rad_tgt = np.max(np.linalg.norm(ctr_st_pos, axis=1)) * rfact
    # zdist = abs(stpos[2].max() - stpos[2].min()) * sim.cosmo.lcMpc * 1e3
    # dx = 1.0 / 2**sim.levelmax

    # vis_dir_ctr_st_pos = rot.apply(ctr_st_pos)
    vis_dir_ctr_st_pos = np.asarray(
        [
            np.dot(dv2, ctr_st_pos.T),
            np.dot(dv3, ctr_st_pos.T),
        ]
    ).T

    img = np.zeros((nbins, nbins))

    img = binned_statistic_2d(
        vis_dir_ctr_st_pos[:, 0],
        vis_dir_ctr_st_pos[:, 1],
        plot_qty,
        "sum",
        bins=[(planx_bins - 0.5) * rad_tgt, (plany_bins - 0.5) * rad_tgt],
    )[0]

    plimg = ax.imshow(
        img.T,
        origin="lower",
        extent=np.asarray([-rad_tgt, rad_tgt, -rad_tgt, rad_tgt]) * l_pMpc * 1e3,
        cmap="gray",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )

    cb = plt.colorbar(plimg, ax=ax, label=cb_label)

    # ax.scatter(vis_dir_ctr_st_pos[:, 0], vis_dir_ctr_st_pos[:, 1])

    ax.set_ylabel("y [kpc]")
    ax.set_xlabel("x [kpc]")

    ax.text(
        0.05,
        0.95,
        f"z = {zed:.2f}",
        transform=ax.transAxes,
        color="white",
        fontsize=16,
    )

    # # plot hagn galaxies
    # plot_zoom_gals(
    #     ax,
    #     snap,
    #     sim,
    #     gal_pos,
    #     rgal,
    #     zdist,
    #     hm=hm,
    #     brick_fct=read_zoom_brick,
    # )

    # plot hagn BHs
    # plot_zoom_BHs(ax, snap, sim, gal_pos, rgal, zdist)

    ax.set_facecolor("black")

    print(f"rank {rank} saving file {outf}")

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
