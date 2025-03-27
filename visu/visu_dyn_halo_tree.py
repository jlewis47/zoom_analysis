import numpy as np
from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from matplotlib import patheffects as pe
import os

from hagn.utils import get_hagn_sim

from zoom_analysis.constants import *
from zoom_analysis.zoom_helpers import starting_hid_from_hagn

from zoom_analysis.stars.dynamics import *

from zoom_analysis.halo_maker.assoc_fcts import (
    find_zoom_tgt_gal,
    get_central_gal_for_hid,
    get_gal_props_snap,
    get_halo_props_snap,
    get_halo_assoc_file,
    get_assoc_pties_in_tree,
    smooth_props,
)
from zoom_analysis.stars import sfhs

from zoom_analysis.read.read_data import read_data_ball


# from zoom_analysis.visu.visu_fct_bckp import (
from zoom_analysis.visu.visu_fct import (
    # make_amr_img,
    # make_amr_img_smooth,
    # make_yt_img,
    basis_from_vect,
    plot_zoom_BHs,
    plot_zoom_gals,
    plot_zoom_halos,
    # plot_stars,
    plot_fields,
)

from zoom_analysis.trees.tree_reader import read_tree_file_rev
from scipy.interpolate import UnivariateSpline

# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_DynBondiGravSinkMass_SE"
# sim_dir = (
# "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_gravSink_SE"
# )
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd_drag"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE"
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
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowSFE_DynBondiGravSinkMass"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_minMassDynGrav"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_kicks"
# sim_dir = "/data102/jlewis/sims/lvlmax_21/mh1e12/id180130_model6_eps0p05"
# sim_dir = "/data103/jlewis/sims/lvlmax_21/mh1e12/id180130_256"
# sim_dir = "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_drag"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890"
sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289"
# sim_dir = "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_frcAccrt"

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
max_zed = 8.0

delta_t = 4  # Myr
every_snap = True  # try and interpolate between tree nodes if not found

rad_fact = 5.0  # fraction of radius to use as plot window
use_r50 = True
fixed_r_ckpc = -1

plot_win_str = str(rad_fact).replace(".", "p")
if fixed_r_ckpc > 0:
    plot_win_str = f"{fixed_r_ckpc}ckpc"


overwrite = True
clean = False  # no markers for halos/bhs
annotate = False
gal_markers = True
halo_markers = True

only_main_stars = True

if only_main_stars:
    main_st_str = "mainStars"
else:
    main_st_str = ""

pdf = False
# zdist = 50  # ckpc
# hm = "HaloMaker_DM_dust"


hagn_sim = get_hagn_sim()

vmin = vmax = None
cmap = None
cb = True


mode = "sum"
marker_color = "white"
#
# field = "density"
# cmap = "magma"
# vmin = 1e-26
# vmax = 1e-20

# field = "temperature"
# vmin = 1e3
# vmax = 1e7
# cmap = "plasma"


# field = "metallicity"
# vmin = "1e-5"
# vmax = "1e-1"
# cmap = "YlOrRd"

# field = "dust_bin01"
# field = "dust_bin02"
# field = "dust_bin03"
# field = "dust_bin04"

# field = "DTM"
# vmin = 1e-5
# vmax = 1
# cmap = "YlGnBu"
# mode = "mean"
#
field = "stellar mass"
cmap = "gray"
vmin = 6e4
vmax = 1e9
marker_color = "r"

# field = "stellar age"
# vmin = 1e1  # Myr
# vmax = 1e3
# cmap = "Spectral_r"
# mode = "mean"

# # field = "SFR1"
# field = "SFR10"
# field = "SFR100"
# # # field = "SFR300"
# # field = "SFR500"
# # # field = "SFR1000"
# mode = "mean"
# cmap = "hot"
# vmin = 1e1
# vmax = 1e4

# field = "dm mass"
# cmap = "viridis"


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

# print(avail_aexps)

tgt_zed = max(tgt_zed, 1.0 / avail_aexps[-1] - 1.0)

print(tgt_zed)

avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

print(intID)

hid_start, halos, galaxies, start_aexp, found = starting_hid_from_hagn(
    zstt, sim, hagn_sim, intID, avail_aexps, avail_times, ztgt=tgt_zed
)

start_zed = 1.0 / start_aexp - 1.0

# print(hid_start, halos, galaxies, start_zed)

# print(hid_start, start_zed)
# print(halos)
# print(galaxies)
# gid = galaxies["gids"][galaxies["mass"].argmax()]
# print(start_aexp)

# gid, gal_props = find_zoom_tgt_gal(sim, tgt_zed, pure_thresh=0.9999, debug=False)

# r50 = gal_props["r50"]


# print(tree_aexps)


# print(l)

directions = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
# directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
# directions = [
#     [0.52704628, -0.52704628, -0.66666667],
#     [-0.70710678, -0.70710678, 0.0],
#     [-0.47140452, 0.47140452, -0.74535599],
# ]

# rascas_dir = [0.52704628, -0.52704628, -0.66666667]
# directions = basis_from_vect(rascas_dir)


tree_name = os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat")
byte_file = os.path.join(sim.path, "TreeMakerDM_dust")

tree_hids, tree_datas, tree_aexps = read_tree_file_rev(
    tree_name,
    byte_file,
    start_zed,
    [hid_start],
    # tree_type="halo",
    tgt_fields=["m", "x", "y", "z", "r"],
    debug=False,
    star=False,
)
tree_times = sim.cosmo_model.age(1.0 / tree_aexps - 1.0).value * 1e3  # Myr

# print(list(zip(tree_hids, tree_datas["m"])))
# print(tree_hids)

filt = tree_datas["x"][0] != -1

for key in tree_datas:
    tree_datas[key] = tree_datas[key][0][filt]
tree_hids = tree_hids[0][filt]
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


rgal = "rgal"
if fixed_r_ckpc > 0:
    rgal = ""

outdir = os.path.join(
    sim_dir, "maps_own_tree", "halo", f"{plot_win_str}{rgal:s}", f"{main_st_str:s}"
)

if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

option_str = ""
if annotate and not clean:
    option_str += "_annotate"
elif clean:
    option_str += "_clean"

gal_props_tree = get_assoc_pties_in_tree(
    sim, tree_aexps, tree_hids, assoc_fields=["r50", "rmax", "mass", "pos", "host hid"]
)

gal_exps = gal_props_tree["aexps"]
r50s = gal_props_tree["r50"]
rmaxs = gal_props_tree["rmax"]
masses = gal_props_tree["mass"]
poss = gal_props_tree["pos"]
hids = gal_props_tree["host hid"]

test_fig_pos = plt.figure()
test_ax_pos = test_fig_pos.add_subplot(111)
test_ax_pos.plot(gal_exps, poss[:, 0])
test_ax_pos.plot(gal_exps, poss[:, 1])
test_ax_pos.plot(gal_exps, poss[:, 2])
# test_ax.set_ylim(0, 1)
test_fig_r = plt.figure()
test_ax_r = test_fig_r.add_subplot(111)
test_ax_r.plot(gal_exps, r50s)
# test_ax.set_ylim(0, 1)

smooth_gal_props = smooth_props(gal_props_tree)

test_ax_r.plot(gal_exps, smooth_gal_props["r50"])

test_fig_pos.savefig("test_smooth_pos")
test_fig_r.savefig("test_smooth_r")

prev_mass = -1  # masses[-1]
prev_rad = -1  # 0.05 * rmaxs[-1]
prev_pos = -1  # poss[-1]

# for snap, aexp, time in zip(snaps[:], aexps[:], times[:]):
for istep, (snap, aexp, time) in enumerate(zip(snaps[::-1], aexps[::-1], times[::-1])):

    # if snap != 166:
    #     continue

    if not os.path.exists(get_halo_assoc_file(sim.path, snap)):
        continue

    zed = 1.0 / aexp - 1.0
    if zed > max_zed:
        continue

    # if snap != 185:
    #     continue

    if fixed_r_ckpc <= 0:
        if use_r50:
            rstr = "r50"
        else:
            rstr = "rmax"
    else:
        rstr = ""

    fout = os.path.join(
        outdir,
        f"{field.replace(' ','_')}_disk_{snap:04d}_{plot_win_str}{rstr}{option_str:s}.png",
    )

    if os.path.isfile(fout) and not overwrite:
        continue

    l_pMpc = l * aexp

    cur_hid = tree_hids[np.argmin(np.abs(tree_aexps - aexp))]

    cur_gid, hosted_gals = get_central_gal_for_hid(
        sim,
        cur_hid,
        snap,
        main_stars=only_main_stars,
        # debug=True,
        prev_mass=prev_mass,
        prev_rad=prev_rad,
        prev_pos=prev_pos,
    )

    if cur_gid is None:
        continue

    # print(cur_hid, cur_gid, tgt_pos, "%e" % hosted_gals["mass"].max(), tgt_rad)

    # print(snap, cur_gid)
    _, cur_gal_props = get_gal_props_snap(
        sim_dir, snap, cur_gid, main_stars=only_main_stars
    )

    aexp_arg = np.argmin(np.abs(aexp - gal_props_tree["aexps"]))

    cur_r50 = smooth_gal_props["r50"][aexp_arg]

    if not use_r50:
        cur_rad = smooth_gal_props["rmax"][aexp_arg]
    else:
        cur_rad = cur_r50

    # cur_mass = masses[aexp_arg]
    # cur_pos = poss[aexp_arg]
    cur_mass = cur_gal_props["mass"]
    cur_pos = cur_gal_props["pos"]
    tgt_pos = cur_pos  # centre on the galaxy we found...
    # tgt_pos = cur_posfg

    prev_pos = cur_pos
    prev_mass = cur_mass
    # prev_rad = cur_rad

    # print(cur_hid, cur_gid, cur_pos, tgt_pos, "%e" % cur_mass, cur_rmax)

    # tgt_pos -= zero_point
    # do edge reflections
    # tgt_pos[tgt_pos < 0] += 1
    # tgt_pos[tgt_pos > 1] -= 1
    # print(tgt_pos)

    # print(cur_rmax, "%.1e" % cur_mass)

    # fig, ax = plt.subplots(1, 1, figsize=(10, 16))

    # use grid spec, make a 3x3 grid
    fig = plt.figure(figsize=(16, 10))  # , layout="constrained")
    grid = plt.GridSpec(2, 3, hspace=0, wspace=0, figure=fig)

    ax_wide = fig.add_subplot(grid[:, :2])
    ax_face = fig.add_subplot(grid[0, 2])
    ax_disk = fig.add_subplot(grid[1, 2])

    ax_face.tick_params(
        bottom=False,
        labelbottom=False,
        left=False,
        labelleft=False,
        right=True,
        labelright=True,
        top=True,
        labeltop=True,
    )
    ax_disk.tick_params(
        bottom=True,
        labelbottom=True,
        left=False,
        labelleft=False,
        right=True,
        labelright=True,
    )

    # data_path = os.path.join(sim.path, "amr2cell", f"output_{snap:05d}/out_amr2cell")

    # if not os.path.exists(data_path):
    #     continue

    # print(snap)
    # print(tgt_pos, tgt_rad)
    # print(1.0 / aexp - 1, 1.0 / tree_aexps[tree_arg] - 1, tree_arg)
    # if snap != 308:
    #     continue

    # print(snap, tgt_pos, tgt_rad, zdist, hagn_l_pMpc)

    # rad_tgt = cur_rmax * rad_fact

    rad_tgt = cur_rad * rad_fact

    if fixed_r_ckpc > 0:
        rad_tgt = fixed_r_ckpc / (sim.cosmo.lcMpc * 1e3)

    rad_tgt_wide = rad_tgt * 2
    zdist = rad_tgt / 1 * sim.cosmo.lcMpc * 1e3

    # print(tgt_pos, tgt_rad, rad_tgt, zdist)
    # print(tgt_pos - tgt_rad * 0.15, tgt_pos + tgt_rad * 0.15)

    # print(rad_tgt * 5, rad_tgt * 5 * sim.cosmo.lcMpc * 1e3)

    stars = read_data_ball(
        sim,
        snap,
        tgt_pos,
        rad_tgt,
        host_halo=cur_hid,
        data_types=["stars"],
        tgt_fields=[
            "pos",
            "vel",
            "mass",
            "age",
            "metallicity",
        ],
    )

    ages = stars["age"]

    Zs = stars["metallicity"]

    masses = sfhs.correct_mass(hagn_sim, ages, stars["mass"], Zs)

    star_bulk_vel = np.linalg.norm(
        np.average(stars["vel"], axis=0, weights=masses)
    )  # km/s
    star_bulk_vel = (
        star_bulk_vel * (3600 * 24 * 365 * 1e6) * (1e3 / 3.08e16 / 1e3)
    )  # kpc/Myr

    # print("star bulk vel", star_bulk_vel)
    if istep > 0:
        prev_rad = abs(star_bulk_vel * (time - times[istep - 1]) * 1.5)
        print(f"dist travelled cst rate since last snap: {prev_rad:.2f} kpc")
        prev_rad = prev_rad / sim.cosmo.lcMpc / 1e3  # ckpc->code

    ang_mom = compute_ang_mom(masses, stars["pos"], stars["vel"], tgt_pos)
    norm_ang_mom = np.linalg.norm(ang_mom)

    dir_face_on = ang_mom / norm_ang_mom
    dir_edge_on = np.cross(dir_face_on, directions[0])
    dir_edge_on /= np.linalg.norm(dir_edge_on)

    cb_args_main = {"location": "left"}

    args_main = {
        "cb": True,
        "cb_args": cb_args_main,
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "mode": mode,
        "transpose": True,
        "color": marker_color,
        "hid": int(hids[aexp_arg]),
    }
    args = {
        "cb": False,
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "mode": mode,
        "transpose": True,
        "color": marker_color,
        "hid": int(hids[aexp_arg]),
        "plot_text": False,
    }

    img = plot_fields(
        field, fig, ax_face, aexp, dir_face_on, tgt_pos, rad_tgt, sim, **args
    )
    img = plot_fields(
        field, fig, ax_disk, aexp, dir_edge_on, tgt_pos, rad_tgt, sim, **args
    )
    img = plot_fields(
        field, fig, ax_wide, aexp, directions, tgt_pos, rad_tgt_wide, sim, **args_main
    )
    ax_wide.text(
        0.05,
        0.9,
        "z = %.2f" % zed,
        color=marker_color,
        transform=ax_wide.transAxes,
        path_effects=[pe.withStroke(linewidth=1, foreground="black")],
        ha="left",
        size=20,
        zorder=999,
    )

    if not clean:

        circ = Circle(
            (
                (zoom_ctr[0] - tgt_pos[0]) * sim.cosmo.lcMpc * 1e3,
                (zoom_ctr[1] - tgt_pos[1]) * sim.cosmo.lcMpc * 1e3,
            ),
            zoom_r * sim.cosmo.lcMpc * 1e3,
            fill=False,
            edgecolor=marker_color,
            lw=2,
            ls="--",
            zorder=999,
        )
        circ = Circle(
            (
                (0) * sim.cosmo.lcMpc * 1e3,
                (0) * sim.cosmo.lcMpc * 1e3,
            ),
            cur_r50 * sim.cosmo.lcMpc * 1e3,
            fill=False,
            edgecolor=marker_color,
            lw=2,
            ls=":",
            zorder=999,
        )

        ax_wide.add_patch(circ)

        # plot zoom galaxies
        if gal_markers:
            plot_zoom_gals(
                ax_wide,
                snap,
                sim,
                tgt_pos,
                rad_tgt,
                zdist,
                hm="HaloMaker_stars2_dp_rec_dust",
                gal_markers=gal_markers,
                annotate=annotate,
                direction=directions[0],
                # transpose=True,
                **args,
            )

        # plot zoom hals
        if halo_markers:
            plot_zoom_halos(
                ax_wide,
                snap,
                sim,
                tgt_pos,
                rad_tgt,
                zdist,
                hm="HaloMaker_DM_dust",
                gal_markers=gal_markers,
                annotate=annotate,
                direction=directions[0],
                # transpose=True,
                **args,
            )

        # plot zoom BHs
        # try:
        plot_zoom_BHs(
            ax_wide,
            snap,
            sim,
            tgt_pos,
            rad_tgt,
            zdist,
            direction=directions[0],
            **args,
        )
        # except (ValueError, AssertionError):
        # pass

    ax_wide.tick_params(top=True, labeltop=True, left=False, labelleft=False)
    ax_wide.set_ylabel("")

    ax_face.yaxis.set_label_position("right")
    ax_face.set_xlabel("")
    ax_disk.yaxis.set_label_position("right")

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
