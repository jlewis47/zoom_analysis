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
    colored_line,
)

from zoom_analysis.trees.tree_reader import (
    read_tree_file_rev_correct_pos,
    read_tree_file_rev,
)
from scipy.interpolate import UnivariateSpline

# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_DynBondiGravSinkMass_SE"
# sim_dir = (
# "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_gravSink_SE"
# )
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictestSF_lowSNe"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictestSF_lowSNe_highAGNeff"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_XtremeLowSFE_stgNHboost_strictSF"
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
# sim_dir = "/data103/jlewis/sims/lvlmax_22/mh1e12/id180130"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289"
# sim_dir = "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_frcAccrt"
# sim_dir = "/data102/jlewis/sims/lvlmax_20/mh1e12/id242756"
sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756"

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
# fixed_r_ckpc = 10
fixed_r_ckpc = 50

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


smass_args = {
    # "field":"stellar mass",
    "transpose": True,
    "cmap": "gray",
    "vmin": 6e4,
    "vmax": 1e9,
    "marker_color": "r",
    "mode": "sum",
}

stage_arg = {
    # "field":"stellar age",
    "transpose": True,
    "cmap": "Spectral_r",
    "vmin": 1,
    "vmax": 1e4,
    "marker_color": "grey",
    "mode": "mean",
}

sfr_arg = {
    # "field":"stellar age",
    "transpose": True,
    "cmap": "hot",
    "vmin": 1e1,
    "vmax": 3e3,
    "marker_color": "white",
    "mode": "mean",
}

density_args = {
    # "field":"density",
    "cmap": "magma",
    "transpose": True,
    "vmin": 1e-26,
    "vmax": 1e-20,
    "marker_color": "white",
    "mode": "sum",
}

temperature_args = {
    # "field":"temperature",
    "cmap": "plasma",
    "transpose": True,
    "vmin": 1e3,
    "vmax": 1e7,
    "marker_color": "white",
    "mode": "mean",
}

vel_args = {
    # "field":"velocity",
    "cmap": "bwr",
    "transpose": True,
    "vmin": -1e3,
    "vmax": 1e3,
    "subtract_mean": True,
    "marker_color": "white",
    "mode": "mean",
}

mach_args = {
    # "field":"velocity",
    "cmap": "bwr",
    "transpose": True,
    "vmin": 0,
    "vmax": 2,
    "log": False,
    # "subtract_mean":True,
    "marker_color": "white",
    "mode": "mean",
}

alpha_vir_args = {
    # "field":"velocity",
    "cmap": "YlGnBu",
    "transpose": True,
    # "vmin":-1e3,
    # "vmax":1e3,
    # "subtract_mean":True,
    "marker_color": "white",
    "mode": "mean",
}


args_main = {
    "cb": True,
    # "transpose":True,
    "cb_args": {"cb_args_main": {"location": "right"}},
}
args = {
    "cb": True,
    # "transpose": True,
    "color": marker_color,
    "plot_text": False,
}

dens_args = {**density_args, **args_main}
smass_args = {**smass_args, **args}
stage_arg = {**stage_arg, **args}
sfr_arg = {**sfr_arg, **args}
vel_args = {**vel_args, **args}
temperature_args = {**temperature_args, **args}
mach_args = {**mach_args, **args}
alpha_vir_args = {**alpha_vir_args, **args}


arg_dicts = [dens_args, smass_args, stage_arg, sfr_arg, temperature_args]

# fig = plt.figure(figsize=(16, 10))  # , layout="constrained")
# grid = plt.GridSpec(3, 3, hspace=0, wspace=0, figure=fig)

# ax_wide = fig.add_subplot(grid[:, :2])


"""
density,stellar mass,
SFR,age,temperature
"""


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

start_snap = sim.get_closest_snap(zed=start_zed)

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

# tree_hids, tree_datas, tree_aexps = read_tree_file_rev_correct_pos(
#     tree_name,
#     sim,
#     start_snap,
#     byte_file,
#     start_zed,
#     [hid_start],
#     # tree_type="halo",
#     tgt_fields=["m", "x", "y", "z", "r"],
#     verbose=True,
#     star=False,
# )
tree_hids, tree_datas, tree_aexps = read_tree_file_rev(
    tree_name,
    byte_file,
    start_zed,
    [hid_start],
    # tree_type="halo",
    tgt_fields=["m", "x", "y", "z", "r"],
    # verbose=True,
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

# colors for time of plot
# time_colors = plt.cm.viridis(np.linspace(0, 1, len(snaps)))
sim_times_normed = (sim_times - sim_times.min()) / (sim_times.max() - sim_times.min())
time_colors = plt.cm.viridis(sim_times_normed)
# time_colors = [rgb2hex(time_color,keep_alpha=True) for time_color in time_colors]

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


# test_ax.set_ylim(0, 1)

smooth_gal_props = smooth_props(gal_props_tree)

prev_mass = -1  # masses[-1]
prev_rad = -1  # 0.05 * rmaxs[-1]
prev_pos = -1  # poss[-1]

stellar_masses = np.zeros(len(snaps))
sfrs = np.zeros(len(snaps))

# for snap, aexp, time in zip(snaps[:], aexps[:], times[:]):
for istep, (snap, aexp, time) in enumerate(zip(snaps, aexps, times)):

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
        f"SFR_combined_{snap:04d}_{plot_win_str}{rstr}{option_str:s}.png",
    )

    if os.path.isfile(fout) and not overwrite:
        continue

    l_pMpc = l * aexp

    cur_hid = tree_hids[np.argmin(np.abs(tree_aexps - aexp))]

    # cur_gid, hosted_gals = get_central_gal_for_hid(
    #     sim,
    #     cur_hid,
    #     snap,
    #     main_stars=only_main_stars,
    #     # debug=True,
    #     prev_mass=prev_mass,
    #     prev_rad=prev_rad,
    #     prev_pos=prev_pos,
    # )

    # fig,axs = plt.subplots(3,3,figsize=(30,30),width_ratios=[1,1,1],height_ratios=[1,1,1], layout="constrained")
    fig, axs = plt.subplots(
        2,
        3,
        figsize=(30, 30),
        width_ratios=[1, 1, 1],
        height_ratios=[1, 1],
        layout="constrained",
    )

    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    ax_dens = axs[0, 0]
    ax_mass = axs[0, 1]
    ax_sfr = axs[0, 2]
    ax_age = axs[1, 0]
    ax_temp = axs[1, 1]
    # ax_vel = axs[1,2]
    ax_sm_sfr = axs[1, 2]
    # ax_alpha_vir = axs[2,0]
    # ax_mach = axs[2,1]
    # axs[2,2].axis("off")

    # print(cur_hid, cur_gid, tgt_pos, "%e" % hosted_gals["mass"].max(), tgt_rad)

    # print(snap, cur_gid)
    # _, cur_gal_props = get_gal_props_snap(
    #     sim_dir, snap, cur_gid, main_stars=only_main_stars
    # )

    aexp_arg = np.argmin(np.abs(aexp - gal_props_tree["aexps"]))

    cur_r50 = smooth_gal_props["r50"][aexp_arg]

    if not use_r50:
        cur_rad = smooth_gal_props["rmax"][aexp_arg]
    else:
        cur_rad = cur_r50

    cur_gid = gal_props_tree["gids"][aexp_arg]

    # cur_mass = masses[aexp_arg]
    # cur_pos = poss[aexp_arg]
    cur_mass = gal_props_tree["mass"][aexp_arg]
    cur_pos = gal_props_tree["pos"][aexp_arg]
    tgt_pos = cur_pos  # centre on the galaxy we found...
    # tgt_pos = cur_posfg

    prev_pos = cur_pos
    prev_mass = cur_mass
    # prev_rad = cur_rad

    # # use grid spec, make a 3x3 grid
    # fig = plt.figure(figsize=(16, 10))  # , layout="constrained")
    # grid = plt.GridSpec(2, 3, hspace=0, wspace=0, figure=fig)

    # ax_wide = fig.add_subplot(grid[:, :2])
    # ax_face = fig.add_subplot(grid[0, 2])
    # ax_disk = fig.add_subplot(grid[1, 2])

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

    zdist = rad_tgt / 1 * sim.cosmo.lcMpc * 1e3

    try:
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
    except FileNotFoundError:
        continue

    stars_in_3r50 = (
        np.linalg.norm(stars["pos"] - tgt_pos[None, :], axis=1) < 3 * cur_r50
    )

    ages = stars["age"]

    Zs = stars["metallicity"]

    stmasses = sfhs.correct_mass(sim, ages, stars["mass"], Zs)

    mass_in_3r50 = np.sum(stmasses[stars_in_3r50])
    sfr100_in_3r50 = np.sum(stmasses[stars_in_3r50][ages[stars_in_3r50] < 100] / 100.0)

    stellar_masses[istep] = mass_in_3r50
    sfrs[istep] = sfr100_in_3r50

    # ax_sm_sfr.plot(stellar_masses[:istep], sfrs[:istep], color=, label="Stellar mass in 2r50")
    lines = colored_line(
        stellar_masses[: istep + 1],
        sfrs[: istep + 1],
        c=sim_times / 1e3,
        ax=ax_sm_sfr,
        cmap="viridis",
        lw=3,
    )
    ax_sm_sfr.scatter(stellar_masses[istep], sfrs[istep], color="k", marker="+", s=150)

    star_bulk_vel = np.linalg.norm(
        np.average(stars["vel"], axis=0, weights=stmasses)
    )  # km/s
    star_bulk_vel = (
        star_bulk_vel * (3600 * 24 * 365 * 1e6) * (1e3 / 3.08e16 / 1e3)
    )  # kpc/Myr

    # print("star bulk vel", star_bulk_vel)
    if istep > 0:
        prev_rad = abs(star_bulk_vel * (time - times[istep - 1]) * 1.5)
        print(f"dist travelled cst rate since last snap: {prev_rad:.2f} kpc")
        prev_rad = prev_rad / sim.cosmo.lcMpc / 1e3  # ckpc->code

    ang_mom = compute_ang_mom(stmasses, stars["pos"], stars["vel"], tgt_pos)
    norm_ang_mom = np.linalg.norm(ang_mom)

    dir_face_on = ang_mom / norm_ang_mom
    dir_edge_on = np.cross(dir_face_on, directions[0])
    dir_edge_on /= np.linalg.norm(dir_edge_on)

    possible_dirs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

    closest_to_face = np.argmax([np.abs(np.dot(dir_face_on, d)) for d in possible_dirs])

    dir_choice = possible_dirs[closest_to_face]

    for arg in [
        dens_args,
        smass_args,
        stage_arg,
        sfr_arg,
        temperature_args,
        vel_args,
        mach_args,
        alpha_vir_args,
    ]:
        arg["hid"] = cur_hid

    img_dens = plot_fields(
        "density", fig, ax_dens, aexp, dir_choice, tgt_pos, rad_tgt, sim, **dens_args
    )

    img_mass = plot_fields(
        "stellar mass",
        fig,
        ax_mass,
        aexp,
        dir_choice,
        tgt_pos,
        rad_tgt,
        sim,
        **smass_args,
    )

    img_sfr = plot_fields(
        "SFR100", fig, ax_sfr, aexp, dir_choice, tgt_pos, rad_tgt, sim, **sfr_arg
    )

    img_age = plot_fields(
        "stellar age", fig, ax_age, aexp, dir_choice, tgt_pos, rad_tgt, sim, **stage_arg
    )

    img_temp = plot_fields(
        "temperature",
        fig,
        ax_temp,
        aexp,
        dir_choice,
        tgt_pos,
        rad_tgt,
        sim,
        **temperature_args,
    )
    # img_mach = plot_fields(

    #     "mach", fig, ax_mach, aexp, dir_choice, tgt_pos, rad_tgt, sim, **mach_args
    # )
    # img_alpha_vir= plot_fields(

    #     "alpha_vir", fig, ax_alpha_vir, aexp, dir_choice, tgt_pos, rad_tgt, sim, **alpha_vir_args
    # )

    # img_vel = plot_fields(
    #     "velocity", fig, ax_vel, aexp, dir_choice, tgt_pos, rad_tgt, sim, **vel_args
    # )

    # temperature_args_max = temperature_args.copy()
    # temperature_args_max["mode"] = "max"

    # img_temp_max = plot_fields(
    #     "temperature", fig, ax_vel, aexp, dir_choice, tgt_pos, rad_tgt, sim, **temperature_args_max)

    ax_dens.text(
        0.05,
        0.9,
        "z = %.2f" % zed,
        color=marker_color,
        transform=ax_dens.transAxes,
        path_effects=[pe.withStroke(linewidth=1, foreground="black")],
        ha="left",
        size=20,
        zorder=999,
    )

    if not clean:

        # circ = Circle(
        #     (
        #         (zoom_ctr[0] - tgt_pos[0]) * sim.cosmo.lcMpc * 1e3,
        #         (zoom_ctr[1] - tgt_pos[1]) * sim.cosmo.lcMpc * 1e3,
        #     ),
        #     zoom_r * sim.cosmo.lcMpc * 1e3,
        #     fill=False,
        #     edgecolor=marker_color,
        #     lw=2,
        #     ls="--",
        #     zorder=999,
        # )
        for iax, ax in enumerate(np.ravel(axs)[:-1]):
            circ = Circle(
                (
                    (0) * sim.cosmo.lcMpc * 1e3,
                    (0) * sim.cosmo.lcMpc * 1e3,
                ),
                cur_r50 * sim.cosmo.lcMpc * 1e3 * 3,
                fill=False,
                edgecolor=arg_dicts[iax]["marker_color"],
                lw=2,
                ls=":",
                zorder=999,
            )

            ax.add_patch(circ)

            ax.grid(True, alpha=0.33)  # Set grid lines with transparency

        # plot zoom galaxies
        if gal_markers:
            plot_zoom_gals(
                ax_dens,
                snap,
                sim,
                tgt_pos,
                rad_tgt,
                zdist,
                hm="HaloMaker_stars2_dp_rec_dust",
                gal_markers=gal_markers,
                annotate=annotate,
                direction=dir_choice,
                # transpose=True,
                **args,
            )

        # plot zoom hals
        if halo_markers:
            plot_zoom_halos(
                ax_dens,
                snap,
                sim,
                tgt_pos,
                rad_tgt,
                zdist,
                hm="HaloMaker_DM_dust",
                gal_markers=gal_markers,
                annotate=annotate,
                direction=dir_choice,
                # transpose=True,
                **args,
            )

        # plot zoom BHs
        # try:
        plot_zoom_BHs(
            ax_dens,
            snap,
            sim,
            tgt_pos,
            rad_tgt,
            zdist,
            direction=dir_choice,
            **args,
        )
        # except (ValueError, AssertionError):
        # pass

    # ax_wide.tick_params(top=True, labeltop=True, left=False, labelleft=False)
    # ax_wide.set_ylabel("")

    # ax_face.yaxis.set_label_position("right")
    # ax_face.set_xlabel("")
    # ax_disk.yaxis.set_label_position("right")

    ax_sm_sfr.set_xlabel("Stellar mass in 3 x r50, [$M_{\odot}$]")
    ax_sm_sfr.set_ylabel("SFR100 in 3 x r50, [$M_{\odot}$ Myr$^{-1}$]")

    ax_sm_sfr.set_xscale("log")
    ax_sm_sfr.set_yscale("log")

    ax_sm_sfr.grid(True, alpha=0.5)

    ax_sm_sfr.set_xlim(
        1e7,
    )

    cb = plt.colorbar(lines, ax=ax_sm_sfr, label="Time [Gyr]")

    # #give cb dual redshift axis
    # cax = cb.ax
    # cax2 = cax.twinx()
    # cax_ticks = cax.get_yticks()
    # cax_tick_labels = cax.get_yticklabels()
    # cax.set_yticks(cax_ticks)
    # cax2.set_yticks(cax_ticks)
    # cax2.set_yticklabels(["%.1f"%(1./sim_aexps[np.argmin(sim_times-float(t.get_text())*1e3)]-1) for t in cax_tick_labels])

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
