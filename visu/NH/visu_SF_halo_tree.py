import numpy as np
from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from f90_tools.star_reader import read_part_ball_NH

from matplotlib import patheffects as pe


# from matplotlib.colors import LogNorm
# from matplotlib.patches import Circle
import os

from hagn.utils import get_hagn_sim, get_nh_sim
from hagn.tree_reader import read_tree_rev
from hagn.catalogues import (
    get_nh_cats_h5,
    convert_cat_units,
    get_cat_hids,
    make_super_cat,
)

# from hagn.IO import read_hagn_snap_brickfile, read_hagn_sink_bin

from zoom_analysis.constants import *
from zoom_analysis.halo_maker.read_treebricks import read_brickfile, read_zoom_brick
from zoom_analysis.sinks.sink_reader import (
    check_if_superEdd,
    find_massive_sink,
    gid_to_sid,
    # find_zoom_massive_central_sink,
    hid_to_sid,
    read_sink_bin,
)

from zoom_analysis.stars.dynamics import compute_ang_mom

from zoom_analysis.stars.sfhs import correct_mass

from zoom_analysis.visu.visu_fct import (
    # make_amr_img,
    make_amr_img_smooth,
    plot_stars,
    # make_yt_img,
    plot_zoom_BHs,
    plot_zoom_gals,
    plot_zoom_halos,
    plot_trail,
    plot_fields,
    colored_line,
)

from f90_tools.star_reader import read_part_ball_NCdust, read_part_ball_hagn

# from zoom_analysis.halo_maker.read_treebricks import read_zoom_stars
from zoom_analysis.halo_maker.assoc_fcts import (
    # find_zoom_tgt_gal,
    # get_gal_assoc_file,
    # find_snaps_with_gals,
    find_snaps_with_halos,
)

from zoom_analysis.sinks.sink_reader import (
    get_sink_mhistory,
    # find_zoom_central_massive_sink,
)


hagn = False
delta_aexp = 0.05


# for getting hagn plots... need to lookup massive hagn sink id. Can use one of the codes in ../sinks
sim = get_nh_sim()
# bh_fct =
# gal_fct =
# read_star_fct = read_part_ball_hagn
read_star_fct = read_part_ball_NH
hm_dm = "TREE_DM"
hm = "TREE_STARS_AdaptaHOP_dp_SCnew_gross"
# intID = 242756
sim_dir = os.path.join("/data101/jlewis/nh")


def gal_fct(snap, sim, hm, **kwargs):
    return read_zoom_brick(
        snap, sim, hm, sim_path="/data7b/NewHorizon", galaxy=True, star=True
    )


def bh_fct(path):

    return read_sink_bin(path, hagn=True)


name = sim.name


# zoom_ctr = sim.zoom_ctr


snaps = sim.snap_numbers
sim_aexps = sim.get_snap_exps(param_save=False)
sim_times = sim.get_snap_times(param_save=False)
sim_snaps = sim.snap_numbers


# zstt = 2.0
tgt_zed = 1.0
# max_zed = 1.1
max_zed = 6.0

delta_t = 4  # Myr
every_snap = True  # try and interpolate between tree nodes if not found

rad_fact = 5.0  # fraction of radius to use as plot window
use_rvir = False
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

last_hid = 21742


sim_aexps = sim.get_snap_exps(param_save=False)  # [::-1]
sim_times = sim.get_snap_times(param_save=False)  # [::-1]
sim_snaps = sim.snap_numbers  # [::-1]

last_snap = 446
last_aexp = sim.get_snap_exps(446, param_save=False)[0]


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
nstar_bins = 1000
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

nh_tree_hids, nh_tree_datas, nh_tree_aexps = read_tree_rev(
    1.0 / last_aexp - 1.0,
    [last_hid],
    tree_type="halo",
    # [gid],
    # tree_type="gal",
    target_fields=["m", "x", "y", "z", "r"],
    sim="nh",
    float_dtype=np.float32,
)

nh_tree_times = sim.cosmo_model.age(1.0 / nh_tree_aexps - 1.0).value * 1e3  # Myr

# print(list(zip(tree_hids, tree_datas["m"])))
# print(tree_hids)

filt = nh_tree_datas["x"][0] != -1

for key in nh_tree_datas:
    nh_tree_datas[key] = nh_tree_datas[key][0][filt]
tree_hids = nh_tree_hids[0][filt]
tree_aexps = nh_tree_aexps[filt]
tree_times = nh_tree_times[filt]
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

# print(nh_tree_datas["r"])

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

nsteps = len(tree_aexps)

reffs = np.zeros(nsteps)
rvirs = np.zeros(nsteps)
mass = np.zeros(nsteps)
pos = np.zeros((nsteps, 3))

for istep, (aexp, time) in enumerate(zip(nh_tree_aexps, nh_tree_times)):

    nh_snap = sim_snaps[np.argmin(np.abs(sim_aexps - aexp))]

    try:
        super_cat = make_super_cat(
            nh_snap, outf="/data101/jlewis/nh/super_cats", sim="nh", overwrite=True
        )
        print(istep, 1.0 / aexp - 1)
    except FileNotFoundError:
        print("No super cat")
        continue

    gal_pties = get_cat_hids(super_cat, [int(nh_tree_hids[0][istep])])

    # print(gal_pties.keys())

    if len(gal_pties["mgal"]) == 0:
        continue

    mass[istep] = gal_pties["mgal"]
    pos[istep, :] = [gal_pties["x"][0], gal_pties["y"][0], gal_pties["z"][0]]
    reffs[istep] = gal_pties["rgal"]
    rvirs[istep] = gal_pties["rvir"]

# test_ax.set_ylim(0, 1)

# smooth_gal_props = smooth_props(gal_props_tree)

prev_mass = -1  # masses[-1]
prev_rad = -1  # 0.05 * rmaxs[-1]
prev_pos = -1  # poss[-1]

stellar_masses = np.zeros(len(snaps))
sfrs = np.zeros(len(snaps))

# for snap, aexp, time in zip(snaps[:], aexps[:], times[:]):
for istep, (snap, aexp, time) in enumerate(zip(snaps, sim_aexps, sim_times)):

    # if snap != 166:
    #     continue

    zed = 1.0 / aexp - 1.0
    if zed > max_zed:
        continue

    # if snap != 185:
    #     continue

    if fixed_r_ckpc <= 0:
        if use_rvir:
            rstr = "rvir"
        else:
            rstr = "rgal"
    else:
        rstr = ""

    fout = os.path.join(
        outdir,
        f"SFR_combined_{snap:04d}_{plot_win_str}{rstr}{option_str:s}.png",
    )

    if os.path.isfile(fout) and not overwrite:
        continue

    l_pMpc = sim.cosmo.lcMpc * aexp

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

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

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

    aexp_arg = np.argmin(np.abs(aexp - nh_tree_aexps))

    cur_r50 = reffs[aexp_arg]

    if not use_rvir:
        cur_rad = rvirs[aexp_arg]
    else:
        cur_rad = cur_r50

    # cur_gid = gal_props_tree["gids"][aexp_arg]

    # cur_mass = masses[aexp_arg]
    # cur_pos = poss[aexp_arg]
    cur_mass = mass[aexp_arg]
    if cur_mass == 0:
        continue
    cur_pos = pos[aexp_arg]
    tgt_pos = cur_pos  # centre on the galaxy we found...
    # tgt_pos = cur_posfg

    prev_pos = cur_pos
    prev_mass = cur_mass
    # prev_rad = cur_rad

    rad_tgt = cur_rad * rad_fact

    if fixed_r_ckpc > 0:
        rad_tgt = fixed_r_ckpc / (sim.cosmo.lcMpc * 1e3)

    zdist = rad_tgt * 1 * sim.cosmo.lcMpc * 1e3  # ckpc

    try:
        stars = read_star_fct(
            sim,
            snap,
            tgt_pos,
            rad_tgt,
            ["birth_time", "metallicity", "mass", "pos", "vel"],
            fam=2,
        )

    except FileNotFoundError:
        continue

    stars_in_3r50 = (
        np.linalg.norm(stars["pos"] - tgt_pos[None, :], axis=1) < 3 * cur_r50
    )

    if stars_in_3r50.sum() == 0:
        continue

    stages = stars["age"]

    print(stages.min(), stages.max(), stages.mean())

    stZs = stars["metallicity"]

    stmasses = correct_mass(sim, stages, stars["mass"], stZs)

    sfrs100 = stmasses[stages < 100.0] / 100.0

    mass_in_3r50 = np.sum(stmasses[stars_in_3r50])
    sfr100_in_3r50 = np.sum(
        stmasses[stars_in_3r50][stages[stars_in_3r50] < 100] / 100.0
    )

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
        prev_rad = abs(star_bulk_vel * (time - sim_times[istep - 1]) * 1.5)
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

    img_dens = make_amr_img_smooth(
        fig,
        ax_dens,
        "density",
        snap,
        sim,
        tgt_pos,
        rad_tgt,
        zdist,
        NH_read=True,
        **dens_args,
    )
    img_temp = make_amr_img_smooth(
        fig,
        ax_temp,
        "temperature",
        snap,
        sim,
        tgt_pos,
        rad_tgt,
        zdist,
        NH_read=True,
        **temperature_args,
    )

    plot_stars(
        fig,
        ax_mass,
        sim,
        aexp,
        directions,
        nstar_bins,
        stmasses,
        stars["pos"],
        tgt_pos,
        rad_tgt * (sim.cosmo.lcMpc * 1e3),
        **smass_args,
    )

    plot_stars(
        fig,
        ax_sfr,
        sim,
        aexp,
        directions,
        nstar_bins,
        sfrs100,
        stars["pos"],
        tgt_pos,
        rad_tgt * (sim.cosmo.lcMpc * 1e3),
        **sfr_arg,
    )

    plot_stars(
        fig,
        ax_age,
        sim,
        aexp,
        directions,
        nstar_bins,
        stages,
        stars["pos"],
        tgt_pos,
        rad_tgt * (sim.cosmo.lcMpc * 1e3),
        **stage_arg,
    )

    # img_dens = plot_fields(
    #     "density", fig, ax_dens, aexp, dir_choice, tgt_pos, rad_tgt, sim, **dens_args
    # )

    # img_mass = plot_fields(
    #     "stellar mass", fig, ax_mass, aexp, dir_choice, tgt_pos, rad_tgt, sim, **smass_args
    # )

    # img_sfr = plot_fields(
    #     "SFR100", fig, ax_sfr, aexp, dir_choice, tgt_pos, rad_tgt, sim, **sfr_arg
    # )

    # img_age = plot_fields(
    #     "stellar age", fig, ax_age, aexp, dir_choice, tgt_pos, rad_tgt, sim, **stage_arg
    # )

    # img_temp = plot_fields(

    #     "temperature", fig, ax_temp, aexp, dir_choice, tgt_pos, rad_tgt, sim, **temperature_args
    # )
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
                hm="TREE_STARS_AdaptaHOP_dp_SCnew_gross",
                gal_markers=gal_markers,
                annotate=annotate,
                direction=directions[0],
                brick_fct=gal_fct,
                transpose=True,
                # **args,
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
                hm="TREE_DM",
                gal_markers=gal_markers,
                annotate=annotate,
                direction=directions[0],
                transpose=True,
                # brick_fct = halo_fct,
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
            sink_read_fct=bh_fct,
            direction=directions[0],
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
