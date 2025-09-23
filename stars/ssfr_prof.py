from zoom_analysis.stars import sfhs

# from zoom_analysis.zoom_helpers import decentre_coordinates
# from zoom_analysis.stars.star_reader import yt_read_star_ball, read_part_ball
from zoom_analysis.stars.leja_quench_fit import sfr_ridge_leja22
from zoom_analysis.halo_maker.read_treebricks import (
    # read_brickfile,
    # convert_brick_units,
    convert_star_units,
    read_zoom_stars,
    # convert_star_time,
)
from zoom_analysis.read.read_data import read_data_ball

from dynamics import extract_nh_kinematics

# from zoom_analysis.trees.tree_reader import read_tree_file_rev as read_tree_fev_sim
from zoom_analysis.trees.tree_reader import (
    convert_adaptahop_pos,
    read_tree_file_rev_correct_pos as read_tree_fev_sim,
)
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import (
    find_zoom_tgt_gal,
    get_central_gal_for_hid,
    get_gal_props_snap,
    get_halo_props_snap,
    get_halo_assoc_file,
    get_assoc_pties_in_tree,
    smooth_props,
    find_snaps_with_gals,
    get_gal_assoc_file,
    get_halo_assoc_file,
)
from zoom_analysis.halo_maker.read_treebricks import read_zoom_stars

from zoom_analysis.zoom_helpers import find_starting_position, starting_hid_from_hagn

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D

import os
import numpy as np
import h5py

# from hagn.association import gid_to_stars
from hagn.utils import get_hagn_sim, adaptahop_to_code_units
from hagn.tree_reader import read_tree_rev, interpolate_tree_position
from hagn.catalogues import get_halos_cat, make_super_cat, get_cat_hids, get_cat_gids

from astropy.cosmology import Planck18 as cosmo

from hagn.catalogues import make_super_cat, get_cat_hids

# from f90_tools.star_reader import read_part_ball_hagn, read_part_ball_NCdust

import matplotlib.pyplot as plt


# sim_ids = [
#     "id2757_995pcnt_mstel/",
#     "id19782_500pcnt_mstel/",pip install setuptools
#     "id13600_005pcnt_mstel/",
# ]
# sdir = "/home/jlewis/sims/proposal_zooms/ExtractZoom/mh1e12/"
hagn_snap = 197  # z=2 in the full res box
HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
delta_t = 50  # Myr
zstt = 2.0


# sdir=    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380"
# sdir=    "/data103/jlewis/sims/lvlmax_22/mh1e12/id52380_NClike"
# sdir=    "/data103/jlewis/sims/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF"
# sdir=    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"
# sdir=    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE/"
# sdir=    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel"
# sdir=    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE"
# sdir=    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost"
# sdir=    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF"
sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictestSF_lowSNe"
# sdir=    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id112288"
# sdir=    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130"
# sdir=    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sdir=    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel"
# sdir=    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE"
# sdir=    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE_stgNHboost_stricterSF"


yax2 = None

hagn_sim = get_hagn_sim()
hagn_sim.init_cosmo()
l_hagn = hagn_sim.unit_l(hagn_sim.aexp_stt) / (ramses_pc * 1e6) / hagn_sim.aexp_stt
# colors = []

hagn_snaps = hagn_sim.snap_numbers
hagn_aexps = hagn_sim.get_snap_exps(param_save=False)
hagn_times = hagn_sim.cosmo_model.age(1.0 / hagn_aexps - 1.0).value * 1e3

tgt_zed = 1.0 / hagn_aexps[hagn_snaps == hagn_snap] - 1.0
tgt_time = cosmo.age(tgt_zed).value


last_hagn_id = -1
isim = 0

# overwrite = True
# # overwrite = False
main_stars = True
# main_stars = False

if main_stars:
    main_star_str = "mainStars"
else:
    main_star_str = ""

zoom_ls = [
    "-",
    "--",
    ":",
    "-.",
    (0, (1, 10)),
    (0, (3, 5, 1, 5)),
    (0, (1, 2, 1, 2, 5, 3)),
] * 15

lines = []
labels = []

sfr_dt = 100  # Myr

last_simID = None
l = None

zoom_style = 0


if sdir[-1] == "/":
    sdir = sdir[:-1]
name = sdir.split("/")[-1].strip()

# sim_id = name[2:].split("_")[0]

# sim_ids = ["id74099", "id147479", "id242704"]

# c = "tab:blue"

# get the galaxy in HAGN
# print(name)
# print(name[2:].split("_"))
intID = int(name[2:].split("_")[0])
# gid = int(make_super_cat(197, outf="/data101/jlewis/hagn/super_cats")["gid"][0])

hagn_tree_hids, hagn_tree_datas, hagn_tree_aexps = read_tree_rev(
    tgt_zed,
    [intID],
    tree_type="halo",
    # [gid],
    # tree_type="gal",
    target_fields=["m", "x", "y", "z", "r"],
    sim="hagn",
)

for key in hagn_tree_datas:
    hagn_tree_datas[key] = hagn_tree_datas[key][0][:]

hagn_sim.init_cosmo()

hagn_tree_times = hagn_sim.cosmo_model.age(1.0 / hagn_tree_aexps - 1.0).value * 1e3
# print("halo id: ", intID)
# print(intID)

sim = ramses_sim(sdir, nml="cosmo.nml")

sim_snaps = sim.get_snaps(mini_snaps=True)[1]
sim_aexps = sim.get_snap_exps()
sim_times = sim.get_snap_times()
sim.init_cosmo()
sim_times = sim.cosmo_model.age(1.0 / sim_aexps - 1.0).value * 1e3

# last sim_aexp
# valid_steps = hagn_tree_aexps < (1.0 / (tgt_zed + 1.0))  # sim.aexp_end
# nsteps = np.sum(valid_steps)

nbins = 10
rbins_ckpc = np.logspace(-1, 2, nbins)

# steps = [6,5,4,3.5,3,2.75,2.5,2.375,2.25,2.2,2.15,2.1,2.05,2.0,1.9] # z
steps = np.round(np.arange(2, 7, 0.2, dtype=np.float32)[::-1], decimals=1)
uniq_rounded_avail_zeds = np.unique(np.round(1.0 / sim_aexps - 1, decimals=1))
# print(uniq_rounded_avail_zeds,steps)
# print(np.in1d(steps,uniq_rounded_avail_zeds))
# print(np.in1d(np.round(1./sim_aexps-1,decimals=1),steps))
steps = np.intersect1d(steps, uniq_rounded_avail_zeds)
nsteps = len(steps)

# print(steps,nsteps)

mstel_profs = np.zeros((nsteps, nbins))
sfr_profs = np.zeros((nsteps, nbins))
vrot_profs = np.zeros((nsteps, nbins))
disp_profs = np.zeros((nsteps, nbins))

# hagn_mstel_profs = np.zeros((nsteps, nbins))
# hagn_sfr_profs = np.zeros((nsteps, nbins))

datas_path = os.path.join(sim.path, "computed_data", main_star_str)
fout_h5 = os.path.join(datas_path, f"stellar_history.h5")
read_existing = True

if not os.path.exists(datas_path):
    os.makedirs(datas_path)

# find last output with assoc files
# assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
# assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

assoc_file_nbs = find_snaps_with_gals(sim_snaps, sim.path)

avail_aexps = np.intersect1d(
    sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
)

if len(avail_aexps) == 0:
    print("No available aexps")


avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

prev_mass = -1
prev_pos = -1
prev_rad = -1
# prev_pos = None


hid_start, halo_dict, gal_dict, start_aexp, found = starting_hid_from_hagn(
    zstt,
    sim,
    hagn_sim,
    intID,
    avail_aexps,
    avail_times,
    ztgt=1.0 / avail_aexps.max() - 1,  # ztgt = np.min(steps)
)

start_snap = sim.get_closest_snap(aexp=start_aexp)
# print(hid_start, halo_dict, gal_dict, start_aexp, found)

# print(hagn_ctr)

# print(hagn_ctr)

# hid_start = int(hosted_gals["hid"][np.argmax(hosted_gals["mass"])])

# load sim tree
sim_halo_tree_rev_fname = os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat")
if not os.path.exists(sim_halo_tree_rev_fname):
    print("No tree file")


sim_zeds = 1.0 / avail_aexps - 1

# print(
#     1.0 / start_aexp - 1.0,
# )

# sim_tree_hids, sim_tree_datas, sim_tree_aexps = read_tree_fev_sim(
#     sim_halo_tree_rev_fname,
#     fbytes=os.path.join(sim.path, "TreeMakerDM_dust"),
#     zstart=1.0 / start_aexp - 1.0,
#     tgt_ids=[hid_start],
#     star=False,
# )
sim_tree_hids, sim_tree_datas, sim_tree_aexps = read_tree_fev_sim(
    sim_halo_tree_rev_fname,
    sim,
    start_snap,
    fbytes=os.path.join(sim.path, "TreeMakerDM_dust"),
    zstart=1.0 / start_aexp - 1.0,
    tgt_ids=[hid_start],
    star=False,
)

found_steps = sim_tree_hids[0] > 0

sim_tree_times = sim.cosmo_model.age(1.0 / sim_tree_aexps - 1).value * 1e3

# print(sim_tree_times)
# print(sim_tree_hids)

gal_props_tree = get_assoc_pties_in_tree(
    sim,
    sim_tree_aexps,
    sim_tree_hids[0],
    assoc_fields=[
        "gids",
        "r50",
        "rmax",
        "mass",
        "pos",
        "host hid",
        "host mass",
    ],
)

# print(gal_props_tree)

r50s = gal_props_tree["r50"]
rmaxs = gal_props_tree["rmax"]
masses = gal_props_tree["mass"]
poss = gal_props_tree["pos"]
hids = gal_props_tree["host hid"]
gal_exps = gal_props_tree["aexps"]

lsteps = len(hids)


smooth_gal_props = smooth_props(gal_props_tree)

for istep, tgt_zed in enumerate(steps):

    if tgt_zed < sim_zeds.min() - 0.1:
        print(f"z={tgt_zed:.2f} not in available aexps")
        continue

    tgt_aexp = 1.0 / (tgt_zed + 1.0)

    # hagn_snap = hagn_snaps[np.argmin(np.abs(hagn_aexps - tgt_aexp))]

    # try:
    #     super_cat = make_super_cat(
    #         hagn_snap, outf="/data101/jlewis/hagn/super_cats"
    #     )  # , overwrite=True)
    # except FileNotFoundError:
    #     print("No super cat")
    #     continue

    # gal_pties = get_cat_hids(super_cat, [int(hagn_tree_hids[0][istep])])

    # # # gal_pties = get_cat_gids(super_cat, [int(hagn_tree_hids[0][istep])])

    # if len(gal_pties["gid"]) == 0:
    #     print("No galaxy")
    #     continue

    # sfr_dt_hagn = [10, 100, 1000][
    #     np.argmin(np.abs(sfr_dt - np.array([10, 100, 1000])))
    # ]

    # # sim_snap = sim.get_closest_snap(aexp=aexp)

    # print(aexp, avail_aexps)

    # hid_start, halo_dict, start_hosted_gals, found, start_aexp = (
    #     find_starting_position(
    #         sim,
    #         avail_aexps[:],
    #         hagn_tree_aexps,
    #         hagn_tree_datas,
    #         hagn_tree_times,
    #         avail_times[:],

    #     )
    # )

    snap = sim.get_closest_snap(aexp=tgt_aexp)

    if not os.path.exists(get_halo_assoc_file(sim.path, snap)):
        print("no halo assoc file")
        continue

    if not os.path.exists(get_gal_assoc_file(sim.path, snap)):
        print("no gal assoc file")
        continue

    # assoc_file = assoc_files[assoc_file_nbs == snap]
    # if len(assoc_file) == 0:
    #     print(f"No assoc file for at z={1./aexp-1:.1f},snap={snap:d}")
    #     continue

    sim_tree_arg = np.argmin(np.abs(sim_tree_aexps - tgt_aexp))
    cur_snap_hid = sim_tree_hids[0][sim_tree_arg]
    if cur_snap_hid in [-1, 0]:
        print(f"No halo in tree at z={1./tgt_aexp-1:.1f},snap={snap:d}")
        continue

    # print()

    # gid, gal_dict = get_central_gal_for_hid(
    #     sim,
    #     cur_snap_hid,
    #     snap,
    #     prev_mass=prev_mass,
    #     prev_pos=prev_pos,
    #     prev_rad=prev_rad,
    #     main_stars=main_stars,
    #     # verbose=True,  # , prev_pos=prev_pos
    #     # debug=True,
    # )
    sim_tree_arg = np.argmin(np.abs(tgt_aexp - smooth_gal_props["aexps"]))

    # if gid == None:
    #     print(
    #         f"No central galaxy in halo {cur_snap_hid} at z={1./aexp-1:.1f},snap={snap:d}"
    #     )
    #     # prev_pos = -1
    #     # prev_rad = -1
    #     # prev_mass = -1
    #     continue
    # else:
    #     print(
    #         f'z={1./aexp-1:.1f},snap={snap:d},hid={cur_snap_hid},gid={gid},fpure={gal_dict["host purity"]}'
    #     )

    # print(gal_dict)

    # tgt_pos = gal_dict["pos"]
    tgt_pos = gal_props_tree["pos"][sim_tree_arg]
    # print(np.linalg.norm(tgt_pos - prev_pos))
    prev_pos = tgt_pos
    # tgt_r = gal_dict["r50"] * 3
    # tgt_r = smooth_gal_props["r50"][sim_tree_arg] * 3

    tgt_r = rbins_ckpc.max() / sim.cosmo.lcMpc / 1e3  # ckpc->code

    try:
        stars = read_data_ball(
            sim,
            snap,
            tgt_pos,
            tgt_r,
            host_halo=cur_snap_hid,
            data_types=["stars"],
            tgt_fields=["pos", "metallicity", "age", "mass", "vel"],
        )
    except FileNotFoundError:
        print(
            f"no data for stars, check output folder for snap {snap:d}, and availability of compressed halogal data"
        )
        continue

    if type(stars) == type(None):
        print("no stars")
        continue

    ages = stars["age"]
    Zs = stars["metallicity"]
    masses = stars["mass"]

    masses = masses = sfhs.correct_mass(sim, ages, masses, Zs)

    for ibin, rbin in enumerate(rbins_ckpc):

        in_r = np.linalg.norm(stars["pos"] - tgt_pos, axis=1) < (
            rbin / sim.cosmo.lcMpc / 1e3
        )

        if np.sum(in_r) == 0:
            continue

        rot_props, kin_props = extract_nh_kinematics(
            masses[in_r],
            stars["pos"][in_r],
            stars["vel"][in_r],
            tgt_pos,
        )

        vrot_profs[istep, ibin] = rot_props["Vrot"]
        disp_profs[istep, ibin] = rot_props["disp"]

        mstel_profs[istep, ibin] = np.sum(masses[in_r])
        yng = ages[in_r] < sfr_dt
        sfr_profs[istep, ibin] = np.sum(masses[in_r][yng]) / sfr_dt


fig, ax = plt.subplots(2, 3, sharex=True, figsize=(15, 10))

# colors = plt.cm.viridis((np.asarray(steps)-np.min(steps))/(np.max(steps)-np.min(steps)))
colors = plt.cm.viridis(np.linspace(0, 1, len(steps)))

for istep in range(len(steps)):

    (c,) = ax[0, 0].plot(
        rbins_ckpc,
        mstel_profs[istep],
        c=colors[istep],
        linestyle=zoom_ls[zoom_style],
        label=f"z={steps[istep]}",
    )
    ax[0, 1].plot(
        rbins_ckpc,
        sfr_profs[istep] / 1e6,
        color=c.get_color(),
        linestyle=zoom_ls[zoom_style],
    )

    ax[1, 0].plot(
        rbins_ckpc,
        sfr_profs[istep] / mstel_profs[istep] / 1e6,
        color=c.get_color(),
        linestyle=zoom_ls[zoom_style],
    )
    ax[1, 1].plot(
        rbins_ckpc,
        np.abs(vrot_profs[istep]) / disp_profs[istep],
        color=c.get_color(),
        linestyle=zoom_ls[zoom_style],
    )


ax[0, 0].set_xscale("log")
ax[0, 0].set_yscale("log")
ax[0, 0].set_xlabel("r [ckpc]")
ax[0, 0].set_ylabel("total M* <r [Msun]")

ax[0, 1].set_xscale("log")
ax[0, 1].set_yscale("log")
ax[0, 1].set_xlabel("r [ckpc]")
ax[0, 1].set_ylabel("total SFR <r [Msun/yr]")

ax[1, 0].set_xscale("log")
ax[1, 0].set_yscale("log")
ax[1, 0].set_xlabel("r [ckpc]")
ax[1, 0].set_ylabel("total sSFR <r [yr^-1]")

ax[1, 1].set_xscale("log")
ax[1, 1].set_xlabel("r [ckpc]")
ax[1, 1].set_ylabel("Vrot/sigma")

fig_dir = os.path.join("figs", main_star_str)
f_fig = os.path.join(fig_dir, f"star_profile_{sim.name}.png")

# get lines for legend from an axis
handles, labels = ax[0, 0].get_legend_handles_labels()

step_strs = [f"z={s:.2f}" for s in steps]
ax[1, 2].legend(handles, step_strs, framealpha=0.0, loc="center", fontsize=14)
ax[1, 2].axis("off")
ax[0, 2].axis("off")

print(f"wrote {f_fig}")

fig.savefig(f_fig)
