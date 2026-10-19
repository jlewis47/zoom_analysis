
from collections.abc import Iterable

from zoom_analysis.read.read_data import read_data_ball
from zoom_analysis.sinks.sink_reader import  snap_to_coarse_step, read_sink_bin, read_sink_output
from zoom_analysis.sinks.sink_writer import write_sink_bin, write_sink_output

# from zoom_analysis.trees.tree_reader import read_tree_file_rev as read_tree_rev_sim
from zoom_analysis.trees.tree_reader import (
    convert_adaptahop_pos,
    read_tree_file_rev_correct_pos as read_tree_rev_sim,
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

from matplotlib import gridspec, pyplot as plt


def merge_common_keys(dict1, dict2):
    """
    Concatenate the values of common keys from dict2 into dict1.

    Rules:
      - np.ndarray + np.ndarray -> np.concatenate(axis=0)
      - list + iterable -> list
      - tuple + iterable -> tuple
      - other iterables -> list
      - scalars are treated as length-1 sequences
    """
    result = dict1.copy()

    for key in dict1.keys() & dict2.keys():

        print(key)

        left = dict1[key]
        right = dict2[key]

        # Preserve ndarray dimensionality
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            result[key] = np.concatenate((left, right), axis=0)
            continue

        def to_list(x):
            if isinstance(x, (str, bytes)):
                return [x]
            if isinstance(x, Iterable):
                return list(x)
            return [x]

        merged = to_list(left) + to_list(right)

        if isinstance(left, tuple):
            result[key] = tuple(merged)
        elif isinstance(left, list):
            result[key] = np.array(merged)
        else:
            result[key] = np.array(merged)

    return result

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

sdir="/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380"


tgt_snap=105
tgt_mass = 1e6

hagn_sim = get_hagn_sim()
hagn_sim.init_cosmo()
l_hagn = hagn_sim.unit_l(hagn_sim.aexp_stt) / (ramses_pc * 1e6) / hagn_sim.aexp_stt
# colors = []

hagn_snaps = hagn_sim.snap_numbers
hagn_aexps = hagn_sim.get_snap_exps(param_save=False)
hagn_times = hagn_sim.cosmo_model.age(1.0 / hagn_aexps - 1.0).value * 1e3

tgt_zed = 1.0 / hagn_aexps[hagn_snaps == hagn_snap] - 1.0
tgt_time = cosmo.age(tgt_zed).value


sfr_min, ssfr_min, sfh_min = np.inf, np.inf, np.inf
sfr_max, ssfr_max, sfh_max = -np.inf, -np.inf, -np.inf

last_hagn_id = -1
isim = 0

# overwrite = True
# overwrite = False
main_stars = True
# main_stars = False

if main_stars:
    main_star_str = "mainStars"
else:
    main_star_str = ""

name = sdir.split("/")[-1].strip()

# sim_id = name[2:].split("_")[0]

# sim_ids = ["id74099", "id147479", "id242704"]

# c = "tab:blue"

# get the galaxy in HAGN

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

print(hagn_tree_hids)

for key in hagn_tree_datas:
    hagn_tree_datas[key] = hagn_tree_datas[key][0][:]

hagn_sim.init_cosmo()

hagn_tree_times = hagn_sim.cosmo_model.age(1.0 / hagn_tree_aexps - 1.0).value * 1e3
# print("halo id: ", intID)
# print(intID)

sim = ramses_sim(sdir, nml="cosmo.nml")

print(name, intID, sim.name, sim.path)

sim_snaps = sim.get_snaps()[1]
sim_aexps = sim.get_snap_exps()
sim_times = sim.get_snap_times()
sim.init_cosmo()
sim_times = sim.cosmo_model.age(1.0 / sim_aexps - 1.0).value * 1e3

# last sim_aexp
# valid_steps = hagn_tree_aexps < (1.0 / (tgt_zed + 1.0))  # sim.aexp_end
# nsteps = np.sum(valid_steps)

nsteps = len(sim_snaps)
nstep_tree_hagn = len(hagn_tree_aexps)

# find last output with assoc files
# assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
# assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

assoc_file_nbs = find_snaps_with_gals(sim_snaps, sim.path)

avail_aexps = np.intersect1d(
    sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
)

avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

prev_mass = -1
prev_pos = -1
prev_rad = -1
# prev_pos = None

# hagn tree loop
for istep, (aexp, time) in enumerate(zip(hagn_tree_aexps, hagn_tree_times)):

    hagn_snap = hagn_snaps[np.argmin(np.abs(hagn_aexps - aexp))]

    try:
        super_cat = make_super_cat(
            hagn_snap, outf="/data101/jlewis/hagn/super_cats"
            , overwrite=False)
    except FileNotFoundError:
        print("No super cat")
        continue

    gal_pties = get_cat_hids(super_cat, [int(hagn_tree_hids[0][istep])])

    # # gal_pties = get_cat_gids(super_cat, [int(hagn_tree_hids[0][istep])])

    if len(gal_pties["gid"]) == 0:
        print("No galaxy")
        continue

    # hagn_file = os.path.join(sim.path, f"stellar_history_{hagn_snap}.h5")

    # if not os.path.exists(hagn_file) or overwrite_hagn:

    # stars = gid_to_stars(
    #     gal_pties["gid"][0],
    #     hagn_snap,
    #     hagn_sim,
    #     ["mass", "birth_time", "metallicity"],
    # )

    # ages = stars["age"]
    # Zs = stars["metallicity"]

    # masses = sfhs.correct_mass(hagn_sim, ages, stars["mass"], Zs)

    # mstel_hagn[istep] = stars["mass"].sum()
    # sfr_hagn[istep] = np.sum(masses[ages < 1000] / 1e3)  # Msun/Myr
    # time_hagn[istep] = hagn_tree_times[istep]

    # # print(gal_pties)
    # if len(gal_pties["mgal"]) == 0:
    #     continue

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


hid_start, halo_dict, gal_dict, start_aexp, found = starting_hid_from_hagn(
    zstt, sim, hagn_sim, intID, avail_aexps, avail_times, debug=True
)

start_snap = sim.get_closest_snap(aexp=start_aexp)
# print(hid_start, halo_dict, gal_dict, start_aexp, found)

# print(hagn_ctr)

# print(hagn_ctr)

# hid_start = int(hosted_gals["hid"][np.argmax(hosted_gals["mass"])])

# load sim tree
sim_halo_tree_rev_fname = os.path.join(
    sim.path, "TreeMakerDM_dust", "tree_rev.dat"
)

sim_zeds = 1.0 / avail_aexps - 1

sim_tree_hids, sim_tree_datas, sim_tree_aexps = read_tree_rev_sim(
    sim_halo_tree_rev_fname,
    sim,
    start_snap,
    fbytes=os.path.join(sim.path, "TreeMakerDM_dust"),
    zstart=1.0 / start_aexp - 1.0,
    tgt_ids=[hid_start],
    star=False,
    verbose=True,
    # overwrite=overwrite,
    speed_lim=1e5,
    # speed_lim=np.inf,
    fpure_min=0.999,
)

found_steps = sim_tree_hids[0] > 0

# test_fig = plt.figure()
# test_ax = test_fig.add_subplot(111)
# test_ax.plot(sim_tree_aexps[found_steps], sim_tree_datas["x"][0][found_steps])
# test_ax.plot(sim_tree_aexps[found_steps], sim_tree_datas["y"][0][found_steps])
# test_ax.plot(sim_tree_aexps[found_steps], sim_tree_datas["z"][0][found_steps])
# # test_ax.set_ylim(0, 1)
# test_fig.savefig("test_smooth_pos")
# test_fig = plt.figure()
# test_ax = test_fig.add_subplot(111)
# test_ax.plot(sim_tree_aexps[found_steps], sim_tree_datas["r"][0][found_steps])
# # test_ax.set_ylim(0, 1)
# test_fig.savefig("test_smooth_r")

# print(sim_tree_datas["m"])

# print(sim_tree_hids)
# print(sim_tree_datas)

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
        # "velocity",
        "host hid",
        "host mass",
    ],
)

tgt_aexp = sim.get_snap_exps(tgt_snap)[0]

gal_tree_arg = np.argmin(np.abs(tgt_aexp-gal_props_tree['aexps']))

pos = gal_props_tree['pos'][gal_tree_arg]
# vel = gal_props_tree['vel'][gal_tree_arg]
hid = gal_props_tree['host hid'][gal_tree_arg]

#get sinkprops nb
stepnb,found = snap_to_coarse_step(tgt_snap,sim)

assert found, 'problem finding corresponding coarse step'

fdesc = os.path.join(sim.output_path,f"output_{tgt_snap:05d}","sink_file_descriptor.txt")
# fdesc = os.path.join(".","sink_file_descriptor.txt")

# cur_sinks = read_sink_bin(os.path.join(sim.sink_path,f"sink_{stepnb:05d}.dat"))
cur_sinks = read_sink_output(os.path.join(sim.output_path,f"output_{tgt_snap:05d}",f"sink_{tgt_snap:05d}.out00001"),fdesc)

# convert_sink_units(cur_sinks, tgt_aexp, sim)

last_sink_id = cur_sinks['identity'].max()

unit_l = sim.unit_l(tgt_aexp)
unit_d = sim.unit_d(tgt_aexp)
unit_t = sim.unit_t(tgt_aexp)
unit_m = unit_d * unit_l**3 / 2e33


lbox = 1.
four_dx = 4 / 2**19 * lbox
#about 1 ckpc

stars = read_data_ball(sim,tgt_snap,pos,tgt_r=four_dx, host_halo=int(hid),data_types=['stars'],tgt_fields=['vel','mass'])

msrd_vel=np.average(stars['vel'],weights=stars['mass'],axis=0)


my_new_sink={}

my_new_sink["identity"] =[ last_sink_id + 1]
my_new_sink["mass"] =[ tgt_mass / unit_m]
my_new_sink["position_x"] =[ pos[0] ]
my_new_sink["position_y"] =[ pos[1] ]
my_new_sink["position_z"] =[ pos[2] ]
my_new_sink["velocity_x"] =[ msrd_vel[0] / (unit_l / unit_t * 1e-5)]
my_new_sink["velocity_y"] =[ msrd_vel[1] / (unit_l / unit_t * 1e-5)]
my_new_sink["velocity_z"] =[ msrd_vel[2] / (unit_l / unit_t * 1e-5)]
my_new_sink["birth_time"]=[sim.get_snap_times(tgt_snap)[0] / (unit_t * 3600 * 24 * 365)]
my_new_sink["dMsmbh"]=[0]
my_new_sink["dMBH_coarse"]=[0]
my_new_sink["dMEd_coarse"]=[0]
my_new_sink["Esave"]=[0]
my_new_sink["jsink_x"]=[0.0]
my_new_sink["jsink_y"]=[0.0]
my_new_sink["jsink_z"]=[0.0]
my_new_sink["spin_x"]=[0.0]
my_new_sink["spin_y"]=[0.0]
my_new_sink["spin_z"]=[0.0]
my_new_sink["spin_magnitude"]=[0]
my_new_sink["eps_sink"]=[0]

#get anon keys
anon_keys = np.sort([k for k in cur_sinks if k.startswith('sink_stat')])

for k in anon_keys:
    if k not in my_new_sink:
        my_new_sink[k] = [0]


new_sinks = merge_common_keys(cur_sinks,my_new_sink)
# print(new_sinks['nsink'],new_sinks['aexp'])

new_sinks['nsink'] = np.int32(new_sinks['nsink'] + 1)
new_sinks['nindsink'] = np.int32(new_sinks['nindsink'] + 1)

write_sink_output(f"fake_sink_outs/new_sink_{stepnb:05d}.dat",fdesc,new_sinks)


# read_sink_bin(os.path.join(sim.sink_path,f"sink_{stepnb:05d}.dat"),debug=True)

test_sinks = read_sink_output(f"fake_sink_outs/new_sink_{stepnb:05d}.dat",fdesc,debug=True)