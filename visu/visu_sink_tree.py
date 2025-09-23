import numpy as np
from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt

# from matplotlib.colors import LogNorm
# from matplotlib.patches import Circle
import os

from hagn.utils import get_hagn_sim
from hagn.IO import read_hagn_snap_brickfile, read_hagn_sink_bin
from scipy.spatial import KDTree


from zoom_analysis.constants import *
from zoom_analysis.halo_maker.read_treebricks import read_zoom_brick
from zoom_analysis.sinks.sink_reader import (
    check_if_superEdd,
    # find_zoom_massive_central_sink,
    hid_to_sid,
    read_sink_bin,
)

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
)

from f90_tools.star_reader import read_part_ball_NCdust, read_part_ball_hagn

# from zoom_analysis.halo_maker.read_treebricks import read_zoom_stars
from zoom_analysis.halo_maker.assoc_fcts import (
    # find_zoom_tgt_gal,
    # get_gal_assoc_file,
    # find_snaps_with_gals,
    find_snaps_with_halos,
    get_halo_props_snap,
)

from zoom_analysis.sinks.sink_reader import (
    get_sink_mhistory,
    # find_zoom_central_massive_sink,
)
from zoom_analysis.zoom_helpers import (
    decentre_coordinates,
    recentre_coordinates,
    starting_hid_from_hagn,
)

import argparse

# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479_old"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_old"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_kicks"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_smooth_ref"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag"
# sim_dir = "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_drag"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_lowerSFE"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_minMassDynGrav"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_minMassDynHydroGravBHL"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_pdrag"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_vlowSFE"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag"
# sim_dir = "/data102/jlewis/sims/lvlmax_21/mh1e12/id180130_model6_eps0p05"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE_lowSN"
# sim_dir = "/data103/jlewis/sims/lvlmax_21/mh1e12/id180130_256"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_smooth_ref"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowSFE_DynBondiGravSinkMass"
# sim_dir = "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth"
# sim_dir = "/data103/jlewis/sims/lvlmax_20/mh1e12/id18
# sim_dir = (
#     "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_boostNH_resimBoostFriction"
# )
# sim_dir = "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_frcAccrt"
# sim_dir = "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_early_refine"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_lowerSFE_lowNsink"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_low_dboost"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_eagn_T0p15"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_lowerSFE_NHboost"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_stricterSF"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictestSF_lowSNe"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_SuperMegaSF_midSNe"
sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_XtremeLowSFE_stgNHboost_strictSF"
# sim_dir="/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_stricterSF_radioHeavy"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099_inter"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id112288"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_noHydroDragB4z4merge"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE"
# sim_dir = "/data103/jlewis/sims/lvlmax_22/mh1e12/id52380_NClike"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh"
# sim_dir = "/data103/jlewis/sims/lvlmax_22/mh1e12/id138140"

argParse = argparse.ArgumentParser()

argParse.add_argument("--sid", type=int, default=None, help="Sink ID to follow")
argParse.add_argument("--hid", type=int, default=None, help="Halo ID of sink to follow")
argParse.add_argument("--ztgt", type=float, default=None, help="Target redshift")

args = argParse.parse_args()

sid_tgt = args.sid
hid_tgt = args.hid
ztgt = args.ztgt

hagn = False
delta_aexp = 0.05

intID = int(sim_dir.split("/")[-1].split("_")[0][2:])

if not hagn:
    sim = ramses_sim(sim_dir, nml="cosmo.nml")
    bh_fct = read_sink_bin
    gal_fct = read_zoom_brick
    read_star_fct = read_part_ball_NCdust
    hm = "HaloMaker_stars2_dp_rec_dust/"
    hm_dm = "HaloMaker_DM_dust/"

else:
    # for getting hagn plots... need to lookup massive hagn sink id. Can use one of the codes in ../sinks
    sim = get_hagn_sim()
    bh_fct = read_hagn_sink_bin
    gal_fct = read_hagn_snap_brickfile
    read_star_fct = read_part_ball_hagn
    hm = None
    # intID = 242756
    sim_dir = os.path.join("/data101/jlewis/hagn", "maps", "halos", f"{intID}")

name = sim.name


# zoom_ctr = sim.zoom_ctr


snaps = sim.snap_numbers
sim_aexps = sim.get_snap_exps(param_save=False)
sim_times = sim.get_snap_times(param_save=False)
sim_snaps = sim.snap_numbers


zstt = 2.0
# tgt_zed = 2.0
max_zed = 10.0

# delta_t = 5  # Myr
# every_snap = True  # try and interpolate between tree nodes if not found

rad_fact = 1.0  # fraction of radius to use as plot window
plot_win_str = str(rad_fact).replace(".", "p")

overwrite = False
# overwrite = True
clean = False  # no markers for halos/bhs
gal_markers = True
halo_markers = True
pdf = False
annotate = False
# zdist = 50  # ckpc
# rad_tgt = 50  # ckpc

zdist = 25  # ckpc
rad_tgt = 25  # ckpc

dv1 = [0, 0, 1]
dv2 = [0, 1, 0]
dv3 = [1, 0, 0]

nstar_bins = 500
# hagn_sim = get_hagn_sim()


field = "density"
vmin = 1e-26
vmax = 1e-21
op = np.sum
log_color = True

# field = "ilevel"
# vmin = 16
# vmax = sim.namelist["amr_params"]["levelmax"]
# op = np.max
# log_color = False

stvmin = 1e5
stvmax = 1e10

# field = "temperature"
# vmin = 1e3
# vmax = 1e8

# field = "pressure"
# vmin = 10  # k
# vmax = 1e5
# tree_hids, tree_datas, tree_aexps = read_tree_rev(
#     tree_path, tgt_zed, tree_type="halo", target_fields=["m", "x", "y", "z", "r"]
# )


# sim = ramses_sim(sim_dir, nml="cosmo.nml")
sim.init_cosmo()
l = sim.unit_l(sim.aexp_stt) / (ramses_pc * 1e6) / sim.aexp_stt


# sim_aexps = sim.get_snap_exps()  # [::-1]
# sim_times = sim.get_snap_times()  # [::-1]
# sim_snaps = sim.snap_numbers  # [::-1]


h_snaps = find_snaps_with_halos(snaps, sim.path)[:-1]
h_aexps = sim.get_snap_exps(h_snaps)
h_times = sim.get_snap_times(h_snaps)

print(h_snaps)


# print(h_snaps[closest_arg], h_aexps[closest_arg], hid_start)
if hid_tgt != None:
    print(f"Using specified halo id: {hid_tgt:d}")
    assert ztgt != None, "Need to specify target redshift alongside halo id"

decal = len(h_aexps) - 1
if ztgt == None:
    ztgt = 1.0 / h_aexps[decal] - 1

sink_files = os.listdir(sim.sink_path)
sink_fnbs = np.asarray([int(f.split("_")[1].split(".")[0]) for f in sink_files])
last_sink_file = sink_files[np.argmax(sink_fnbs) - 1]

last_sinks = read_sink_bin(
    os.path.join(sim.sink_path, last_sink_file), tgt_fields=["identity", "mass"]
)
last_sink_aexp = last_sinks["aexp"]
last_sink_zed = 1.0 / last_sink_aexp - 1.0

# closest_arg = np.argmin(np.abs(1.0 / h_aexps - 1.0 - last_sink_aexp))
# mass = last_sinks["mass"]
# mass_order = np.sort(mass)
# mass_arg = np.where(mass == mass_order[-4])[0][0]
# sid = last_sinks["identity"][mass_arg]

find_sink_zed = max(ztgt, last_sink_zed)

hagn_sim = get_hagn_sim()


last_snap_aexp = h_aexps[-2]
last_snap_zed = 1.0 / last_snap_aexp - 1.0


print(last_snap_zed, find_sink_zed)




found = False

# while not found and decal > 0:

print(1.0 / h_aexps[decal] - 1)


if hid_tgt == None:

    hid_start, halos, galaxies, start_aexp, found = starting_hid_from_hagn(
        zstt,
        sim,
        hagn_sim,
        intID,
        h_aexps[:decal],
        h_times[:decal],
        ztgt=ztgt,
    )

else:
    hid_start = hid_tgt
    start_aexp = 1.0 / (ztgt + 1)
    found = True

print(hid_start, start_aexp, found)

sim_snap = sim.get_closest_snap(aexp=start_aexp)

# # print(tgt_zed, last_snap_zed, find_sink_zed)

# find_sink_zed = 1.0 / h_aexps[:decal][closest_arg] - 1.0

# print(hid_start, sim_snap)
if sid_tgt == None:
    print(
        "no specified sink id, trying to follow most massive at last snap in zoom target halo"
    )
    sid, found = hid_to_sid(sim, hid_start, sim_snap, debug=False)
else:
    found = True
    sid = sid_tgt
# if not found:
#     decal -= 1

# sid = 3

# assert found, "Didn't find a sink at "

print("Found starting massive sink:", sid, "at z=", 1.0 / h_aexps[decal] - 1)

# sid = find_zoom_massive_central_sink(sim, 1.0 / (find_sink_zed + 1.0))
# sid = 120

sink_hist = get_sink_mhistory(
    sid,
    sim_snap,
    sim,
    hagn=hagn,
    out_keys=[
        ("position", 3),
        ("mass", 1),
        ("dMBH_coarse", 1),
        ("dMEd_coarse", 1),
        ("dens", 1),
        ("csound", 1),
        ("vrel", 1),
    ],
)

print(f"found sink mass {sink_hist["mass"].max():.1e}")

sink_hist_time = sim.cosmo_model.age(sink_hist["zeds"]).value * 1e3  # Myr

print("Got sink history")


if check_if_superEdd(sim):
    sink_growth = np.min([sink_hist["dMBH_coarse"], sink_hist["dMBH_coarse"]], axis=0)
else:
    sink_growth = sink_hist["dMBH_coarse"]

# print(sink_hist["zeds"].min(), sink_hist["zeds"].max())

rad_tgt_str = str(rad_tgt).replace(".", "p")


outdir = os.path.join(
    sim_dir,
    "maps_own_tree",
    "sinks",
    f"{sid:d}_r{rad_tgt_str}ckpc",
)


if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

option_str = ""
if annotate and not clean:
    option_str += "_annotate"
elif clean:
    option_str += "_clean"


for snap, aexp, time in zip(h_snaps[::-1], h_aexps[::-1], h_times[::-1]):

    print(snap, 1.0 / aexp - 1)

    zed = 1.0 / aexp - 1.0
    if zed > max_zed or zed < find_sink_zed:
        print("outside of redshift range")
        continue

    fout = os.path.join(
        outdir,
        f"{field[:4]}_{snap:05d}_{rad_tgt_str}ckpc{option_str:s}.png",
    )

    if os.path.isfile(fout) and not overwrite:
        print("file exists... skipping")
        continue

    l_pMpc = l * aexp

    # zed_diff = np.abs(zed - sink_hist["zeds"])
    # # print(zed_diff.min(), zed, sink_hist["zeds"][np.argmin(zed_diff)])
    # if np.all(zed_diff > 0.1):
    #     continue
    # sink_hist_arg = np.argmin(zed_diff)

    aexp_diff = np.abs(aexp - 1.0 / (sink_hist["zeds"] + 1.0))
    # print(zed_diff.min(), zed, sink_hist["zeds"][np.argmin(zed_diff)])
    if np.all(aexp_diff > delta_aexp):
        print("no sink data for this snap")
        continue

    sink_hist_arg = np.argmin(aexp_diff)

    if (
        sink_hist["zeds"][sink_hist_arg] > zed + delta_aexp
        or zed > sink_hist["zeds"].max()
    ):
        print("outside of sink history")
        continue

    print(zed, sink_hist["zeds"][sink_hist_arg])

    # is_snap = sink_hist["snap"] == snap
    # if np.sum(is_snap) == 0:
    #     continue
    # sink_hist_arg = np.where(is_snap)[0]

    # print(zed, sink_hist["zeds"][sink_hist_arg])
    tgt_pos = sink_hist["position"][sink_hist_arg]  # * sim.cosmo.lcMpc * 1e3
    cur_mass = sink_hist["mass"][sink_hist_arg]
    cur_mdot = sink_growth[sink_hist_arg]
    cur_dens = sink_hist["dens"][sink_hist_arg]
    cur_cs = sink_hist["csound"][sink_hist_arg]
    cur_vrel = sink_hist["vrel"][sink_hist_arg]

    # print(snaps)

    # if not os.path.exists(get_gal_assoc_file(sim.path, snap)):
    #     print("no association file")
    #     continue

    # try:
    #     gid, gprops = find_zoom_tgt_gal(sim, 1.0 / aexp - 1.0)
    # except IndexError:
    #     print("no galaxy found")
    #     continue

    fig, axs = plt.subplots(
        3,
        2,
        figsize=(20, 20),
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    # plt.subplots_adjust(wspace=0.3, hspace=0.0)

    ax_img, ax_stars, ax_mass, ax_dens, ax_growth, ax_cs = axs.flatten()

    # zoom_stars = read_zoom_stars(sim, snap, gid)
    # zoom_stars = read_star_fct(
    #     sim,
    #     snap,
    #     tgt_pos,
    #     rad_tgt / (sim.cosmo.lcMpc * 1e3),
    #     ["birth_time", "metallicity", "mass", "pos"],
    #     fam=2,
    # )

    # ages = zoom_stars["age"]
    # Zs = zoom_stars["metallicity"]
    # masses = zoom_stars["mass"]

    # masses = correct_mass(sim, ages, masses, Zs)
    # stpos = zoom_stars["pos"]

    # xbins = np.linspace(stpos[:, 1].min(), stpos[:, 1].max(), nstar_bins)
    # ybins = np.linspace(stpos[:, 2].min(), stpos[:, 2].max(), nstar_bins)

    # ctr_st = np.mean(stpos, axis=0)
    # rad_gal = np.max(np.linalg.norm(np.asarray(stpos) - ctr_st, axis=1))

    # smallest possible value is 5 stellar particle masses
    # if len(masses) > 0:
    #     vmin_stmass = masses.min() * 5

    # plot_stars(
    #     fig,
    #     ax_stars,
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
    #     vmax=1e6,
    #     # vmin=vmin_stmass,
    # )

    halos_snap = get_halo_props_snap(sim.path, snap)

    rvirs = np.asarray(
        [halos_snap[halo_k]["rvir"] for halo_k in halos_snap if "halo_" in halo_k]
    )
    hpos = np.asarray(
        [halos_snap[halo_k]["pos"] for halo_k in halos_snap if "halo_" in halo_k]
    )

    fpure = halos_snap["fpure"] > 1 - 1e-4

    h_tree = KDTree([tgt_pos], boxsize=1 + 1e-6)

    sink_args = h_tree.query_ball_point(hpos[fpure], rvirs[fpure])

    for isink, sink_arg in enumerate(sink_args):
        if len(sink_arg) > 0:
            contains_sink = isink
            break

    cur_hid = halos_snap["hid"][fpure][contains_sink]

    stimg = plot_fields(
        "stellar mass",
        fig,
        ax_stars,
        aexp,
        [dv1, dv2, dv3],
        tgt_pos,
        rad_tgt / (sim.cosmo.lcMpc * 1e3),
        sim,
        cb=True,
        vmin=stvmin,
        # vmax=stvmax,
        hid=cur_hid,
        transpose=True,
        cmap="grey",
        log=True,
    )

    # # fix, should subtract the historical position
    # sink_hist_pos_dv1 = np.dot(sink_hist["position"][:, :] - tgt_pos[:], dv1)
    # sink_hist_pos_dv2 = np.dot(sink_hist["position"][:, :] - tgt_pos[:], dv2)
    # sink_hist_pos_dv3 = np.dot(sink_hist["position"][:, :] - tgt_pos[:], dv3)

    # # plot main sink trail in red
    # plot_trail(
    #     ax_stars,
    #     np.transpose([sink_hist_pos_dv2, sink_hist_pos_dv3]) * sim.cosmo.lcMpc * 1e3,
    #     sink_hist_time,
    #     time,
    #     50,
    # )

    # # find snap that was more than 50 Myr ago
    # prev_time = time - 50
    # prev_snap = sim.get_closest_snap(time=prev_time)

    # # snaps between now and then
    # possible_snaps = np.arange(prev_snap - 1, snap + 1)
    # halo_snaps = find_snaps_with_halos(snaps, sim.path)
    # tgt_snaps = np.intersect1d(possible_snaps, halo_snaps)
    # tgt_aexps = sim.get_snap_exps(tgt_snaps)
    # tgt_times = sim.get_snap_times(tgt_snaps)

    # gal_ctrs = np.zeros((len(tgt_snaps), 3))

    # for i, (tgt_snap, tgt_aexp) in enumerate(zip(tgt_snaps, tgt_aexps)):
    #     try:
    #         gid, gprops = find_zoom_tgt_gal(sim, 1.0 / tgt_aexp - 1.0)
    #     except IndexError:
    #         continue

    #     gal_ctrs[i] = gprops["pos"]

    # zero_ctrs = np.any(gal_ctrs > 0, axis=1)

    # gal_hist_pos_dv1 = np.dot(gal_ctrs[zero_ctrs, :] - tgt_pos[:], dv1)
    # gal_hist_pos_dv2 = np.dot(gal_ctrs[zero_ctrs, :] - tgt_pos[:], dv2)
    # gal_hist_pos_dv3 = np.dot(gal_ctrs[zero_ctrs, :] - tgt_pos[:], dv3)

    # # loop over snaps separating cur snap and that one 50 Myr ago
    # # for each get zoom ctr gal and pos

    # # get gal coords into right basis and centroid

    # # plot main galaxy trail in blue
    # plot_trail(
    #     ax_stars,
    #     np.transpose([gal_hist_pos_dv2, gal_hist_pos_dv3]) * sim.cosmo.lcMpc * 1e3,
    #     tgt_times[zero_ctrs],
    #     time,
    #     50,
    #     color="b",
    # )

    # ax_img.text(
    #     0.05,
    #     0.05,
    #     f"M={cur_mass:.1e} Msun\nMdot={cur_mdot:.1e} Msun/yr",
    #     # transform=ax_img.transAxes,
    #     color="white",
    # )

    # rad_tgt = cur_r50 * rad_fact
    zdist = rad_tgt  # / 1 * sim.cosmo.lcMpc * 1e3

    # print(tgt_pos, rad_tgt / (sim.cosmo.lcMpc * 1e3), zdist)
    # print(tgt_pos - tgt_rad * 0.15, tgt_pos + tgt_rad * 0.15)

    # make_amr_img_smooth(
    #     fig,
    #     ax_img,
    #     field,
    #     snap,
    #     sim,
    #     tgt_pos,
    #     rad_tgt / (sim.cosmo.lcMpc * 1e3),
    #     # zdist=-1,
    #     zdist=zdist,
    #     debug=False,
    #     vmin=vmin,
    #     vmax=vmax,
    #     # vmax=1e-22,
    #     cb=True,
    #     direction=dv1,
    # )

    img = plot_fields(
        field,
        fig,
        ax_img,
        aexp,
        [dv1, dv2, dv3],
        tgt_pos,
        rad_tgt / (sim.cosmo.lcMpc * 1e3),
        sim,
        cb=True,
        vmin=vmin,
        vmax=vmax,
        transpose=True,
        cmap="magma",
        log=log_color,
        hid=cur_hid,
        op=op,
    )

    if not clean:

        for plot_ax, color in zip([ax_img, ax_stars], ["white", "red"]):
            # plot_ax.scatter(
            #     0, 0, s=200, c="r", marker="+", label="HAGN Halo center", zorder=999, lw=1
            # )

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

            # plot_ax.add_patch(circ)

            # plot zoom galaxies

            if plot_ax == ax_img:
                legend = True
            else:
                legend = False

            if gal_markers:
                plot_zoom_gals(
                    plot_ax,
                    snap,
                    sim,
                    tgt_pos,
                    rad_tgt / (1e3 * sim.cosmo.lcMpc),
                    zdist,
                    hm=hm,
                    annotate=annotate,
                    brick_fct=gal_fct,
                    legend=legend,
                    color=color,
                    direction=dv1,
                    transpose=True,
                )

            # plot zoom halos
            if halo_markers:
                plot_zoom_halos(
                    plot_ax,
                    snap,
                    sim,
                    tgt_pos,
                    rad_tgt / (1e3 * sim.cosmo.lcMpc),
                    zdist,
                    hm=hm_dm,
                    annotate=annotate,
                    brick_fct=gal_fct,
                    legend=legend,
                    color=color,
                    direction=dv1,
                    transpose=True,
                )

            # plot zoom BHs
            try:
                plot_zoom_BHs(
                    plot_ax,
                    snap,
                    sim,
                    tgt_pos,
                    rad_tgt,
                    zdist,
                    sink_read_fct=bh_fct,
                    legend=legend,
                    annotate=annotate,
                    color=color,
                    direction=dv1,
                    transpose=True,
                )
            except (ValueError, AssertionError):
                pass

    # now put a single maker at current time and mass
    # ax_mass.axvline(zed, color="r", lw=1.5)

    ax_growth.plot(sink_hist["zeds"], sink_growth, color="tab:blue", lw=3)
    ax_growth.scatter(
        zed,
        cur_mdot,
        color="r",
        s=100,
        zorder=999,
    )
    ax_growth.set_yscale("log")
    ax_growth.grid()

    ax_growth.set_xlabel("redshift")
    ax_growth.set_ylabel("BH growth rate, $M_\odot$/yr")
    ax_growth.set_xlim(xmax=max_zed, xmin=sink_hist["zeds"].min())
    ax_growth.invert_xaxis()

    ax_mass.plot(
        sink_hist["zeds"],
        sink_hist["mass"],
        color="tab:blue",
        lw=3,
    )
    ax_mass.scatter(
        zed,
        cur_mass,
        s=100,
        c="r",
        zorder=999,
    )
    ax_mass.set_yscale("log")
    ax_mass.grid()

    ax_mass.set_xlabel("redshift")
    ax_mass.set_ylabel("BH mass, $M_\odot$")

    ax_mass.set_xlim(xmax=max_zed, xmin=sink_hist["zeds"].min())
    ax_mass.invert_xaxis()

    ax_dens.plot(
        sink_hist["zeds"],
        sink_hist["dens"],
        color="tab:blue",
        lw=3,
    )

    ax_dens.scatter(zed, cur_dens, s=100, c="r", zorder=999)

    ax_dens.set_yscale("log")
    ax_dens.grid()

    ax_dens.set_xlabel("redshift")
    ax_dens.set_ylabel("cloud density, H/cm$^3$")
    ax_dens.set_xlim(xmax=max_zed, xmin=sink_hist["zeds"].min())
    ax_dens.invert_xaxis()

    ax_cs.plot(
        sink_hist["zeds"],
        sink_hist["vrel"],
        color="tab:blue",
        lw=3,
    )

    ax_cs.scatter(
        zed,
        cur_vrel,
        s=100,
        c="r",
        zorder=999,
    )

    ax_cs.set_yscale("log")
    ax_cs.grid()

    ax_cs.set_xlabel("redshift")
    ax_cs.set_ylabel("vrel, km/s")
    ax_cs.set_xlim(xmax=max_zed, xmin=sink_hist["zeds"].min())
    ax_cs.invert_xaxis()

    plt.tight_layout()

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
