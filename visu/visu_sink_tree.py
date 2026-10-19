import numpy as np
from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt
from zoom_analysis.visu.visu_fct import basis_from_vect
# from matplotlib.colors import LogNorm
# from matplotlib.patches import Circle
import os

from hagn.utils import get_hagn_sim
from hagn.IO import read_hagn_snap_brickfile, read_hagn_sink_bin
from scipy.spatial import KDTree


from zoom_analysis.halo_maker.read_treebricks import (
    convert_star_units,
    read_zoom_brick,
    read_gal_stars,
    read_zoom_stars,
)


from zoom_analysis.constants import *
from zoom_analysis.halo_maker.read_treebricks import read_zoom_brick

from zoom_analysis.sinks.sink_reader import (
    check_if_superEdd,
    # find_zoom_massive_central_sink,
    hid_to_sid,
    read_sink_bin,
    read_sink_bin_NH,
    sid_to_hid

)
from zoom_analysis.read.read_data import read_data_ball

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

from f90_tools.star_reader import read_part_ball_NCdust, read_part_ball_hagn, read_part_ball_NH

# from zoom_analysis.halo_maker.read_treebricks import read_zoom_stars
from zoom_analysis.halo_maker.assoc_fcts import (
    # find_zoom_tgt_gal,
    # get_gal_assoc_file,
    # find_snaps_with_gals,
    find_snaps_with_halos,
    get_gal_props_snap,
    get_halo_props_snap,
)

from zoom_analysis.sinks.sink_reader import (
    get_sink_mhistory,
    read_sink_snap
    # find_zoom_central_massive_sink,
)
from zoom_analysis.zoom_helpers import (
    decentre_coordinates,
    recentre_coordinates,
    starting_hid_from_hagn,
)

import argparse

# def read_galbricks_NH(snap,sim,sim):

#     fstar = ""

#     stars = read_gal_stars(fstar)

#     convert_star_units(stars, snap, sim)    

#     return(stars)


argParse = argparse.ArgumentParser()


argParse.add_argument("sim_dir", type=str, help="path to sim")
argParse.add_argument("--sid", type=int, default=None, help="Sink ID to follow")
argParse.add_argument("--hid", type=int, default=None, help="Halo ID of sink to follow")
argParse.add_argument("--gid", type=int, default=None, help="Gal ID of sink to follow")
argParse.add_argument("--ztgt", type=float, default=None, help="Target redshift")
argParse.add_argument("--win", type=float, default=25, help="window size in ckpc")
argParse.add_argument("--annotate", action="store_true",  help="annotate halos/galaxies/BHs")
argParse.add_argument("--overwrite", action="store_true",  help="overwrite existing plots")

args = argParse.parse_args()

sid_tgt = args.sid
hid_tgt = args.hid
gid_tgt = args.gid
ztgt = args.ztgt
win_size = args.win
sim_dir = args.sim_dir
annotate=args.annotate
overwrite=args.overwrite

hagn = False
delta_aexp = 0.05
delta_time = 10 #Myr

if sim_dir.split("/")[-1].startswith('id'):
    intID = int(sim_dir.split("/")[-1].split("_")[0][2:])
else:
    intID = sim_dir.split("/")[-1]

nh = ("NH_for_rascas" in sim_dir) or ("newhorizon" in sim_dir.lower())

if not hagn and not nh:
    sim = ramses_sim(sim_dir, nml="cosmo.nml")
    bh_fct = read_sink_bin
    gal_fct = read_zoom_brick
    read_star_fct = read_part_ball_NCdust
    
    hm = "HaloMaker_stars2_dp_rec_dust/"
    hm_dm = "HaloMaker_DM_dust/"
   

elif nh: #my copy/link to NH compatible with my codes
    sim = ramses_sim(sim_dir, nml="cosmo.nml")
    bh_fct = read_sink_bin_NH
    # gal_fct = read_galbricks_NH
    gal_fct = read_zoom_brick
    read_star_fct = read_part_ball_NH
        
    hm_dm = "TREE_DM"
    hm = "TREE_STARS_AdaptaHOP_dp_SCnew_gross"

elif hagn:
    # for getting hagn plots... need to lookup massive hagn sink id. Can use one of the codes in ../sinks
    sim = get_hagn_sim()
    bh_fct = read_hagn_sink_bin
    gal_fct = read_hagn_snap_brickfile
    read_star_fct = read_part_ball_hagn
    
    hm = None
    # intID = 242756
    sim_dir = os.path.join("/data101/jlewis/hagn", "maps", "halos", f"{intID}")

else:
    print("didn't recognize simulation, maybe problems reading data")


def read_data(*args,**kwargs):

    return read_data_ball(*args,read_part_ball_fct=read_star_fct,**kwargs)

name = sim.name


# zoom_ctr = sim.zoom_ctr


snaps = sim.snap_numbers
sim_aexps = sim.get_snap_exps(param_save=False)
sim_times = sim.get_snap_times(param_save=False)
sim_snaps = sim.snap_numbers


# zstt = 5.0
# zstt = 2.0
# tgt_zed = 2.0

max_zed = np.inf
# max_zed = 8.0

# delta_t = 5  # Myr
# every_snap = True  # try and interpolate between tree nodes if not found

rad_fact = 1.0  # fraction of radius to use as plot window
plot_win_str = str(rad_fact).replace(".", "p")

# overwrite = False
# overwrite = True
clean = False  # no markers for halos/bhs
gal_markers = True
halo_markers = True
pdf = False
# annotate = False
# zdist = 50  # ckpc
# rad_tgt = 50  # ckpc

zdist = win_size  # ckpc
rad_tgt = win_size  # ckpc

dv1 = [0, 0, 1]
# dv2 = [0, 1, 0]
# dv3 = [1, 0, 0]

directions = list(basis_from_vect(dv1))

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

snaps_with_halos = find_snaps_with_halos(snaps, sim.path)[:]
h_snaps = snaps_with_halos
h_aexps = sim.get_snap_exps(h_snaps)
h_times = sim.get_snap_times(h_snaps)

# print(h_snaps)

# print(h_snaps[closest_arg], h_aexps[closest_arg], hid_start)
if hid_tgt != None:
    print(f"Using specified halo id: {hid_tgt:d}")
    assert ztgt != None, "Need to specify target redshift alongside halo id"

decal = 0
if ztgt == None:
    ztgt = 1.0 / h_aexps[decal-1] - 1
else:
    decal = np.argmin(np.abs(h_aexps-1/(ztgt+1)))

sink_files = os.listdir(sim.sink_path)
sink_fnbs = np.asarray([int(f.split("_")[1].split(".")[0]) for f in sink_files])
last_sink_file = sink_files[np.argmax(sink_fnbs)]

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

last_snap_aexp = h_aexps[-1]
last_snap_zed = 1.0 / last_snap_aexp - 1.0

# print(last_snap_zed, find_sink_zed, ztgt, last_sink_zed)

found = False

# while not found and decal > 0:

print(1.0 / h_aexps[decal-1] - 1)

assert hid_tgt is None or gid_tgt is None, "cannot specify both hid and gid number"

if hid_tgt == None and gid_tgt==None and type(intID) in [int,np.int32,np.int64]:

    try:
        hid_start, halos, galaxies, start_aexp, found = starting_hid_from_hagn(
            ztgt,
            sim,
            hagn_sim,
            intID,
            h_aexps[:decal],
            h_times[:decal],
            ztgt=ztgt,
        )
    except (IndexError):
        found=False

elif gid_tgt is not None:

    hid_start = get_gal_props_snap(sim.path,sim.get_closest_snap(zed=ztgt),gid=gid_tgt)['host hid']
    start_aexp = 1.0 / (ztgt + 1)
    found = True

elif hid_tgt is not None:
    hid_start = hid_tgt
    start_aexp = 1.0 / (ztgt + 1)
    found = True

stt_snap=sim.get_closest_snap(zed=ztgt)
start_aexp = sim.get_snap_exps(stt_snap)[0]
sim_snap = sim.get_closest_snap(aexp=start_aexp)

if sid_tgt == None:
    print(
        "no specified sink id, trying to follow most massive at last snap in zoom target halo"
    )
    
    halos = get_halo_props_snap(sim.path, sim_snap)
    pure = halos['fpure']>0.9999
    hid_start = halos['hid'][pure][np.argmax(halos['mvir'][pure])]
    sid, found, step = hid_to_sid(sim, hid_start, sim_snap, debug=False)
else:
    found = True
    sid = sid_tgt


if not found : #assume most massive sink
    print('looking for most massive sink')
    
    sinks = read_sink_snap(sim, snap = stt_snap)
    sid = sinks['identity'][np.argmax(sinks['mass'])]
    sim_snap = np.max(stt_snap)
    found=True
    # hid_start = sid_to_hid(sim, sid, stt_snap)

    # start_aexp = h_aexps.max()

    # hprops = get_halo_props_snap(sim.path,sim_snap)

    # pure = hprops['fpure']>0.9999
    # hid_start = hprops['hid'][pure][np.argmax(hprops['mvir'][pure])]

    

# print(hid_start, start_aexp, found)

# # print(tgt_zed, last_snap_zed, find_sink_zed)

# find_sink_zed = 1.0 / h_aexps[:decal][closest_arg] - 1.0

# print(hid_start, sim_snap)


# sid = 3

# assert found, "Didn't find a sink at "

# print(found)

print("Found starting massive sink:", sid, "at z=", 1.0 / h_aexps[decal-1] - 1)

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
    debug=False
)

print(f"found sink mass {sink_hist["mass"].max():.1e}")

sink_hist_time = sim.cosmo_model.age(sink_hist["zeds"]).value * 1e3  # Myr

print("Got sink history")

dists2sink = np.full_like(sink_hist_time,-1)

if check_if_superEdd(sim):
    sink_growth = np.min([sink_hist["dMEd_coarse"], sink_hist["dMBH_coarse"]], axis=0)
else:
    sink_growth = sink_hist["dMBH_coarse"]
sink_growth = np.max([sink_growth, np.full_like(sink_growth,1e-32)],axis=0)
sink_edd = sink_hist["dMEd_coarse"]

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

print('computing distance evolution')

sink_times = sim.cosmo_model.age(sink_hist['zeds']).value * 1e3

for snap, aexp, time in zip(h_snaps[::-1], h_aexps[::-1], h_times[::-1]):

    print(snap, 1.0 / aexp - 1)

    zed = 1.0 / aexp - 1.0
    if zed > max_zed or zed < find_sink_zed-1e-3:
        print("outside of redshift range")
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
        sink_times[sink_hist_arg] > time + delta_time
        or zed > sink_hist["zeds"].max()
        # sink_hist["zeds"][sink_hist_arg] > zed + delta_aexp
        # or zed > sink_hist["zeds"].max()
    ):
        print("outside of sink history")
        continue


    # print(zed, sink_hist["zeds"][sink_hist_arg])
    tgt_pos = sink_hist["position"][sink_hist_arg]  # * sim.cosmo.lcMpc * 1e3
    cur_mass = sink_hist["mass"][sink_hist_arg]
    cur_mdot = sink_growth[sink_hist_arg]
    cur_dens = sink_hist["dens"][sink_hist_arg]
    cur_cs = sink_hist["csound"][sink_hist_arg]
    cur_vrel = sink_hist["vrel"][sink_hist_arg]

    if not snap in snaps_with_halos:
        print("no halos for this snap")
        continue

    
    halos_snap = get_halo_props_snap(sim.path, snap)


    rvirs = np.asarray(
        [halos_snap[halo_k]["rvir"] for halo_k in halos_snap if "halo_" in halo_k]
    )
    hpos = np.asarray(
        [halos_snap[halo_k]["pos"] for halo_k in halos_snap if "halo_" in halo_k]
    )

    fpure = halos_snap["fpure"] > 1 - 1e-4

    # h_tree = KDTree([tgt_pos], boxsize=1 + 1e-6)
    # sink_args = h_tree.query_ball_point(hpos[fpure], rvirs[fpure])

    hdists = np.linalg.norm(hpos[fpure]-tgt_pos,axis=1)
    sink_args = np.where(hdists<rvirs[fpure])[0]

    # for isink, sink_arg in enumerate(sink_args):
    #     if len(sink_arg) > 0:
    #         contains_sink = isink
    #         break

    hmasses = halos_snap["mvir"][fpure][sink_args]

    possible_hids = halos_snap["hid"][fpure][sink_args]

    if len(possible_hids)>0:
        # print('No hids')
        # continue

        cur_hid = possible_hids[np.argmax(hmasses)]

        hprops,cur_gals = get_halo_props_snap(sim.path,snap,hids=cur_hid,hosted_gals=True)
        if 'gids' in cur_gals: #halos have galaxies
            cur_gids = cur_gals['gids']
            cur_gals = get_gal_props_snap(sim.path,snap,cur_gids)

            gals_pos = cur_gals['pos']
            gals_r= cur_gals['rmax']


            if type(gals_r) == np.ndarray:
                try:
                    gal2sink_dist = np.linalg.norm(gals_pos.T-tgt_pos,axis=1)
                    sink_in_gal_args = np.where(gal2sink_dist<gals_r*3)[0]

                    host_arg = sink_in_gal_args[np.argmax(cur_gals['mass'][sink_in_gal_args])]

                    dists2sink[sink_hist_arg] = gal2sink_dist[host_arg] * sim.cosmo.lcMpc*1e3
                except (IndexError,ValueError):
                    print('no nearby galaxy! setting distance to -1')
                    # break

            else:
                dists2sink[sink_hist_arg] = np.linalg.norm(gals_pos.T-tgt_pos) * sim.cosmo.lcMpc*1e3

    
print('making plots')

for snap, aexp, time in zip(h_snaps[::-1], h_aexps[::-1], h_times[::-1]):

    print(snap, 1.0 / aexp - 1)

    zed = 1.0 / aexp - 1.0
    if zed > max_zed or zed < find_sink_zed-1e-3:
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

    

    # print(zed, sink_hist["zeds"][sink_hist_arg])

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
    cur_dist2sink = dists2sink[sink_hist_arg]
    


    cur_hid=None
    

    if snap in snaps_with_halos:

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

        contains_sink=False
        for isink, sink_arg in enumerate(sink_args):
            if len(sink_arg) > 0:
                contains_sink = isink
                break

        if not contains_sink:
            print("no pure halo close to sink")
            continue

        cur_hid = halos_snap["hid"][fpure][contains_sink]    # print(snaps)

    fig, axs = plt.subplots(
        4,
        2,
        figsize=(20, 20),
        gridspec_kw={"height_ratios": [4, 1, 1,1]},
    )

    # plt.subplots_adjust(wspace=0.3, hspace=0.0)

    ax_img, ax_stars, ax_mass, ax_dens, ax_growth, ax_vrel, ax_dist, ax_cs = axs.flatten()

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


    try:

        stimg = plot_fields(
            "stellar mass",
            fig,
            ax_stars,
            aexp,
            directions,
            tgt_pos,
            rad_tgt / (sim.cosmo.lcMpc * 1e3),
            sim,
            cb=True,
            vmin=stvmin,
            # vmax=stvmax,
            hid=cur_hid,
            transpose=False,
            cmap="grey",
            log=True,
            read_ball_fct = read_data
        )
    except FileNotFoundError:
        print('missing files when plotting image')
        continue

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

    try:
        img = plot_fields(
            field,
            fig,
            ax_img,
            aexp,
            directions,
            tgt_pos,
            rad_tgt / (sim.cosmo.lcMpc * 1e3),
            sim,
            cb=True,
            vmin=vmin,
            vmax=vmax,
            transpose=False,
            cmap="magma",
            log=log_color,
            hid=cur_hid,
            op=op,
            read_ball_fct = read_data
        )
    except FileNotFoundError:
        print('missing files when plotting image')
        continue

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
                    zdist/ (1e3 * sim.cosmo.lcMpc),
                    hm=hm,
                    annotate=annotate,
                    brick_fct=gal_fct,
                    legend=legend,
                    color=color,
                    direction=directions[0],
                    transpose=False,
                    units=1e3 * sim.cosmo.lcMpc
                )

            # plot zoom halos
            if halo_markers:
                plot_zoom_halos(
                    plot_ax,
                    snap,
                    sim,
                    tgt_pos,
                    rad_tgt / (1e3 * sim.cosmo.lcMpc),
                    zdist/ (1e3 * sim.cosmo.lcMpc),
                    hm=hm_dm,
                    annotate=annotate,
                    brick_fct=gal_fct,
                    legend=legend,
                    color=color,
                    direction=directions[0],
                    transpose=False,
                    units=1e3 * sim.cosmo.lcMpc
                )

            # plot zoom BHs
            # try:
            plot_zoom_BHs(
                plot_ax,
                snap,
                sim,
                tgt_pos,
                rad_tgt/ (1e3 * sim.cosmo.lcMpc),
                zdist/ (1e3 * sim.cosmo.lcMpc),
                sink_read_fct=bh_fct,
                legend=legend,
                annotate=annotate,
                color=color,
                direction=directions[0],
                transpose=False,
                units=1e3 * sim.cosmo.lcMpc
            )
            # except (ValueError, AssertionError):
            #     pass

    # now put a single maker at current time and mass
    # ax_mass.axvline(zed, color="r", lw=1.5)

    zmin = np.min(sink_hist["zeds"][sink_hist["zeds"]>0])


    zargs = np.argsort(sink_hist["zeds"])
    ax_growth.plot(sink_hist["zeds"][zargs], sink_growth[zargs], color="tab:blue", lw=3)
    ax_growth.plot(sink_hist["zeds"][zargs], sink_edd[zargs], color="tab:blue", lw=3,ls=':')
    ax_growth.scatter(
        zed,
        cur_mdot,
        color="r",
        s=100,
        zorder=999,
    )
    ax_growth.set_yscale("log")
    ax_growth.grid()
    ax_growth.set_ylim(ymin=max(1e-6,sink_growth.min()),)

    max_zed_plot = min(max_zed, sink_hist["zeds"].max())

    ax_growth.set_xlabel("redshift")
    ax_growth.set_ylabel("BH growth rate, $M_\odot$/yr")
    ax_growth.set_xlim(xmax=max_zed_plot, xmin=zmin)
    ax_growth.invert_xaxis()

    ax_mass.plot(
        sink_hist["zeds"][zargs],
        sink_hist["mass"][zargs],
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

    ax_mass.set_xlim(xmax=max_zed_plot, xmin=zmin)
    ax_mass.invert_xaxis()

    ax_dens.plot(
        sink_hist["zeds"][zargs],
        sink_hist["dens"][zargs],
        color="tab:blue",
        lw=3,
    )

    ax_dens.scatter(zed, cur_dens, s=100, c="r", zorder=999)

    ax_dens.set_yscale("log")
    ax_dens.grid()

    ax_dens.set_xlabel("redshift")
    ax_dens.set_ylabel("cloud density, H/cm$^3$")
    ax_dens.set_xlim(xmax=max_zed_plot, xmin=zmin)
    ax_dens.invert_xaxis()

    ax_vrel.plot(
        sink_hist["zeds"][zargs],
        sink_hist["vrel"][zargs],
        color="tab:blue",
        lw=3,
    )

    ax_vrel.scatter(
        zed,
        cur_vrel,
        s=100,
        c="r",
        zorder=999,
    )

    ax_vrel.set_yscale("log")
    ax_vrel.grid()

    ax_vrel.set_xlabel("redshift")
    ax_vrel.set_ylabel("vrel, km/s")
    ax_vrel.set_xlim(xmax=max_zed_plot, xmin=zmin)
    ax_vrel.invert_xaxis()

    ax_cs.plot(
        sink_hist["zeds"][zargs],
        sink_hist["csound"][zargs],
        color="tab:blue",
        lw=3,
    )

    ax_cs.scatter(
        zed,
        cur_cs,
        s=100,
        c="r",
        zorder=999,
    )

    ax_cs.set_yscale("log")
    ax_cs.grid()

    ax_cs.set_xlabel("redshift")
    ax_cs.set_ylabel("sound speed, km/s")
    ax_cs.set_xlim(xmax=max_zed_plot, xmin=zmin)
    ax_cs.invert_xaxis()


    ax_dist.plot(
        sink_hist["zeds"][dists2sink>=0],
        dists2sink[dists2sink>=0] / (1+sink_hist["zeds"][dists2sink>=0]) ,
        color="tab:blue",
        lw=3,
    )

    ax_dist.scatter(
        zed,
        cur_dist2sink/(zed+1),
        s=100,
        c="r",
        zorder=999,
    )

    ax_dist.set_yscale("log")
    ax_dist.grid()

    ax_dist.set_xlabel("redshift")
    ax_dist.set_ylabel("distance from host galaxy, kpc")
    ax_dist.set_xlim(xmax=max_zed_plot, xmin=zmin)
    ax_dist.invert_xaxis()
    ax_dist.axhline(sim.cosmo.lcMpc/2**sim.levelmax*1e3,c='k',ls='--')

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
