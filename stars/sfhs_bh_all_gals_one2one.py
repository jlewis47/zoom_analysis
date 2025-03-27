from zoom_analysis.read.read_data import read_data_ball
from zoom_analysis.sinks.agn_models import zoom_injection
from zoom_analysis.sinks.sink_reader import get_sink_mhistory, hid_to_sid
from zoom_analysis.stars import sfhs
from zoom_analysis.zoom_helpers import decentre_coordinates, recentre_coordinates

# from zoom_analysis.stars.star_reader import yt_read_star_ball, read_part_ball
from zoom_analysis.stars.leja_quench_fit import sfr_ridge_leja22
from zoom_analysis.halo_maker.read_treebricks import (
    read_brickfile,
    convert_brick_units,
    convert_star_units,
    read_zoom_stars,
)
from scipy.spatial import KDTree
from zoom_analysis.trees.tree_reader import (
    read_tree_file_rev_correct_pos as read_tree_fev_sim,
)
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import (
    get_assoc_pties_in_tree,
    get_halo_assoc_file,
    get_halo_props_snap,
    find_zoom_tgt_gal,
    find_zoom_tgt_halo,
    get_central_gal_for_hid,
    get_gal_props_snap,
    smooth_props,
)

from scipy.ndimage import median_filter

# from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from scipy.spatial import KDTree

import os
import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# from zoom_analysis.sink_reader import *

# get colour cycler
# from cycler import cycler

# planck cosmo
from astropy.cosmology import Planck18 as cosmo

from hagn.catalogues import make_super_cat, get_cat_hids


# delta_t = 100  # Myr

# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id21892_leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id180130_leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id242704"  # _leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id242756"  # _leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id74099"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sdir = "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_drag"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel"
# sdir=     # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_lowerSFE"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_noAGN"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh"
# sdir=         # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/too_high_nsink/id242704"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/too_high_nsink/id242756"  # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704"  # _leastcoarse"
# sdir  "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756"  # _leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242704"  # _leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242756"  # _leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242704_lower_nstar"  # _leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242756_lower_nstar"  # _leastcoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel"  # _leastcoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel"  # _leastcoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE"  # _leastcoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_lowerSFE_NHboost"  # _leastcoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost"  # _leastcoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"  # _leastcoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380"  # _leastcoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289"  # _leastcoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id147479"  # _leastcoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2"  # _leastcoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag"  # _leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh"  # _leastcoarse"
# sdir=     "/data103/jlewis/sims/lvlmax_22/mh1e12/id180130"  # _leastcoarse"
# sdir=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099"

# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892"
# sdir =
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_evenlesscoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479"
# print(explr_dirs)

sim_dirs=[
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_lowerSFE_NHboost",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",
    ]

yax2 = None

tgt_zed = 2.2
tgt_time = cosmo.age(tgt_zed).value
rad_fact=2.0

delta_aexp = 0.005

mlim = 1e10
# mlim = 1e10

#pick one sim and find all galaxies above a certain mass
#store their host ids and gids, and find the halos in the other sims!

sim = ramses_sim(sim_dirs[0], nml="cosmo.nml")

# sim_snaps = sim.snap_numbers
tgt_snap = sim.get_closest_snap(zed=tgt_zed)
real_start_zed = 1.0 / sim.get_snap_exps(tgt_snap)[0] - 1.0

gprops = get_gal_props_snap(sim.path, tgt_snap)
hprops = get_halo_props_snap(sim.path, tgt_snap)

gmass = gprops["mass"]
fpure = gprops["host purity"] > 0.9999
central = gprops["central"] == 1
massive = (gmass > mlim)*(fpure) * (central)

hosts = gprops["host hid"][massive]
gids = gprops["gids"][massive]

hids = hprops["hid"]
found_hids,hid_args,host_args= np.intersect1d(hids,hosts,return_indices=True)
hpos = np.asarray([hprops[f"halo_{hid:07d}"]['pos'] for hid in found_hids])
if np.all(sim.zoom_ctr!=[0.5,0.5,0.5]):
    hpos = recentre_coordinates(hpos,sim.path,sim.zoom_ctr)
rvir = np.asarray([hprops[f"halo_{hid:07d}"]['rvir'] for hid in found_hids])
# hpos = hprops["pos"][hosts]
hmass = gprops["host mass"][massive]


all_hids = np.zeros((len(sim_dirs), len(hosts)), dtype=np.int32)-1
all_gids = np.zeros((len(sim_dirs), len(hosts)), dtype=np.int32)-1
all_snaps = np.zeros((len(sim_dirs)), dtype=np.int32)
all_hmasses = np.zeros((len(sim_dirs), len(hosts)), dtype=np.float32)
all_zeds = np.zeros((len(sim_dirs)), dtype=np.float32)

all_hids[0] = hosts
all_gids[0] = gids
all_snaps[0]= tgt_snap
all_hmasses[0] = hmass
all_zeds[0] = real_start_zed

full_args = np.all(all_hmasses>0,axis=0)



for isim, sdir in enumerate(sim_dirs[:]):

    if isim==0:continue

    sim = ramses_sim(sdir, nml="cosmo.nml")

    # sim_snaps = sim.snap_numbers
    tgt_snap = sim.get_closest_snap(zed=tgt_zed)
    if abs(tgt_zed-(1./sim.get_snap_exps(tgt_snap)-1))>0.2:
        continue
    all_snaps[isim] = tgt_snap
    real_start_zed = 1.0 / sim.get_snap_exps(tgt_snap)[0] - 1.0
    all_zeds[isim] = real_start_zed

    gprops = get_gal_props_snap(sim.path, tgt_snap)
    hprops = get_halo_props_snap(sim.path, tgt_snap)

    loc_hosts = gprops["host hid"]
    found_hids,hid_args,host_args= np.intersect1d(hprops['hid'],loc_hosts,return_indices=True)

    loc_gids = gprops["gids"]
    loc_hmasses = gprops["host mass"]
    loc_hpos = np.asarray([hprops[f"halo_{hid:07d}"]['pos'] for hid in found_hids])
    if np.all(sim.zoom_ctr!=[0.5,0.5,0.5]):
        loc_hpos = recentre_coordinates(loc_hpos,sim.path,sim.zoom_ctr)
    loc_fpure = gprops["host purity"]
    # loc_rvirs = gprops["host rvir"]

    fpure_args = np.where(loc_fpure > 0.9999)[0]

    central_args = np.where(gprops["central"] == 1)[0]

    htree = KDTree(loc_hpos,boxsize=1+1e-6)

    nearby =  htree.query_ball_point(hpos,rvir*2)

    for iarg,args in enumerate(nearby):

        if type(args)!=int:
            if len(args)==0:
                continue

        pure_and_loc_args = np.intersect1d(np.intersect1d(fpure_args,args), central_args)

        if type(pure_and_loc_args)!=int:
            if len(pure_and_loc_args)==0:
                continue

        # loc_arg = np.argmax(loc_hmasses[pure_and_loc_args])
        loc_arg = np.argmin(np.abs(loc_hmasses[pure_and_loc_args]-hmass[iarg]))
        all_hids[isim,iarg] = loc_hosts[pure_and_loc_args[loc_arg]]
        all_gids[isim,iarg] = loc_gids[pure_and_loc_args[loc_arg]]
        all_hmasses[isim,iarg] = loc_hmasses[pure_and_loc_args[loc_arg]]
    

for ihalo in range(len(hosts)):
        

    # setup plot
    fig, ax = plt.subplots(3, 3, figsize=(16, 16), sharex=True, layout="constrained")

    # map for plots

    # Mstar, SFR, sSFR
    # MBH, mdot_bh, nrg injection
    # density, vrel, mgas


    # mlim = 8e9


    last_hagn_id = -1
    isim = 0

    # zoom_ls = ["-", "--", ":", "-.", (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10))]
    lines = []
    labels = []


    last_simID = None
    l = None

    zoom_style = 0


    name = sdir.split("/")[-1].strip()

    for isim,sdir in enumerate(sim_dirs):

        hid = all_hids[isim,ihalo]
        gid = all_gids[isim,ihalo]

        if hid==-1 or gid==-1:
            continue

        sim = ramses_sim(sdir, nml="cosmo.nml")

        sim_snaps = sim.snap_numbers
        sim_aexps = sim.get_snap_exps()
        sim.init_cosmo()
        sim_times = sim.cosmo_model.age(1.0 / sim_aexps - 1.0).value * 1e3

        # last sim_aexp
        # valid_steps = hagn_tree_aexps < (1.0 / (tgt_zed + 1.0))  # sim.aexp_end
        # nsteps = np.sum(valid_steps)

        nsteps = len(sim_snaps)

        mstel_zoom = np.zeros(nsteps, dtype=np.float32)
        mgas_zoom = np.zeros(nsteps, dtype=np.float32)
        sfr_zoom = np.zeros(nsteps, dtype=np.float32)
        time_zoom = np.zeros(nsteps, dtype=np.float32)

        # find last output with assoc files
        assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
        assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

        avail_aexps = np.intersect1d(
            sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
        )
        avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

        avail_snaps = np.intersect1d(sim_snaps, assoc_file_nbs)

        avail_zeds = 1.0 / avail_aexps - 1.0

        real_start_snap = all_snaps[isim]
        real_start_zed = 1.0 / sim.get_snap_exps(real_start_snap)[0] - 1.0

        gal_props = get_gal_props_snap(sim.path, real_start_snap)

        # print(gal_props["gids"][fpure][mass_ok])

        # mstel_zoom = np.zeros((len(avail_times),nb_gals_to_plot), dtype=np.float32)
        mstel_zoom = np.zeros((len(avail_times)), dtype=np.float32)
        mvir_zoom = np.zeros((len(avail_times)), dtype=np.float32)
        mgas_zoom = np.zeros((len(avail_times)), dtype=np.float32)
        # sfr_zoom = np.zeros((len(avail_times),nb_gals_to_plot), dtype=np.float32)
        sfr_zoom = np.zeros((len(avail_times)), dtype=np.float32)
        # time_zoom = np.zeros((len(avail_times),nb_gals_to_plot), dtype=np.float32)
        time_zoom = np.zeros((len(avail_times)), dtype=np.float32)

        # tree_gids, tree_datas, tree_aexps = read_tree_fev_sim(
        tree_hids, tree_datas, tree_aexps = read_tree_fev_sim(
            # os.path.join(sim.path, "TreeMakerstars2_dp_rec_dust", "tree_rev.dat"),
            # fbytes=os.path.join(sim.path, "TreeMakerstars2_dp_rec_dust"),
            os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat"),
            sim,
            real_start_snap,
            fbytes=os.path.join(sim.path, "TreeMakerDM_dust"),
            zstart=real_start_zed,
            tgt_ids=[hid],
            # tgt_ids=[gid],
            star=False,
            # star=True,
        )

        if len(tree_aexps)==0:continue

        # ok_step = tree_gids[0] > 0
        ok_step = tree_hids[0] > 0

        tree_hids = tree_hids[0][ok_step]
        # tree_gids = tree_gids[0][ok_step]
        tree_aexps = tree_aexps[ok_step]

        # print(tree_hids, tree_datas)
        tree_times = sim.cosmo_model.age(1.0 / tree_aexps - 1.0).value * 1e3  # Myr

        sim_times = sim.get_snap_times()

        try:
            gal_props_tree = get_assoc_pties_in_tree(
                sim,
                tree_aexps,
                tree_hids,
                assoc_fields=[
                    "gids",
                    "r50",
                    "rmax",
                    "mass",
                    "pos",
                    "host hid",
                    "host mass",
                    "sfr100",
                ],
            )
        except KeyError:
            continue

        smooth_gal_props = smooth_props(gal_props_tree)

        for k in tree_datas:
            tree_datas[k] = tree_datas[k][0][ok_step]

        sid, found_sink = hid_to_sid(sim, hid, real_start_snap)

        for istep, (time, aexp, snap) in enumerate(
            zip(avail_times, avail_aexps, avail_snaps)
        ):

            if np.min(np.abs(aexp - tree_aexps)) > delta_aexp:
                continue

            tree_arg = np.argmin(np.abs(tree_aexps - aexp))
            props_arg = np.argmin(np.abs(gal_props_tree["aexps"] - aexp))

            cur_hid = tree_hids[tree_arg]

            hprops, hosted_gals = get_halo_props_snap(sim.path, snap, cur_hid)
            if hosted_gals == {}:
                print(f"step:{istep:d}, no hosted gals, skipping")
                continue
            cur_gid = hosted_gals["gids"][hosted_gals["mass"].argmax()]

            # cur_gid = tree_gids[tree_arg]

            # _, cur_gal_props = get_gal_props_snap(sim.path, snap, gid=cur_gid)

            # print(snap, cur_gid)

            cur_gal_props = {
                key: smooth_gal_props[key][props_arg] for key in smooth_gal_props
            }

            cur_r50 = cur_gal_props["r50"]

            cur_pos = gal_props_tree["pos"][props_arg]

            try:
                datas = read_data_ball(
                    sim,
                    snap,
                    cur_pos,
                    cur_r50 * rad_fact,
                    host_halo=cur_hid,
                    data_types=["stars", "gas"],
                    tgt_fields=["ilevel", "density", "mass", "metallicity", "age"],
                )
            except AssertionError:
                print(f"step:{istep:d}, no data, skipping")
                continue

            gas_data = datas["gas"]

            if gas_data == None:
                print(f"step:{istep:d}, no gas data, skipping")
                continue

            cell_vols_ccm = (sim.cosmo.lcMpc / 2 ** gas_data["ilevel"] * 3.08e24) ** 3.0
            gas_mass_msun = gas_data["density"] * (cell_vols_ccm / 1.989e33)

            star_data = datas["stars"]

            if star_data == None:
                print(f"step:{istep:d}, no star data, skipping")
                continue

            masses = sfhs.correct_mass(
                sim, star_data["age"], star_data["mass"], star_data["metallicity"]
            )

            mstel_zoom[istep] = masses.sum()  # msun
            mvir_zoom[istep] = cur_gal_props["mvir"]  # msun
            tlim = 100.0  # Myr
            sfr_zoom[istep] = np.sum(masses[star_data["age"] < tlim]) / tlim  # msun/Myr
            time_zoom[istep] = time  # Myr
            mgas_zoom[istep] = gas_mass_msun.sum()  # msun

        # print(mstel_zoom)

        ssfr_zoom = sfr_zoom / mstel_zoom

        # print(f"zoom style is {zoom_ls[zoom_style]}")

        pos_mass = mstel_zoom > 0

        (l,) = ax[0, 0].plot(time_zoom[pos_mass], mstel_zoom[pos_mass])
        ax[0, 0].plot(time_zoom[pos_mass], mvir_zoom[pos_mass], c=l.get_color(), ls=":")
        ax[1, 0].plot(time_zoom[pos_mass], sfr_zoom[pos_mass])
        ax[2, 0].plot(time_zoom[pos_mass], ssfr_zoom[pos_mass])

        ax[2, 2].plot(time_zoom[pos_mass], mgas_zoom[pos_mass])

        if found_sink:

            sink_hist = get_sink_mhistory(
                sid,
                real_start_snap,
                sim,
                out_keys=[
                    ("position", 3),
                    ("spins", 3),
                    ("mass", 1),
                    ("dMBH_coarse", 1),
                    ("dMsmbh", 1),
                    ("dMEd_coarse", 1),
                    ("dens", 1),
                    ("csound", 1),
                    ("vrel", 1),
                ],
            )

            zoom_injection(sink_hist, sim)

            sink_times = sim.cosmo_model.age(sink_hist["zeds"]).value * 1e3

            sink_found = sink_hist["mass"] > 0

            cur_dt = np.abs(np.median(np.diff(sink_times[sink_found])))
            tgt_dt = max(np.median(np.diff(time_zoom[pos_mass][::-1])), 100)

            scale = int(np.ceil(tgt_dt / cur_dt))
            # median filter all properties
            sink_hist_smooth = {
                # k: np.convolve(v, np.ones(scale) / scale, mode="same")
                k: median_filter(v, size=scale)
                for k, v in sink_hist.items()
                if len(v.shape) == 1
            }

            # cuml_eagn = np.asarray(
            #     [
            #         np.trapz(
            #             np.float64(sink_hist["EAGN"][sink_found][:i])[::-1],
            #             sink_times[sink_found][:i][::-1] * 1e6 * (3600 * 24 * 365),
            #         )
            #         for i in range(2, len(sink_times[sink_found]))
            #     ]
            # )[::-1]

            order_time = np.argsort(sink_times[sink_found])
            ordered_nrg = sink_hist["EAGN"][sink_found][order_time]
            ordered_time = sink_times[sink_found][order_time]

            cuml_eagn = np.zeros(len(ordered_nrg) - 1)

            for i in range(2, len(ordered_nrg)):

                cuml_eagn[i - 2] = np.trapz(
                    np.float64(ordered_nrg[:i]),
                    ordered_time[:i] * 1e6 * (3600 * 24 * 365),
                )

            (l,) = ax[0, 1].plot(sink_times[sink_found], sink_hist["mass"][sink_found])
            ax[2, 1].plot(ordered_time[1:], cuml_eagn / 1e30)

            ax[1, 1].plot(
                sink_times[sink_found],
                sink_hist["dMsmbh"][sink_found],
                lw=0.4,
                alpha=0.2,
                ls=":",
            )
            ax[0, 2].plot(
                sink_times[sink_found],
                sink_hist["dens"][sink_found],
                lw=0.4,
                alpha=0.2,
                ls=":",
            )
            ax[1, 2].plot(
                sink_times[sink_found],
                sink_hist["vrel"][sink_found],
                lw=0.4,
                alpha=0.2,
                ls=":",
            )

            ax[1, 1].plot(
                sink_times[sink_found],
                sink_hist_smooth["dMsmbh"][sink_found],
                lw=1,
                c=l.get_color(),
            )
            ax[0, 2].plot(
                sink_times[sink_found],
                sink_hist_smooth["dens"][sink_found],
                lw=1,
                c=l.get_color(),
            )
            ax[1, 2].plot(
                sink_times[sink_found],
                sink_hist_smooth["vrel"][sink_found],
                lw=1,
                c=l.get_color(),
            )

        label = f"{sim.name}: hid={hid:d},gid={gid:d}"
        if type(sid)!=int and found_sink:
            label += f",sid={sid:d}"
        labels.append(label)
        lines.append(l)


        ax[0, 0].legend(
            [Line2D([], [], c="k", ls="-"), Line2D([], [], c="k", ls=":")],
            ["Stellar mass", "DM Host virial mass"],
            framealpha=0.0,
        )

        for a in ax:
            for b in a:
                b.tick_params(
                    axis="both",
                    which="both",
                    bottom=True,
                    top=True,
                    left=True,
                    right=True,
                    direction="in",
                )

                # b.set_xscale("log")
                b.set_yscale("log")

                b.grid()


    # y2 = ax[0].twiny()
    # y2.set_xlim(xlim)
    # zlabels = [f"{cosmo.age(zed).value:.1f}" for zed in ax[0].get_xticks()]
    # # y2.set_xticks(ax[0].get_xticks())
    # y2.set_xticklabels(zlabels)
    # y2.set_xlabel("redshift")


    ax[-1, -1].legend(
        lines, labels, framealpha=0.0, ncol=2, title=f"z={real_start_zed:.2f}"
    )  # , handlelength=3)

    zstr = f"{real_start_zed:.1f}".replace(".", "p")
    outdir=f"./figs/sfhs_all_gals_BH_one2one/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fout = f"{outdir:s}sfhs_all_gals_BH_{ihalo:d}_z{zstr}.png"
    print(f"saving {fout:s}")


    ax[0, 0].set_ylabel("Mass [Msun]")
    ax[0, 0].set_ylim(
        5e6,
    )
    ax[1, 0].set_ylabel("SFR [Msun/Myr]")

    ax[2, 0].set_ylabel("sSFR [1/Myr]")
    ax[2, 0].set_xlabel("Time [Myr]")

    ax[0, 1].set_ylabel("MBH [Msun]")
    ax[1, 1].set_ylabel("dMBH [Msun/Myr]")
    ax[2, 1].set_ylabel("cumulative EAGN [E30 J]")
    # ax[2, 1].set_ylim(
    #     1e14,
    # )

    ax[0, 2].set_ylabel("Density [g/cm^3]")
    ax[1, 2].set_ylabel("vrel [km/s]")
    ax[2, 2].set_ylabel("Mgas [Msun]")
    ax[2, 2].set_xlabel("Time [Myr]")

    fig.savefig(fout)
