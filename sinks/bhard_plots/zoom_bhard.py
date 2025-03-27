# from f90nml import read
from hagn.utils import get_hagn_sim
from zoom_analysis.sinks import sink_histories, sink_reader
from zoom_analysis.constants import ramses_pc

from zoom_analysis.zoom_helpers import check_in_all_sims_vol

# from zoom_analysis.halo_maker.assoc_fcts import get_gal_assoc_file, get_gal_props_snap

# from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# from scipy.spatial import cKDTree
from scipy.stats import binned_statistic

import os
import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# from zoom_analysis.sink_reader import *

from hagn.association import gid_to_stars
from hagn.utils import get_hagn_sim, adaptahop_to_code_units
from hagn.tree_reader import read_tree_rev
from hagn.catalogues import get_halos_cat, make_super_cat, get_cat_hids

# get colour cycler
# from cycler import cycler

# planck cosmo
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
from astropy import units as u

# from hagn.catalogues import make_super_cat

# fig, ax = sink_histories.setup_bh_plot()


tgt_zed = 2.25
fpure = 1.0 - 1e-4

HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
delta_t = 100  # Myr

sim_dirs = [
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE_highAGNtherm/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highSFE_lowSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE_lowSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_low_dboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_minMassDynGrav",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowSFE_DynBondiGravSinkMass",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_lowerSFE_drag/",
    # "/data102/jlewis/sims/lvlmax20/mh1e12/id180130_superEdd_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_accBoost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE_lowNsink/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_pdrag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_noAGN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_high_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nrg_SN",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_lowerSFE_NHboost",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_pdrag",
]

# list of all available pyplot markers:
# https://matplotlib.org/stable/api/markers_api.html
markers = [
    "o",
    "v",
    "^",
    "<",
    ">",
    "s",
    "p",
    "P",
    "*",
    "h",
    "H",
    "X",
    "D",
    "d",
    "|",
    "_",
]

# sink_key = "dMBH_coarse"
sink_key = "dMsmbh"

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

ax.grid()

sink_mdot_bins = np.logspace(-8, 7, num=25)


for isim, sim_dir in enumerate(sim_dirs):

    print(sim_dir)
    sim = ramses_sim(sim_dir, nml="cosmo.nml")  # , verbose=True)

    name = sim.name
    print(name)

    snaps = sim.get_snaps(full_snaps=True, mini_snaps=True)[1]
    if len(snaps) == 0:
        continue

    intID = int(sim.name.split("_")[0][2:])

    snap = sim.get_closest_snap(zed=tgt_zed)
    sim_zed = 1.0 / sim.get_snap_exps(snap)[0] - 1
    aexp = 1.0 / (sim_zed + 1)

    # hagn_sim = get_hagn_sim()
    # hagn_snap = hagn_sim.get_closest_snap(zed=tgt_zed)

    # # super_cat = make_super_cat(
    # #     hagn_snap, outf="/data101/jlewis/hagn/super_cats"
    # # )  # , overwrite=True)

    # # gal_pties = get_cat_hids(super_cat, [intID])

    # hagn_zed = 1.0 / hagn_sim.aexps[hagn_sim.snap_numbers == hagn_snap][0] - 1

    # mbh, dmbh, t_sink, fmerger, t_merge = sink_histories.get_bh_stuff(
    #     hagn_sim, intID, hagn_zed
    # )

    # l = sink_histories.plot_bh_stuff(
    #     ax,
    #     mbh,
    #     dmbh,
    #     t_sink,
    #     fmerger,
    #     t_merge,
    #     0,
    #     cosmo,
    #     label=name,
    #     lw=1,
    # )

    # c = l[0].get_color()
    # find last assoc_file

    coarse_step = sink_reader.snap_to_coarse_step(snap, sim)

    # sim_coarse_steps, sim_coarse_zeds, sim_coarse_times = sink_reader.get_coarse_dts(
    #     sim
    # )

    # prev_step = coarse_step - 1
    # cur_time = sim_coarse_times[sim_coarse_steps == coarse_step][0]
    # prev_time = sim_coarse_times[sim_coarse_steps == prev_step][0]

    # coarse_dt = cur_time - prev_time

    sfile = os.path.join(sim.sink_path, f"sink_{coarse_step:05d}.dat")
    sinks = sink_reader.read_sink_bin(sfile)

    sink_reader.convert_sink_units(sinks, sim.get_snap_exps(snap)[0], sim)

    label = sim.name
    if len(sim_dirs) > 1:
        label += f", z={sim_zed:.2f}"

    # print(sinks.keys())

    sink_pos = np.transpose(sinks["position"].T)
    vol_args,min_vol = check_in_all_sims_vol(sink_pos, sim, sim_dirs)        

    # dsmbh = sinks[sink_key]
    # rates = dsmbh / coarse_dt

    rates = sinks["dMsmbhdt_coarse"]
    rates = np.min([rates, sinks["dMEd_coarse"]], axis=0)[vol_args]

    counts = np.histogram(sinks[sink_key][vol_args], bins=sink_mdot_bins)[0]
    med_bin = binned_statistic(
        sinks[sink_key][vol_args], sinks[sink_key][vol_args], bins=sink_mdot_bins, statistic="median"
    )[0]

    # vol, check_vol = sim.get_volume()

    ax.plot(
        # sink_mdot_bins[:-1],
        med_bin,
        counts / np.diff(np.log10(sink_mdot_bins)) / min_vol,
        label=label,
    )
    # print(counts)

# print(sinks[sink_key].sum() / vol / aexp**3)

hagn_sim = get_hagn_sim()

hagn_snaps = hagn_sim.snap_numbers
hagn_aexps = np.asarray(hagn_sim.aexps)
hagn_zeds = 1.0 / hagn_aexps - 1

tgt_hagn_snap = hagn_snaps[np.argmin(np.abs(hagn_aexps - aexp))]
tgt_hagn_aexp = hagn_aexps[np.argmin(np.abs(hagn_aexps - aexp))]
tgt_hagn_zed = 1.0 / tgt_hagn_aexp - 1

try:
    coarse_step = sink_reader.snap_to_coarse_step(tgt_hagn_snap, hagn_sim)
except AssertionError:
    print(f"Failed to get coarse step for snap {tgt_hagn_snap}")


sfile = os.path.join(hagn_sim.sink_path, f"sink_{coarse_step:05d}.dat")
sinks = sink_reader.read_sink_bin(sfile, hagn=True)

sink_reader.convert_sink_units(sinks, tgt_hagn_aexp, hagn_sim)

sink_pos = sinks["position"]

vol_args,min_vol = check_in_all_sims_vol(sink_pos, sim, sim_dirs)        

# in_vol = check_vol(sink_pos)

for k in sinks.keys():
    pty = sinks[k]
    if type(pty) not in [float, int, np.float32, np.float64, np.int32]:
        sinks[k] = pty[vol_args]
    else:
        pass

rates = sinks["dMsmbhdt_coarse"]
rates = np.min([rates, sinks["dMEd_coarse"]], axis=0)

counts = np.histogram(sinks[sink_key], bins=sink_mdot_bins)[0]
med_bin = binned_statistic(
    sinks[sink_key], sinks[sink_key], bins=sink_mdot_bins, statistic="median"
)[0]

# vol = hagn_sim.cosmo.lcMpc**3

ax.plot(
    med_bin,
    counts / np.diff(np.log10(sink_mdot_bins)) / min_vol,
    label=f"HAGN (z={tgt_hagn_zed:.2f} same volume)",
    ls="--",
    color="k",
    lw=1,
)

# print(counts)


ax.tick_params(
    axis="both",
    which="both",
    bottom=True,
    # top=True,
    top=False,
    left=True,
    right=True,
    direction="in",
)

ax.grid()

ax.set_xlabel("Accretion rate [M$_\odot$/yr]")
ax.set_ylabel("Density [Mpc$^{-3}$ dex$^{-1}$]")

ax.set_yscale("log")
ax.set_xscale("log")
title_txt = ""
if len(sim_dirs) == 1:
    title_txt += f"z={sim_zed:.2f}"
    # ax.text(0.05, 0.95, f"z={sim_zed:.2f}", transform=ax.transAxes)
# also print purity threshold
# ax.text(0.05, 0.9, f"fpure > {fpure:.3f}", transform=ax.transAxes)
title_txt += f" fpure > {fpure:.3f}"

# y2 = ax.twiny()
# # y2.set_xlim(ax.get_xlim())
# # y2.set_xticks(ax.get_xticks())
# # xlim = ax.get_xlim()
# y2.set_xticklabels(
#     [
#         "%.1f" % z_at_value(sim.cosmo_model.age, time_label * u.Gyr, zmax=np.inf)
#         for time_label in ax.get_xticks()
#     ]
# )
# y2.set_xlabel("redshift")

# outdir = "./bhsmr_plots"
# if not os.path.exists(outdir):
# os.makedirs(outdir)

# NH_smbh(ax)

ax.legend(framealpha=0.0, title=title_txt)
if tgt_zed != None:
    fig.savefig(f"zoom_BHARD_z{tgt_zed:.2f}.png")
else:
    fig.savefig("zoom_BHARD.png")


fig, ax = plt.subplots(1, 1, figsize=(8, 8))

ax.grid()


zmax = 0

for isim, sim_dir in enumerate(sim_dirs):

    sim = ramses_sim(sim_dir, nml="cosmo.nml")  # , verbose=True)

    name = sim.name
    print(f"{name} evol")

    snaps = sim.get_snaps(full_snaps=True, mini_snaps=True)[1]
    if len(snaps) == 0:
        continue

    intID = int(sim.name.split("_")[0][2:])

    snaps = sim.snap_numbers
    aexps = np.asarray(sim.get_snap_exps(snaps))
    zeds = 1.0 / aexps - 1

    tot_acc_rate = np.zeros_like(snaps, dtype=float)

    vol, check_vol = sim.get_volume()

    # get dts
    coarse_info = sink_reader.get_coarse_dts(sim)

    for isnap, (snap, aexp, zed) in enumerate(zip(snaps, aexps, zeds)):

        try:
            coarse_step = sink_reader.snap_to_coarse_step(snap, sim)
            sfile = os.path.join(sim.sink_path, f"sink_{coarse_step:05d}.dat")
            sinks = sink_reader.read_sink_bin(sfile)

            sink_reader.convert_sink_units(sinks, snap, sim, coarse_info=coarse_info)
        except AssertionError:
            print(f"Failed to get coarse step for snap {snap}")
            continue


        # print(isnap, sinks[sink_key].sum() / vol)

        # dMsmbh = sinks["dMsmbh"]
        # prev_step = coarse_step - 1
        # cur_time = sim_coarse_times[sim_coarse_steps == coarse_step][0]
        # prev_time = sim_coarse_times[sim_coarse_steps == prev_step][0]
        # coarse_dt = cur_time - prev_time
        # rates = dMsmbh / coarse_dt
        # rates = np.min([sinks[sink_key], sinks["dMEd_coarse"]], axis=0)
        rates = sinks["dMsmbhdt_coarse"]
        rates = np.min([rates, sinks["dMEd_coarse"]], axis=0)

        tot_acc_rate[isnap] = rates.sum() / vol  # / aexp**3
        # tot_acc_rate[isnap] = sinks[sink_key].sum() / vol  # / aexp**3

    # print(tot_acc_rate)


    non_nul = tot_acc_rate > 0

    if non_nul.sum()==0:continue
    
    zmax = max(zmax, zeds[non_nul].max())

    ax.plot(
        zeds[non_nul],
        tot_acc_rate[non_nul],
        label=name,
    )

print("hagn evol")

hagn_sim = get_hagn_sim()

hagn_snaps = hagn_sim.snap_numbers
hagn_aexps = np.asarray(hagn_sim.aexps)
hagn_zeds = 1.0 / hagn_aexps - 1

tot_acc_rate = np.zeros_like(hagn_snaps, dtype=float)

# vol = hagn_sim.cosmo.lcMpc**3


coarse_info = sink_reader.get_coarse_dts(hagn_sim)

for isnap, (snap, aexp, zed) in enumerate(zip(hagn_snaps, hagn_aexps, hagn_zeds)):

    if zed < 2:
        continue

    try:
        coarse_step = sink_reader.snap_to_coarse_step(snap, hagn_sim)
    except AssertionError:
        print(f"Failed to get coarse step for snap {snap}")
        continue

    sfile = os.path.join(hagn_sim.sink_path, f"sink_{coarse_step:05d}.dat")
    sinks = sink_reader.read_sink_bin(sfile, hagn=True)

    sink_reader.convert_sink_units(sinks, aexp, hagn_sim, coarse_info=coarse_info)

    sink_pos = sinks["position"]
    if len(sink_pos.shape) == 1:
        continue

    in_vol = check_vol(sink_pos)

    for k in sinks.keys():
        pty = sinks[k]
        if type(pty) not in [float, int, np.float32, np.float64, np.int32]:
            sinks[k] = pty[in_vol]
        else:
            pass

    # rates = np.min([sinks[sink_key], sinks["dMEd_coarse"]], axis=0)

    # dMsmbh = sinks["dMsmbh"]
    # cur_dt = sim_coarse_times[sim_coarse_steps == coarse_step][0]
    # prev_dt = sim_coarse_times[sim_coarse_steps == (coarse_step - 1)][0]
    # coarse_dt = cur_dt - prev_dt

    # rates = dMsmbh / coarse_dt
    rates = sinks["dMsmbhdt_coarse"]
    rates = np.min([rates, sinks["dMEd_coarse"]], axis=0)

    tot_acc_rate[isnap] = rates.sum() / vol  # / aexp**3

# print(tot_acc_rate)

non_nul = tot_acc_rate > 0

zmax = max(zmax, hagn_zeds[non_nul].max())

ax.plot(
    hagn_zeds[non_nul],
    tot_acc_rate[non_nul],
    label="HAGN (same volume)",
    ls="--",
    color="k",
    lw=1,
)

ax.fill_between(
    [1, 9],
    [2e-5, 2e-5],
    [2e-4, 2e-4],
    color="gray",
    alpha=0.5,
    label="JWST (Atkins+24, Yang+23)",
)

ax.set_xlim(2, zmax)
ax.set_ylim(
    1e-10,
)

ax.grid()


ax.legend(framealpha=0.0, title=title_txt)
ax.set_xlabel("Redshift")
ax.set_ylabel("Total SMBH accretion rate [M$_\odot/yr/cMpc$^3]")

ax.set_yscale("log")

fig.savefig("zoom_BHARD_evol.png")
