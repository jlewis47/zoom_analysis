# from f90nml import read
from turtle import title
import h5py
from zoom_analysis.sinks import sink_histories, sink_reader
from zoom_analysis.constants import ramses_pc

# from zoom_analysis.halo_maker.assoc_fcts import get_gal_assoc_file

# from scipy.stats import binned_statistic


# from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

import os
import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# from zoom_analysis.sink_reader import *

from hagn.association import gid_to_stars
from hagn.utils import get_hagn_sim, adaptahop_to_code_units
from hagn.tree_reader import read_tree_rev
from hagn.catalogues import get_halos_cat, make_super_cat, get_cat_hids

from zoom_analysis.zoom_helpers import check_in_all_sims_vol

# get colour cycler
# from cycler import cycler

# planck cosmo
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
from astropy import units as u

from zoom_analysis.visu.visu_fct import get_gal_props_snap

# from hagn.catalogues import make_super_cat

# fig, ax = sink_histories.setup_bh_plot()

# tgt_zed = 4.0

fpure = 1.0 - 1e-4

HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
delta_t = 100  # Myr

sim_dirs = [
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_coreen",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_gravSink",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_DynBondiGravSinkMass",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_fullresIC",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_kicks",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_smooth_ref",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_low_dboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_minMassDynGrav",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_vlowSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_high_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_accBoost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE_lowNsink",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highSFE_lowSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE_lowSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_pdrag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nrg_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_high_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_high_thermal_eff",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE_highAGNtherm",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_noAGN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756",  # _leastcoarse",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel",  # _leastcoarse",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",  # _leastcoarse"
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE",  # _leastcoarse"
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_lowerSFE_NHboost",  # _leastcoarse"
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost",  # _leastcoarse"
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_noHydroDragB4z4merge",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
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

fig_mstar, ax_mstar = plt.subplots(1, 1, figsize=(8, 8))
fig_sfr, ax_sfr = plt.subplots(1, 1, figsize=(8, 8))


for isim, sim_dir in enumerate(sim_dirs):

    sim = ramses_sim(sim_dir, nml="cosmo.nml")

    name = sim.name

    intID = int(sim.name.split("_")[0][2:])

    found = False
    decal = len(sim.snap_numbers) - 1

    # print(found, decal, len(sim.snap_numbers))

    sfr_dense = np.zeros(len(sim.snap_numbers))
    smass_dense = np.zeros(len(sim.snap_numbers))

    vol = sim.get_volume()[0]

    sim_times = sim.get_snap_times()
    sim_aexps = sim.get_snap_exps()

    print(f"Working on {name}")

    for isnap, (sim_snap, sim_aexp, sim_time) in enumerate(
        zip(sim.snap_numbers, sim_aexps, sim_times)
    ):

        try:
            gal_dict = get_gal_props_snap(sim.path, sim_snap, main_stars=False)
        except FileNotFoundError:
            # print(f"Failed to read gal props for {sim_snap}")
            continue


        vol_args,min_vol = check_in_all_sims_vol(gal_dict["pos"].T,sim,sim_dirs)

        print(f"Working on snap {sim_snap}, z={1/sim_aexp-1}")

        host_purity = gal_dict["host purity"]
        central_cond = gal_dict["central"] == 1
        pure_cond = (host_purity > fpure) * central_cond * vol_args
        gal_mass = gal_dict["mass"][pure_cond]
        gal_pos = gal_dict["pos"][:, pure_cond]
        rmax = gal_dict["rmax"][pure_cond]
        halo_mass = gal_dict["host mass"][pure_cond]

        sfr = gal_dict["sfr100"][pure_cond]/1e6

        sfr_dense[isnap] = np.sum(sfr) / min_vol
        smass_dense[isnap] = np.sum(gal_mass) / min_vol

    ax_mstar.plot(
        sim_times,
        smass_dense,
        label=name,
        marker=markers[isim],
        markevery=5,
        markersize=8,
        linestyle="-",
    )

    ax_sfr.plot(
        sim_times,
        sfr_dense,
        label=name,
        marker=markers[isim],
        markevery=5,
        markersize=8,
        linestyle="-",
    )



#now for HAGN

hagn_sim = get_hagn_sim()

hagn_snaps = hagn_sim.snap_numbers
hagn_aexps = hagn_sim.aexps
hagn_zeds = 1.0 / hagn_aexps - 1
hagn_times = hagn_sim.get_snap_times(param_save=False)

hagn_sfr_dense = np.zeros(len(hagn_snaps))
hagn_smass_dense = np.zeros(len(hagn_snaps))

for isnap, (hagn_snap, hagn_aexp, hagn_zed) in enumerate(
    zip(hagn_snaps,hagn_aexps,hagn_zeds)
):
    
    if hagn_zed<2.0:continue

    try:
        hagn_gals = make_super_cat(hagn_snap)
    except FileNotFoundError:
        continue

    # print(hagn_gals.keys())

    hagn_gal_pos = np.transpose([hagn_gals["x"], hagn_gals["y"], hagn_gals["z"]])

    vol_args,min_vol = check_in_all_sims_vol(hagn_gal_pos, hagn_sim, sim_dirs)

    if vol_args.sum() > 0:
        sfr100 = hagn_gals['sfr100'][vol_args]
        mgals= hagn_gals['mgal'][vol_args]


        hagn_sfr_dense[isnap] = np.sum(sfr100) / min_vol
        hagn_smass_dense[isnap] = np.sum(mgals) / min_vol


ax_mstar.plot(
    hagn_times,
    hagn_smass_dense,
    label="HAGN",
    marker="x",
    markevery=5,
    markersize=8,
    linestyle="-",
    color='k',
)
ax_sfr.plot(
    hagn_times,
    hagn_sfr_dense,
    label="HAGN",
    marker="x",
    markevery=5,
    markersize=8,
    linestyle="-",
    color='k',
)

ax_mstar.set_yscale("log")
ax_mstar.set_ylabel("Stellar mass density [M$_\odot$ cMpc$^{-3}$]")
ax_mstar.set_xlabel("Time [Myr]")
ax_mstar.legend(framealpha=0.0)
ax_mstar.set_xlim(50, 3.4e3)
ax_mstar.grid()

# double axis with sim_aexp
ax2 = ax_mstar.twiny()
ax2.set_xlabel("redshift")
# ax2.set_xlim(xticks2.min(), xticks2.max())
xticks2 = ax_mstar.get_xticks()
# ax_mstar.set_xlim(xticks2.min(), xticks2.max())
ax_mstar.set_xticks(xticks2)
ax2.set_xticks(xticks2)
zed_ticks = np.asarray(
    [z_at_value(cosmo.age, (max(age, 1.0)) * u.Myr) for age in xticks2]
)
ax2.set_xticklabels(np.round(zed_ticks, decimals=2))


mstar_fname = os.path.join("mstar_dense.png")
fig_mstar.savefig(mstar_fname)

ax_sfr.set_yscale("log")
ax_sfr.set_ylabel("SFR density [M$_\odot$ yr$^{-1}$ cMpc$^{-3}$]")
ax_sfr.set_xlabel("Time [Myr]")
ax_sfr.legend(framealpha=0.0)
ax_sfr.set_xlim(50, 3.4e3)
ax_sfr.grid()

# double axis with sim_aexp
ax2 = ax_sfr.twiny()
ax2.set_xlabel("redshift")
xticks2 = ax_sfr.get_xticks()
# ax_sfr.set_xlim(xticks2.min(), xticks2.max())
ax_sfr.set_xticks(xticks2)
ax2.set_xticks(xticks2)
zed_ticks = np.asarray(
    [z_at_value(cosmo.age, (max(age, 1.0)) * u.Myr) for age in xticks2]
)
ax2.set_xticklabels(np.round(zed_ticks, decimals=2))


sfr_fname = os.path.join("sfr_dense.png")
fig_sfr.savefig(sfr_fname)
