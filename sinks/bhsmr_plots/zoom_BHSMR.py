from zoom_analysis.sinks import sink_histories, sink_reader
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import get_gal_assoc_file, get_gal_props_snap

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

from zoom_analysis.zoom_helpers import check_in_all_sims_vol

from zoom_analysis.sinks.sink_constraints import (
    calculate_reines15_mbh_bulges,
    calculate_reines15_mbh_local,
    NH_smbh,
)

# from hagn.catalogues import make_super_cat

# fig, ax = sink_histories.setup_bh_plot()


# tgt_zed = 2.0
# tgt_zed = 2.1
# tgt_zed = 2.2
# tgt_zed = 2.4
# tgt_zed = 2.5
# tgt_zed = 2.6
# tgt_zed = 2.7
# tgt_zed = 2.8
# tgt_zed = 2.9
# tgt_zed = 3.0
# tgt_zed = 3.1
# tgt_zed = 3.5
# tgt_zed = 3.2
# tgt_zed = 3.3
# tgt_zed = 3.4
# tgt_zed = 3.5
# tgt_zed = 3.7
# tgt_zed = 3.8
# tgt_zed = 4.0
# tgt_zed = 4.05
# tgt_zed = 4.1
# tgt_zed = 4.2
# tgt_zed = 4.25
# tgt_zed = 4.4
tgt_zed = 4.5
# tgt_zed = 4.6
# tgt_zed = 4.7
# tgt_zed = 4.75
# tgt_zed = 4.8
# tgt_zed = 4.93
# tgt_zed = 5.0
# tgt_zed = 5.3
# tgt_zed = 5.4
# tgt_zed = 5.5
# tgt_zed = 5.7
# tgt_zed = 5.8
# tgt_zed = 6.0
# tgt_zed = 6.20
# tgt_zed = 6.53
# tgt_zed = 6.6
# tgt_zed = 6.78
# tgt_zed = 6.8
# tgt_zed = 7.0
# tgt_zed = 7.18626797394214067502
# tgt_zed = 7.33
# tgt_zed = 7.37
fpure = 1.0 - 1e-4

HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
delta_t = 100  # Myr

sim_dirs = [
    # "/data101/jlewis/sims/NH_for_rascas",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_NClike",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_model5_100eps",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_model0_0p05eps",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_lowSN_strictSF",
    # "/data103/jlewis/sims/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF_lowSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF_halfSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130_novrel_lowerSFE_stgNHboost_strictSF",

    # "/data101/jlewis/sims/NH_for_rascas",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id36755",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_NClike",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_model5_100eps",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_model0_0p02eps",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_model0_0p05eps",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_model0_0p10eps",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_manual_sink1D7",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_no_sink_desc",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_merge_main_branch",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_sink_desc_goodfix",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_boostAcc",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_boostAcc_lowerSFE",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_boostAcc_novrel",    
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_boostAcc_lowKin",    
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_boostAcc_lowTherm",    
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_boostAcc_superEdd",    
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_boostAcc_noSpin",    
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_form_stellar_peak",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_form_stellar_peak_boostAcc",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_form_stellar_peak_boostAcc_1d7_dynMass",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_sink_desc_goodfix",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_lowAGNtherm",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_fullresIC",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE/",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE_stgNHboost_stricterSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE_highAGNtherm/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE/",
    # "/data103/jlewis/sims/lvlmax_22/mh1e12/id180130/",
    # "/data103/jlewis/sims/lvlmax_21/mh1e12/id180130_256/",
    # "/data102/jlewis/sims/lvlmax_21/mh1e12/id180130_model6_eps0p05/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id180130_model5/",
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
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_no_sink_desc",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_sink_desc",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_sink_desc_fix",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_sink_desc_coarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_sink_desc_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_sink_desc_recursive",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_no_sink_desc",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_cst_comov",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE_lowNsink/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_pdrag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_noAGN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_high_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nrg_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_strict_massive_seeds",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id36755",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_lowerSFE_NHboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_stricterSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictestSF_lowSNe",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictestSF_lowSNe_highAGNeff",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_stricterSF_radioHeavy",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_medSFE_stgNHboost_stricterSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_XtremeSF_lowSNe",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_MegaSF_lowSNe",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_SuperMegaSF_midSNe",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_highSFE_stgNHboost_strictestSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_XtremeLowSFE_stgNHboost_strictSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_SuperLowSFE_stgNHboost_strictSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_stricterSF_SEdd2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_pdrag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id112288",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288_novrel_lowerSFE_stgNHboost_strictSF",
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
    "X",
    "D",
    "d",
    "|",
    "_",
]

xlim=(np.inf,-np.inf)
ylim=(np.inf,-np.inf)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))


gal_mass_bins = np.logspace(6, 12, 12)

all_same_sim = True

if ramses_sim(sim_dirs[0], nml="cosmo.nml", full_check=False).name.startswith('id'):
    prev_ID = int(
        ramses_sim(sim_dirs[0], nml="cosmo.nml", full_check=False).name.split("_")[0][2:]
    )
else:
    prev_ID = ramses_sim(sim_dirs[0], nml="cosmo.nml", full_check=False).name

for sim_dir in sim_dirs[1:]:

    sim = ramses_sim(sim_dir, nml="cosmo.nml", full_check=False)
    if sim.name.startswith('id'):
        intID = int(sim.name.split("_")[0][2:])
    else:
        intID = sim.name
        if intID == "NH_for_rascas":
            intID = "NewHorizon"

    all_same_sim = all_same_sim * (intID == prev_ID)

    prev_ID = intID


for isim, sim_dir in enumerate(sim_dirs):

    sim = ramses_sim(
        sim_dir, nml="cosmo.nml", full_snaps=False, tar_snaps=False, mini_snaps=True
    )

    name = sim.name

    print(name)

    if sim.name.startswith('id'):
        intID = int(sim.name.split("_")[0][2:])
    else:
        intID = sim.name

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
    found = False
    decal = 0

    z_dists = np.abs(1.0 / sim.aexps - 1.0 - tgt_zed)
    args = np.argsort(z_dists)

    while not found and decal < len(sim.snap_numbers):
        arg = args[decal]
        sim_snap = sim.snap_numbers[arg]
        sim_zed = 1.0 / sim.aexps[arg] - 1

        gfile = get_gal_assoc_file(sim_dir, sim_snap)

        found_cond = os.path.isfile(gfile) * (z_dists[arg] < 0.1)
        print(sim_snap, sim_zed, decal, gfile)

        if found_cond:
            found = True
        else:
            decal += 1

    if not found:
        continue

    gal_dict = get_gal_props_snap(sim.path, sim_snap, main_stars=False)
    # print(gal_dict.keys())

    gal_pos = gal_dict["pos"].T

    if len(sim_dirs)>1:
        if all_same_sim:
            vol_args, min_vol = check_in_all_sims_vol(gal_pos, sim, sim_dirs)

            if vol_args.sum() == 0:
                raise ValueError("No galaxies in volume")

    # print(isim, vol_args.sum())

    # centred_gal_pos = decentre_coordinates(gal_dict["pos"].T,sim.path)

    # vol_args = [check_vol(decentred_gal_pos) for check_vol in check_vols]

    # vol_args = np.all(vol_args, axis=0)

    host_purity = gal_dict["host purity"]
    central_cond = gal_dict["central"] == 1
    pure_cond = (host_purity > fpure) * central_cond
    if len(sim_dirs)>1:
        if all_same_sim:
            pure_cond *= vol_args
    gal_mass = gal_dict["mass"][pure_cond]
    gal_pos = gal_dict["pos"][:, pure_cond]
    rmax = gal_dict["rmax"][pure_cond]
    gids = gal_dict["gids"][pure_cond]

    print(f"I found {pure_cond.sum():d} galaxies with purity > {fpure:.4f},")
    print(f"and {len(host_purity)-pure_cond.sum()} galaxies with purity < {fpure:.4f}")

    sink_mass = np.zeros_like(gal_mass)

    step=None

    for igal, gid in enumerate(gids):
        try:
            sid, found, step = sink_reader.gid_to_sid(sim, gid, sim_snap, step=step, gdict=gal_dict)
        except AssertionError:
            found = False

        if not found:
            print(gid,np.log10(gal_mass[igal]),f'No sink')
            continue

        print(gid,np.log10(gal_mass[igal]),sid)

        sink_props = sink_reader.get_sink(sid, sim_snap, sim)
        if "mass" in sink_props:
            sink_mass[igal] = sink_props["mass"]

    print(np.mean(sink_mass / gal_mass))

    # for igal, (m, pos, rmax) in enumerate(zip(gal_mass, gal_pos.T, rmax)):
    # print(m, pos, rmax)

    # try:
    #     sink_props = sink_reader.find_massive_sink(pos, sim_snap, sim, rmax)

    #     # sink_dict = sink_reader.get_sink_mhistory(sid, sim_snap, sim)
    #     sink_mass[igal] = sink_props["mass"]
    # except ValueError:
    #     print("No sink found")

    label = sim.name
    if len(sim_dirs) > 1:
        label += f", z={sim_zed:.2f}"

    sca = ax.scatter(gal_mass, sink_mass, label=label, alpha=0.5, marker=markers[isim])

    if xlim[0]> gal_mass.min():
        xlim = (gal_mass.min()*0.9, xlim[1])
    if xlim[1]< gal_mass.max():
        xlim = (xlim[0], gal_mass.max()*1.1)
    if ylim[0]> sink_mass.min()*0.9:
        ylim = (sink_mass.min(), ylim[1])
    if ylim[1]< sink_mass.max():
        ylim = (ylim[0], sink_mass.max()*1.1)

    if ylim[0]< sim.namelist['smbh_params']['mseed']:
        ylim = (sim.namelist['smbh_params']['mseed']*0.9, ylim[1])


    if ylim[1]<1e7:
        ylim = (ylim[0], 1e7)



    pos = sink_mass > 0

    if pos.sum() == 0:
        continue

    avg, _, _ = binned_statistic(
        gal_mass[pos], sink_mass[pos], bins=gal_mass_bins, statistic="mean"
    )
    med_bin, _, _ = binned_statistic(
        gal_mass[pos], gal_mass[pos], bins=gal_mass_bins, statistic="median"
    )
    ax.plot(med_bin, avg, c=sca.get_facecolor(), lw=1, ls="--")


reines_masses_bulges = calculate_reines15_mbh_bulges(gal_mass_bins)
reines_masses_local = calculate_reines15_mbh_local(gal_mass_bins)

ax.plot(
    gal_mass_bins[gal_mass_bins > 1e8],
    reines_masses_bulges[gal_mass_bins > 1e8],
    label="Reines,Volonteri 2015 $M_{BH}/M_{\star}$",
    c="k",
    lw=0.6,
)
ax.plot(
    gal_mass_bins[gal_mass_bins > 1e8],
    reines_masses_local[gal_mass_bins > 1e8],
    # label="Reines+ 2015 $M_{BH}/M_{\star}$",
    c="k",
    lw=0.6,
)
# ax.plot(
#     gal_mass_bins,
#     calculate_reines_mbh(gal_mass_bins) * 1.05,
#     r"Reines+ 2015 $M_{BH}/M_{\star}$",
#     verticalalignment="bottom",
#     horizontalalignment="left",
#     fontsize=6,
#     rotation=angle,
#     # rotation_mode="anchor",
#     # transform_rotates_text=True,
#     zorder=np.inf,
# )




# now for HAGN

hagn_sim = get_hagn_sim()

hagn_snap = hagn_sim.get_closest_snap(zed=tgt_zed)
hagn_aexp = hagn_sim.get_snap_exps(hagn_snap)[0]
hagn_zed = 1.0 / hagn_aexp - 1


if all_same_sim:
    try: 

        hagn_gals = make_super_cat(hagn_snap)

        # print(hagn_gals.keys())

        hagn_gal_pos = np.transpose([hagn_gals["x"], hagn_gals["y"], hagn_gals["z"]])

        vol_args, min_vol = check_in_all_sims_vol(hagn_gal_pos, hagn_sim, sim_dirs)

        if np.sum(vol_args) > 0:

            if "mbh" in hagn_gals:

                mbhs = hagn_gals["mbh"][vol_args]
                mgals = hagn_gals["mgal"][vol_args]

                ax.scatter(
                    mgals,
                    mbhs,
                    label=f"HAGN, z={hagn_zed:.2f}",
                    alpha=0.5,
                    marker="x",
                    color="k",
                )


                if xlim[0]> mgals.min():
                    xlim = (mgals.min()*0.9, xlim[1])
                if xlim[1]< mgals.max():
                    xlim = (xlim[0], mgals.max()*1.1)
                if ylim[0]> mbhs.min()*0.9:
                    ylim = (mbhs.min(), ylim[1])
                if ylim[1]< mbhs.max():
                    ylim = (ylim[0], mbhs.max()*1.1)                

                # now avg stats
                avg, _, _ = binned_statistic(
                    mgals, mbhs, bins=gal_mass_bins, statistic=np.nanmean
                )
                med_bin, _, _ = binned_statistic(
                    mgals, mgals, bins=gal_mass_bins, statistic=np.nanmedian
                )
                ax.plot(med_bin, avg, c="k", lw=1, ls="--")  # ,label=f"HAGN, z={hagn_zed:.2f}")
    except FileNotFoundError:
        pass

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


ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.grid()

ax.set_xlabel("Stellar mass [M$_\odot$]")
ax.set_ylabel("BH Mass [M$_\odot$]")

ax.set_yscale("log")
ax.set_xscale("log")
title_txt = ""
if len(sim_dirs) == 1:
    title_txt += f"z={sim_zed:.2f}"
    # ax.text(0.05, 0.95, f"z={sim_zed:.2f}", transform=ax.transAxes)
# also print purity threshold
# ax.text(0.05, 0.9, f"fpure > {fpure:.3f}", transform=ax.transAxes)
title_txt += f" fpure > {fpure:.4f}"

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

NH_smbh(ax)

if all_same_sim:
    name_prefix = f"{intID}_"
else:
    name_prefix = ""

ax.legend(framealpha=0.0, title=title_txt)
if tgt_zed != None:
    fig.savefig(name_prefix + f"zoom_BHSMR_z{tgt_zed:.2f}.png")
else:
    fig.savefig(name_prefix + "zoom_BHSMR.png")
