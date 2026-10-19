from matplotlib.lines import Line2D
from gremlin.read_sim_params import ramses_sim
from zoom_analysis.sinks.sink_reader import (
    get_sink_mhistory,
    find_massive_sink,
    read_sink_bin,
    read_sink_snap,
    snap_to_coarse_step,
    check_if_superEdd,
)
from zoom_analysis.halo_maker.assoc_fcts import (
    find_snaps_with_halos,
    get_halo_props_snap,
    get_gal_props_snap,
    find_star_ctr_period,
    # find_zoom_tgt_halo,
    # get_central_gal_for_hid,
    # get_gal_props_snap,
)

from zoom_analysis.zoom_helpers import find_starting_position, decentre_coordinates
from gremlin.read_sim_params import ramses_sim
from zoom_analysis.constants import ramses_pc
from zoom_analysis.trees.tree_reader import read_tree_file_rev as read_tree_fev_sim
from zoom_analysis.trees.tree_reader import convert_adaptahop_pos

from zoom_analysis.halo_maker.read_treebricks import read_zoom_stars

# from hagn.IO import read_hagn_snap_brickfile, get_hagn_brickfile_stpids
from hagn.tree_reader import read_tree_rev, interpolate_tree_position
from hagn.catalogues import make_super_cat, get_cat_hids
from hagn.utils import get_hagn_sim

import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.spatial import KDTree
import h5py


# tgt_zed = 4.0
# tgt_zed = 4.3
# tgt_zed = 4.70
# tgt_zed = 4.60
# tgt_zed = 5.0
# tgt_zed = 5.8
tgt_zed = 6.0
# tgt_zed = 6.5
# tgt_zed = 7.0


sim_paths = [
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_coreen",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_low_dboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_minMassDynGrav",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictestSF_lowSNe",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_sink_desc_goodfix",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_vlowSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_high_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd",
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
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_sqrt_rr_merge",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_merge_main_branch",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_boostAcc",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_boostAcc_superEdd",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_boostAcc_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_boostAcc_lowerSFE",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_form_stellar_peak",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_form_stellar_peak_boostAcc",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_form_stellar_peak_boostAcc_1d7_dynMass",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_manual_sink1D7",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_lowSN_strictSF/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_model0_0p02eps",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_model0_0p05eps",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_model0_0p10eps",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_model5_100eps",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_NClike",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF_halfSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
]


fig,ax=plt.subplots()

for sdir in sim_paths:

    sim = ramses_sim(sdir)

    print(sim.name)

    snap = sim.get_closest_snap(zed=tgt_zed)

    if abs((1/sim.get_snap_exps(snap)[0]-1)-tgt_zed) > 0.1:
        continue

    try:
        gals = get_gal_props_snap(sim.path,snap)
    except:
        continue

    pure = gals['host purity']>0.9999

    sinks = read_sink_snap(sim,snap)

    sink_pos_tree = KDTree(sinks['position'])

    my_gal_masses = gals['mass'][pure]
    my_sink_masses = np.zeros(np.sum(pure)) - 1
    my_sink_dists = np.zeros(np.sum(pure)) - 1

    for igal,gid in enumerate(gals['gids'][pure]):

        rad = 3 * gals['r50'][pure][igal]
        pos = gals['pos'][:,pure][:,igal]

        nearby = sink_pos_tree.query_ball_point(pos,rad)

        if np.sum(nearby)>0:

            nearby_ids = sinks['identity'][nearby]
            nearby_masses = sinks['mass'][nearby]
            nearby_pos = sinks['position'][nearby]

            arg_mass = np.argmax(nearby_masses)

            my_sink_masses[igal] = nearby_masses[arg_mass]
            my_sink_dists[igal] = np.linalg.norm(nearby_pos[arg_mass]-pos)



    # print(list(zip(gals['mass'][pure],my_sink_masses,my_sink_dists)))


    ok = my_sink_dists>-1

    # sc=ax.scatter(my_sink_masses[ok],my_sink_dists[ok] * sim.cosmo.lcMpc*1e3,label=sim.name)
    sc=ax.scatter(my_gal_masses[ok],my_sink_dists[ok] * sim.cosmo.lcMpc*1e3,label=sim.name)
    ax.axhline(np.mean(my_sink_dists[ok]) * sim.cosmo.lcMpc*1e3,c=sc.get_facecolors(), ls='--')


ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Distance to host galaxy, ckpc')
# ax.set_xlabel('Sink mass, Msun')
ax.set_xlabel('Gal mass, Msun')

fig.suptitle(f'z={tgt_zed:.1f}')

ax.grid()


ax.legend(framealpha=0.0)
fig.savefig(f'test_sink_mass_dist_{str(tgt_zed).replace('.','p'):s}')