from zoom_analysis.catalogue.read_cat import read_cat

# from zoom_analysis.stars import sfhs
# from zoom_analysis.zoom_helpers import decentre_coordinates
# from zoom_analysis.stars.star_reader import yt_read_star_ball, read_part_ball
# from zoom_analysis.stars.leja_quench_fit import sfr_ridge_leja22
# from zoom_analysis.halo_maker.read_treebricks import (
#     # read_brickfile,
#     # convert_brick_units,
#     # convert_star_units,
#     read_zoom_stars,
# )

from scipy.stats import binned_statistic

# from zoom_analysis.trees.tree_reader import read_tree_file_rev as read_tree_fev_sim
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import (
    # find_zoom_tgt_gal,
    # find_zoom_tgt_halo,
    # get_central_gal_for_hid,
    get_gal_assoc_file,
    find_snaps_with_gals,
    get_gal_props_snap,
)

# from zoom_analysis.zoom_helpers import find_starting_position, starting_hid_from_hagn

from gremlin.read_sim_params import ramses_sim

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import os
import numpy as np
import h5py

# from hagn.association import gid_to_stars
from hagn.utils import get_hagn_sim  # , adaptahop_to_code_units

# from hagn.tree_reader import read_tree_rev, interpolate_tree_position
from hagn.catalogues import get_halos_cat, make_super_cat, get_cat_hids, get_cat_gids

from astropy.cosmology import Planck18 as cosmo

from hagn.catalogues import make_super_cat  # , get_cat_hids

# from f90_tools.star_reader import read_part_ball_hagn, read_part_ball_NCdust

from matplotlib import colors


def shuntov24_all(z):

    mshuntov_d = "/data101/jlewis/constraints/mshuntov_smf_24/v3.1.1"

    # lookup zeds
    avail_files = np.asarray(os.listdir(mshuntov_d))
    zstrs = np.asarray([np.float32(f[2:].split("-")) for f in avail_files])

    ok_z_num = np.empty(len(avail_files), dtype=bool)
    ok_z_num[:] = False

    for ifile, zstr in enumerate(zstrs):
        ok_z_num[ifile] = z >= zstr[0] and z <= zstr[1]

    if not np.any(ok_z_num):
        return {"logM": np.asarray([]), "Fi": np.asarray([])}
    elif np.sum(ok_z_num) > 1:
        bin_ctrs = np.mean(zstrs, axis=1)
        ok_z_num[:] = False
        ok_z_num[np.argmin(np.abs(bin_ctrs - z))] = True

    zdir = avail_files[ok_z_num][0]

    z_ok_files = np.asarray(os.listdir(os.path.join(mshuntov_d, zdir)))

    sf_file = [f for f in z_ok_files if "SMF-all" in f]
    if len(sf_file) == 0:
        return {"logM": np.asarray([]), "Fi": np.asarray([])}

    return np.genfromtxt(os.path.join(mshuntov_d, zdir, sf_file[0]), names=True)


def shuntov24_qu(z):

    mshuntov_d = "/data101/jlewis/constraints/mshuntov_smf_24/v3.1.1"

    # lookup zeds
    avail_files = np.asarray(os.listdir(mshuntov_d))
    zstrs = np.asarray([np.float32(f[2:].split("-")) for f in avail_files])

    ok_z_num = np.empty(len(avail_files), dtype=bool)
    ok_z_num[:] = False

    for ifile, zstr in enumerate(zstrs):
        ok_z_num[ifile] = z >= zstr[0] and z < zstr[1]

    if not np.any(ok_z_num):
        return {"logM": np.asarray([]), "Fi": np.asarray([])}
    elif np.sum(ok_z_num) > 1:
        bin_ctrs = np.mean(zstrs, axis=1)
        ok_z_num[:] = False
        ok_z_num[np.argmin(np.abs(bin_ctrs - z))] = True

    zdir = avail_files[ok_z_num][0]

    z_ok_files = np.asarray(os.listdir(os.path.join(mshuntov_d, zdir)))

    qu_file = [f for f in z_ok_files if "SMF-Quies" in f]
    if len(qu_file) == 0:
        return {"logM": np.asarray([]), "Fi": np.asarray([])}

    return np.genfromtxt(os.path.join(mshuntov_d, zdir, qu_file[0]), names=True)


def schechter(m, phi1, mstar, alpha1):
    return (
        np.log(10)
        * np.exp(-(10 ** (m - mstar)))
        * phi1
        * (10 ** (m - mstar)) ** (alpha1 + 1)
    )


def dbl_schechter(m, phi1, phi2, mstar, alpha1, alpha2):
    return (
        np.log(10)
        * np.exp(-(10 ** (m - mstar)))
        * (
            phi1 * (10 ** (m - mstar)) ** (alpha1 + 1)
            + phi2 * (10 ** (m - mstar)) ** (alpha2 + 1)
        )
    )


def mcleod_sf(m, z):

    zbins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.25]
    zbin_wdth = 0.5

    valid_z = (z >= zbins[0] - 0.5 * zbin_wdth) * (z <= zbins[-1] + 0.5 * zbin_wdth)

    if not valid_z:
        return np.zeros_like(m)

    zbin = np.argmin(np.abs(z - zbins))

    mstars = [10.72, 10.77, 10.77, 10.83, 10.84, 10.82, 11.02]
    phi1s = [-3.15, -3.07, -3.13, -3.33, -3.43, -3.52, -4.19]
    alpha1s = [-1.45, -1.41, -1.43, -1.51, -1.47, -1.52, -1.85]
    mlims = [8.0, 8.5, 8.6, 8.7, 9.6, 9.8, 10.2]

    return schechter(m, 10 ** phi1s[zbin], mstars[zbin], alpha1s[zbin]) * (
        m > mlims[zbin]
    )


def mcleod_quiescent(m, z):

    zbins = [1.0, 1.5, 2.0, 2.5, 3.0, 3.25]
    zbin_wdth = 0.5

    valid_z = (z >= zbins[0] - 0.5 * zbin_wdth) * (z <= zbins[-1] + 0.5 * zbin_wdth)

    if not valid_z:
        return np.zeros_like(m)

    mstars = [10.59, 10.48, 10.43, 10.52, 10.45, 10.40]
    phi1s = [-2.44, -2.79, -2.91, -3.19, -3.51, -3.77, -4.33]
    alpha1s = [0.19, 0.41, 0.69, 0.32, 0.71, 0.49]
    phi2s = [-3.80, -3.97, -4.23, None, None, None]
    alpha2s = [-1.49, -1.32, -1.13, None, None, None]
    mlims = [8.5, 8.6, 8.7, 9.6, 9.8, 10.2]

    zbin = np.argmin(np.abs(z - zbins))

    if phi2s[zbin] == None:
        return schechter(m, 10 ** phi1s[zbin], mstars[zbin], alpha1s[zbin]) * (
            m > mlims[zbin]
        )
    else:
        return dbl_schechter(
            m,
            10 ** phi1s[zbin],
            10 ** phi2s[zbin],
            mstars[zbin],
            alpha1s[zbin],
            alpha2s[zbin],
        ) * (m > mlims[zbin])


def hamadch_quiescent(m, z):

    zbins = [0.5, 1.0, 1.5, 2.0]
    zbin_wdth = 0.5

    valid_z = (z >= zbins[0] - 0.5 * zbin_wdth) * (z <= zbins[-1] + 0.5 * zbin_wdth)

    if not valid_z:
        return np.zeros_like(m)

    zbin = np.argmin(np.abs(z - zbins))

    mstars = [10.7, 10.73, 10.67, 10.71]
    phi1s = [-2.7, -3.0, -3.17, -3.37]
    alpha1s = [0.3, 0.19, 0.22, 0.02]
    phi2s = [-3.46, -4.55, -5.07, -7.25]
    alpha2s = [-1.3, -1.53, -1.54, -2.6]
    mlims = [7.7, 7.9, 8.0, 8.5]

    return dbl_schechter(
        m,
        10 ** phi1s[zbin],
        10 ** phi2s[zbin],
        mstars[zbin],
        alpha1s[zbin],
        alpha2s[zbin],
    ) * (m > mlims[zbin])


def hamadch_sf(m, z):

    zbins = [0.5, 1.0, 1.5, 2.0]
    zbin_wdth = 0.5

    valid_z = (z >= zbins[0] - 0.5 * zbin_wdth) * (z <= zbins[-1] + 0.5 * zbin_wdth)

    if not valid_z:
        return np.zeros_like(m)

    zbin = np.argmin(np.abs(z - zbins))

    mstars = [10.93, 10.95, 10.94, 10.97]
    phi1s = [-2.92, -3.05, -3.11, -3.17]
    alpha1s = [-1.39, -1.4, -1.4, -1.39]
    mlims = [7.7, 7.9, 8.0, 8.5]

    return schechter(m, 10 ** phi1s[zbin], mstars[zbin], alpha1s[zbin]) * (
        m > mlims[zbin]
    )


# sim_ids = [
#     "id2757_995pcnt_mstel/",
#     "id19782_500pcnt_mstel/",pip install setuptools
#     "id13600_005pcnt_mstel/",
# ]
# sdir = "/home/jlewis/sims/proposal_zooms/ExtractZoom/mh1e12/"
hagn_snap = 197  # z=2 in the full res box
HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
delta_t = 100  # Myr
zstt = 2.0

sdirs = [
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_coreen",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_DynBondiGravSinkMass",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowSFE_gravSink",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowSFE_DynBondiGravSinkMass",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_fullresIC",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_kicks",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_smooth_ref",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_boostNH",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_early_refine",
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
    # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_drag",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_lowerSFE_drag",
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
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_XtremeLowSFE_stgNHboost_strictSF",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictestSF_lowSNe"
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099",
    # "/data102/jlewis/sims/lvlmax_21/mh1e12/id180130_model6_eps0p05",
    # "/data103/jlewis/sims/lvlmax_21/mh1e12/id180130_256",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost_lowerSFE",
    "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_noHydroDragB4z4merge",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288",
    "/data103/jlewis/sims/lvlmax_22/mh1e12/id180130",
]


yax2 = None

hagn_sim = get_hagn_sim()
hagn_sim.init_cosmo()
l_hagn = hagn_sim.unit_l(hagn_sim.aexp_stt) / (ramses_pc * 1e6) / hagn_sim.aexp_stt
# colors = []

hagn_snaps = hagn_sim.snap_numbers
hagn_aexps = hagn_sim.get_snap_exps(param_save=False)
hagn_times = hagn_sim.cosmo_model.age(1.0 / hagn_aexps - 1.0).value * 1e3
hagn_zeds = 1.0 / hagn_aexps - 1

tgt_zed = 1.0 / hagn_aexps[hagn_snaps == hagn_snap] - 1.0
tgt_time = cosmo.age(tgt_zed).value

mf_nbins = 12
mass_fct_bins = np.logspace(6, 12, mf_nbins)
mass_fct_bin_sizes = 0.5 * np.diff(mass_fct_bins)


mass_bins = [10**9.5, 10**10.0, 10**10.6]
# mass_bins = [10**9.5, 10**10., np.inf]
# ssfr_bins = [1e-9, 1e-10, 1e-11]
ssfr_bins = [10**-10.5]

zstep = 1.0
zbins_plot = np.arange(2, 6 + zstep, zstep)

time_bins_total = np.arange(100, 3400, 25)
mf_zoom = np.zeros((len(sdirs), mf_nbins - 1, len(zbins_plot)))
mf_hagn = np.zeros((len(sdirs), mf_nbins - 1, len(zbins_plot)))
mf_zoom_pass = np.zeros((len(sdirs), mf_nbins - 1, len(ssfr_bins), len(zbins_plot)))
mf_hagn_pass = np.zeros((len(sdirs), mf_nbins - 1, len(ssfr_bins), len(zbins_plot)))

tot_zoom_quenched_dens = np.zeros(
    (len(time_bins_total), len(mass_bins), len(ssfr_bins))
)
tot_zoom_galdens = np.zeros((len(time_bins_total), len(mass_bins)))
tot_zoom_vol = np.zeros(len(time_bins_total))
tot_hagn_quenched_dens = np.zeros(
    (len(time_bins_total), len(mass_bins), len(ssfr_bins))
)
tot_hagn_galdens = np.zeros((len(time_bins_total), len(mass_bins)))


hagn_closest_z = [
    hagn_zeds[np.argmin(np.abs(z - hagn_zeds))]
    for z in zbins_plot
    if np.abs(z - hagn_zeds).min() < 0.2
]


# zed_bins = []

isim = 0

# overwrite = True
overwrite = False
main_stars = False

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


plot_mbin = 1
plot_ssfr_bin = 0


fig, ax = plt.subplots(2, 2, figsize=(8, 8), layout="constrained")
# fig_ssfr, ax_ssfr = plt.subplots(1, 1, figsize=(8, 8), layout="constrained")

lines = []
labels = []

last_simID = None
l = None

zoom_style = 0
vol_tot = 0

for isim, sdir in enumerate(sdirs):

    name = sdir.split("/")[-1].strip()

    # sim_id = name[2:].split("_")[0]

    # sim_ids = ["id74099", "id147479", "id242704"]

    # c = "tab:blue"

    # get the galaxy in HAGN

    intID = int(name[2:].split("_")[0])
    # gid = int(make_super_cat(197, outf="/data101/jlewis/hagn/super_cats")["gid"][0])

    sim = ramses_sim(sdir, nml="cosmo.nml")

    sim_snaps = sim.snap_numbers
    sim_aexps = sim.get_snap_exps()
    sim_zeds = 1.0 / sim_aexps - 1.0
    sim.init_cosmo()
    sim_times = sim.cosmo_model.age(1.0 / sim_aexps - 1.0).value * 1e3

    zoom_closest_z = [
        sim_zeds[np.argmin(np.abs(z - sim_zeds))]
        for z in zbins_plot
        if np.abs(z - sim_zeds).min() < 0.2
    ]

    # last sim_aexp
    # valid_steps = hagn_tree_aexps < (1.0 / (tgt_zed + 1.0))  # sim.aexp_end
    # nsteps = np.sum(valid_steps)

    hagn_sim = get_hagn_sim()

    nsteps = len(sim_snaps)
    nstep_hagn = len(hagn_aexps)

    mstel_hagn = np.zeros(nstep_hagn, dtype=np.float32)
    mvir_hagn = np.zeros(nstep_hagn, dtype=np.float32)
    sfr_hagn = np.zeros(nstep_hagn, dtype=np.float32)
    ssfr_hagn = np.zeros(nstep_hagn, dtype=np.float32)
    qf_hagn = np.zeros((nstep_hagn, len(mass_bins), len(ssfr_bins)), dtype=np.float32)
    qdens_hagn = np.zeros(
        (nstep_hagn, len(mass_bins), len(ssfr_bins)), dtype=np.float32
    )
    galdens_hagn = np.zeros((nstep_hagn, len(mass_bins)), dtype=np.float32)
    time_hagn = np.zeros(nstep_hagn, dtype=np.float32)

    mstel_zoom = np.zeros(nsteps, dtype=np.float32)
    mvir_zoom = np.zeros(nsteps, dtype=np.float32)
    sfr_zoom = np.zeros(nsteps, dtype=np.float32)
    ssfr_zoom = np.zeros(nsteps, dtype=np.float32)
    qf_zoom = np.zeros((nsteps, len(mass_bins), len(ssfr_bins)), dtype=np.float32)
    qdens_zoom = np.zeros((nsteps, len(mass_bins), len(ssfr_bins)), dtype=np.float32)
    galdens_zoom = np.zeros((nsteps, len(mass_bins)), dtype=np.float32)
    time_zoom = np.zeros(nsteps, dtype=np.float32)

    datas_path = os.path.join(sim.path, "computed_data", main_star_str)
    fout_h5 = os.path.join(datas_path, f"quenched_stats.h5")
    read_existing = True

    if not os.path.exists(datas_path):
        os.makedirs(datas_path)
        read_existing = False
    else:
        if not os.path.exists(fout_h5):
            read_existing = False

    if overwrite:
        read_existing = False

    if read_existing:

        # read the data
        with h5py.File(fout_h5, "r") as f:
            saved_mstel_hagn = f["mstel_hagn"][:]
            saved_sfr_hagn = f["sfr_hagn"][:]
            saved_ssfr_hagn = f["ssfr_hagn"][:]
            saved_qf_hagn = f["qf_hagn"][:]
            saved_qdens_hagn = f["qdens_hagn"][:]
            saved_galdens_hagn = f["galdens_hagn"][:]
            saved_time_hagn = f["time_hagn"][:]

            saved_mstel_zoom = f["mstel_zoom"][:]
            saved_sfr_zoom = f["sfr_zoom"][:]
            saved_ssfr_zoom = f["ssfr_zoom"][:]
            saved_qf_zoom = f["qf_zoom"][:]
            saved_qdens_zoom = f["qdens_zoom"][:]
            saved_galdens_zoom = f["galdens_zoom"][:]
            saved_time_zoom = f["time_zoom"][:]

            saved_vol_cMpc = f["vol_cMpc"][()]

        if saved_time_zoom.max() >= sim_times.max():
            # if we have the same number of values
            # then just read the data
            mstel_zoom = saved_mstel_zoom
            sfr_zoom = saved_sfr_zoom
            ssfr_zoom = saved_ssfr_zoom
            qf_zoom = saved_qf_zoom
            qdens_zoom = saved_qdens_zoom
            galdens_zoom = saved_galdens_zoom
            time_zoom = saved_time_zoom

            mstel_hagn = saved_mstel_hagn
            sfr_hagn = saved_sfr_hagn
            ssfr_hagn = saved_ssfr_hagn
            qf_hagn = saved_qf_hagn
            qdens_hagn = saved_qdens_hagn
            galdens_hagn = saved_galdens_hagn
            time_hagn = saved_time_hagn

            vol_cMpc = saved_vol_cMpc
        else:  # otherwise we need to recompute because something has changed
            read_existing = False

    # print(read_existing, len(saved_mstel_zoom), len(sim_snaps))

    if not read_existing:

        # find last output with assoc files
        # assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
        # assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

        assoc_file_nbs = find_snaps_with_gals(sim_snaps, sim.path)

        avail_aexps = np.intersect1d(
            sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
        )

        if len(avail_aexps) == 0:
            continue

        avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

        prev_mass = -1
        # prev_pos = None

        zoom_ctr = [
            sim.namelist["refine_params"]["xzoom"],
            sim.namelist["refine_params"]["yzoom"],
            sim.namelist["refine_params"]["zzoom"],
        ]
        if "rzoom" in sim.namelist["refine_params"]:
            check_zoom = (
                lambda coords: np.linalg.norm(coords - zoom_ctr, axis=1)
                < sim.namelist["refine_params"]["rzoom"]
            )

            vol_cMpc = (
                4.0
                / 3
                * np.pi
                * (sim.namelist["refine_params"]["rzoom"] * sim.cosmo.lcMpc) ** 3
            )

        elif "azoom" in sim.namelist["refine_params"]:
            if "zoom_shape" in sim.namelist["refine_params"]:
                ellipse = True
                if sim.namelist["refine_params"]["zoom_shape"] == "rectangle":
                    ellipse = False
            else:
                ellipse = True

            a = sim.namelist["refine_params"]["azoom"]
            b = sim.namelist["refine_params"]["bzoom"]
            c = sim.namelist["refine_params"]["czoom"]

            if ellipse:
                check_zoom = (
                    lambda coords: (
                        ((coords[:, 0] - zoom_ctr[0]) / a) ** 2
                        + ((coords[:, 1] - zoom_ctr[1]) / b) ** 2
                        + ((coords[:, 2] - zoom_ctr[2]) / c) ** 2
                    )
                    < 1
                )

                vol_cMpc = 4 / 3 * np.pi * a * b * c * sim.cosmo.lcMpc**3

            else:
                check_zoom = lambda coords: (np.abs(coords[:, 0] - zoom_ctr[0]) < a) * (
                    np.abs(coords[:, 1] - zoom_ctr[1])
                ) < b * (np.abs(coords[:, 2] - zoom_ctr[2]) < c)

                vol_cMpc = a * b * c * sim.cosmo.lcMpc**3

        # hagn tree loop
        for istep, (aexp, time) in enumerate(zip(hagn_aexps, hagn_times)):

            hagn_snap = hagn_snaps[np.argmin(np.abs(hagn_aexps - aexp))]

            try:
                super_cat = make_super_cat(
                    hagn_snap, outf="/data101/jlewis/hagn/super_cats"
                )  # , overwrite=True)
            except (FileNotFoundError, OSError):
                print("No super cat")
                continue

            # print(super_cat.keys())

            hagn_pos = np.transpose([super_cat["x"], super_cat["y"], super_cat["z"]])

            in_zoom = np.where(check_zoom(hagn_pos))[0]

            ssfr_hagn[istep] = np.sum(
                super_cat["sfr100"][in_zoom] / super_cat["mgal"][in_zoom]
            )
            sfr = super_cat["sfr100"][in_zoom]
            mass = super_cat["mgal"][in_zoom]

            ssfr = sfr / mass

            mstel_hagn[istep] = mass.sum()
            mvir_hagn[istep] = super_cat["mhalo"][in_zoom].sum()
            sfr_hagn[istep] = sfr.sum()
            time_hagn[istep] = time

            time_diff = np.abs(time - time_bins_total)
            time_bins_arg = np.argmin(time_diff)
            if time_diff[time_bins_arg] > 20:  # 20 Myr
                time_bins_arg = None

            zed = 1.0 / aexp - 1.0
            if np.any(np.abs(zed - hagn_closest_z) < 0.001):
                iz = np.argmin(np.abs(hagn_closest_z - zed))
                mf_hagn[isim, :, iz], _, _ = binned_statistic(
                    mass, mass, bins=mass_fct_bins, statistic="count"
                )
                # print(isim, istep, aexp, mf_hagn[isim, :, iz])

            for mb, mass_bin in enumerate(mass_bins[:]):
                mass_cond = super_cat["mgal"][in_zoom] > mass_bin
                galdens_hagn[istep, mb] = mass_cond.sum() / vol_cMpc
                for ssfrb, ssfr_bin in enumerate(ssfr_bins[:]):
                    # mass_cond = (super_cat["mgal"][in_zoom] > mass_bin) * (
                    #     super_cat["sfr100"][in_zoom] <= mass_bins[mb + 1]
                    # )

                    ssfr_cut = ssfr < ssfr_bin

                    # in_bin = np.where(
                    #     (mass_cond) * (ssfr < ssfr_bin) * (ssfr > ssfr_bins[ssfrb + 1])
                    # )[0]

                    if ssfr_cut.sum() == 0:
                        continue
                    if np.any(np.abs(zed - hagn_closest_z) < 0.001):
                        iz = np.argmin(np.abs(hagn_closest_z - zed))
                        mf_hagn_pass[isim, :, ssfrb, iz], _, _ = binned_statistic(
                            mass[ssfr_cut],
                            mass[ssfr_cut],
                            bins=mass_fct_bins,
                            statistic="count",
                        )

                    in_bin = np.where((mass_cond) * (ssfr_cut))[0]
                    qf_hagn[istep, mb, ssfrb] = len(in_bin) / mass_cond.sum()
                    qdens_hagn[istep, mb, ssfrb] = len(in_bin) / vol_cMpc

                    if time_bins_arg is not None:

                        tot_hagn_quenched_dens[time_bins_arg, mb, ssfrb] += (
                            qdens_hagn[istep, mb, ssfrb] * vol_cMpc
                        )
                if time_bins_arg is not None:
                    tot_hagn_galdens[time_bins_arg, mb] += (
                        galdens_hagn[istep, mb] * vol_cMpc
                    )

        # # zoom loop
        for istep, (snap, aexp, time) in enumerate(
            zip(sim_snaps, sim_aexps, sim_times)
        ):
            # if time < sim_tree_times.min() - 5:
            # continue

            zed = 1.0 / aexp - 1.0

            if np.all(np.abs(avail_aexps - aexp) > 1e-1):
                print(f"No assoc file for at z={1./aexp-1:.1f},snap={snap:d}")
                continue

            if not os.path.exists(get_gal_assoc_file(sim.path, snap)):
                print("no assoc file")
                continue

            # assoc_file = assoc_files[assoc_file_nbs == snap]
            # if len(assoc_file) == 0:
            #     print(f"No assoc file for at z={1./aexp-1:.1f},snap={snap:d}")
            #     continue

            gprops = get_gal_props_snap(sim.path, snap)

            zoom_pos = gprops["pos"].T

            in_zoom = np.where(
                check_zoom(zoom_pos) * (gprops["host purity"] > (1 - 1e-4))
            )[0]

            gids = gprops["gids"][in_zoom]

            mass = np.zeros(len(gids))
            sfr = np.zeros(len(gids))

            gpos = gprops["pos"].T[in_zoom]

            centrals = gprops["central"][in_zoom]

            for igal, gid in enumerate(gids):

                try:
                    gal_cat = read_cat(
                        sim.path,
                        snap,
                        "galaxy",
                        2.0,
                        gid,
                        tgt_keys=["stars/Mstar_msun", "stars/sfr100_Msun_per_yr"],
                    )
                except (ValueError, FileNotFoundError, KeyError, BlockingIOError):
                    continue
                mass[igal] = gal_cat["stars/Mstar_msun"]
                sfr[igal] = gal_cat["stars/sfr100_Msun_per_yr"]

            # print(np.log10(mass.sum()), np.log10(sfr.sum()))
            # print(
            #     np.log10(gprops["mass"][in_zoom].sum()),
            #     np.log10(gprops["sfr100"][in_zoom].sum()) - 6,
            # )
            # mass = gprops["mass"][in_zoom]
            # sfr = gprops["sfr100"][in_zoom] / 1e6
            ssfr = sfr / mass

            rmax = np.max(np.linalg.norm(gpos - zoom_ctr, axis=1))
            rmax_vol = (rmax * sim.cosmo.lcMpc) ** 3 * 4 / 3 * np.pi

            zoom_vol = min(rmax_vol, vol_cMpc)

            mstel_zoom[istep] = mass.sum()
            mvir_zoom[istep] = gprops["host mass"][in_zoom].sum()
            sfr_zoom[istep] = sfr.sum()
            ssfr_zoom[istep] = ssfr.sum()
            time_zoom[istep] = time

            time_diff = np.abs(time - time_bins_total)
            time_bins_arg = np.argmin(time_diff)
            if time_diff[time_bins_arg] > 20:  # 20 Myr
                time_bins_arg = None

            z = 1.0 / aexp - 1
            if np.any(np.abs(z - zoom_closest_z) < 0.001):
                iz = np.argmin(np.abs(zed - zoom_closest_z))
                mf_zoom[isim, :, iz], _, _ = binned_statistic(
                    mass, mass, bins=mass_fct_bins, statistic="count"
                )

            for mb, mass_bin in enumerate(mass_bins[:-1]):
                mass_cond = (mass > mass_bin) * (centrals == True)
                galdens_zoom[istep, mb] = mass_cond.sum() / zoom_vol
                for ssfrb, ssfr_bin in enumerate(ssfr_bins[:-1]):
                    # mass_cond = (mass > mass_bin) * (mass <= mass_bins[mb + 1])

                    ssfr_cut = ssfr < ssfr_bin

                    if ssfr_cut.sum() == 0:
                        continue

                    # in_bin = np.where(
                    #     (mass_cond) * (ssfr < ssfr_bin) * (ssfr > ssfr_bins[ssfrb + 1])
                    # )[0]

                    if np.any(np.abs(z - zoom_closest_z) < 0.001):
                        iz = np.argmin(np.abs(zed - zoom_closest_z))
                        mf_zoom_pass[isim, :, ssfrb, iz], _, _ = binned_statistic(
                            mass[ssfr_cut],
                            mass[ssfr_cut],
                            bins=mass_fct_bins,
                            statistic="count",
                        )

                    in_bin = np.where((mass_cond) * (ssfr_cut))[0]
                    qf_zoom[istep, mb, ssfrb] = len(in_bin) / mass_cond.sum()
                    qdens_zoom[istep, mb, ssfrb] = len(in_bin) / zoom_vol

                    # print(ssfr[in_bin])

                    if time_bins_arg is not None:
                        tot_zoom_quenched_dens[time_bins_arg, mb, ssfrb] += (
                            qdens_zoom[istep, mb, ssfrb] * zoom_vol
                        )
                if time_bins_arg is not None:
                    tot_zoom_galdens[time_bins_arg, mb] += (
                        galdens_zoom[istep, mb] * zoom_vol
                    )
            tot_zoom_vol[time_bins_arg] += zoom_vol

    print(vol_cMpc, zoom_vol)

    vol_tot += vol_cMpc
    # vol_tot_zoom += zoom_vol

    color = None
    if last_simID == intID:
        color = l[0].get_color()
        zoom_style += 1
    else:
        zoom_style = 0

    mcut_str = f"M_{mass_bins[plot_mbin]:.1e}"
    ssfr_cur_str = f"sSFR_{ssfr_bins[plot_ssfr_bin]:.1e}"

    ax[0, 0].text(
        0.05,
        0.95,
        f"M > {mass_bins[plot_mbin]:.1e}",
        transform=ax[0, 0].transAxes,
    )
    ax[0, 0].text(
        0.05,
        0.9,
        f"{ssfr_bins[plot_ssfr_bin]:.1e} < sSFR",
        transform=ax[0, 0].transAxes,
    )

    if zoom_style == 0:
        l = ax[0, 0].plot(time_hagn, mstel_hagn, ls="-", lw=1.0)
        ax[0, 0].plot(time_hagn, mvir_hagn, ls=":", lw=1.0, color=l[0].get_color())
        ax[0, 1].plot(time_hagn, sfr_hagn, ls="-", color=l[0].get_color(), lw=1.0)
        # for mb in range(len(mass_bins) - 1):
        # for ssfrb in range(len(ssfr_bins) - 1):
        ax[1, 0].plot(
            time_hagn,
            qf_hagn[:, plot_mbin, plot_ssfr_bin],
            ls="-",
            color=l[0].get_color(),
            lw=1.0,
        )
        ax[1, 1].plot(
            time_hagn,
            qdens_hagn[:, plot_mbin, plot_ssfr_bin],
            ls="-",
            color=l[0].get_color(),
            lw=1.0,
        )
        # ax[1, 0].plot(time_hagn, qf_hagn, ls="-", color=l[0].get_color(), lw=1.0)
        # ax[1, 1].plot(time_hagn, qdens_hagn, ls="-", color=l[0].get_color(), lw=1.0)

    ax[0, 0].plot(
        time_zoom,
        mstel_zoom,
        # ls=zoom_ls[zoom_style],
        color=l[0].get_color(),
        lw=2.0,
    )

    mvir_pos = mvir_zoom > 0
    ax[0, 0].plot(
        time_zoom[mvir_pos],
        mvir_zoom[mvir_pos],
        # ls=zoom_ls[zoom_style],
        color=l[0].get_color(),
        lw=2.0,
        ls=":",
    )
    ax[0, 1].plot(
        time_zoom,
        sfr_zoom,
        # ls=zoom_ls[zoom_style],
        color=l[0].get_color(),
        lw=2.0,
    )
    # for mb in range(len(mass_bins) - 1):
    # for ssfrb in range(len(ssfr_bins) - 1):
    ax[1, 0].plot(
        time_zoom,
        qf_zoom[:, plot_mbin, plot_ssfr_bin],
        # ls=zoom_ls[zoom_style],
        color=l[0].get_color(),
        lw=2.0,
    )
    ax[1, 1].plot(
        time_zoom,
        qdens_zoom[:, plot_mbin, plot_ssfr_bin],
        # ls=zoom_ls[zoom_style],
        color=l[0].get_color(),
        lw=2.0,
    )
    # ax[1, 0].plot(time_zoom, qf_zoom, ls=zoom_ls[zoom_style], color=l[0].get_color(),, lw=2.0)
    # ax[1, 1].plot(time_zoom, qdens_zoom, ls=zoom_ls[zoom_style], color=l[0].get_color(),, lw=2.0)

    lvlmax = sim.namelist["amr_params"]["levelmax"]

    labels.append(sim.name + f" {lvlmax}")
    lines.append(l)

    last_simID = intID

    with h5py.File(fout_h5, "w") as f:

        f.create_dataset(
            "mstel_hagn", data=mstel_hagn, compression="lzf", dtype=np.float32
        )
        f.create_dataset("sfr_hagn", data=sfr_hagn, compression="lzf", dtype=np.float32)
        f.create_dataset(
            "ssfr_hagn", data=ssfr_hagn, compression="lzf", dtype=np.float32
        )
        f.create_dataset("qf_hagn", data=qf_hagn, compression="lzf", dtype=np.float32)
        f.create_dataset(
            "qdens_hagn", data=qdens_hagn, compression="lzf", dtype=np.float32
        )
        f.create_dataset(
            "time_hagn", data=time_hagn, compression="lzf", dtype=np.float32
        )
        f.create_dataset(
            "galdens_hagn",
            data=galdens_hagn,
            compression="lzf",
            dtype=np.float32,
        )

        f.create_dataset(
            "mstel_zoom", data=mstel_zoom, compression="lzf", dtype=np.float32
        )
        f.create_dataset("sfr_zoom", data=sfr_zoom, compression="lzf", dtype=np.float32)
        f.create_dataset(
            "ssfr_zoom", data=ssfr_zoom, compression="lzf", dtype=np.float32
        )
        f.create_dataset("qf_zoom", data=qf_zoom, compression="lzf", dtype=np.float32)
        f.create_dataset(
            "qdens_zoom", data=qdens_zoom, compression="lzf", dtype=np.float32
        )
        f.create_dataset(
            "time_zoom", data=time_zoom, compression="lzf", dtype=np.float32
        )
        f.create_dataset(
            "galdens_zoom",
            data=galdens_zoom,
            compression="lzf",
            dtype=np.float32,
        )

        f.create_dataset("vol_cMpc", data=vol_cMpc, dtype=np.float32)

if len(sdirs) > 1:
    # total curves

    total_zoom_quenched_frac = (
        tot_zoom_quenched_dens[:, :, plot_ssfr_bin] / tot_zoom_galdens
    )
    total_hagn_quenched_frac = (
        tot_hagn_quenched_dens[:, :, plot_ssfr_bin] / tot_hagn_galdens
    )

    non_zero_hagn = (
        tot_hagn_galdens[:, plot_mbin] > 0
    )  # only plot filled bins where there were galaxies
    non_zero_zoom = tot_zoom_galdens[:, plot_mbin] > 0

    if np.any(non_zero_zoom):
        ax[1, 0].plot(
            time_bins_total[non_zero_zoom],
            total_zoom_quenched_frac[non_zero_zoom, plot_mbin],
            ls="-",
            color="k",
            lw=2.0,
        )
        ax[1, 1].plot(
            time_bins_total[non_zero_zoom],
            tot_zoom_quenched_dens[non_zero_zoom, plot_mbin, plot_ssfr_bin]
            / tot_zoom_vol[non_zero_zoom],
            ls="-",
            color="k",
            lw=2.0,
        )

    if np.any(non_zero_hagn):
        ax[1, 0].plot(
            time_bins_total[non_zero_hagn],
            total_hagn_quenched_frac[non_zero_hagn, plot_mbin],
            ls="-",
            color="k",
            lw=1.0,
        )

        ax[1, 1].plot(
            time_bins_total[non_zero_hagn],
            tot_hagn_quenched_dens[non_zero_hagn, plot_mbin, plot_ssfr_bin] / vol_tot,
            ls="-",
            color="k",
            lw=1.0,
        )

ax[0, 0].set_ylabel(r"$M_{\star}$ [$M_{\odot}$]")
ax[0, 1].set_ylabel(r"SFR [$M_{\odot}$ yr$^{-1}$]")
ax[1, 0].set_ylabel("Quiescent Fraction")
ax[1, 1].set_ylabel("Quiescent Density, $N_{\mathrm{q}}$ [cMpc$^{-3}$]")
ax[1, 0].set_xlabel("Time [Myr]")
ax[1, 1].set_xlabel("Time [Myr]")

ax[0, 0].set_yscale("log")
ax[0, 1].set_yscale("log")
ax[1, 0].set_yscale("log")
ax[1, 1].set_yscale("log")

ax[0, 0].legend(lines, labels, framealpha=0.0, ncol=2)  # , handlelength=3)

for iter_axis in ax.flatten():
    iter_axis.grid()
    iter_axis.tick_params(which="both", direction="in", right=True, top=True)
    iter_axis.set_xlim(200, 3400)

fig_dir = os.path.join("figs", main_star_str)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
f_fig = os.path.join(fig_dir, f"vol_star_stat_{mcut_str:s}_{ssfr_cur_str:s}.png")

print(f"Saving figure to {f_fig}")

fig.savefig(f_fig)


Vtot = 0
tot_mf_hagn = np.zeros((len(mass_fct_bins) - 1, len(zbins_plot)))
tot_mf_zoom = np.zeros((len(mass_fct_bins) - 1, len(zbins_plot)))
tot_pass_mf_hagn = np.zeros((len(mass_fct_bins) - 1, len(ssfr_bins), len(zbins_plot)))

tot_pass_mf_zoom = np.zeros((len(mass_fct_bins) - 1, len(ssfr_bins), len(zbins_plot)))


zbin_colors = plt.cm.viridis(np.linspace(0, 1, len(zbins_plot)))

for isim, sdir in enumerate(sdirs):

    fig_mfs, ax_mfs = plt.subplots(1, 1, figsize=(8, 8), layout="constrained")
    sim = ramses_sim(sdir)
    sim.init_cosmo()
    sim_times = sim.get_snap_times()

    time_bins = sim.cosmo_model.age(zbins_plot).value * 1e3

    zoom_ctr = [
        sim.namelist["refine_params"]["xzoom"],
        sim.namelist["refine_params"]["yzoom"],
        sim.namelist["refine_params"]["zzoom"],
    ]
    if "rzoom" in sim.namelist["refine_params"]:
        check_zoom = (
            lambda coords: np.linalg.norm(coords - zoom_ctr, axis=1)
            < sim.namelist["refine_params"]["rzoom"]
        )

        vol_cMpc = (
            4.0
            / 3
            * np.pi
            * (sim.namelist["refine_params"]["rzoom"] * sim.cosmo.lcMpc) ** 3
        )

    elif "azoom" in sim.namelist["refine_params"]:
        if "zoom_shape" in sim.namelist["refine_params"]:
            ellipse = True
            if sim.namelist["refine_params"]["zoom_shape"] == "rectangle":
                ellipse = False
        else:
            ellipse = True

        a = sim.namelist["refine_params"]["azoom"]
        b = sim.namelist["refine_params"]["bzoom"]
        c = sim.namelist["refine_params"]["czoom"]

        if ellipse:
            check_zoom = (
                lambda coords: (
                    ((coords[:, 0] - zoom_ctr[0]) / a) ** 2
                    + ((coords[:, 1] - zoom_ctr[1]) / b) ** 2
                    + ((coords[:, 2] - zoom_ctr[2]) / c) ** 2
                )
                < 1
            )

            vol_cMpc = 4 / 3 * np.pi * a * b * c * sim.cosmo.lcMpc**3

        else:
            check_zoom = lambda coords: (np.abs(coords[:, 0] - zoom_ctr[0]) < a) * (
                np.abs(coords[:, 1] - zoom_ctr[1])
            ) < b * (np.abs(coords[:, 2] - zoom_ctr[2]) < c)

            vol_cMpc = a * b * c * sim.cosmo.lcMpc**3

    Vtot += vol_cMpc

    cur_mf_zoom = (
        mf_zoom[isim, :, :] / np.diff(np.log10(mass_fct_bins))[:, np.newaxis] / vol_cMpc
    )
    cur_mf_hagn = (
        mf_hagn[isim, :, :] / np.diff(np.log10(mass_fct_bins))[:, np.newaxis] / vol_cMpc
    )

    cur_pass_mf_zoom = (
        mf_zoom_pass[isim, :, :, :]
        / np.diff(np.log10(mass_fct_bins))[:, np.newaxis, np.newaxis]
        / vol_cMpc
    )
    cur_pass_mf_hagn = (
        mf_hagn_pass[isim, :, :, :]
        / np.diff(np.log10(mass_fct_bins))[:, np.newaxis, np.newaxis]
        / vol_cMpc
    )

    tot_mf_hagn += mf_hagn[isim, :, :]
    tot_pass_mf_hagn += mf_hagn_pass[isim, :, :, :]

    tot_mf_zoom += mf_zoom[isim, :, :]
    tot_pass_mf_zoom += mf_zoom_pass[isim, :, :, :]

    for it, ttgt in enumerate(time_bins):

        c = zbin_colors[it]

        ax_mfs.plot(
            mass_fct_bins[:-1] + mass_fct_bin_sizes,
            cur_mf_hagn[:, it],
            ls="-",
            color=c,
            lw=1.0,
        )
        ax_mfs.plot(
            mass_fct_bins[:-1] + mass_fct_bin_sizes,
            cur_mf_zoom[:, it],
            ls="-",
            color=c,
            lw=2.0,
        )
        ax_mfs.plot(
            mass_fct_bins[:-1] + mass_fct_bin_sizes,
            cur_pass_mf_hagn[:, plot_ssfr_bin, it],
            ls="--",
            color=c,
            lw=1.0,
        )
        ax_mfs.plot(
            mass_fct_bins[:-1] + mass_fct_bin_sizes,
            cur_pass_mf_zoom[:, plot_ssfr_bin, it],
            ls="--",
            color=c,
            lw=2.0,
        )

    ax_mfs.set_yscale("log")
    ax_mfs.set_xscale("log")

    ax_mfs.set_xlabel(r"$M_{\star}$ [$M_{\odot}$]")
    ax_mfs.set_ylabel(r"$\Phi$ [$cMpc^{-3} dex^{-1}$]")

    ax_mfs.grid()

    plt.colorbar(
        plt.cm.ScalarMappable(
            norm=colors.Normalize(vmin=zbins_plot.min(), vmax=zbins_plot.max()),
            cmap=plt.cm.viridis,
        ),
        ax=ax_mfs,
        label="Redshift",
    )

    fig_mfs.savefig(f"figs/pass_mf_{sim.name:s}_{ssfr_cur_str:s}.png")


tot_mf_hagn /= np.diff(np.log10(mass_fct_bins))[:, np.newaxis] * Vtot
tot_mf_zoom /= np.diff(np.log10(mass_fct_bins))[:, np.newaxis] * Vtot

tot_pass_mf_hagn /= np.diff(np.log10(mass_fct_bins))[:, np.newaxis, np.newaxis] * Vtot
tot_pass_mf_zoom /= np.diff(np.log10(mass_fct_bins))[:, np.newaxis, np.newaxis] * Vtot


nplots = len(time_bins)

nrows = 2
ncols = int(np.ceil(nplots / nrows))

if nrows * ncols < nplots:
    ncols += 1

if nrows * nplots == nplots:
    ncols += 1

overstep = nrows * ncols - nplots

fig_tot_mfs, ax_tot_mfs = plt.subplots(
    nrows,
    ncols,
    figsize=(4 * ncols + 2, 4 * nrows),
    layout="constrained",
    sharex=True,
    sharey=True,
)

ax_tot_mfs = np.ravel(ax_tot_mfs)

unity_dens = 0.1 / (np.mean(np.diff(np.log10(mass_fct_bins))) * Vtot)

ax_tot_mfs[0].set_ylim(unity_dens, 1)


for it, ttgt in enumerate(time_bins):

    c = zbin_colors[it]
    plot_z = zbins_plot[it]

    plot_aexp = 1.0 / (plot_z + 1.0)

    ax_tot_mfs[it].plot(
        mass_fct_bins[:-1] + mass_fct_bin_sizes,
        tot_mf_hagn[:, it],
        ls="-",
        color=c,
        lw=1.0,
    )
    ax_tot_mfs[it].plot(
        mass_fct_bins[:-1] + mass_fct_bin_sizes,
        tot_mf_zoom[:, it],
        ls="-",
        color=c,
        lw=2.0,
    )
    ax_tot_mfs[it].plot(
        mass_fct_bins[:-1] + mass_fct_bin_sizes,
        tot_pass_mf_hagn[:, plot_ssfr_bin, it],
        ls="--",
        color=c,
        lw=1.0,
    )
    ax_tot_mfs[it].plot(
        mass_fct_bins[:-1] + mass_fct_bin_sizes,
        tot_pass_mf_zoom[:, plot_ssfr_bin, it],
        ls="--",
        color=c,
        lw=2.0,
    )

    mbins_cstr = mass_fct_bins[:-1] + mass_fct_bin_sizes

    hmdc_sf = hamadch_sf(np.log10(mbins_cstr), zbins_plot[it])  # * plot_aexp**3
    hmdc_qu = hamadch_quiescent(np.log10(mbins_cstr), zbins_plot[it])  # * plot_aexp**3
    ax_tot_mfs[it].plot(
        mbins_cstr,
        hmdc_sf + hmdc_qu,
        ls="-",
        color="k",
        lw=1.0,
        marker="o",
        markerfacecolor="none",
        markeredgecolor="k",
        markevery=1,
    )
    ax_tot_mfs[it].plot(
        mbins_cstr,
        hmdc_qu,
        ls="--",
        color="k",
        lw=1.0,
        marker="o",
        markerfacecolor="none",
        markeredgecolor="k",
        markevery=1,
    )

    mcld_sf = mcleod_sf(np.log10(mbins_cstr), zbins_plot[it])  # * plot_aexp**3
    mcld_qu = mcleod_quiescent(np.log10(mbins_cstr), zbins_plot[it])  # * plot_aexp**3
    ax_tot_mfs[it].plot(
        mbins_cstr,
        mcld_sf + mcld_qu,
        ls="-",
        color="k",
        lw=1.0,
        marker="v",
        markerfacecolor="none",
        markeredgecolor="k",
        markevery=1,
    )
    ax_tot_mfs[it].plot(
        mbins_cstr,
        mcld_qu,
        ls="--",
        color="k",
        lw=1.0,
        marker="v",
        markerfacecolor="none",
        markeredgecolor="k",
        markevery=1,
    )

    pl_shuntov_all = shuntov24_all(zbins_plot[it])
    pl_shuntov_qu = shuntov24_qu(zbins_plot[it])

    ax_tot_mfs[it].plot(
        10 ** pl_shuntov_all["logM"],
        pl_shuntov_all["Fi"],
        ls="-",
        color="k",
        lw=1.0,
        marker="D",
        markerfacecolor="none",
        markeredgecolor="k",
    )
    ax_tot_mfs[it].plot(
        10 ** pl_shuntov_qu["logM"],
        pl_shuntov_qu["Fi"],
        ls="--",
        color="k",
        lw=1.0,
        marker="D",
        markerfacecolor="none",
        markeredgecolor="k",
    )

    ax_tot_mfs[it].set_yscale("log")
    ax_tot_mfs[it].set_xscale("log")

    if it % ncols == 0:
        ax_tot_mfs[it].set_ylabel(r"$\Phi$ [$cMpc^{-3} dex^{-1}$]")
    if it >= nplots - ncols:
        ax_tot_mfs[it].set_xlabel(r"$M_{\star}$ [$M_{\odot}$]")

    ax_tot_mfs[it].grid()

plt.colorbar(
    plt.cm.ScalarMappable(
        norm=colors.Normalize(vmin=zbins_plot.min(), vmax=zbins_plot.max()),
        cmap=plt.cm.viridis,
    ),
    ax=ax_tot_mfs,
    label="Redshift",
)

for i in range(nplots, overstep + nplots):
    ax_tot_mfs[i].axis("off")

lines = [
    Line2D([], [], ls="-", color="k", lw=2.0),
    Line2D([], [], ls="--", color="k", lw=2.0),
    Line2D([], [], ls="-", color="k", lw=1.0),
    Line2D([], [], ls="--", color="k", lw=1.0),
    Line2D(
        [],
        [],
        ls="-",
        color="k",
        lw=1.0,
        marker="o",
        markerfacecolor="none",
        markeredgecolor="k",
    ),
    Line2D(
        [],
        [],
        ls="-",
        color="k",
        lw=1.0,
        marker="D",
        markerfacecolor="none",
        markeredgecolor="k",
    ),
    Line2D(
        [],
        [],
        ls="-",
        color="k",
        lw=1.0,
        marker="v",
        markerfacecolor="none",
        markeredgecolor="k",
    ),
]
labels = [
    "All zoom galaxies",
    "Passive zoom galaxies",
    "All HAGN galaxies",
    "Passive HAGN galaxies",
    "Hamadouche+24",
    "Shuntov+24",
    "Mcleod+21",
]


ax_tot_mfs[-1].legend(lines, labels, framealpha=0.0, prop={"size": 12})


fig_tot_mfs.suptitle(f"Mass Functions sSFR<{ssfr_bins[plot_ssfr_bin]:.1e} $yr^{-1}$")

fig_tot_mfs.savefig(f"figs/pass_mf_tot_{ssfr_cur_str:s}.png")


fig_qf_tot, ax_qf_tot = plt.subplots(1, 1, figsize=(8, 8), layout="constrained")

ax_qf_tot.tick_params(which="both", direction="in", right=True, top=True)

for it, ttgt in enumerate(time_bins):

    c = zbin_colors[it]

    ax_qf_tot.plot(
        mass_fct_bins[:-1] + mass_fct_bin_sizes,
        tot_pass_mf_hagn[:, plot_ssfr_bin, it] / tot_mf_hagn[:, it],
        ls="--",
        color=c,
        lw=1.0,
    )
    ax_qf_tot.plot(
        mass_fct_bins[:-1] + mass_fct_bin_sizes,
        tot_pass_mf_zoom[:, plot_ssfr_bin, it] / tot_mf_zoom[:, it],
        ls="--",
        color=c,
        lw=2.0,
    )

ax_qf_tot.set_yscale("log")
ax_qf_tot.set_xscale("log")

ax_qf_tot.set_xlabel(r"$M_{\star}$ [$M_{\odot}$]")
ax_qf_tot.set_ylabel(r"$f_{\mathrm{q}}$")
ax_qf_tot.grid()

plt.colorbar(
    plt.cm.ScalarMappable(
        norm=colors.Normalize(vmin=zbins_plot.min(), vmax=zbins_plot.max()),
        cmap=plt.cm.viridis,
    ),
    ax=ax_qf_tot,
    label="Redshift",
)

fig_qf_tot.savefig(f"figs/pass_qf_tot_{ssfr_cur_str:s}.png")
