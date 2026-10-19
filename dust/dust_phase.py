# from f90nml import read
from shutil import which
from f90_tools.star_reader import read_part_ball_NCdust
from zoom_analysis.stars import sfhs
from zoom_analysis.zoom_helpers import decentre_coordinates, find_starting_position
from zoom_analysis.halo_maker.read_treebricks import (
    read_zoom_stars,
)
from zoom_analysis.trees.tree_reader import read_tree_file_rev as read_tree_fev_sim
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import (
    # find_zoom_tgt_halo,
    get_central_gal_for_hid,
    get_gal_props_snap,
    get_halo_props_snap,
)
from zoom_analysis.dust.gas_reader import gas_pos_rad, code_to_cgs
from scipy.stats import binned_statistic_2d

# from zoom_analysis.halo_maker.assoc_fcts import find_star_ctr_period

from compress_zoom.read_compressd import read_compressed_target, check_for_compressd

# from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D

# import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# from plot_stuff import setup_plots

from scipy.spatial import cKDTree

import os
import numpy as np

from hagn.utils import get_hagn_sim

# from hagn.tree_reader import read_tree_rev, interpolate_tree_position

# planck cosmo
# from astropy.cosmology import Planck18 as cosmo
# from astropy.cosmology import z_at_value
# from astropy import units as u
import h5py


# setup_plots()

if not os.path.exists("./figs"):
    os.makedirs("./figs", exist_ok=True)

# tgt_zed = 3.5149170041138174
# tgt_zed = 3.488
# tgt_zed = 3.439
tgt_zed = 3.36146335787232156548
overwrite = False
frac = 0.1  # fraction of rvir to use
dens_thresh = 0.1  # Hpcc for the dust and gas properties
mlim = 1e6  # Msun
fpure_lim = 1 - 1e-4

dens_bins = np.logspace(-5, 4, 23)
temp_bins = np.logspace(0.5, 9, 23)

tgt_zedstr = f"{tgt_zed:.2f}".replace(".", "p")


# sim_ids = [
#     "id2757_995pcnt_mstel/",
#     "id19782_500pcnt_mstel/",pip install setuptools
#     "id13600_005pcnt_mstel/",
# ]
# sdir = "/home/jlewis/sims/proposal_zooms/ExtractZoom/mh1e12/"

HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
delta_t = 25  # Myr

# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_evenlesscoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892"
sdirs = [
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id21892_leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id180130_leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id242704",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id242756",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id74099",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi/id180130_meanBondi_higher_nmax",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi/id180130_meanBondi_Sconstant",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
    # "/data102/jlewis/sims/lvlmax_21/mh1e12/id180130_256",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictestSF_lowSNe",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_XtremeLowSFE_stgNHboost_strictSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288_novrel_lowerSFE_stgNHboost_strictSF",
    "/data103/jlewis/sims/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF",
    "/data102/jlewis/sims/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF_high_sconstant",
    "/data102/jlewis/sims/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF_Vhigh_sconstant",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF_VVhigh_sconstant",
]
# sdir =
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892"
# sdir =
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_evenlesscoarse"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756"
# sdir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479"
# print(explr_dirs)


msun_to_g = 1.989e33  # g
pmass = 1.67e-24  # g
MgoverSil = 0.141
FeoverSil = 0.324
SioverSil = 0.163
OoverSil = 0.372


markers = ["o", "s", "D", "P", "X", "v", "^", "<", ">", "p", "*", "h", "H", "+", "x"]
colors = ["b", "g", "r", "c", "m", "y", "k"]


yax2 = None

last_hagn_id = -1
isim = 0

zoom_ls = ["-", "--", ":", "-."]

lines = []
labels = []

mstel_max = -np.inf
mstel_min = np.inf

hagn_sim = get_hagn_sim()

xmax = -np.inf
xmin = np.inf

for isim, sdir in enumerate(sdirs):

    fig_dtm, ax_dtm = plt.subplots(
        2, 2, figsize=(18, 18), sharey=True, layout="constrained"
    )
    plt.subplots_adjust(wspace=0.0)
    plt.suptitle(f"z={tgt_zedstr:s}, {frac:.1f}Rvir")

    sim = ramses_sim(sdir, nml="cosmo.nml")

    print("Reading sim:", sim.name)

    closest_snap = sim.get_closest_snap(zed=tgt_zed)
    closest_aexp = sim.get_snap_exps(closest_snap)[0]
    closest_zed = 1 / closest_aexp - 1.0

    print("Closest snap:", closest_snap, "at z=", closest_zed)

    if abs(tgt_zed - closest_zed) > 0.5:
        print("No close snap")
        continue

    zstr = f"{closest_zed:.2f}".replace(".", "p")

    name = sdir.split("/")[-1]

    print("Calculating dust properties")
    intID = int(name[2:].split("_")[0])

    gal_dict = get_gal_props_snap(sim.path, closest_snap)

    gal_args = (gal_dict["host purity"] > fpure_lim) * (gal_dict["mass"] > mlim)

    for k in gal_dict.keys():
        if len(gal_dict[k].shape) == 2:
            gal_dict[k] = gal_dict[k][:, gal_args]
        else:
            gal_dict[k] = gal_dict[k][gal_args]

    ngals = gal_args.sum()

    gal_arg = np.argmax(gal_dict["mass"])

    host_halo = gal_dict["host hid"][gal_arg]

    try:
        hal_dict = get_halo_props_snap(sim.path, closest_snap, host_halo)[
            f"halo_{host_halo:07d}"
        ]
    except KeyError:
        continue

    tgt_pos = hal_dict["pos"]
    tgt_r = hal_dict["rvir"] * frac

    load_amr = True

    if not check_for_compressd(sim, closest_snap):
        # if (
        #     load_amr
        #     and closest_snap in sim.get_snaps(full_snaps=True, mini_snaps=False)[1]
        # ):
        print("No compressed file")
        load_amr = True

        code_cells = gas_pos_rad(
            # sim, snap, [1, 6, 16, 17, 18, 19], ctr_stars, extent_stars
            sim,
            closest_snap,
            [1, 5, 6, 8, 9, 10, 11, 13, 16, 17, 18, 19],
            tgt_pos,
            tgt_r,
        )

        cells = code_to_cgs(sim, closest_snap, code_cells)

        print(" Read gas")

        stars = read_part_ball_NCdust(
            sim,
            closest_snap,
            tgt_pos,
            tgt_r,
            tgt_fields=["mass", "birth_time", "metallicity"],
            fam=2,
        )
        print(" Read stars")
    else:
        # print("reading compressed file")
        load_amr = False

        datas = read_compressed_target(sim, closest_snap, hid=host_halo)

        # stars = datas["stars"]
        # # keep only stars within tgt_r of tgt_pos
        # st_tree = cKDTree(stars["pos"], boxsize=1 + 1e-6)
        # st_pos_args = st_tree.query_ball_point(tgt_pos, tgt_r)

        # for field in stars:
        #     stars[field] = stars[field][st_pos_args]

        cells = datas["gas"]

        # print(cells)
        # keep only cells within tgt_r of tgt_pos
        cell_tree = cKDTree(
            np.transpose([cells["x"], cells["y"], cells["z"]]), boxsize=1 + 1e-6
        )
        cell_pos_args = cell_tree.query_ball_point(tgt_pos, tgt_r)

        for field in cells:
            cells[field] = np.float64(cells[field][cell_pos_args])

        if len(cell_pos_args) > 1:
            if len(cells["density"]) == 0:
                print("No cells")
                continue

    # ages = stars["age"]
    # Zs = stars["metallicity"]

    # masses = sfhs.correct_mass(hagn_sim, ages, stars["mass"], Zs)

    # star_pos, ctr_stars, extent_stars = find_star_ctr_period(stars["pos"])

    # unit_d = sim.unit_d(closest_aexp)

    dens_hpcc = np.float64(cells["density"]) / 1.67e-24 * 0.76
    # print(dens_hpcc.min(), dens_hpcc.max())
    # dens_gas = dens_hpcc > dens_thresh

    gas_tot_dens = (
        cells["dust_bin01"]
        + cells["dust_bin02"]
        + cells["dust_bin03"]
        + cells["dust_bin04"]
    )

    gas_DTG = gas_tot_dens / (cells["density"] / sim.unit_d(closest_aexp))

    gas_DTM = gas_tot_dens / (gas_tot_dens + cells["metallicity"])
    gas_DTMC = (cells["dust_bin01"] + cells["dust_bin02"]) / (
        gas_tot_dens + cells["metallicity"]
    )
    gas_DTMSi = (cells["dust_bin03"] + cells["dust_bin04"]) / (
        gas_tot_dens + cells["metallicity"]
    )
    gas_fsmall = (cells["dust_bin01"] + cells["dust_bin03"]) / gas_tot_dens

    fC = (cells["dust_bin01"] + cells["dust_bin02"]) / gas_tot_dens
    fSi = (cells["dust_bin03"] + cells["dust_bin04"]) / gas_tot_dens

    ax = np.ravel(ax_dtm)

    counts = binned_statistic_2d(
        dens_hpcc,
        cells["temperature"],
        dens_hpcc,
        "count",
        bins=[dens_bins, temp_bins],
    )[0]

    Xgrid, Ygrid = np.meshgrid(np.log10(dens_bins[:-1]), np.log10(temp_bins[:-1]))

    for iplot, (plot_data, label) in enumerate(
        zip(
            # [gas_DTM, gas_DTG, gas_DTMC, gas_DTMSi], ["DTM", "DTG", "DTM, C", "DTM, SI"]
            # [gas_DTM, gas_DTG, fC, fSi],
            # [gas_DTM, gas_DTG, fC, fSi],
            [gas_DTM, gas_DTG, gas_fsmall, fC],
            ["DTM", "DTG", "fsmall", "fC"],
        )
    ):

        # weights = dens_hpcc
        weights = dens_hpcc * (1.0 / 2 ** cells["ilevel"]) ** 3.0

        weighted_sum = binned_statistic_2d(
            dens_hpcc,
            cells["temperature"],
            plot_data * weights,
            "sum",
            bins=[dens_bins, temp_bins],
        )[0]
        sum_weights = binned_statistic_2d(
            dens_hpcc,
            cells["temperature"],
            weights,
            "sum",
            bins=[dens_bins, temp_bins],
        )[0]

        img = weighted_sum / sum_weights

        # if label == "DTG":
        #     vmin, vmax = 0.002, 0.010
        # else:
        #     vmin, vmax = None, None

        imgpl = ax[iplot].imshow(
            img.T,
            origin="lower",
            extent=np.log10([dens_bins[0], dens_bins[-1], temp_bins[0], temp_bins[-1]]),
            # vmin=vmin,
            # vmax=vmax,
        )

        cbar = plt.colorbar(imgpl, ax=ax[iplot], fraction=0.047, pad=0.04)
        ax[iplot].set_title(label)

        imgpl = ax[iplot].contour(
            Xgrid,
            Ygrid,
            counts,
            levels=[100, 1000, 10000],
            linestyles=["solid", "dashed", "dotted"],
            colors="red",
        )

        ax[iplot].set_xlabel(r"$\rho$, Hpcc")
        ax[iplot].set_ylabel(r"T/$\mu$, K")

    fig_dtm.savefig(os.path.join("figs/", f"dust_phase_{sim.name}.png"))
