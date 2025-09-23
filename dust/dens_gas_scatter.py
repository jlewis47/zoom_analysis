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
# tgt_zed = 3.36146335787232156548
tgt_zed = 3.31284701427126730076

overwrite = False
frac = 0.1  # fraction of rvir to use
dens_thresh = 0.1  # Hpcc for the dust and gas properties
mlim = 1e6  # Msun
fpure_lim = 1 - 1e-4

tgt_zedstr = f"{tgt_zed:.2f}".replace(".", "p")

fig_dtm, ax_dtm = plt.subplots(1, 3, figsize=(18, 6), sharey=True, layout="constrained")
plt.subplots_adjust(wspace=0.0)
plt.suptitle(
    f"z={tgt_zedstr:s}, {frac:.1f}Rvir, >{dens_thresh:.1f}Hpcc, Mgal>{mlim:.1e}Msun"
)
fig_dtmCs, ax_dtmCs = plt.subplots(
    1, 3, figsize=(18, 6), sharey=True, layout="constrained"
)
plt.subplots_adjust(wspace=0.0)
plt.suptitle(
    f"z={tgt_zedstr:s}, {frac:.1f}Rvir, >{dens_thresh:.1f}Hpcc, Mgal>{mlim:.1e}Msun"
)
fig_dtmCl, ax_dtmCl = plt.subplots(
    1, 3, figsize=(18, 6), sharey=True, layout="constrained"
)
plt.subplots_adjust(wspace=0.0)
plt.suptitle(
    f"z={tgt_zedstr:s}, {frac:.1f}Rvir, >{dens_thresh:.1f}Hpcc, Mgal>{mlim:.1e}Msun"
)
fig_dtmSis, ax_dtmSis = plt.subplots(
    1, 3, figsize=(18, 6), sharey=True, layout="constrained"
)
plt.subplots_adjust(wspace=0.0)
plt.suptitle(
    f"z={tgt_zedstr:s}, {frac:.1f}Rvir, >{dens_thresh:.1f}Hpcc, Mgal>{mlim:.1e}Msun"
)
fig_dtmSil, ax_dtmSil = plt.subplots(
    1, 3, figsize=(18, 6), sharey=True, layout="constrained"
)
plt.subplots_adjust(wspace=0.0)
plt.suptitle(
    f"z={tgt_zedstr:s}, {frac:.1f}Rvir, >{dens_thresh:.1f}Hpcc, Mgal>{mlim:.1e}Msun"
)

fig_depl, ax_depl = plt.subplots(
    3, 2, figsize=(9, 13), sharey=True, layout="constrained", sharex=True
)
plt.subplots_adjust(wspace=0.0)
plt.suptitle(
    f"z={tgt_zedstr:s}, {frac:.1f}Rvir, >{dens_thresh:.1f}Hpcc, Mgal>{mlim:.1e}Msun"
)


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
    # "/data103/jlewis/sims/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF",
    # "/data102/jlewis/sims/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF_high_sconstant",
    # "/data102/jlewis/sims/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF_Vhigh_sconstant",
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
    dust_file = os.path.join(sdir, f"dens_dust_data_{closest_snap:d}.h5")

    if os.path.exists(dust_file) and not overwrite:
        print("Reading from file")
        with h5py.File(dust_file, "r") as f:
            mstel_zoom = f["mstel_zoom"][:]
            sfr10_zoom = f["sfr10_zoom"][:]
            sfr100_zoom = f["sfr100_zoom"][:]

            volumes = f["volumes"][:]
            mZ_zoom = f["mZ_zoom"][:]
            mG_zoom = f["mG_zoom"][:]

            mdCs = f["mCs_zoom"][:]
            mdCl = f["mCl_zoom"][:]
            mdSs = f["mSs_zoom"][:]
            mdSl = f["mSl_zoom"][:]

            fdMg_zoom = f["fdMg_zoom"][:]
            fdFe_zoom = f["fdFe_zoom"][:]
            fdO_zoom = f["fdO_zoom"][:]
            fdSi_zoom = f["fdSi_zoom"][:]
            fdC_zoom = f["fdC_zoom"][:]

    else:
        print("Calculating dust properties")
        intID = int(name[2:].split("_")[0])

        gal_dict = get_gal_props_snap(sim.path, closest_snap)

        gal_args = (gal_dict["host purity"] > fpure_lim) * (gal_dict["mass"] > mlim)

        ngals = gal_args.sum()

        mstel_zoom = np.zeros(ngals, dtype=np.float32)
        sfr10_zoom = np.zeros(ngals, dtype=np.float32)
        sfr100_zoom = np.zeros(ngals, dtype=np.float32)

        volumes = np.zeros(ngals, dtype=np.float64)
        mG_zoom = np.zeros(ngals, dtype=np.float32)
        mZ_zoom = np.zeros(ngals, dtype=np.float32)

        mdCs = np.zeros(ngals, dtype=np.float32)
        mdCl = np.zeros(ngals, dtype=np.float32)
        mdSs = np.zeros(ngals, dtype=np.float32)
        mdSl = np.zeros(ngals, dtype=np.float32)

        fdMg_zoom = np.zeros(ngals, dtype=np.float32)
        fdFe_zoom = np.zeros(ngals, dtype=np.float32)
        fdO_zoom = np.zeros(ngals, dtype=np.float32)
        fdSi_zoom = np.zeros(ngals, dtype=np.float32)
        fdC_zoom = np.zeros(ngals, dtype=np.float32)

        for igal, gal_arg in enumerate(np.where(gal_args)[0][:]):

            print(f"Galaxy {igal+1}/{ngals}")

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
                    [1, 6, 8, 9, 10, 11, 13, 16, 17, 18, 19],
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

                stars = datas["stars"]
                # keep only stars within tgt_r of tgt_pos
                st_tree = cKDTree(stars["pos"], boxsize=1 + 1e-6)
                st_pos_args = st_tree.query_ball_point(tgt_pos, tgt_r)

                for field in stars:
                    stars[field] = stars[field][st_pos_args]

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

            ages = stars["age"]
            Zs = stars["metallicity"]

            masses = sfhs.correct_mass(hagn_sim, ages, stars["mass"], Zs)

            # star_pos, ctr_stars, extent_stars = find_star_ctr_period(stars["pos"])

            # unit_d = sim.unit_d(closest_aexp)

            mstel_zoom[igal] = masses.sum()
            sfr10_zoom[igal] = masses[ages < 10].sum() / 10
            sfr100_zoom[igal] = masses[ages < 100].sum() / 100

            dens_hpcc = np.float64(cells["density"]) / 1.67e-24 * 0.76
            # print(dens_hpcc.min(), dens_hpcc.max())
            dens_gas = dens_hpcc > dens_thresh

            if dens_gas.sum() == 0:
                print("No dense gas")
                continue

            for field in cells:
                cells[field] = cells[field][dens_gas]

            l_hagn_cm_comov = sim.cosmo.lcMpc * closest_aexp * 1e6 * ramses_pc
            vols = (2 ** -cells["ilevel"] * l_hagn_cm_comov) ** 3
            volumes[igal] = np.sum(vols)

            gas_mass = cells["density"] * vols / msun_to_g

            mG_zoom[igal] = np.sum(gas_mass)

            mdCs[igal] = np.sum(cells["dust_bin01"] * gas_mass)

            mdCl[igal] = np.sum(cells["dust_bin02"] * gas_mass)

            mdSs[igal] = np.sum(cells["dust_bin03"] * gas_mass) / SioverSil

            mdSl[igal] = np.sum(cells["dust_bin04"] * gas_mass) / SioverSil

            mZ_zoom[igal] = np.sum(cells["metallicity"] * gas_mass)

            mO = np.sum(gas_mass * cells["chem_O"])
            mFe = np.sum(gas_mass * cells["chem_Fe"])
            mMg = np.sum(gas_mass * cells["chem_Mg"])
            mC = np.sum(gas_mass * cells["chem_C"])
            mSi = np.sum(gas_mass * cells["chem_Si"])

            mdC = mdCs + mdCl
            mdS = mdSs + mdSl
            md = mdC + mdS

            mdMg = mdS * MgoverSil
            mdFe = mdS * FeoverSil
            mdSi = mdS * SioverSil
            mdO = mdS * OoverSil

            fdMg_zoom[igal] = 1 - mdMg[igal] / mMg
            fdFe_zoom[igal] = 1 - mdFe[igal] / mFe
            fdO_zoom[igal] = 1 - mdO[igal] / mO
            fdSi_zoom[igal] = 1 - mdSi[igal] / mSi
            fdC_zoom[igal] = 1 - mdC[igal] / mC

        with h5py.File(dust_file, "w") as f:
            f.create_dataset("mstel_zoom", data=mstel_zoom)
            f.create_dataset("sfr10_zoom", data=sfr10_zoom)
            f.create_dataset("sfr100_zoom", data=sfr100_zoom)
            f.create_dataset("mCs_zoom", data=mdCs)
            f.create_dataset("mCl_zoom", data=mdCl)
            f.create_dataset("mSs_zoom", data=mdSs)
            f.create_dataset("mSl_zoom", data=mdSl)
            f.create_dataset("fdMg_zoom", data=fdMg_zoom)
            f.create_dataset("fdFe_zoom", data=fdFe_zoom)
            f.create_dataset("fdO_zoom", data=fdO_zoom)
            f.create_dataset("fdSi_zoom", data=fdSi_zoom)
            f.create_dataset("fdC_zoom", data=fdC_zoom)
            f.create_dataset("mZ_zoom", data=mZ_zoom)
            f.create_dataset("mG_zoom", data=mG_zoom)
            f.create_dataset("volumes", data=volumes)

    # lines.append(Line2D([0, 1], [0, 1], ls="-", c=l.get_color()))
    lines.append(Line2D([], [], marker=markers[isim], c=colors[isim], ls=""))
    labels.append(name)

    # dtm figure

    tot_md = mdCs + mdCl + mdSs + mdSl
    dtms = [md / (mZ_zoom + md) for md in [mdCs, mdCl, mdSs, mdSl]]

    tot_dtm = tot_md / (mZ_zoom + tot_md)

    ax_dtm[0].scatter(
        mstel_zoom,
        tot_dtm,
        marker=markers[isim],
        facecolors=colors[isim],
        edgecolors=colors[isim],
    )
    ax_dtm[0].set_xlabel("Stellar mass, $M_{\odot}$")
    ax_dtm[0].set_ylabel("Dust-to-metals ratio")

    ax_dtm[1].scatter(
        sfr10_zoom,
        tot_dtm,
        marker=markers[isim],
        facecolors=colors[isim],
        edgecolors=colors[isim],
    )
    ax_dtm[1].set_xlabel("SFR10, $M_{\odot}/Myr$")
    # ax_dtm[1].set_ylabel("Dust-to-metals ratio")

    ax_dtm[2].scatter(
        mZ_zoom / mG_zoom,
        tot_dtm,
        marker=markers[isim],
        facecolors=colors[isim],
        edgecolors=colors[isim],
    )
    ax_dtm[2].set_xlabel("Metallicity")
    # ax_dtm[2].set_ylabel("Dust-to-metals ratio")

    ax_dtmCs[0].scatter(
        mstel_zoom,
        dtms[0],
        marker=markers[isim],
        facecolors=colors[isim],
        edgecolors=colors[isim],
    )
    ax_dtmCs[0].set_xlabel("Stellar mass, $M_{\odot}$")
    ax_dtmCs[0].set_ylabel("Dust-to-metals ratio, small carbonaceous")

    ax_dtmCs[1].scatter(
        sfr10_zoom,
        dtms[0],
        marker=markers[isim],
        facecolors=colors[isim],
        edgecolors=colors[isim],
    )
    ax_dtmCs[1].set_xlabel("SFR10, $M_{\odot}/Myr$")
    # ax_dtmCs[1].set_ylabel("Dust-to-metals ratio")

    ax_dtmCs[2].scatter(
        mZ_zoom / mG_zoom,
        dtms[0],
        marker=markers[isim],
        facecolors=colors[isim],
        edgecolors=colors[isim],
    )
    ax_dtmCs[2].set_xlabel("Metallicity")
    # ax_dtm[2].set_ylabel("Dust-to-metals ratio")

    ax_dtmCl[0].scatter(
        mstel_zoom,
        dtms[1],
        marker=markers[isim],
        facecolors=colors[isim],
        edgecolors=colors[isim],
    )
    ax_dtmCl[0].set_xlabel("Stellar mass, $M_{\odot}$")
    ax_dtmCl[0].set_ylabel("Dust-to-metals ratio, large carbonaceous")

    ax_dtmCl[1].scatter(
        sfr10_zoom,
        dtms[1],
        marker=markers[isim],
        facecolors=colors[isim],
        edgecolors=colors[isim],
    )
    ax_dtmCl[1].set_xlabel("SFR10, $M_{\odot}/Myr$")
    # ax_dtmCl[1].set_ylabel("Dust-to-metals ratio")

    ax_dtmCl[2].scatter(
        mZ_zoom / mG_zoom,
        dtms[1],
        marker=markers[isim],
        facecolors=colors[isim],
        edgecolors=colors[isim],
    )
    ax_dtmCl[2].set_xlabel("Metallicity")
    # ax_dtm[2].set_ylabel("Dust-to-metals ratio")

    ax_dtmSis[0].scatter(
        mstel_zoom,
        dtms[2],
        marker=markers[isim],
        facecolors=colors[isim],
        edgecolors=colors[isim],
    )
    ax_dtmSis[0].set_xlabel("Stellar mass, $M_{\odot}$")
    ax_dtmSis[0].set_ylabel("Dust-to-metals ratio, small silicates")

    ax_dtmSis[1].scatter(
        sfr10_zoom,
        dtms[2],
        marker=markers[isim],
        facecolors=colors[isim],
        edgecolors=colors[isim],
    )
    ax_dtmSis[1].set_xlabel("SFR10, $M_{\odot}/Myr$")
    # ax_dtmSis[1].set_ylabel("Dust-to-metals ratio")

    ax_dtmSis[2].scatter(
        mZ_zoom / mG_zoom,
        dtms[2],
        marker=markers[isim],
        facecolors=colors[isim],
        edgecolors=colors[isim],
    )
    ax_dtmSis[2].set_xlabel("Metallicity")
    # ax_dtm[2].set_ylabel("Dust-to-metals ratio")

    ax_dtmSil[0].scatter(
        mstel_zoom,
        dtms[3],
        marker=markers[isim],
        facecolors=colors[isim],
        edgecolors=colors[isim],
    )
    ax_dtmSil[0].set_xlabel("Stellar mass, $M_{\odot}$")
    ax_dtmSil[0].set_ylabel("Dust-to-metals ratio, large silicates")

    ax_dtmSil[1].scatter(
        sfr10_zoom,
        dtms[3],
        marker=markers[isim],
        facecolors=colors[isim],
        edgecolors=colors[isim],
    )
    ax_dtmSil[1].set_xlabel("SFR10, $M_{\odot}/Myr$")
    # ax_dtmSil[1].set_ylabel("Dust-to-metals ratio")

    ax_dtmSil[2].scatter(
        mZ_zoom / mG_zoom,
        dtms[3],
        marker=markers[isim],
        facecolors=colors[isim],
        edgecolors=colors[isim],
    )
    ax_dtmSil[2].set_xlabel("Metallicity")
    # ax_dtmSil[2].set_ylabel("Dust-to-metals ratio")

    for a_dtm in [ax_dtm, ax_dtmCs, ax_dtmCl, ax_dtmSis, ax_dtmSil]:
        for a in a_dtm:
            a.grid()
            a.set_xscale("log")
            # a.set_yscale("log")
            a.tick_params(which="both", direction="in", right=True, top=True)

    dens_hpcc = (np.float64(mG_zoom) * msun_to_g / pmass) / (volumes)

    ax_depl[0, 0].scatter(dens_hpcc, fdMg_zoom, marker=markers[isim], c=colors[isim])
    ax_depl[0, 0].set_xlabel("Gas density, $H_{pcc}$")
    ax_depl[0, 0].set_ylabel("Depletion factor, Mg")

    ax_depl[0, 1].scatter(dens_hpcc, fdFe_zoom, marker=markers[isim], c=colors[isim])
    ax_depl[0, 1].set_xlabel("Gas density, $H_{pcc}$")
    ax_depl[0, 1].set_ylabel("Depletion factor, Fe")

    ax_depl[1, 0].scatter(dens_hpcc, fdO_zoom, marker=markers[isim], c=colors[isim])
    ax_depl[1, 0].set_xlabel("Gas density, $H_{pcc}$")
    ax_depl[1, 0].set_ylabel("Depletion factor, O")

    ax_depl[1, 1].scatter(dens_hpcc, fdSi_zoom, marker=markers[isim], c=colors[isim])
    ax_depl[1, 1].set_xlabel("Gas density, $H_{pcc}$")
    ax_depl[1, 1].set_ylabel("Depletion factor, Si")

    ax_depl[2, 0].scatter(dens_hpcc, fdC_zoom, marker=markers[isim], c=colors[isim])
    ax_depl[2, 0].set_xlabel("Gas density, $H_{pcc}$")
    ax_depl[2, 0].set_ylabel("Depletion factor, C")

    for a in ax_depl.flatten():
        a.grid()
        a.set_xscale("log")
        a.set_yscale("log")
        a.tick_params(which="both", direction="in", right=True, top=True)


# draw to update positions
# fig_depl.draw(fig_depl.canvas.get_renderer())


ax_depl[2, 1].axis("off")

fig_depl.draw(fig_depl.canvas.get_renderer())

ax_depl[0, 0].legend(lines, labels, framealpha=0.0)

cur_pos = ax_depl[2][0].get_position(original=False).bounds
print(cur_pos)
tgt_pos = np.copy(list(cur_pos))
print(tgt_pos)
tgt_pos[1] = ax_depl[2][1].get_position().bounds[1] - 0.0175
tgt_pos[0] = (cur_pos[0] + ax_depl[2][1].get_position(original=False).bounds[0]) * 0.5
# tgt_pos[2] = (cur_pos[2] + ax_depl[2][1].get_position().bounds[2]) * 0.5
# tgt_pos[3] = (cur_pos[3] + ax_depl[2][1].get_position().bounds[3]) * 0.5
print(tgt_pos)
ax_depl[2][0].set_position(tgt_pos, which="both")

fig_depl.savefig(f"figs/depletion_{tgt_zedstr:s}.png")

for a_dtm in [ax_dtm, ax_dtmCs, ax_dtmCl, ax_dtmSis, ax_dtmSil]:
    a_dtm[0].legend(lines, labels, framealpha=0.0)
fig_dtm.savefig(f"figs/dtm_{tgt_zedstr:s}.png")
fig_dtmCs.savefig(f"figs/dtm_Cs{tgt_zedstr:s}.png")
fig_dtmCl.savefig(f"figs/dtm_Cl{tgt_zedstr:s}.png")
fig_dtmSis.savefig(f"figs/dtm_Sis{tgt_zedstr:s}.png")
fig_dtmSil.savefig(f"figs/dtm_Sil{tgt_zedstr:s}.png")
