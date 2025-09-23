# from f90nml import read
from f90_tools.star_reader import read_part_ball_NCdust
from zoom_analysis.stars import sfhs
from zoom_analysis.zoom_helpers import decentre_coordinates, find_starting_position
from zoom_analysis.halo_maker.read_treebricks import (
    read_zoom_stars,
)
from zoom_analysis.read.read_data import read_data_ball
from zoom_analysis.trees.tree_reader import read_tree_file_rev_correct_pos as read_tree_fev_sim
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import (
    # find_zoom_tgt_halo,
    get_assoc_pties_in_tree,
    get_central_gal_for_hid,
    smooth_props,
)
from zoom_analysis.dust.gas_reader import gas_pos_rad
from zoom_analysis.halo_maker.assoc_fcts import find_star_ctr_period

# from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# from scipy.spatial import cKDTree

import os
import numpy as np

from hagn.utils import get_hagn_sim
from hagn.tree_reader import read_tree_rev, interpolate_tree_position

# planck cosmo
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
from astropy import units as u
import h5py


HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"


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
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756",  # _leastcoarse",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
]

MgoverSil = 0.141
FeoverSil = 0.324
SioverSil = 0.163
OoverSil = 0.372

tgt_zed = 2.0
delta_t = 25  # Myr

msun_to_g = 1.989e33
Pmass_g = 1.6726219e-24

overwrite = True

# setup plot


# plt.subplots_adjust(hspace=0.0)

dens_bins = [0.1, 1, 10, 100]  # H/ccm
figs = []
axs = []
for ibin in range(len(dens_bins)):
    loc_figs, loc_axs = plt.subplots(
        4, 2, figsize=(8, 14), sharex=True, layout="constrained"
    )
    axs.append(np.ravel(loc_axs))
    figs.append(loc_figs)

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

for sdir in sdirs:
    name = sdir.split("/")[-1]
    dust_file = os.path.join(sdir, "fdepl_data.h5")
    sim = None

    if os.path.exists(dust_file) and not overwrite:
        with h5py.File(dust_file, "r") as f:
            mstel_zoom = f["mstel_zoom"][:]
            fdMg_zoom = f["fdMg_zoom"][:]
            fdFe_zoom = f["fdFe_zoom"][:]
            fdO_zoom = f["fdO_zoom"][:]
            fdSi_zoom = f["fdSi_zoom"][:]
            fdC_zoom = f["fdC_zoom"][:]
            tgt_times = f["tgt_times"][:]
    else:
        intID = int(name[2:].split("_")[0])

        hagn_tree_hids, hagn_tree_datas, hagn_tree_aexps = read_tree_rev(
            tgt_zed,
            [intID],
            tree_type="halo",
            target_fields=["m", "x", "y", "z", "r"],
            sim="hagn",
        )

        for k in hagn_tree_datas.keys():
            hagn_tree_datas[k] = hagn_tree_datas[k][0]

        hagn_sim.init_cosmo()
        l_hagn = (
            hagn_sim.unit_l(hagn_sim.aexp_stt) / (ramses_pc * 1e6) / hagn_sim.aexp_stt
        )
        hagn_tree_times = (
            hagn_sim.cosmo_model.age(1.0 / hagn_tree_aexps - 1.0).value * 1e3
        )

        sim = ramses_sim(sdir, nml="cosmo.nml")

        sim_snaps = sim.get_snaps(mini_snaps=True)
        sim_aexps = sim.get_snap_exps()
        sim.init_cosmo()
        sim_times = sim.cosmo_model.age(1.0 / sim_aexps - 1.0).value * 1e3

        tgt_times = np.arange(sim_times.min(), sim_times.max() + delta_t, delta_t)

        tgt_snaps = np.asarray(
            [sim.get_closest_snap(time=tgt_time) for tgt_time in tgt_times]
        )

        tgt_aexps = sim.get_snap_exps(tgt_snaps)
        tgt_times = sim.cosmo_model.age(1.0 / tgt_aexps - 1.0).value * 1e3

        nsteps = len(tgt_snaps)

        mstel_zoom = np.zeros(nsteps, dtype=np.float32)
        fdMg_zoom = np.zeros((nsteps, len(dens_bins)), dtype=np.float32)
        fdFe_zoom = np.zeros((nsteps, len(dens_bins)), dtype=np.float32)
        fdO_zoom = np.zeros((nsteps, len(dens_bins)), dtype=np.float32)
        fdSi_zoom = np.zeros((nsteps, len(dens_bins)), dtype=np.float32)
        fdC_zoom = np.zeros((nsteps, len(dens_bins)), dtype=np.float32)

        assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
        assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

        avail_aexps = sim.get_snap_exps(assoc_file_nbs, param_save=False)
        if not hasattr(sim, "cosmo_model"):
            sim.init_cosmo()
        avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

        hid_start, halo_dict, hosted_gals, found, start_aexp = find_starting_position(
            sim,
            avail_aexps,
            hagn_tree_aexps,
            hagn_tree_datas,
            hagn_tree_times,
            avail_times,
        )

        if not found:
            continue

        sim_halo_tree_rev_fname = os.path.join(
            sim.path, "TreeMakerDM_dust", "tree_rev.dat"
        )
        if not os.path.exists(sim_halo_tree_rev_fname):
            print("No tree file")
            continue

        tgt_zeds = 1.0 / tgt_aexps - 1

        sim_tree_hids, sim_tree_datas, sim_tree_aexps = read_tree_fev_sim(
            sim_halo_tree_rev_fname,
            sim,
            sim.get_closest_snap(aexp=start_aexp),
            fbytes=os.path.join(sim.path, "TreeMakerDM_dust"),
            zstart=1.0 / start_aexp - 1.0,
            tgt_ids=[hid_start],
            star=False,
        )

        gal_tree_props = get_assoc_pties_in_tree(sim, sim_tree_aexps, sim_tree_hids[0])

        smooth_tree_props = smooth_props(gal_tree_props)

        # print(sim_tree_hids)

        for k in sim_tree_datas.keys():
            sim_tree_datas[k] = sim_tree_datas[k][0]

        sim_tree_times = sim.cosmo_model.age(tgt_zeds).value * 1e3

        for istep, (snap, aexp, time) in enumerate(
            zip(tgt_snaps, tgt_aexps, tgt_times)
        ):
            if np.all(np.abs(avail_aexps - aexp) > 0.05):
                print("No assoc file for this aexp")
                continue

            assoc_file = assoc_files[assoc_file_nbs == snap]
            if len(assoc_file) == 0:
                print("No assoc file for this snap")
                continue

            sim_tree_arg = np.argmin(np.abs(sim_tree_aexps - aexp))
            cur_snap_hid = sim_tree_hids[0][sim_tree_arg]

            print(snap, aexp, 1 / aexp - 1, sim_tree_aexps[sim_tree_arg], cur_snap_hid)

            if cur_snap_hid <= 0:
                print("No halo")
                continue

            print(np.log10(sim_tree_datas["m"][sim_tree_arg]))

            gid, gal_dict = get_central_gal_for_hid(
                sim, cur_snap_hid, snap, verbose=True
            )
            if gid == None:
                print("No central galaxy")
                continue

            print("reading data")

            tgt_pos = gal_dict["pos"]
            # tgt_r = gal_dict["r50"]

            aexp_arg = np.argmin(np.abs(aexp - smooth_tree_props["aexps"]))
            # tgt_pos = smooth_tree_props["pos"][aexp_arg]
            tgt_r = smooth_tree_props["r50"][aexp_arg]

            # # stars = read_zoom_stars(sim, snap, gid)
            # stars = read_part_ball_NCdust(
            #     sim,
            #     snap,
            #     tgt_pos,
            #     tgt_r,
            #     tgt_fields=["mass", "birth_time", "metallicity"],
            #     fam=2,
            # )

            # cells = gas_pos_rad(
            #     sim,
            #     snap,
            #     # [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19],
            #     [1, 8, 9, 10, 11, 13, 16, 17, 18, 19],
            #     # ctr_stars,
            #     # extent_stars,
            #     tgt_pos,
            #     tgt_r,
            # )

            datas = read_data_ball(
                sim,
                snap,
                tgt_pos,
                tgt_r,
                host_halo=cur_snap_hid,
                data_types=["stars", "gas"],
                tgt_fields=["mass","age","metallicity","density","dust_bin01","dust_bin02","dust_bin03","dust_bin04","chem_O","chem_Fe","chem_C","chem_Si","chem_Mg"],
            )
            cells = datas["gas"]
            stars = datas["stars"]

            ages = stars["age"]
            Zs = stars["metallicity"]

            masses = sfhs.correct_mass(hagn_sim, ages, stars["mass"], Zs)

            # star_pos, ctr_stars, extent_stars = find_star_ctr_period(stars["pos"])

            # unit_d = sim.unit_d(aexp)

            mstel_zoom[istep] = masses.sum()

            # print(cells.keys())

            l_hagn_cm_comov = l_hagn * aexp * 1e6 * ramses_pc
            volumes = (2 ** -cells["ilevel"] * l_hagn_cm_comov) ** 3

            # fdep = 1 - Mdust/Mgas

            dens_Hpcc = cells["density"] / Pmass_g
            # print(mstel_zoom[istep], dens_Hpcc.mean())

            for ibin, dens_bin in enumerate(dens_bins):

                tgt_cells_cond = np.abs(dens_Hpcc - dens_bin) / dens_bin < 0.1

                tgt_cells = {}
                for k in cells.keys():
                    tgt_cells[k] = cells[k][tgt_cells_cond]
                tgt_volumes = volumes[tgt_cells_cond]

                # mgas = np.sum(tgt_cells["density"] * tgt_volumes)
                # mmet = np.sum(tgt_cells["density"] * tgt_volumes * tgt_cells["metallicity"])
                mdCs = np.sum(
                    tgt_cells["density"] * tgt_volumes * tgt_cells["dust_bin01"]
                )
                mdCl = np.sum(
                    tgt_cells["density"] * tgt_volumes * tgt_cells["dust_bin02"]
                )
                mdSs = (
                    np.sum(tgt_cells["density"] * tgt_volumes * tgt_cells["dust_bin03"])
                    / SioverSil
                )
                mdSl = (
                    np.sum(tgt_cells["density"] * tgt_volumes * tgt_cells["dust_bin04"])
                    / SioverSil
                )

                # mH = np.sum(tgt_cells["density"] * tgt_volumes * tgt_cells["chem_H"])
                mO = np.sum(tgt_cells["density"] * tgt_volumes * tgt_cells["chem_O"])
                mFe = np.sum(tgt_cells["density"] * tgt_volumes * tgt_cells["chem_Fe"])
                mMg = np.sum(tgt_cells["density"] * tgt_volumes * tgt_cells["chem_Mg"])
                mC = np.sum(tgt_cells["density"] * tgt_volumes * tgt_cells["chem_C"])
                mSi = np.sum(tgt_cells["density"] * tgt_volumes * tgt_cells["chem_Si"])
                # mN = np.sum(tgt_cells["density"] * tgt_volumes * tgt_cells["chem_N"])
                # mS = np.sum(tgt_cells["density"] * tgt_volumes * tgt_cells["chem_S"])

                mdC = mdCs + mdCl
                mdS = mdSs + mdSl
                md = mdC + mdS

                mdMg = mdS * MgoverSil
                mdFe = mdS * FeoverSil
                mdSi = mdS * SioverSil
                mdO = mdS * OoverSil

                fdMg_zoom[istep, ibin] = 1 - mdMg / mMg
                fdFe_zoom[istep, ibin] = 1 - mdFe / mFe
                fdO_zoom[istep, ibin] = 1 - mdO / mO
                fdSi_zoom[istep, ibin] = 1 - mdSi / mSi
                fdC_zoom[istep, ibin] = 1 - mdC / mC

                print(
                    1 / aexp - 1,
                    time,
                    dens_bins[ibin],
                    fdMg_zoom[istep, ibin],
                    fdFe_zoom[istep, ibin],
                    fdO_zoom[istep, ibin],
                    fdSi_zoom[istep, ibin],
                    fdC_zoom[istep, ibin],
                )

        with h5py.File(dust_file, "w") as f:
            f.create_dataset("mstel_zoom", data=mstel_zoom, compression="lzf")
            f.create_dataset("fdMg_zoom", data=fdMg_zoom, compression="lzf")
            f.create_dataset("fdFe_zoom", data=fdFe_zoom, compression="lzf")
            f.create_dataset("fdO_zoom", data=fdO_zoom, compression="lzf")
            f.create_dataset("fdSi_zoom", data=fdSi_zoom, compression="lzf")
            f.create_dataset("fdC_zoom", data=fdC_zoom, compression="lzf")
            f.create_dataset("tgt_times", data=tgt_times, compression="lzf")

    for ibin in range(len(dens_bins)):

        (l,) = axs[ibin][0].plot(
            tgt_times / 1e3, mstel_zoom, ls=zoom_ls[isim], label=name
        )

        axs[ibin][1].plot(
            tgt_times / 1e3, fdMg_zoom[:, ibin], ls=zoom_ls[isim], c=l.get_color()
        )
        axs[ibin][2].plot(
            tgt_times / 1e3, fdFe_zoom[:, ibin], ls=zoom_ls[isim], c=l.get_color()
        )
        axs[ibin][3].plot(
            tgt_times / 1e3, fdO_zoom[:, ibin], ls=zoom_ls[isim], c=l.get_color()
        )
        axs[ibin][4].plot(
            tgt_times / 1e3, fdSi_zoom[:, ibin], ls=zoom_ls[isim], c=l.get_color()
        )
        axs[ibin][5].plot(
            tgt_times / 1e3, fdC_zoom[:, ibin], ls=zoom_ls[isim], c=l.get_color()
        )

        if ibin == 0:
            lines.append(Line2D([0, 1], [0, 1], ls="-", c=l.get_color()))
            labels.append(name)

        times_nonzero_stmass = tgt_times[mstel_zoom > 0]

        xmin = min(xmin, times_nonzero_stmass.min() / 1e3)
        xmax = max(xmax, times_nonzero_stmass.max() / 1e3)

if xmin == np.inf:
    xmin = None
if xmax == -np.inf:
    xmax = None


for ibin in range(len(dens_bins)):

    axs[ibin][-1].legend(
        lines,
        labels,
        framealpha=0.0,
        ncol=3,
        title="nH = {:.1f} H/ccm".format(dens_bins[ibin]),
    )
    axs[ibin][-1].axis("off")
    axs[ibin][-2].axis("off")

    # ax[-1].axis("off")
    # cur_leg = ax[-1].get_legend()
    # ax[-1].legend(
    #     cur_leg.legendHandles
    #     + [
    #         Line2D([0], [0], color="k", linestyle="-"),
    #         Line2D([0], [0], color="k", ls="--"),
    #     ],
    #     sim_ids + ["HAGN", "zoom"],
    # )

    axs[ibin][0].set_ylim(
        1e2,
    )

    axs[ibin][0].set_xlim(xmin, xmax)

    axs[ibin][0].tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=True,
        left=True,
        right=True,
        direction="in",
    )

    axs[ibin][1].tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=True,
        right=True,
        direction="in",
    )

    axs[ibin][2].tick_params(
        axis="both",
        which="both",
        bottom=True,
        top=False,
        left=True,
        right=True,
        direction="in",
    )

    axs[ibin][4].tick_params(
        axis="both", which="both", direction="in", bottom=True, labelbottom=True
    )
    axs[ibin][5].tick_params(
        axis="both", which="both", direction="in", bottom=True, labelbottom=True
    )

    axs[ibin][0].set_yscale("log")

    for a in axs[ibin][1:-2]:
        a.set_ylim(0, 1)

    for a in axs[ibin]:
        # a.set_yscale("log")
        a.grid()

    xlim = axs[ibin][0].get_xlim()
    axs[ibin][0].set_xlim(xlim)

    axs[ibin][0].set_ylabel("Stellar mass [M$_\odot$]")
    axs[ibin][1].set_ylabel("f$_{depl}$ [Mg]")
    axs[ibin][2].set_ylabel("f$_{depl}$ [Fe]")
    axs[ibin][3].set_ylabel("f$_{depl}$ [O]")
    axs[ibin][4].set_ylabel("f$_{depl}$ [Si]")
    axs[ibin][5].set_ylabel("f$_{depl}$ [C]")

    axs[ibin][4].set_xlabel("Time [Gyr]")
    axs[ibin][5].set_xlabel("Time [Gyr]")

    y2 = axs[ibin][0].twiny()
    y2.set_xlim(xlim)
    # y2.set_xticks(ax.get_xticks())
    y2.set_xticklabels(
        [f"{cosmo.age(zed).value:.1f}" for zed in axs[ibin][-1].get_xticks()]
    )
    y2.set_xlabel("redshift")

    y2 = axs[ibin][1].twiny()
    y2.set_xlim(xlim)
    # y2.set_xticks(ax.get_xticks())
    y2.set_xticklabels(
        [f"{cosmo.age(zed).value:.1f}" for zed in axs[ibin][-1].get_xticks()]
    )
    y2.set_xlabel("redshift")

    bin_str = f"{dens_bins[ibin]:.1f}".replace(".", "p")

    figs[ibin].savefig(f"figs/fdepl_trees_{bin_str}.png")
