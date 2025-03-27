from zoom_analysis.stars import sfhs

# from zoom_analysis.zoom_helpers import decentre_coordinates
# from zoom_analysis.stars.star_reader import yt_read_star_ball, read_part_ball
# from zoom_analysis.stars.leja_quench_fit import sfr_ridge_leja22
# from zoom_analysis.halo_maker.read_treebricks import (
# read_brickfile,
# convert_brick_units,
# convert_star_units,
# read_zoom_stars,
# )
from zoom_analysis.trees.tree_reader import read_tree_file_rev as read_tree_fev_sim
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import (
    # find_zoom_tgt_gal,
    # find_zoom_tgt_halo,
    # get_central_gal_for_hid,
    get_halo_props_snap,
    # get_gal_props_snap,
    get_r50,
)

import h5py

from zoom_analysis.zoom_helpers import find_starting_position


# from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# from scipy.spatial import cKDTree

import os
import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# from zoom_analysis.sink_reader import *

from hagn.association import gid_to_stars
from hagn.utils import get_hagn_sim, adaptahop_to_code_units
from hagn.tree_reader import read_tree_rev, interpolate_tree_position
from hagn.catalogues import get_halos_cat, make_super_cat, get_cat_hids, get_cat_gids

# get colour cycler
# from cycler import cycler

# planck cosmo
from astropy.cosmology import Planck18 as cosmo

from hagn.catalogues import make_super_cat, get_cat_hids

from f90_tools.star_reader import read_part_ball_hagn, read_part_ball_NCdust


# sim_ids = [
#     "id2757_995pcnt_mstel/",
#     "id19782_500pcnt_mstel/",pip install setuptools
#     "id13600_005pcnt_mstel/",
# ]
# sdir = "/home/jlewis/sims/proposal_zooms/ExtractZoom/mh1e12/"
hagn_snap = 197  # z=2 in the full res box
HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
delta_t = 100  # Myr


sdirs = [
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_kicks",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_high_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_high_thermal_eff",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_noAGN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704",  # _leastcoarse",
    # # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
]

# overwrite = False
overwrite = True

# setup plot
fig, ax = plt.subplots(3, 2, sharex=True, figsize=(16, 8))

yax2 = None

hagn_sim = get_hagn_sim()
hagn_sim.init_cosmo()
l_hagn = hagn_sim.unit_l(hagn_sim.aexp_stt) / (ramses_pc * 1e6) / hagn_sim.aexp_stt
# colors = []

hagn_snaps = hagn_sim.snap_numbers
hagn_aexps = hagn_sim.get_snap_exps(param_save=False)
hagn_times = hagn_sim.cosmo_model.age(1.0 / hagn_aexps - 1.0).value * 1e3

tgt_zed = 1.0 / hagn_aexps[hagn_snaps == hagn_snap] - 1.0
tgt_time = cosmo.age(tgt_zed).value

last_hagn_id = -1
isim = 0

zoom_ls = ["-", "--", ":", "-.", (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10))] * 15
lines = []
labels = []

last_simID = None
l = None

zoom_style = 0

for sdir in sdirs:

    name = sdir.split("/")[-1].strip()

    # get the galaxy in HAGN
    intID = int(name[2:].split("_")[0])
    # gid = int(make_super_cat(197, outf="/data101/jlewis/hagn/super_cats")["gid"][0])

    hagn_tree_hids, hagn_tree_datas, hagn_tree_aexps = read_tree_rev(
        tgt_zed,
        [intID],
        tree_type="halo",
        # [gid],
        # tree_type="gal",
        target_fields=["m", "x", "y", "z", "r"],
        sim="hagn",
    )

    for key in hagn_tree_datas:
        hagn_tree_datas[key] = hagn_tree_datas[key][0][:]

    hagn_sim.init_cosmo()

    hagn_tree_times = hagn_sim.cosmo_model.age(1.0 / hagn_tree_aexps - 1.0).value * 1e3
    # print("halo id: ", intID)
    # print(intID)

    sim = ramses_sim(sdir, nml="cosmo.nml")

    sim_snaps = sim.snap_numbers
    sim_aexps = sim.get_snap_exps()
    sim.init_cosmo()
    sim_times = sim.cosmo_model.age(1.0 / sim_aexps - 1.0).value * 1e3

    # last sim_aexp
    # valid_steps = hagn_tree_aexps < (1.0 / (tgt_zed + 1.0))  # sim.aexp_end
    # nsteps = np.sum(valid_steps)

    nsteps = len(sim_snaps)
    nstep_tree_hagn = len(hagn_tree_aexps)

    mstel_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
    r50_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
    dens_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
    time_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)

    mstel_zoom = np.zeros(nsteps, dtype=np.float32)
    r50_zoom = np.zeros(nsteps, dtype=np.float32)
    dens_zoom = np.zeros(nsteps, dtype=np.float32)
    mfrac_zoom_gals = np.zeros(nsteps, dtype=np.float32)
    mfrac_zoom = np.zeros(nsteps, dtype=np.float32)
    time_zoom = np.zeros(nsteps, dtype=np.float32)
    aexp_zoom = np.zeros(nsteps, dtype=np.float32)

    datas_path = os.path.join(sim.path, "computed_data")
    fout_h5 = os.path.join(datas_path, f"morpho_inHalo.h5")
    read_existing = True

    if not os.path.exists(datas_path):
        os.makedirs(datas_path, exist_ok=True)
        read_existing = False
    else:
        if not os.path.exists(fout_h5):
            read_existing = False

    if read_existing and not overwrite:

        # read the data
        with h5py.File(fout_h5, "r") as f:

            saved_mstel_zoom = f["mstel_zoom"][:]
            saved_r50_zoom = f["r50_zoom"][:]
            saved_dens_zoom = f["dens_zoom"][:]
            saved_mfrac_zoom = f["mfrac_zoom"][:]
            saved_mfrac_zoom_gals = f["mfrac_zoom_gals"][:]
            saved_time_zoom = f["time_zoom"][:]
            saved_aexp_zoom = f["aexp_zoom"][:]

            saved_mstel_hagn = f["mstel_hagn"][:]
            saved_r50_hagn = f["r50_hagn"][:]
            saved_dens_hagn = f["dens_hagn"][:]
            saved_time_hagn = f["time_hagn"][:]

        if len(saved_mstel_zoom) == len(sim_snaps):

            mstel_zoom = saved_mstel_zoom
            r50_zoom = saved_r50_zoom
            dens_zoom = saved_dens_zoom
            mfrac_zoom = saved_mfrac_zoom
            mfrac_zoom_gals = saved_mfrac_gals_zoom
            time_zoom = saved_time_zoom
            aexp_zoom = saved_aexp_zoom

            mstel_hagn = saved_mstel_hagn
            r50_hagn = saved_r50_hagn
            dens_hagn = saved_dens_hagn
            time_hagn = saved_time_hagn
        else:
            read_existing = False

    if not read_existing or overwrite:

        # find last output with assoc files
        assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
        assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

        avail_snaps = np.intersect1d(sim_snaps, assoc_file_nbs)

        avail_aexps = np.intersect1d(
            sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
        )
        avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

        # hagn tree loop
        for istep, (aexp, time) in enumerate(zip(hagn_tree_aexps, hagn_tree_times)):

            hagn_snap = hagn_snaps[np.argmin(np.abs(hagn_aexps - aexp))]

            try:
                super_cat = make_super_cat(
                    hagn_snap, outf="/data101/jlewis/hagn/super_cats"
                )  # , overwrite=True)
            except FileNotFoundError:
                print("No super cat")
                continue

            gal_pties = get_cat_hids(super_cat, [int(hagn_tree_hids[0][istep])])

            if len(gal_pties["hid"]) == 0:
                continue

            # print(gal_pties["hx"])
            tgt_pos = np.asarray(
                [gal_pties["hx"][0], gal_pties["hy"][0], gal_pties["hz"][0]]
            )
            tgt_rad = gal_pties["rvir"]

            tgt_rad_ckpc = tgt_rad * hagn_sim.cosmo.lcMpc * 1e3

            stars = read_part_ball_hagn(
                hagn_sim,
                hagn_snap,
                tgt_pos,
                tgt_rad,
                ["pos", "mass", "birth_time", "metallicity"],
                fam=2,
            )

            # print(stars.keys())

            ages = stars["age"]
            Zs = stars["metallicity"]

            masses = sfhs.correct_mass(hagn_sim, ages, stars["mass"], Zs)

            mstel_hagn[istep] = masses.sum()

            r50_hagn[istep] = get_r50(stars["pos"], masses, tgt_pos)
            time_hagn[istep] = time

            dens_hagn[istep] = mstel_hagn[istep] / tgt_rad_ckpc**3 / 4.0 * 3.0 / np.pi

            # # sim_snap = sim.get_closest_snap(aexp=aexp)

            # print(aexp, avail_aexps)

        hid_start, halo_dict, hostel_gals, found, start_aexp = find_starting_position(
            sim,
            avail_aexps,
            hagn_tree_aexps,
            hagn_tree_datas,
            hagn_tree_times,
            avail_times,
        )

        if not found:
            continue

        # print(hagn_ctr)

        # print(hagn_ctr)

        # hid_start = int(hosted_gals["hid"][np.argmax(hosted_gals["mass"])])

        # load sim tree
        sim_halo_tree_rev_fname = os.path.join(
            sim.path, "TreeMakerDM_dust", "tree_rev.dat"
        )
        sim_gal_tree_rev_fname = os.path.join(
            sim.path, "TreeMakerstars2_dp_rec_dust", "tree_rev.dat"
        )
        if not os.path.exists(sim_halo_tree_rev_fname):
            print("No tree file")
            continue

        sim_zeds = 1.0 / avail_aexps - 1

        sim_tree_hids, sim_tree_halo_datas, sim_tree_halo_aexps = read_tree_fev_sim(
            sim_halo_tree_rev_fname,
            fbytes=os.path.join(sim.path, "TreeMakerDM_dust"),
            zstart=1.0 / start_aexp - 1.0,
            tgt_fields=["m_father"],  # only collect hids
            tgt_ids=[hid_start],
            star=False,
        )

        last_assoc_snap = np.max(avail_snaps)
        last_assoc_aexp = avail_aexps[np.argmax(avail_snaps)]
        last_assoc_hid = sim_tree_hids[0][
            np.argmin(np.abs(sim_tree_halo_aexps - last_assoc_aexp))
        ]

        sim_tree_times = sim.cosmo_model.age(sim_zeds).value * 1e3

        # # zoom loop
        for istep, (snap, aexp, time) in enumerate(
            zip(sim_snaps, sim_aexps, sim_times)
        ):
            # if time < sim_tree_times.min() - 5:
            # continue

            if np.all(np.abs(avail_aexps - aexp) > 1e-1):
                print(f"No assoc file for at z={1./aexp-1:.1f},snap={snap:d}")
                continue

            assoc_file = assoc_files[assoc_file_nbs == snap]
            if len(assoc_file) == 0:
                print(f"No assoc file for at z={1./aexp-1:.1f},snap={snap:d}")
                continue

            sim_halo_tree_arg = np.argmin(np.abs(sim_tree_halo_aexps - aexp))
            cur_snap_hid = sim_tree_hids[0][sim_halo_tree_arg]

            if cur_snap_hid in [0, -1]:
                continue

            mfrac = 1.0 - sim_tree_halo_datas["m_father"][0][sim_halo_tree_arg] / 100.0

            hprops, hgals = get_halo_props_snap(sim.path, snap, hid=cur_snap_hid)

            if hgals != {}:
                mass_oomb = np.zeros(len(hgals["gids"]))  # OutOFMainBranch

                for igal, hgal_id in enumerate(hgals["gids"]):

                    try:
                        sim_tree_gids, sim_tree_gal_datas, sim_tree_gal_aexps = (
                            read_tree_fev_sim(
                                sim_gal_tree_rev_fname,
                                fbytes=os.path.join(
                                    sim.path, "TreeMakerstars2_dp_rec_dust"
                                ),
                                zstart=aexp,
                                tgt_ids=[hgal_id],
                                star=True,
                                tgt_fields=["m_father"],
                            )
                        )
                    except IndexError:
                        continue

                    sim_gal_tree_arg = np.argmin(np.abs(sim_tree_gal_aexps - aexp))

                    mass_oomb[igal] = (
                        1.0
                        - sim_tree_gal_datas["m_father"][0][sim_gal_tree_arg] / 100.0
                    ) * hgals["mass"][igal]

                mfrac_zoom_gals[istep] = mass_oomb.sum() / hprops["mvir"]

            tgt_pos = hprops["pos"]
            tgt_r = hprops["rvir"]

            tgt_r_ckpc = tgt_r * sim.cosmo.lcMpc * 1e3

            stars = read_part_ball_NCdust(
                sim,
                snap,
                tgt_pos,
                tgt_r,
                tgt_fields=["pos", "mass", "birth_time", "metallicity"],
                fam=2,
            )
            # print(stars.keys())

            if len(stars["mass"]) == 0:
                print(f"No stars in halo {cur_snap_hid} at snap {snap}")
                continue

            ages = stars["age"]
            Zs = stars["metallicity"]

            masses = sfhs.correct_mass(hagn_sim, ages, stars["mass"], Zs)

            mstel_zoom[istep] = masses.sum()
            r50_zoom[istep] = get_r50(stars["pos"], masses, tgt_pos)
            dens_zoom[istep] = mstel_zoom[istep] / tgt_r_ckpc**3 / 4.0 * 3.0 / np.pi
            mfrac_zoom[istep] = mfrac
            time_zoom[istep] = time
            aexp_zoom[istep] = aexp

            print(
                snap,
                1.0 / aexp - 1,
                mstel_zoom[istep],
                r50_zoom[istep],
                dens_zoom[istep],
                mfrac,
                mfrac_zoom_gals[istep],
            )

    color = None
    if last_simID == intID:
        color = l.get_color()
        zoom_style += 1
    else:
        zoom_style = 0

    (l,) = ax[0, 0].plot(
        time_zoom / 1e3, mstel_zoom, ls=zoom_ls[zoom_style], lw=2.0, color=color
    )

    nn_zero_r50 = r50_zoom > 0

    ax[1, 0].plot(
        time_zoom[nn_zero_r50] / 1e3,
        r50_zoom[nn_zero_r50] * sim.cosmo.lcMpc * 1e3 * aexp_zoom[nn_zero_r50],
        ls=zoom_ls[zoom_style],
        lw=2.0,
        color=color,
    )
    ax[1, 1].plot(
        time_zoom / 1e3,
        dens_zoom,
        ls=zoom_ls[zoom_style],
        lw=2.0,
        color=color,
    )

    ax[1, 1].set_yscale("log")

    valid_mfrac = (mfrac_zoom > 0) * (mfrac_zoom < 1.0)

    ax[2, 0].plot(
        time_zoom[valid_mfrac] / 1e3,
        mfrac_zoom[valid_mfrac],
        ls=zoom_ls[zoom_style],
        lw=2.0,
        color=color,
    )

    valid_mfrac_gal = (mfrac_zoom_gals > 0) * (mfrac_zoom_gals < 1.0)

    ax[0, 1].plot(
        time_zoom[valid_mfrac] / 1e3,
        mfrac_zoom_gals[valid_mfrac_gal],
        ls=zoom_ls[zoom_style],
        lw=2.0,
        color=color,
    )

    labels.append(sim.name)
    lines.append(l)

    last_simID = intID

    if not read_existing or overwrite:

        with h5py.File(fout_h5, "w") as dest:

            dest.create_dataset("mstel_zoom", data=mstel_zoom, compression="lzf")
            dest.create_dataset("r50_zoom", data=r50_zoom, compression="lzf")
            dest.create_dataset("dens_zoom", data=dens_zoom, compression="lzf")
            dest.create_dataset("mfrac_zoom", data=mfrac_zoom, compression="lzf")
            dest.create_dataset("time_zoom", data=time_zoom, compression="lzf")
            dest.create_dataset("aexp_zoom", data=aexp_zoom, compression="lzf")

            dest.create_dataset("mstel_hagn", data=mstel_hagn, compression="lzf")
            dest.create_dataset("r50_hagn", data=r50_hagn, compression="lzf")
            dest.create_dataset("dens_hagn", data=dens_hagn, compression="lzf")
            dest.create_dataset("time_hagn", data=time_hagn, compression="lzf")


# ax[0, 0].set_xlim(0.5, 3.316)

ax[0, 0].set_yscale("log")

ax[2, 0].set_ylim(-0.05, 1.05)

# ax[2].set_ylim(-0.05, 2.0)

# ax[2].set_ylim(-0.05, 1.05)
# ax[1].set_yscale("log")
# ax[0].set_xlim(0.5, 2.5)

# add key to existing legend

# ax[-1].plot([], [], "k-", label="HAGN")
# ax[-1].plot([], [], "k-", label="zoom")


# cur_leg = ax[-1].get_legend()
# ax[-1].legend(
#     cur_leg.legendHandles
#     + [
#         Line2D([0], [0], color="k", linestyle="-"),
#         Line2D([0], [0], color="k", ls="--"),
#     ],
#     sim_ids + ["HAGN", "zoom"],
# )

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

plt.subplots_adjust(hspace=0.0)


# sfrs_leja = sfr_ridge_leja22(tgt_zed, mstel_zoom)
# ax[0].plot(mass_bins, sfrs_leja / 10.0 / mstel_zoom, "k--")
# ax[0].annotate(
#     mass_bins[0], sfrs_leja[0] / 10.0 / mstel_zoom, "0.1xMS of Leja+22", color="k"
# )

ax[0, 0].set_ylabel(r"$M_{\star}$ [$M_{\odot}$]")
ax[1, 0].set_ylabel(r"$R_{50}$ [kpc]")
ax[1, 1].set_ylabel(r"Stellar density,[M$_{\odot}$ kpc$^{-3}$]")
ax[2, 0].set_ylabel(r"$\delta m_{halo}_{\mathrm{minor}}$")
ax[0, 1].set_ylabel(r"$\delta m_{gal}_{\mathrm{minor}}$")


y2 = ax[0, 0].twiny()
y2.set_xlim(ax[0, 0].get_xlim())
zlabels = [f"{cosmo.age(zed).value:.1f}" for zed in ax[0, 0].get_xticks()]
# y2.set_xticks(ax[0].get_xticks())
y2.set_xticklabels(zlabels)
y2.set_xlabel("redshift")

ax[2, 1].legend(lines, labels, framealpha=0.0, ncol=2, loc="center right")
ax[2, 1].axis("off")

fig.savefig(f"morpho_compHAGN_trees_inHalo.png")
