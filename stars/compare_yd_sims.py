from zoom_analysis.constants import ramses_pc
from zoom_analysis.stars import sfhs
from zoom_analysis.halo_maker.read_treebricks import (
    read_brickfile,
    convert_brick_units,
    read_gal_stars,
    convert_star_units,
)

# from zoom_analysis.stars import star_reader
from zoom_analysis.sinks.sink_reader import (
    find_massive_sink,
    get_sink_mhistory,
    convert_sink_units,
    # read_sink_bin,
    # snap_to_coarse_step,
)

from gremlin.read_sim_params import ramses_sim

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# from hagn.utils import get_hagn_sim, adaptahop_to_code_units
# from hagn.tree_reader import read_tree_rev
# from hagn.catalogues import get_halos_cat, make_super_cat, get_cat_hids
# from f90_tools.star_reader import read_part_ball_hagn, read_part_ball_YDBondi


sim_ids = [
    "BondiNoVrel",
    "ClassicBondi",
    "MeanBondi",
    "MeanBondiVrelNonZero",
    "NoAGN",
    # "id147479_low_nstar",
    # "id147479_high_nsink",
]
# sim_ids = ["id74099", "id147479", "id242704"]


sdirs = [
    "/data102/dubois/ZoomIMAGE/Dust",
    "/data102/dubois/ZoomIMAGE/Dust",
    "/data102/dubois/ZoomIMAGE/Dust",
    "/data102/dubois/ZoomIMAGE/Dust",
    "/data102/dubois/ZoomIMAGE/Dust",
]

overwrite = False

# stellar mass, bh mass
# ssfr, mdot
fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex="col")
plt.subplots_adjust(hspace=0.0, wspace=0.0)

plotz_window = (2, 8)

for sim_id, sim_dir in zip(sim_ids, sdirs):

    sim = ramses_sim(
        os.path.join(sim_dir, sim_id), nml="cosmo.nml", output_path="OUTPUT_DIR"
    )
    print(sim.name)

    zoom_ctr = sim.zoom_ctr
    zoom_r = sim.namelist["refine_params"]["rzoom"]

    # star part

    # find galmaker runs
    galmaker_dir = os.path.join(sim_dir, sim_id, "TREE_STARS_AdaptaHOP_dp_SCnew_gross")
    if not os.path.exists(galmaker_dir):
        print("didn't find galmaker dir")
        continue
    gals_dirs = np.asarray([d for d in os.listdir(galmaker_dir) if "GAL_" in d])
    gal_snaps = np.asarray([int(d.split("_")[-1]) for d in gals_dirs])

    prev_gal_pos = zoom_ctr

    snaps = sim.snap_numbers
    aexps = sim.get_snap_exps(param_save=False)
    zeds = 1.0 / aexps - 1.0
    times = sim.get_snap_times(param_save=False)

    # filter out the snaps that are not in the gal dirs
    mask = np.isin(snaps, gal_snaps)
    snaps = snaps[mask]
    aexps = aexps[mask]
    times = times[mask]

    outd = os.path.join("/data101/jlewis/YD_accretion_tests", sim.name, "prods")
    if not os.path.exists(outd):
        os.makedirs(outd, exist_ok=True)
    outf = os.path.join(outd, "star_formation_history")

    if not os.path.isfile(outf) or overwrite:

        sms = np.zeros(len(snaps), dtype="f4")
        ssfr = np.zeros(len(snaps), dtype="f4")
        gal_pos = np.zeros((len(snaps), 3), dtype="f8")
        gal_r = np.zeros(len(snaps), dtype="f4")

        for isnap, (snap, aexp, time) in enumerate(zip(snaps, aexps, times)):

            print(
                f"...output:{snap:d}, time:{time:.1f} Myr, redshift:{1./aexp-1.:.2f}",
                end="\n",
            )

            # load stars
            gals = read_brickfile(
                os.path.join(galmaker_dir, f"tree_bricks{snap:03d}"), star=True
            )
            convert_brick_units(gals, sim)

            brick_gal_pos = np.asarray(
                [gals["positions"]["x"], gals["positions"]["y"], gals["positions"]["z"]]
            )

            gal_mass = gals["hosting info"]["hmass"]
            gal_hids = gals["hosting info"]["hid"]
            centrals = gals["hosting info"]["hlvl"] == 1

            # find most massive gal nearest to zoom_ctr and then the last pos where we found a galaxy
            dist_pos = np.sqrt(
                (brick_gal_pos[0] - prev_gal_pos[0]) ** 2
                + (brick_gal_pos[1] - prev_gal_pos[1]) ** 2
                + (brick_gal_pos[2] - prev_gal_pos[2]) ** 2
            )
            # in centre 20%
            found = False
            pcnt_rad = 0.1
            while not found:
                centre = dist_pos < pcnt_rad * zoom_r

                central_centrals = np.where(centre * centrals)[0]

                if len(central_centrals) > 0:
                    found = True
                else:
                    pcnt_rad += 0.1

            central_centrals_mass = gal_mass[central_centrals]

            target_arg = central_centrals[np.argmax(central_centrals_mass)]
            target_hid = gal_hids[target_arg]

            gal_pos[isnap, :] = brick_gal_pos[:, target_arg]

            gal_r[isnap] = gals["virial properties"]["rvir"][target_arg]

            # print(gal_pos[isnap], gal_r[isnap])

            prev_gal_pos = gal_pos[isnap, :]

            fname_stars = os.path.join(
                galmaker_dir, f"GAL_{snap:05d}", f"gal_stars_{target_hid:07d}"
            )
            gal_stars = read_gal_stars(
                fname_stars, tgt_fields=["mpart", "Zpart", "agepart"]
            )
            convert_star_units(
                gal_stars,
                snap,
                sim,
                # cosmo_fname="/data101/jlewis/friedman/MeanBondiVrelNonZero/friedman.txt",
            )

            loc_sms, loc_sfr, loc_ssfr, _, _ = sfhs.get_sf_stuff(
                gal_stars, 1.0 / aexp - 1, sim
            )

            sms[isnap], ssfr[isnap] = loc_sms[0], loc_ssfr[0]
            # print(loc_sms, sms[isnap])
            # print(loc_sfr)
            # print(loc_ssfr, ssfr[isnap])
            if isnap > 0:
                dx = np.sqrt(np.sum((gal_pos[isnap, :] - prev_gal_pos) ** 2))
                if dx > gal_r[isnap]:
                    print(
                        f"galaxy moved by {dx*sim.cosmo.lcMpc*1e3:.2e} kpc (more than HM radius)"
                    )
                    print(
                        f"cur mass:{sms[isnap]: .2e} Msun, prev mass: {sms[isnap-1]:.2e} Msun"
                    )
        with h5py.File(outf, "w") as f:
            f.create_dataset("aexps", data=aexps, compression="lzf")
            f.create_dataset(name="times", data=times, compression="lzf")
            f.create_dataset("zeds", data=zeds, compression="lzf")
            f.create_dataset("sms Msun", data=sms, compression="lzf")
            f.create_dataset("ssfr Myr^-1", data=ssfr, compression="lzf")
            f.create_dataset("gal positions", data=gal_pos, compression="lzf")
            f.create_dataset("gal radii", data=gal_r, compression="lzf")

    else:

        with h5py.File(outf, "r") as f:
            aexps = f["aexps"][:]
            times = f["times"][:]
            zeds = f["zeds"][:]
            sms = f["sms Msun"][:]
            ssfr = f["ssfr Myr^-1"][:]
            gal_pos = f["gal positions"][:]
            gal_r = f["gal radii"][:]

            # print(sim.name, list(zip(zeds, ssfr)))

    if "smbh_params" in sim.namelist:

        # using last snap galaxy position to get bh mass
        # load sinks
        sid = find_massive_sink(gal_pos[-1, :], snaps[-1], sim, rmax=gal_r[-1])

        # if sid is not None:
        # # get coarse step
        # step = snap_to_coarse_step(snap, sim)
        # # load sink data
        # fname_sink = os.path.join(sim.sink_path, f"sink_{step:05d}.dat")
        # sink = read_sink_bin(fname_sink)

        sink = get_sink_mhistory(sid["identity"], snaps[-1], sim)
        # convert_sink_units(sink, sim)

        bhms = sink["mass"]
        mdots = sink["dMBH_coarse"]
        zeds_bh = sink["zeds"]

        zcond = (zeds_bh > plotz_window[0]) * (zeds_bh <= plotz_window[1])

        ax[0, 1].plot(
            zeds_bh[zcond],
            bhms[zcond],
            lw=3,
        )
        # (l,) = ax[1, 1].plot(zeds_bh, mdots)
        # smooth mdots
        kern_size = 50
        mdots_smooth = np.convolve(mdots, np.ones(kern_size) / kern_size, mode="same")[
            ::kern_size
        ]
        zeds_smooth = np.convolve(zeds_bh, np.ones(kern_size) / kern_size, mode="same")[
            ::kern_size
        ]

        zcond = (zeds_smooth > plotz_window[0]) * (zeds_smooth <= plotz_window[1])

        ax[1, 1].plot(
            zeds_smooth[zcond],
            mdots_smooth[zcond],
            # color=l.get_color(),
            lw=3,
        )

    else:
        print("no agns in this run")

    zeds = 1.0 / aexps - 1

    zcond = (zeds > plotz_window[0]) * (zeds <= plotz_window[1])

    ax[0, 0].plot(
        zeds[zcond],
        sms[zcond],
        label=sim_id,
        lw=3,
    )
    ax[1, 0].plot(zeds[zcond], ssfr[zcond], lw=3)


ax[0, 0].set_ylabel("Stellar Mass, [M$_\odot$]")
ax[0, 1].set_ylabel("BH Mass, [M$_\odot$]")
ax[1, 0].set_ylabel("sSFR, [$\mathrm{Myr^{-1}}$]")
ax[1, 1].set_ylabel("BH Accretion Rate, [?]")  # [M$_\odot$ yr$^-1$]")

ax[0, 1].yaxis.set_label_position("right")
ax[1, 1].yaxis.set_label_position("right")

for ia, a in enumerate(np.ravel(ax)):
    a.set_xlabel("z")
    a.set_yscale("log")
    # a.grid()

    a.set_xlim(max(plotz_window), min(plotz_window))
    xticks = np.arange(min(plotz_window), max(plotz_window), 1)[::-1]
    a.set_xticks(xticks)

    plt.draw()

    # dual time x axis
    if ia <= 1:
        a2 = a.twiny()
        a2.set_xlim(a.get_xlim())
        # xticks = a.get_xticks()
        a2.set_xticks(xticks)
        time_labels = [sim.cosmo_model.age(tick_zed).value for tick_zed in xticks]
        a2.set_xticklabels([f"{time:.2f}" for time in time_labels])
        a2.set_xlabel("Time, [Gyr]")

    if ia % 2 != 0:

        a.tick_params(
            axis="y",
            which="both",
            left=False,
            right=True,
            labelleft=False,
            labelright=True,
        )

    if ia > 1:

        a.tick_params(
            axis="y",
            which="both",
            labeltop=False,
            top=False,
        )

    # a.set_xlim(8,3.5)

    # a.
    # a.invert_xaxis()

    a.tick_params(axis="both", which="both", direction="in")

    a.grid()

ax[0, 0].legend(framealpha=0.0)


fig.savefig("yd_mdot_model_comparison.png")
