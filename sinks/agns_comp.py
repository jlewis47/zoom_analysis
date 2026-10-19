from gremlin.read_sim_params import ramses_sim
from sink_reader import get_sink_mhistory, find_massive_sink
from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt
import os
import numpy as np
from hagn.utils import get_hagn_sim
from hagn.IO import read_hagn_snap_brickfile, get_hagn_brickfile_stpids
from hagn.tree_reader import read_tree_rev
from hagn.catalogues import make_super_cat, get_cat_hids
from astropy.cosmology import z_at_value
from astropy import units as u
from zoom_analysis.constants import ramses_pc
from zoom_analysis.zoom_helpers import decentre_coordinates


tgt_zed = 2.0

hagn_snap = 197  # z=2 in the full res box
# HAGN_gal_dir = f"/data40b/Horizon-AGN/TREE_STARS/GAL_{hagn_snap:05d}"
# sim_ids = ["id242704", "id147479", "id147479_highnstar_nbh"]  # "id292074"
sim_ids = [
    # "id242704_old",
    "id242704",
    # "id242704_coarse",
    # "id242704_lesscoarse",
    # "id242704_evenlesscoarse",
    # "id242704_coarse",
    "id242756",
    # "id242756_coarse",
    # "id180130",
    # "id180130_evenlesscoarse",
    "id180130_leastcoarse",
    # "id21892",
    "id21892_leastcoarse",
    # "id242704_eagn_T0p15",
    # "id147479_old",
    "id74099",
    # "id147479",
    # "id74099_inter",
    #    "id147479_high_nsink"
]  # "id292074"
# sim_ids = ["id242704", "id147479", "id74099"]  # "id292074"
sdirs = [
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12",
]

super_cat = make_super_cat(
    197, outf="/data101/jlewis/hagn/super_cats"
)  # , overwrite=True)

fig = plt.figure()
ax = fig.add_subplot(111)

hagn_sim = get_hagn_sim()
hagn_sim.init_cosmo()

l_hagn = hagn_sim.unit_l(hagn_sim.aexp_stt) / (ramses_pc * 1e6) / hagn_sim.aexp_stt

last_hagn_id = -1

zoom_ls = ["--", ":", "-."]
zoom_style = 0


def follow_halo_sinks(
    sim,
    sim_snaps,
    sim_aexps,
    sim_times,
    hagn_tree_aexps,
    hagn_tree_times,
    hagn_ctr,
    hagn_rvir,
    bh_mass,
    bh_max_mass,
):
    for inode, node_aexp in enumerate(hagn_tree_aexps):
        snap_dist = np.abs(hagn_tree_times[inode] - sim_times)
        if np.all(snap_dist > 15):
            # if debug:
            # print("no close enough snaps")
            continue

        # print(sim_times, hagn_tree_times[inode], np.min(snap_dist))

        tree_arg = np.argmin(snap_dist)

        cur_snap = sim_snaps[tree_arg]
        cur_aexp = sim_aexps[tree_arg]

        # hagn_main_branch_id = hagn_tree_hids[np.argmin(np.abs(hagn_tree_aexps - cur_aexp))]

        ctr = hagn_ctr[:, inode]
        ctr = decentre_coordinates(ctr, sim.path)

        try:
            massive_sinks = find_massive_sink(
                # pos, snap, sim, rmax=sim.namelist["refine_params"]["rzoom"]
                # pos,
                ctr,
                cur_snap,
                sim,
                # rmax=0.3 * sim.namelist["refine_params"]["rzoom"],
                rmax=hagn_rvir[inode],
                all_sinks=True,  # get all in rmax
            )
        except ValueError:
            continue

        if len(massive_sinks["mass"]) > 0:
            bh_mass[inode] = np.sum(massive_sinks["mass"])
            bh_max_mass[inode] = np.max(massive_sinks["mass"])


for isim, (sdir, sim_name) in enumerate(zip(sdirs, sim_ids)):
    print(sim_name)

    sim_id = int(sim_name[2:].split("_")[0])

    gal_pties = get_cat_hids(super_cat, [sim_id])

    sim_path = os.path.join(sdir, sim_name)

    sim = ramses_sim(os.path.join(sdirs[0], sim_name), nml="cosmo.nml")

    sim_snaps = sim.snap_numbers
    sim_aexps = sim.get_snap_exps()

    sim_zeds = 1.0 / sim_aexps - 1.0

    sim.init_cosmo()
    sim_times = sim.cosmo_model.age(1.0 / sim_aexps - 1.0).value * 1e3

    hagn_snaps = hagn_sim.snap_numbers
    hagn_aexps = hagn_sim.get_snap_exps(param_save=False)
    hagn_zeds = 1.0 / hagn_aexps - 1.0
    hagn_sim.init_cosmo()
    hagn_times = hagn_sim.cosmo_model.age(1.0 / hagn_aexps - 1.0).value * 1e3

    pos = np.asarray(
        [
            sim.namelist["refine_params"]["xzoom"],
            sim.namelist["refine_params"]["yzoom"],
            sim.namelist["refine_params"]["zzoom"],
        ]
    )

    hagn_tree_hids, hagn_tree_datas, hagn_tree_aexps = read_tree_rev(
        tgt_zed,
        gal_pties["hid"],
        tree_type="halo",
        # gal_pties["gid"],
        # tree_type="gal",
        target_fields=["m", "x", "y", "z", "r"],
    )
    hagn_tree_times = (
        hagn_sim.cosmo_model.age(1.0 / hagn_tree_aexps - 1.0).value * 1e3
    )  # Myr#

    hagn_ctr = np.asarray(
        [
            hagn_tree_datas["x"][0],
            hagn_tree_datas["y"][0],
            hagn_tree_datas["z"][0],
        ]
    )

    hagn_ctr += 0.5 * (l_hagn * hagn_tree_aexps)
    hagn_ctr /= l_hagn * hagn_tree_aexps

    hagn_rvir = hagn_tree_datas["r"][0] / (l_hagn * hagn_tree_aexps)  # * 10

    bh_mass = np.zeros(len(hagn_tree_aexps))
    bh_max_mass = np.zeros(len(hagn_tree_aexps))
    hagn_tree_zeds = 1.0 / hagn_tree_aexps - 1

    # print(
    #     list(
    #         zip(
    #             1.0 / hagn_tree_aexps - 1.0,
    #             ["%.1e" % m for m in hagn_tree_datas["m"][0]],
    #             hagn_tree_datas["r"][0],
    #         )
    #     )
    # )

    # print(massive_sink)
    zeds = 1.0 / hagn_tree_aexps - 1

    # times = sim.cosmo_model.age(zeds).value * 1e3  # Myr

    # print(list(zip(central_sink["mass"], central_sink["zeds"])))
    if last_hagn_id != sim_id:

        last_hagn_id = sim_id

        # hagn_massive_sid = find_massive_sink(
        #     hagn_ctr[:, 0], hagn_snap, hagn_sim, rmax=hagn_rvir[0]
        # )["identity"]

        # # print()

        # hagn_sink_hist = get_sink_mhistory(hagn_massive_sid, hagn_snap, hagn_sim)

        # # print(list(zip(hagn_sink_hist["zeds"], hagn_sink_hist["mass"])))

        # zeds = hagn_sink_hist["zeds"]

        follow_halo_sinks(
            hagn_sim,
            hagn_snaps,
            hagn_aexps,
            hagn_times,
            hagn_tree_aexps,
            hagn_tree_times,
            hagn_ctr,
            hagn_rvir,
            bh_mass,
            bh_max_mass,
        )

        if not hasattr(hagn_sim, "cosmo_model"):
            hagn_sim.init_cosmo()

        times = hagn_sim.cosmo_model.age(zeds).value * 1e3  # Myr

        # (l,) = ax.plot(zeds, hagn_sink_hist["mass"], lw=3)
        (l,) = ax.plot(
            hagn_tree_zeds[bh_mass > 0],
            bh_mass[bh_mass > 0],
            lw=1,
        )

        ax.plot(
            hagn_tree_zeds[bh_max_mass > 0],
            bh_max_mass[bh_max_mass > 0],
            # label="",
            lw=3,
            # label=sim_name,
            c=l.get_color(),
        )

        last_hagn_id = sim_id

        zoom_style = 0

    else:
        zoom_style += 1

    bh_mass = np.zeros(len(hagn_tree_aexps))
    bh_max_mass = np.zeros(len(hagn_tree_aexps))

    follow_halo_sinks(
        sim,
        sim_snaps,
        sim_aexps,
        sim_times,
        hagn_tree_aexps,
        hagn_tree_times,
        hagn_ctr,
        hagn_rvir,
        bh_mass,
        bh_max_mass,
    )

    ax.plot(
        hagn_tree_zeds[bh_mass > 0],
        bh_mass[bh_mass > 0],
        label="",
        lw=1,
        ls=zoom_ls[zoom_style],
        c=l.get_color(),
    )

    ax.plot(
        hagn_tree_zeds[bh_max_mass > 0],
        bh_max_mass[bh_max_mass > 0],
        # label="",
        lw=3,
        ls=zoom_ls[zoom_style],
        color=l.get_color(),
        label=sim_name,
    )

    # gal_pties = get_tgt_HAGN_pties(hids=[intID])

    # now get equivalent for hagn
    # hagn_snap = hagn_sim.snap_numbers[
    # np.argmin(np.abs(hagn_sim.get_snap_exps(param_save=False) - snap_aexp))
    # ]
    # hagn_gals = read_hagn_snap_brickfile(hagn_snap, hagn_sim)
    # pos, rvir, _ = get_hagn_brickfile_stpids(
    #     f"/data40b/Horizon-AGN/TREE_STARS/tree_bricks{hagn_snap:03d}",
    #     gal_pties["gid"],
    #     sim,
    # )


ax.plot([], [], lw=3, ls="-", label="HAGN", c="k")
ax.plot([], [], lw=3, ls="--", label="zoom max mass", c="k")
ax.plot([], [], lw=1.5, ls="--", label="zoom tot mass", c="k")


# second xaxis times...
ax.invert_xaxis()
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
xticks = ax.get_xticks()
ax2.set_xticks(xticks)
zticks = [z_at_value(sim.cosmo_model.age, (x + 1e-3) * u.Gyr).value for x in xticks]
ax2.set_xticklabels(["{:.2f}".format(z) for z in zticks])


# ax2.invert_xaxis()

ax.set_xlabel("z")
ax2.set_xlabel("time [Gyr]")

ax.set_ylabel("BH mass [M$_\odot$]")

ax.set_yscale("log")
# ax.grid()
ax.legend(ncol=2, loc="lower right", framealpha=0.0, fontsize=10)

ax.tick_params(direction="in", top=True, right=True, left=True)
ax2.tick_params(direction="in", top=True, right=True, left=True)

fig.savefig("sink_comp")
