# from f90nml import read
from zoom_analysis.stars import sfhs
from zoom_analysis.constants import ramses_pc

# from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory

from gremlin.read_sim_params import ramses_sim

import os
import numpy as np

from hagn.utils import get_hagn_sim, adaptahop_to_code_units
from hagn.tree_reader import read_tree_rev
from hagn.catalogues import get_halos_cat, make_super_cat, get_cat_hids


from hagn.catalogues import make_super_cat

from f90_tools.star_reader import read_part_ball_hagn, read_part_ball_YDBondi


def follow_hagn_halo(
    delta_t,
    l_hagn,
    sfh,
    sfr,
    ssfr,
    hagn_tree_datas,
    hagn_tree_aexps,
    sim,
    sim_snaps,
    sim_times,
    hagn_tree_times,
    part_ball_fct,
    debug=False,
):
    for inode, node_aexp in enumerate(hagn_tree_aexps):

        # print(inode, node_aexp)
        hagn_ctr = np.asarray(
            [
                hagn_tree_datas["x"][0][inode],
                hagn_tree_datas["y"][0][inode],
                hagn_tree_datas["z"][0][inode],
            ]
        )

        tree_snap_dist = np.abs(sim_times - hagn_tree_times[inode])
        if np.all(tree_snap_dist > 15):
            continue

        cur_snap = sim_snaps[np.argmin(tree_snap_dist)]

        # print(l_hagn)

        hagn_ctr += 0.5 * l_hagn * node_aexp
        hagn_ctr /= l_hagn * node_aexp

        hagn_rvir = hagn_tree_datas["r"][0][inode] / (l_hagn * node_aexp)  # * 3  # * 10

        # hagn_ctr = np.asarray([0.38182679, 0.14534359, 0.14002028])
        # hagn_rvir = 0.00585990

        if debug:
            print(hagn_ctr, hagn_rvir)

        # print(hagn_rvir)

        # stars = yt_read_star_ball(sim, cur_snap, hagn_ctr, hagn_rvir)

        stars = part_ball_fct(
            sim,
            cur_snap,
            hagn_ctr,
            hagn_rvir,
            # ["metallicity", "mass", "birth_time"],
            fam=2,
            debug=debug,
        )

        print(len(stars["mass"]))
        if len(stars["mass"]) == 0:
            continue

        # print(f"Found {stars['mass'].sum():.1e} Msun of stellar mass")

        # print(list(zip(t, sfh)))

        star_ages_filt = stars["age"] < delta_t

        if len(star_ages_filt) == 0:
            # print("no young stars")
            continue

        # sfr[inode] = stars["mass"][star_ages_filt].sum() / delta_t
        # ssfr[inode] = sfr[inode] / sfh[inode]

        # sfh[inode] = sfhs.get_sf_stuff(
        #     stars,
        #     tgt_zed,
        #     sim,
        #     deltaT=delta_t,
        # )[
        #     0
        # ][0]
        # _, sfr[inode], ssfr[inode], _, _ = sfhs.get_sf_stuff(
        #     young_stars,
        #     tgt_zed,
        #     sim,
        #     deltaT=delta_t,
        # )
        # sfh[inode] *= stars["mass"].sum() / young_stars["mass"].sum()
        # ssfr[inode] /= stars["mass"].sum() / young_stars["mass"].sum()

        # print(stars["age"].min(), stars["age"].max(), stars["age"].mean())

        stars["mass"] = sfhs.correct_mass(
            sim, stars["age"], stars["mass"], stars["metallicity"]
        )

        young_stars = {}
        for k in stars.keys():
            # print(len(stars[k]))
            young_stars[k] = stars[k][star_ages_filt]

        sfh[inode] = stars["mass"].sum()

        sfr[inode] = young_stars["mass"].sum() / delta_t

        ssfr[inode] = sfr[inode] / sfh[inode]

        if debug:
            print(f"Found {stars['mass'].sum():.1e} Msun of stellar mass")
            print(f"Found {young_stars['mass'].sum():.1e} Msun of young stellar mass")
            print(f"{sfh[inode]:.1e}, {sfr[inode]:.1e}, {ssfr[inode]:.1e}")


# sim_ids = [
#     "id2757_995pcnt_mstel/",
#     "id19782_500pcnt_mstel/",pip install setuptools
#     "id13600_005pcnt_mstel/",
# ]
# sdir = "/home/jlewis/sims/proposal_zooms/ExtractZoom/mh1e12/"
hagn_snap = 197  # z=2 in the full res box
HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
delta_t = 100  # Myr

sim_ids = [
    # "BondiNoVrel",
    # "ClassicBondi",
    "MeanBondi",
    # "MeanBondiVrelNonZero",
    # "NoAGN",
    # "id147479_low_nstar",
    # "id147479_high_nsink",
]
# sim_ids = ["id74099", "id147479", "id242704"]

super_cat = make_super_cat(
    197, outf="/data101/jlewis/hagn/super_cats"
)  # , overwrite=True)


sdirs = [
    "/data102/dubois/ZoomIMAGE/Dust",
    # "/data102/dubois/ZoomIMAGE/Dust",
    "/data102/dubois/ZoomIMAGE/Dust",
    "/data102/dubois/ZoomIMAGE/Dust",
    "/data102/dubois/ZoomIMAGE/Dust",
]


# setup plot
fig, ax = sfhs.setup_sfh_plot()


yax2 = None

hagn_sim = get_hagn_sim()

hagn_snaps = hagn_sim.snap_numbers
hagn_aexps = hagn_sim.get_snap_exps(param_save=False)

hagn_sim.init_cosmo()
hagn_times = hagn_sim.cosmo_model.age(1.0 / hagn_aexps - 1.0).value * 1e3

tgt_snap = hagn_sim.get_closest_snap(zed=2.0)
tgt_zed = 1.0 / hagn_aexps[hagn_snaps == tgt_snap] - 1
tgt_time = hagn_times[hagn_snaps == tgt_snap]


l_hagn = hagn_sim.cosmo["unit_l"] / (ramses_pc * 1e6) / hagn_sim.aexp_stt
# colors = []


sfr_min, ssfr_min, sfh_min = np.inf, np.inf, np.inf
sfr_max, ssfr_max, sfh_max = -np.inf, -np.inf, -np.inf

last_hagn_id = -1
isim = 0

zoom_ls = ["--", ":", "-.", (0, (3, 5, 1, 5, 1, 5))]

zoom_512_target = 33455, 1, 2.080e12, 0.87481546, 0.24813820, 0.29004672, 0.00275581
hid_512, hid_sub, hid_mvir, hid_x, hid_y, hid_z, hid_rvir = zoom_512_target

# find closest z=2 in supercat
# match mass and position
super_masses = super_cat["mhalo"]
super_x = super_cat["hx"]
super_y = super_cat["hy"]
super_z = super_cat["hz"]
super_rvir = super_cat["rvir"]

pos_dist = np.linalg.norm(
    np.array([super_x, super_y, super_z]).T - np.array([hid_x, hid_y, hid_z]), axis=1
)
mass_dist = np.abs(super_masses - hid_mvir)
rvir_dist = np.abs(super_rvir - hid_rvir)

# print(pos_dist, mass_dist, rvir_dist)
# print(np.argmin(pos_dist), np.argmin(mass_dist), np.argmin(rvir_dist))
combined_dist = (
    pos_dist / pos_dist.max()
) ** 2 + mass_dist / mass_dist.max()  # + rvir_dist
intID = super_cat["hid"][np.argmin(combined_dist)]

# intID = 2050
# c = "tab:blue"

for sim_id, sdir in zip(sim_ids, sdirs):

    # get the galaxy in HAGN
    # intID = int(sim_id[2:].split("_")[0])

    # print("halo id: ", intID)
    print(sim_id)

    gal_pties = get_cat_hids(super_cat, [intID])

    # print(gal_pties)

    hagn_tree_hids, hagn_tree_datas, hagn_tree_aexps = read_tree_rev(
        tgt_zed,
        gal_pties["hid"],
        tree_type="halo",
        target_fields=["m", "x", "y", "z", "r"],
        sim="hagn",
    )
    hagn_tree_times = (
        hagn_sim.cosmo_model.age(1.0 / hagn_tree_aexps - 1.0).value * 1e3
    )  # Myr#

    sim = ramses_sim(os.path.join(sdir, sim_id), nml="cosmo.nml")

    # print(
    #     sim.namelist["refine_params"]["xzoom"],
    #     sim.namelist["refine_params"]["yzoom"],
    #     sim.namelist["refine_params"]["zzoom"],
    # )

    if last_hagn_id != intID:

        zoom_style = 0
        last_hagn_id = intID

        # stars = gid_to_stars(
        #     gal_pties["gid"], hagn_snap, hagn_sim, ["mass", "birth_time", "metallicity"]
        # )

        # print(stars["mass"].min())

        sfh = np.zeros_like(hagn_tree_times, dtype=float)
        sfr = np.zeros_like(hagn_tree_times, dtype=float)
        ssfr = np.zeros_like(hagn_tree_times, dtype=float)

        follow_hagn_halo(
            delta_t,
            l_hagn,
            sfh,
            sfr,
            ssfr,
            hagn_tree_datas,
            hagn_tree_aexps,
            hagn_sim,
            hagn_snaps,
            hagn_times,
            hagn_tree_times,
            read_part_ball_hagn,
            debug=False,
        )

        # sfh, sfr, ssfr, t, cosmo = sfhs.get_sf_stuff(stars, tgt_zed, hagn_sim)

        sfr_min = min(sfr[sfr > 0].min(), sfr_min)
        sfr_max = max(sfr.max(), sfr_max)
        ssfr_min = min(ssfr[ssfr > 0].min(), ssfr_min)
        ssfr_max = max(ssfr.max(), ssfr_max)
        sfh_min = min(sfh[sfh > 0].min(), sfh_min)
        sfh_max = max(sfh.max(), sfh_max)

        # # print(sim_id)
        # # print(gal_pties["masses"], sfh[0])
        # # print(gal_pties["sfrs"], sfr[0])
        # # print(gal_pties["ssfrs"], ssfr[0])

        # l = sfhs.plot_sf_stuff(
        #     ax,
        #     sfh,
        #     sfr,
        #     ssfr,
        #     t,
        #     0,
        #     cosmo,
        #     # label="id74099",
        #     lw=3,
        # )

        l = sfhs.plot_sf_stuff(
            ax,
            sfh[sfh > 0],
            sfr[sfh > 0],
            ssfr[sfh > 0],
            hagn_tree_times[sfh > 0],
            0,
            hagn_sim.cosmo_model,
            # label=sim_id + " zoom",
            # label=sim_id,
            # ls=zoom_ls[zoom_style],
            # ticks=sim_id == sim_ids[-1],
            lw=3,
            # ticks=True,
            # marker="o",
        )

        c = l[0].get_color()

        # print(c)

    else:
        zoom_style += 1

    sim_snaps = sim.snap_numbers
    sim_aexps = sim.get_snap_exps(param_save=False)

    sim.init_cosmo()
    sim_times = sim.cosmo_model.age(1.0 / sim_aexps - 1.0).value * 1e3

    sfh = np.zeros_like(hagn_tree_times, dtype=float)
    sfr = np.zeros_like(hagn_tree_times, dtype=float)
    ssfr = np.zeros_like(hagn_tree_times, dtype=float)

    follow_hagn_halo(
        delta_t,
        l_hagn,
        sfh,
        sfr,
        ssfr,
        hagn_tree_datas,
        hagn_tree_aexps,
        sim,
        sim_snaps,
        sim_times,
        hagn_tree_times,
        read_part_ball_YDBondi,
        debug=False,
    )

    if np.sum(sfr > 0) == 0:
        # print("skip")
        continue

    sfr_min = min(sfr[sfr > 0].min(), sfr_min)
    sfr_max = max(sfr.max(), sfr_max)
    ssfr_min = min(ssfr[ssfr > 0].min(), ssfr_min)
    ssfr_max = max(ssfr.max(), ssfr_max)
    sfh_min = min(sfh[sfh > 0].min(), sfh_min)
    sfh_max = max(sfh.max(), sfh_max)

    #    print(list(zip(t, sfh)))

    sfhs.plot_sf_stuff(
        ax,
        sfh[sfh > 0],
        sfr[sfh > 0],
        ssfr[sfh > 0],
        hagn_tree_times[sfh > 0],
        0,
        sim.cosmo_model,
        color=c,
        # label=sim_id + " zoom",
        label=sim_id,
        ls=zoom_ls[zoom_style],
        ticks=sim_id == sim_ids[-1],
        lw=3,
        # ticks=True,
        # marker="o",
    )


ax[0].set_ylim(ssfr_min * 0.5, ssfr_max * 1.5)
ax[1].set_ylim(sfr_min * 0.5, sfr_max * 1.5)
ax[2].set_ylim(sfh_min * 0.5, sfh_max * 1.5)

# add key to existing legend

ax[-1].plot([], [], "k-", label="HAGN")
ax[-1].plot([], [], "k--", label="zoom")

ax[-1].legend(framealpha=0.0, ncol=3)
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
    a.tick_params(
        axis="both",
        which="both",
        bottom=True,
        top=True,
        left=True,
        right=True,
        direction="in",
    )

# add 2nd yaxis to last plot
# y2 = ax[-1].twiny()
# y2.set_xlim(ax[0].get_xlim())

fig.savefig("sfhs_comp_bondi_HAGNhalotree.png")
