# from f90nml import read
from zoom_analysis.sinks import sink_histories
from zoom_analysis.constants import ramses_pc

# from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

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

from hagn.catalogues import make_super_cat


# sim_ids = [
#     "id2757_995pcnt_mstel/",
#     "id19782_500pcnt_mstel/",pip install setuptools
#     "id13600_005pcnt_mstel/",
# ]
# sdir = "/home/jlewis/sims/proposal_zooms/ExtractZoom/mh1e12/"
hagn_snap = 197  # z=2 in the full res box
HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
delta_t = 100  # Myr

explr_path = f"/data101/jlewis/sims/dust_fid/lvlmax_20/"
explr_dirs = [
    os.path.join(explr_path, d)
    for d in os.listdir(explr_path)
    if os.path.isdir(os.path.join(explr_path, d))
]

# print(explr_dirs)


def go_down_for_ids(explr_dirs):

    # print(dirs)

    names = []

    for explr_dir in explr_dirs:
        # print("looking in %s" % explr_dir)
        if explr_dir.split("/")[-1].strip().startswith("id"):
            names.append(explr_dir)
        else:
            next_dirs = [
                os.path.join(explr_dir, d)
                for d in os.listdir(explr_dir)
                if os.path.isdir(os.path.join(explr_dir, d))
            ]
            if len(next_dirs) > 0:
                names.extend([d for d in go_down_for_ids(next_dirs)])

    return names
    # return [n.split("/")[-1].strip() for n in names]


sdirs = np.asarray(go_down_for_ids(explr_dirs))
# sim_ids = np.asarray([d.split("/")[-1].strip() for d in sdirs])
sim_ids = np.asarray(
    ["id242704", "id242756", "id180130", "id21892", "id74099", "id292074"]
)
colours = np.asarray(
    [
        "tab:grey",
        "tab:olive",
        "tab:red",
        "tab:pink",
        "tab:green",
        "tab:cyan",
        # "tab:orange",
        # "tab:green",
    ]
)
names = np.asarray([d[2:].split("_")[0] for d in sim_ids])
u_names, args = np.unique(names, return_index=True)
sdirs = sdirs[args]
sim_ids = sim_ids[args]
colours = colours[args]
# print(sim_names)

# sim_ids = ["id74099", "id147479", "id242704"]

super_cat = make_super_cat(
    197, outf="/data101/jlewis/hagn/super_cats"
)  # , overwrite=True)


# setup plot
fig, ax = sink_histories.setup_bh_plot()


yax2 = None

hagn_sim = get_hagn_sim()
hagn_sim.init_cosmo()
# colors = []

hagn_snaps = hagn_sim.snap_numbers
hagn_aexps = hagn_sim.get_snap_exps(param_save=False)
hagn_times = hagn_sim.cosmo_model.age(1.0 / hagn_aexps - 1.0).value * 1e3

tgt_zed = 1.0 / hagn_aexps[hagn_snaps == hagn_snap] - 1.0
tgt_time = cosmo.age(tgt_zed).value

l_hagn = (
    hagn_sim.unit_l(hagn_sim.aexp_stt)
    / (ramses_pc * 1e6)
    / hagn_sim.aexp_stt
    * 1.0
    / (1.0 + tgt_zed)
)

last_hagn_id = -1
isim = 0

zoom_ls = ["--", ":", "-."]

# c = "tab:blue"

# print(sim_ids)

for isim, sim_id in enumerate(sim_ids):

    # get the galaxy in HAGN
    intID = int(sim_id[2:].split("_")[0])

    # print("halo id: ", intID)
    print(sim_id)

    gal_pties = get_cat_hids(super_cat, [intID])

    # hagn_tree_hids, hagn_tree_datas, hagn_tree_aexps = read_tree_rev(
    #     tgt_zed,
    #     gal_pties["hid"],
    #     tree_type="halo",
    #     target_fields=["m", "x", "y", "z", "r"],
    # )
    # hagn_tree_times = (
    #     hagn_sim.cosmo_model.age(1.0 / hagn_tree_aexps - 1.0).value * 1e3
    # )  # Myr#

    # print(last_hagn_id != sim_id)

    if last_hagn_id != sim_id.split("_")[0]:

        zoom_style = 0
        last_hagn_id = sim_id.split("_")[0]

        mbh, dmbh, t_sink, fmerger, t_merge = sink_histories.get_bh_stuff(
            hagn_sim, intID, tgt_zed
        )

        if np.any(mbh == -1):
            continue

        # # print(sim_id)
        # # print(gal_pties["masses"], sfh[0])
        # # print(gal_pties["sfrs"], sfr[0])
        # # print(gal_pties["ssfrs"], ssfr[0])

        l = sink_histories.plot_bh_stuff(
            ax,
            mbh,
            dmbh,
            t_sink,
            fmerger,
            t_merge,
            0,
            cosmo,
            label=sim_id.split("_")[0],
            lw=3,
            color=colours[isim],
        )

        c = l[0].get_color()

        # print(c)

    else:
        zoom_style += 1


# add key to existing legend

# ax[-1].plot([], [], "k-", label="HAGN")
# ax[-1].plot([], [], "k--", label="zoom")

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
        # top=True,
        top=False,
        left=True,
        right=True,
        direction="in",
    )

y2 = ax[0].twiny()
xlim = ax[-1].get_xlim()
y2.set_xlim(xlim)
# y2.set_xticks(ax[0].get_xticks())
y2.set_xticklabels([f"{cosmo.age(zed).value:.1f}" for zed in ax[0].get_xticks()])
y2.set_xlabel("redshift")


fig.savefig("sinks_ics_comp.png")
