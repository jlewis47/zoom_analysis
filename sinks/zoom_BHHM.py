# from f90nml import read
from zoom_analysis.sinks import sink_histories, sink_reader
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import find_zoom_tgt_halo

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
from astropy.cosmology import z_at_value
from astropy import units as u

# from hagn.catalogues import make_super_cat

# fig, ax = sink_histories.setup_bh_plot()

tgt_zed = 2.0

HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
delta_t = 100  # Myr

# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099"

sim = ramses_sim(sim_dir, nml="cosmo.nml")

name = sim.name

intID = int(sim.name.split("_")[-1][2:])


# hagn_sim = get_hagn_sim()
# hagn_snap = hagn_sim.get_closest_snap(zed=tgt_zed)

# # super_cat = make_super_cat(
# #     hagn_snap, outf="/data101/jlewis/hagn/super_cats"
# # )  # , overwrite=True)


# # gal_pties = get_cat_hids(super_cat, [intID])

# hagn_zed = 1.0 / hagn_sim.aexps[hagn_sim.snap_numbers == hagn_snap][0] - 1

# mbh, dmbh, t_sink, fmerger, t_merge = sink_histories.get_bh_stuff(
#     hagn_sim, intID, hagn_zed
# )

# l = sink_histories.plot_bh_stuff(
#     ax,
#     mbh,
#     dmbh,
#     t_sink,
#     fmerger,
#     t_merge,
#     0,
#     cosmo,
#     label=name,
#     lw=1,
# )


# c = l[0].get_color()

sim_snap = sim.snap_numbers[-1]
sim_zed = 1.0 / sim.aexps[-1] - 1
hid, halo_dict, halo_gals = find_zoom_tgt_halo(sim, sim_snap)

hpos = halo_dict["pos"]
hrvir = halo_dict["rvir"]

sid = sink_reader.find_massive_sink(hpos, sim_snap, sim, hrvir)["identity"]

sink_dict = sink_reader.get_sink_mhistory(sid, sim_snap, sim)

sim.init_cosmo()
sim_time = sim.cosmo_model.age(sink_dict["zeds"]).value * 1e3

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# colors = plt.cm.viridis(np.linspace(0, 1, len(sink_dict["times"])))

ax.scatter(sim_time, sink_dict["mass"], label=name)  # , c=colors)

ax.legend(framealpha=0.0, ncol=3)
# cur_leg = ax.get_legend()
# ax.legend(
#     cur_leg.legendHandles
#     + [
#         Line2D([0], [0], color="k", linestyle="-"),
#         Line2D([0], [0], color="k", ls="--"),
#     ],
#     sim_ids + ["HAGN", "zoom"],
# )


ax.tick_params(
    axis="both",
    which="both",
    bottom=True,
    # top=True,
    top=False,
    left=True,
    right=True,
    direction="in",
)

ax.grid()

ax.set_xlabel("Time [Myr]")
ax.set_ylabel("BH Mass [M$_\odot$]")

ax.set_yscale("log")

# y2 = ax.twiny()
# # y2.set_xlim(ax.get_xlim())
# # y2.set_xticks(ax.get_xticks())
# # xlim = ax.get_xlim()
# y2.set_xticklabels(
#     [
#         "%.1f" % z_at_value(sim.cosmo_model.age, time_label * u.Gyr, zmax=np.inf)
#         for time_label in ax.get_xticks()
#     ]
# )
# y2.set_xlabel("redshift")


fig.savefig(f"zoom_{name}_BHSMR.png")
