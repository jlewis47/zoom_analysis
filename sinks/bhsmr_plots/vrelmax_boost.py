# from f90nml import read
from zoom_analysis.sinks import sink_histories, sink_reader
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import get_gal_assoc_file, get_gal_props_snap

# from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# from scipy.spatial import cKDTree
from scipy.stats import binned_statistic

import os
import numpy as np


tgt_zed = 5.0
fpure = 1.0 - 1e-4

HAGN_gal_dir = f"/data40b/Horizon-AGN/STARS"
delta_t = 100  # Myr

sim_dirs = [
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE_highAGNtherm/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highSFE_lowSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE_lowSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_low_dboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_minMassDynGrav",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowSFE_DynBondiGravSinkMass",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_lowerSFE_drag/",
    # "/data102/jlewis/sims/lvlmax20/mh1e12/id180130_superEdd_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_accBoost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE_lowNsink/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_pdrag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_noAGN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_high_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nrg_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_pdrag",
]

markers = [
    "o",
    "v",
    "^",
    "<",
    ">",
    "s",
    "p",
    "P",
    "*",
    "h",
    "H",
    "X",
    "D",
    "d",
    "|",
    "_",
]


fig, ax = plt.subplots(1, 1, figsize=(8, 8))

nbins = 6

mstel_bins = np.logspace(6, 11, nbins)

avg_mstellar = np.zeros((nbins - 1, len(sim_dirs)))
std_mstellar = np.zeros((nbins - 1, len(sim_dirs)))
med_mstellar = np.zeros((nbins - 1, len(sim_dirs)))

for isim, sim_dir in enumerate(sim_dirs):

    sim = ramses_sim(sim_dir, nml="cosmo.nml")

    name = sim.name

    print(name)

    intID = int(sim.name.split("_")[0][2:])

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
    # find last assoc_file
    found = False
    decal = -1

    while not found:
        sim_snap = sim.snap_numbers[decal]
        sim_zed = 1.0 / sim.aexps[decal] - 1

        gfile = get_gal_assoc_file(sim_dir, sim_snap)

        found_cond = os.path.isfile(gfile)
        # print(sim_snap, decal, gfile)
        if tgt_zed != None:
            found_cond = found_cond and np.abs(sim_zed - tgt_zed) < 0.25

        if found_cond:
            found = True
        else:
            decal -= 1

    gal_dict = get_gal_props_snap(sim.path, sim_snap, main_stars=False)

    # print(gal_dict.keys())

    host_purity = gal_dict["host purity"]
    central_cond = gal_dict["central"] == 1
    pure_cond = (host_purity > fpure) * central_cond
    gal_mass = gal_dict["mass"][pure_cond]
    gal_pos = gal_dict["pos"][:, pure_cond]
    rmax = gal_dict["rmax"][pure_cond]
    gids = gal_dict["gids"][pure_cond]

    print(f"I found {pure_cond.sum():d} galaxies with purity > {fpure:f},")
    print(f"and {len(host_purity)-pure_cond.sum()} galaxies with purity < {fpure:f}")

    sink_mass = np.zeros_like(gal_mass)

    for igal, gid in enumerate(gids):

        sid, found = sink_reader.gid_to_sid(sim, gid, sim_snap)

        if not found:
            continue

        sink_props = sink_reader.get_sink(sid, sim_snap, sim)
        if "mass" in sink_props:
            sink_mass[igal] = sink_props["mass"]

    # for igal, (m, pos, rmax) in enumerate(zip(gal_mass, gal_pos.T, rmax)):
    # print(m, pos, rmax)

    # try:
    #     sink_props = sink_reader.find_massive_sink(pos, sim_snap, sim, rmax)

    #     # sink_dict = sink_reader.get_sink_mhistory(sid, sim_snap, sim)
    #     sink_mass[igal] = sink_props["mass"]
    # except ValueError:
    #     print("No sink found")

    label = sim.name
    if len(sim_dirs) > 1:
        label += f", z={sim_zed:.2f}"

    mseed = sim.namelist["smbh_params"]["mseed"]

    # only plot galaxies with BHs
    wbh = sink_mass > 0

    sca = ax.scatter(
        gal_mass[wbh],
        sink_mass[wbh] / mseed,
        label=label,
        alpha=0.5,
        marker=markers[isim],
    )
    avg, _, _ = binned_statistic(
        gal_mass[wbh], sink_mass[wbh] / mseed, bins=mstel_bins, statistic="mean"
    )
    std, _, _ = binned_statistic(
        gal_mass[wbh], sink_mass[wbh] / mseed, bins=mstel_bins, statistic="std"
    )
    med_bin, _, _ = binned_statistic(
        gal_mass[wbh], gal_mass[wbh], bins=mstel_bins, statistic="median"
    )
    # ax.plot(med_bin, avg, c=sca.get_facecolor(), lw=1, ls="--")

    avg_mstellar[:, isim] = avg
    std_mstellar[:, isim] = std
    med_mstellar[:, isim] = med_bin

med_bins = np.median(med_mstellar, axis=1)

ax.plot(
    # med_bin + np.diff(mstel_bins) * 0.5,
    med_bins,
    avg_mstellar[:, 1] / avg_mstellar[:, 0],
    lw=2,
    ls="-",
    c="k",
    label="ratio",
)

# ax.plot(
#     # med_bin + np.diff(mstel_bins) * 0.5,
#     med_bins,
#     std_mstellar[:, 1] / std_mstellar[:, 0],
#     lw=2,
#     ls="--",
#     c="k",
#     label="std",
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

ax.set_xlabel("Stellar mass [M$_\odot$]")
ax.set_ylabel("BH mass ratio")

# ax.set_yscale("log")
ax.set_xscale("log")
title_txt = ""
if len(sim_dirs) == 1:
    title_txt += f"z={sim_zed:.2f}"
    # ax.text(0.05, 0.95, f"z={sim_zed:.2f}", transform=ax.transAxes)
# also print purity threshold
# ax.text(0.05, 0.9, f"fpure > {fpure:.3f}", transform=ax.transAxes)
title_txt += f" fpure > {fpure:.3f}"

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

# outdir = "./bhsmr_plots"
# if not os.path.exists(outdir):
# os.makedirs(outdir)

# NH_smbh(ax)

ax.legend(framealpha=0.0, title=title_txt)
if tgt_zed != None:
    fig.savefig(f"zoom_bhmass_avg_ratio_z{tgt_zed:.2f}.png")
else:
    fig.savefig("zoom_bhmass_avg_ratio.png")
