# from f90nml import read
from zoom_analysis.read.read_data import read_data_ball
from zoom_analysis.sinks import sink_histories, sink_reader
from zoom_analysis.sinks.sink_constraints import calculate_bennert21_bmh2disp
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

from zoom_analysis.zoom_helpers import check_in_all_sims_vol

# from hagn.catalogues import make_super_cat

# fig, ax = sink_histories.setup_bh_plot()


tgt_zed = 4.5
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
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE_highAGNtherm/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE/",
    # "/data103/jlewis/sims/lvlmax_22/mh1e12/id180130/",
    # "/data103/jlewis/sims/lvlmax_21/mh1e12/id180130_256/",
    # "/data102/jlewis/sims/lvlmax_21/mh1e12/id180130_model6_eps0p05/",
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
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",  # _leastcoarse",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE",  # _leastcoarse",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_lowerSFE_NHboost",  # _leastcoarse",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_pdrag",
]

# list of all available pyplot markers:
# https://matplotlib.org/stable/api/markers_api.html
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


st_fig, st_ax = plt.subplots(1, 1, figsize=(8, 8))
gas_fig, gas_ax = plt.subplots(1, 1, figsize=(8, 8))

disp_bins = np.logspace(-1,3,10)

for isim, sim_dir in enumerate(sim_dirs):

    sim = ramses_sim(sim_dir, nml="cosmo.nml")

    name = sim.name

    # print(name)

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
    decal = 0

    z_dists = np.abs(1./sim.aexps - 1.0 - tgt_zed)
    args = np.argsort(z_dists)



    while not found and decal > -(len(sim.snap_numbers) - 1):
        arg = args[decal]
        sim_snap = sim.snap_numbers[arg]
        sim_zed = 1.0 / sim.aexps[arg] - 1

        gfile = get_gal_assoc_file(sim_dir, sim_snap)

        found_cond = os.path.isfile(gfile)
        # print(sim_snap, sim_zed, decal, gfile)


        if found_cond:
            found = True
        else:
            decal +=1

    if not found:
        continue

    gal_dict = get_gal_props_snap(sim.path, sim_snap, main_stars=False)
    # print(gal_dict.keys())

    gal_pos = gal_dict["pos"].T

    vol_args,min_vol = check_in_all_sims_vol(gal_pos, sim, sim_dirs)


    # if vol_args.sum() == 0:
    #     raise ValueError("No galaxies in volume")

    # print(isim, vol_args.sum())

    # centred_gal_pos = decentre_coordinates(gal_dict["pos"].T,sim.path)

    # vol_args = [check_vol(decentred_gal_pos) for check_vol in check_vols]

    # vol_args = np.all(vol_args, axis=0)

    host_purity = gal_dict["host purity"]
    central_cond = gal_dict["central"] == 1
    pure_cond = (host_purity > fpure) * central_cond * vol_args
    gal_mass = gal_dict["mass"][pure_cond]
    gal_pos = gal_dict["pos"][:, pure_cond]
    r50 = gal_dict["r50"][pure_cond]*0.5
    rmax = gal_dict["rmax"][pure_cond]
    gids = gal_dict["gids"][pure_cond]
    host_hids = gal_dict["host hid"][pure_cond]

    print(f"I found {pure_cond.sum():d} galaxies with purity > {fpure:.4f},")
    print(f"and {len(host_purity)-pure_cond.sum()} galaxies with purity < {fpure:.4f}")

    sink_mass = np.zeros_like(gal_mass)
    st_disp = np.zeros_like(gal_mass)
    gas_disp= np.zeros_like(gal_mass)

    for igal, gid in enumerate(gids):
        try:
            sid, found = sink_reader.gid_to_sid(sim, gid, sim_snap)
        except AssertionError:
            found = False

        if not found:
            continue

        host_hid = host_hids[igal]

        try:
            datas=read_data_ball(sim, sim_snap, gal_pos.T[igal],r50[igal], host_hid, gid=gid, data_types=['gas','stars'],tgt_fields=['ilevel','density','velocity_x','velocity_y','velocity_z','vel','velocities','mpart','mass'])
        except KeyError:
            continue

        stars = datas['stars']
        if stars!=None:
            st_mkey='mpart' if 'mpart' in stars.keys() else 'mass'
            st_avg_vels2 = np.average(stars['vel']**2,axis=0, weights=stars[st_mkey]**2)
            st_avg2_vels = np.average(stars['vel'],axis=0, weights=stars[st_mkey])**2
            st_disp[igal] = np.sqrt(np.sum(st_avg_vels2 - st_avg2_vels))

        gas = datas['gas']
        if gas!=None:
            gas_masses = gas['density'] * (((sim.cosmo.lcMpc*ramses_pc*1e2*1e6)/(2**gas['ilevel'])) / 2e33) # in Msun
            gas_vel = np.transpose([gas['velocity_x'],gas['velocity_y'],gas['velocity_z']])/1e5
            gas_avg_vels2 = np.average(gas_vel**2,axis=0, weights=gas_masses)
            gas_avg2_vels = np.average(gas_vel,axis=0, weights=gas_masses)**2
            gas_disp[igal] = np.sqrt(np.sum(gas_avg_vels2 - gas_avg2_vels))


        sink_props = sink_reader.get_sink(sid, sim_snap, sim)
        if "mass" in sink_props:
            sink_mass[igal] = sink_props["mass"]

        print(igal, st_disp[igal], gas_disp[igal], sink_mass[igal])
        

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



    sca = st_ax.scatter(st_disp, sink_mass, label=label, alpha=0.5, marker=markers[isim])
    gas_ax.scatter(gas_disp, sink_mass, label=label, alpha=0.5, marker=markers[isim])

    pos = sink_mass > 0

    if pos.sum() == 0:
        continue


    avg, _, _ = binned_statistic(
        gas_disp[pos], sink_mass[pos], bins=disp_bins, statistic="mean"
    )
    med_bin, _, _ = binned_statistic(
        gas_disp[pos], gas_disp[pos], bins=disp_bins, statistic="median"
    )
    gas_ax.plot(med_bin, avg, c=sca.get_facecolor(), lw=1, ls="--")
    avg, _, _ = binned_statistic(
        st_disp[pos], sink_mass[pos], bins=disp_bins, statistic="mean"
    )
    med_bin, _, _ = binned_statistic(
        st_disp[pos], st_disp[pos], bins=disp_bins, statistic="median"
    )
    st_ax.plot(med_bin, avg, c=sca.get_facecolor(), lw=1, ls="--")


# #now for HAGN

# hagn_sim = get_hagn_sim()

# hagn_snap = hagn_sim.get_closest_snap(zed=tgt_zed)
# hagn_aexp = hagn_sim.get_snap_exps(hagn_snap)[0]
# hagn_zed = 1.0 / hagn_aexp - 1

# hagn_gals = make_super_cat(hagn_snap)

# # print(hagn_gals.keys())

# hagn_gal_pos = np.transpose([hagn_gals["x"], hagn_gals["y"], hagn_gals["z"]])

# vol_args,min_vol = check_in_all_sims_vol(hagn_gal_pos, hagn_sim, sim_dirs)

# if vol_args.sum() > 0:
#     mbhs = hagn_gals['mbh'][vol_args]
#     mgals= hagn_gals['mgal'][vol_args]


#     ax.scatter(mgals, mbhs, label=f"HAGN, z={hagn_zed:.2f}", alpha=0.5, marker="x",color='k')

#     #now avg stats
#     avg, _, _ = binned_statistic(
#         mgals, mbhs, bins=gal_mass_bins, statistic=np.nanmean
#     )
#     med_bin, _, _ = binned_statistic(
#         mgals, mgals, bins=gal_mass_bins, statistic=np.nanmedian
#     )
#     ax.plot(med_bin, avg, c="k", lw=1, ls="--")#,label=f"HAGN, z={hagn_zed:.2f}")
gas_ax.set_xlabel("Gas dispersion [km/s]")
st_ax.set_xlabel("Stellar dispersion [km/s]")

for fig,ax in zip([st_fig,gas_fig],[st_ax, gas_ax]):

    bennert21_mass = calculate_bennert21_bmh2disp(disp_bins)

    ax.plot(disp_bins[disp_bins>10], bennert21_mass[disp_bins>10], c="k", ls="--", label="Bennert+21", lw=1)

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

    ax.set_ylabel("BH Mass [M$_\odot$]")

    ax.set_yscale("log")
    ax.set_xscale("log")
    title_txt = ""
    if len(sim_dirs) == 1:
        title_txt += f"z={sim_zed:.2f}"
        # ax.text(0.05, 0.95, f"z={sim_zed:.2f}", transform=ax.transAxes)
    # also print purity threshold
    # ax.text(0.05, 0.9, f"fpure > {fpure:.3f}", transform=ax.transAxes)
    title_txt += f" fpure > {fpure:.4f}"

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

    ax.set_xlim(disp_bins[0], disp_bins[-1])

if tgt_zed != None:
    gas_fig.savefig(f"zoom_BHdisp_gas_z{tgt_zed:.2f}.png")
    st_fig.savefig(f"zoom_BHdisp_stars_z{tgt_zed:.2f}.png")
else:
    gas_fig.savefig("zoom_BHdisp_gas.png")
    st_fig.savefig("zoom_BHdisp_stars.png")
