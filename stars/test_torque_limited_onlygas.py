from matplotlib.colors import LogNorm
from scipy.spatial import KDTree

# from scipy.stats import binned_statistic_2d
from zoom_analysis.read.read_data import read_data_ball
from zoom_analysis.sinks.sink_reader import (
    read_sink_bin,
    snap_to_coarse_step,
    convert_sink_units,
)
from zoom_analysis.stars import sfhs

# from zoom_analysis.zoom_helpers import decentre_coordinates
# from zoom_analysis.stars.star_reader import yt_read_star_ball, read_part_ball
from zoom_analysis.stars.leja_quench_fit import sfr_ridge_leja22
from zoom_analysis.halo_maker.read_treebricks import (
    # read_brickfile,
    # convert_brick_units,
    # convert_star_units,
    read_zoom_stars,
)
from zoom_analysis.trees.tree_reader import read_tree_file_rev as read_tree_fev_sim
from zoom_analysis.constants import ramses_c, ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import (
    # find_zoom_tgt_gal,
    # find_zoom_tgt_halo,
    get_assoc_pties_in_tree,
    get_central_gal_for_hid,
    get_gal_assoc_file,
    find_snaps_with_gals,
    # smooth_radii_tree,
    smooth_props,
)

from dynamics import *


from zoom_analysis.zoom_helpers import find_starting_position

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


import os
import numpy as np
import h5py

from hagn.association import gid_to_stars
from hagn.utils import get_hagn_sim, adaptahop_to_code_units
from hagn.tree_reader import read_tree_rev, interpolate_tree_position
from hagn.catalogues import get_halos_cat, make_super_cat, get_cat_hids, get_cat_gids

from astropy.cosmology import Planck18 as cosmo

from hagn.catalogues import make_super_cat, get_cat_hids

from f90_tools.star_reader import read_part_ball_hagn, read_part_ball_NCdust

from zoom_analysis.sinks.sink_histories import find_massive_sink


def torque_limited(eta_bh, alphaT, fdisk, Mbh, Mdisk, Mgas, R0):
    """
    Masses in Msun
    R0 in pc
    alphaT is a normalisation factor between 1 and 10, for unresolved inflow rates; Alcazar+13 use 5
    """

    f0 = 0.31 * fdisk**2 * (Mdisk / 1e9) ** (-1.0 / 3)
    fgas = Mgas / Mdisk

    Mdot_torque = (
        alphaT
        * fdisk ** (5.0 / 2)
        * (Mbh / 1e8) ** (1.0 / 6)
        * (Mdisk / 1e9) ** 1
        * (R0 / 100) ** (-3.0 / 2)
        * (1 + f0 / fgas) ** (-1)
    )  # yr^-1

    Mdot_bh = (1 - eta_bh) * Mdot_torque

    return Mdot_bh


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
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_coreen",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_kicks",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_pdrag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_high_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_high_thermal_eff",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE_highAGNtherm",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_noAGN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
]


# setup plot
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
ax = np.ravel(axs)

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


tgt_r_kpc = 2.0
n_to_find = 256  # search for min radius that contains this many star and gas particles


last_hagn_id = -1
isim = 0

overwrite = True


zoom_ls = ["-", "--", ":", "-.", (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10))] * 5
lines = []
labels = []


xmin = np.inf

last_simID = None
l = None

zoom_style = 0

for sdir in sdirs:

    name = sdir.split("/")[-1].strip()

    # sim_id = name[2:].split("_")[0]

    # sim_ids = ["id74099", "id147479", "id242704"]

    # c = "tab:blue"

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

    sim = ramses_sim(sdir, nml="cosmo.nml")

    sim_snaps = sim.snap_numbers
    sim_aexps = sim.get_snap_exps()
    sim.init_cosmo()
    sim_times = sim.cosmo_model.age(1.0 / sim_aexps - 1.0).value * 1e3

    # last sim_aexp
    # valid_steps = hagn_tree_aexps < (1.0 / (tgt_zed + 1.0))  # sim.aexp_end
    # nsteps = np.sum(valid_steps)

    nsteps = len(sim_snaps)

    # find last output with assoc files
    # assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
    # assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

    assoc_file_nbs = find_snaps_with_gals(sim_snaps, sim.path)

    avail_aexps = np.intersect1d(
        sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
    )
    avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

    hid_start, halo_dict, start_hosted_gals, found, start_aexp = find_starting_position(
        sim,
        avail_aexps,
        hagn_tree_aexps,
        hagn_tree_datas,
        hagn_tree_times,
        avail_times,
    )

    if not found:
        continue

    # load sim tree
    sim_halo_tree_rev_fname = os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat")
    if not os.path.exists(sim_halo_tree_rev_fname):
        print("No tree file")
        continue

    sim_zeds = 1.0 / avail_aexps - 1

    sim_tree_hids, sim_tree_datas, sim_tree_aexps = read_tree_fev_sim(
        sim_halo_tree_rev_fname,
        fbytes=os.path.join(sim.path, "TreeMakerDM_dust"),
        zstart=1.0 / start_aexp - 1.0,
        tgt_ids=[hid_start],
        star=False,
    )

    sim_tree_hids = sim_tree_hids[0]

    sim_tree_times = sim.cosmo_model.age(1.0 / sim_tree_aexps - 1).value * 1e3

    sim.get_snap_times()

    gal_props_tree = get_assoc_pties_in_tree(
        sim,
        sim_tree_aexps,
        sim_tree_hids,
        assoc_fields=["pos", "r50", "rmax"],
    )

    gal_smoothed_props_tree = smooth_props(gal_props_tree)

    fig, ax = plt.subplots()

    sid = None

    mbhs = np.zeros(nsteps)
    stdisps = np.zeros(nsteps)
    strots = np.zeros(nsteps)
    mdots_coarse = np.zeros(nsteps)
    mdots_torque = np.zeros(nsteps)
    mdots_edd = np.zeros(nsteps)
    sfrs = np.zeros(nsteps)
    fdisks = np.zeros(nsteps)
    mdisks = np.zeros(nsteps)
    mbulges = np.zeros(nsteps)

    # # zoom loop
    for istep, (snap, aexp, time) in enumerate(
        zip(sim_snaps[::-1], sim_aexps[::-1], sim_times[::-1])
    ):

        if np.all(np.abs(avail_aexps - aexp) > 1e-1):
            print(f"No assoc file for at z={1./aexp-1:.1f},snap={snap:d}")
            continue

        if not os.path.exists(get_gal_assoc_file(sim.path, snap)):
            continue

        sim_tree_arg = np.argmin(np.abs(sim_tree_aexps - aexp))
        cur_snap_hid = sim_tree_hids[sim_tree_arg]

        if cur_snap_hid in [-1, 0]:
            print(f"No halo at z={1./aexp-1:.1f},snap={snap:d}")
            continue

        gid, gal_dict = get_central_gal_for_hid(sim, cur_snap_hid, snap)
        if gid == None:
            print(f"No central galaxy at z={1./aexp-1:.1f},snap={snap:d}")
            continue

        gal_pos = gal_dict["pos"]
        # tgt_r = gal_dict["r50"]

        smooth_tree_arg = np.argmin(np.abs(gal_smoothed_props_tree["aexps"] - aexp))

        r50 = gal_smoothed_props_tree["r50"][smooth_tree_arg]

        if sid == None:

            sink = find_massive_sink(
                gal_pos,
                snap,
                sim,
                rmax=r50,
                tgt_fields=[
                    "position",
                    "mass",
                    "dMBH_coarse",
                    "dMEd_coarse",
                    "identity",
                ],
            )

            sid = sink["identity"]

        else:

            coarse_step = snap_to_coarse_step(snap, sim)
            fname_sink = os.path.join(sim.sink_path, f"sink_{coarse_step:05d}.dat")
            sink = read_sink_bin(
                fname_sink,
                # tgt_fields=[
                #     ("position", 3, np.float64),
                #     ("mass", 1, np.float64),
                #     ("dMBH_coarse", 1, np.float64),
                #     # "identity",
                # ],
                sid=sid,
            )

            convert_sink_units(sink, aexp, sim)

        # print(sink.keys())

        sink_mass = sink["mass"]

        tgt_pos = sink["position"]

        tgt_r = tgt_r_kpc / (sim.cosmo.lcMpc * 1e3)

        sim_max_dx = 1.0 / 2 ** sim.namelist["amr_params"]["levelmax"]  # code units

        rad_bins = np.linspace(sim_max_dx * 3, tgt_r, 100)

        # stars = read_zoom_stars(sim, snap, gid)

        # ages = stars["agepart"]
        # Zs = stars["Zpart"]

        # masses = sfhs.correct_mass(hagn_sim, ages, stars["mpart"], Zs)

        # stars = read_part_ball_NCdust(
        #     sim,
        #     snap,
        #     tgt_pos,
        #     tgt_r,
        #     tgt_fields=[
        #         "pos",
        #         "vel",
        #         "mass",
        #         "birth_time",
        #         "metallicity",
        #     ],
        #     fam=2,
        # )

        try:
            gas = read_data_ball(
                sim,
                snap,
                tgt_pos,
                tgt_r,
                host_halo=cur_snap_hid,
                data_types=["gas"],
                tgt_fields=[
                    "pos",
                    "vel",
                    "mass",
                    "age",
                    "metallicity",
                    "density",
                    "ilevel",
                    "velocity_x",
                    "velocity_y",
                    "velocity_z",
                ],
            )
        except AssertionError:
            continue

        # stars = datas["stars"]
        # gas = datas["gas"]

        print(f"z={1./aexp-1:.1f},snap={snap:d}")
        print(
            f"I found {len(gas['x'])} gas particles within {tgt_r_kpc:.1f} kpc"
            # f"I found {len(stars['mass'])} stars and {len(gas['x'])} gas particles within {tgt_r_kpc:.1f} kpc"
        )

        # star_args = []

        # star_pos = stars["pos"]
        # star_tree = KDTree(star_pos, boxsize=1 + 1e-6)
        # st_ibin = 0
        # while len(star_args) < n_to_find and st_ibin < len(rad_bins):
        #     star_args = star_tree.query_ball_point(tgt_pos, rad_bins[st_ibin])
        #     st_ibin += 1

        gas_args = []
        gas_pos = np.transpose([gas["x"], gas["y"], gas["z"]])
        gas_tree = KDTree(gas_pos, boxsize=1 + 1e-6)
        g_ibin = 0
        while len(gas_args) < n_to_find and g_ibin < len(rad_bins):
            gas_args = gas_tree.query_ball_point(tgt_pos, rad_bins[g_ibin])
            g_ibin += 1

        # ibin_max = max(g_ibin, st_ibin) - 1
        rbin_max = rad_bins[g_ibin - 1]

        # star_args = star_tree.query_ball_point(tgt_pos, rbin_max)
        # for k in stars:
        #     stars[k] = stars[k][star_args]
        gas_args = gas_tree.query_ball_point(tgt_pos, rbin_max)
        for k in gas:
            gas[k] = gas[k][gas_args]

        rmax_pc = rbin_max * sim.cosmo.lcMpc * 1e6

        print(f"found {n_to_find} stars and gas particles within {rmax_pc:.1f} pc")

        if gas is None:
            print("didn't find stars or gas")
            continue

        gas_pos = np.transpose([gas["x"], gas["y"], gas["z"]])

        gas_vel = (
            np.transpose([gas["velocity_x"], gas["velocity_y"], gas["velocity_z"]])
            / 1e5
        )
        bulk_vel_gas = np.average(gas_vel, weights=gas["density"], axis=0)

        dxs = 2 ** gas["ilevel"]
        dxs_cm = sim.cosmo.lcMpc / dxs * 1e6 * ramses_pc * aexp
        gas_masses = gas["density"] * (dxs_cm**3 / 2e33)

        rot_props, kin_props = extract_nh_kinematics(
            gas_masses, gas_pos, gas_vel - bulk_vel_gas, tgt_pos, debug=False
        )

        vrot = rot_props["Vrot"]
        disp = rot_props["disp"]

        fbulge = kin_props["fbulge"]
        fdisk = kin_props["fdisk"]
        mdisk = kin_props["Mdisk"]

        print(f"vrot={vrot}, disp={disp}")
        print(f"vrot/disps={vrot/disp}")

        stdisps[istep] = disp
        strots[istep] = vrot
        mdisks[istep] = mdisk
        mbulges[istep] = (mdisk / fdisk) * fbulge
        fdisks[istep] = fdisk

        print(f"f_bulge={fbulge}, f_disk={fdisk}", f"m_disk={mdisk:.1e}")

        rmax_pc = tgt_r * sim.cosmo.lcMpc * 1e6

        print(
            f"eta_bh: {0.1}, alphaT: {5}, fdisk: {fdisk:.1f}, Mbh: {sink['mass']:.1e}, Mdisk: {mdisk:.1e}, Mgas: {gas_masses.sum():.1e}, R0: {rmax_pc:.1e}"
        )

        mdot_torque = torque_limited(
            0.1, 5, fdisk, sink["mass"], mdisk, gas_masses.sum(), rmax_pc
        )

        print(f"mdot_torque={mdot_torque:.1e}")
        print(f"mdot_bondi={sink["dMBH_coarse"]:.1e}")

        mbhs[istep] = sink["mass"]
        mdots_torque[istep] = mdot_torque
        mdots_coarse[istep] = sink["dMBH_coarse"]
        mdots_edd[istep] = sink["dMEd_coarse"]

    #     ax[0].plot(sim_times, mstars, label="Stellar Mass")
    #     ax[0].plot(sim_times, stdisps, label="Velocity Dispersion")
    #     ax[0].plot(sim_times, mdots_coarse, label="Bondi Accretion Rate")
    #     ax[0].plot(sim_times, mdots_torque, label="Torque-limited Accretion Rate")
    #     ax[0].plot(sim_times, sfrs, label="Star Formation Rate")

    #     ax[0].set_ylabel("Mass / Rate")
    #     ax[0].set_yscale("log")
    #     ax[0].legend()

    #     ax[1].plot(sim_times, mdots_torque / sfrs, label="Mdot_torque / SFR")
    #     ax[1].set_ylabel("Mdot_torque / SFR")
    #     ax[1].set_yscale("log")
    #     ax[1].legend()

    #     ax[1].set_xlabel("Time (Myr)")

    # plt.tight_layout()
    # plt.show()

    plot_dir = os.path.join(".", "test_trq_ltd_only_gas", sim.name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # non_zero = np.where((mstars > 0) * (mdots_coarse > 0))[0]
    plot_times = sim_times[::-1]
    # plot_times = sim_times

    time_bins = np.linspace(sim_times.min(), sim_times.max(), 100)
    time_bins_normd = (time_bins - time_bins[0]) / time_bins
    time_colors = plt.colormaps.get_cmap("viridis")(np.digitize(plot_times, time_bins))

    # Plot each quantity in separate figures and save them
    fig1, ax1 = plt.subplots()
    # ax1.plot(plot_times, mstars, label="Stellar Mass")
    ax1.plot(plot_times, mbulges, label="Bulge Mass")
    ax1.plot(plot_times, mdisks, label="Disk Mass")
    ax1.set_ylabel("Mass, $M_{\odot}$")
    ax1.set_yscale("log")
    ax1.set_xlabel("Time, Myr")
    ax1.legend()
    plt.tight_layout()
    fig1.savefig(os.path.join(plot_dir, "stellar_bulge_disk_mass.png"))

    fig2, ax2 = plt.subplots()
    ax2.plot(plot_times, stdisps, label="Velocity Dispersion")
    ax2.plot(plot_times, np.abs(strots), label="Rotation Velocity")
    ax2.set_ylabel("Velocity, $km/s$")
    ax2.set_yscale("log")
    ax2.set_xlabel("Time, Myr")
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(os.path.join(plot_dir, "velocity_dispersion.png"))

    fig3, ax3 = plt.subplots()
    ax3.plot(plot_times, mdots_coarse, label="Bondi")
    ax3.plot(plot_times, mdots_torque, label="Torque-limited")
    ax3.plot(
        plot_times,
        mdots_edd,
        label="Eddington limit",
        c="k",
        alpha=0.8,
        ls="--",
    )
    ax3.set_ylabel("Bondi Accretion Rate,  $M_{\odot}/yr$")
    ax3.set_yscale("log")
    ax3.set_xlabel("Time, Myr")
    ax3.set_ylim(1e-8)
    ax3.legend()
    plt.tight_layout()
    fig3.savefig(os.path.join(plot_dir, "accretion_rates.png"))

    fig4, ax4 = plt.subplots()
    ax4.plot(plot_times, fdisks, label="Disk Fraction")
    ax4.plot(plot_times, 1 - fdisks, label="Bulge Fraction")
    ax4.set_ylabel("Fraction")
    ax4.set_xlabel("Time, Myr")
    ax4.legend()
    plt.tight_layout()
    fig4.savefig(os.path.join(plot_dir, "bulge_disk_fractions.png"))

    fig6, ax6 = plt.subplots()
    ax6.plot(plot_times, mdots_torque / sfrs, label="Mdot_torque / SFR")
    ax6.set_ylabel("$\dot{M}_{torque}$ / SFR")
    ax6.set_yscale("log")
    ax6.set_xlabel("Time, $Myr$")
    ax6.legend()
    plt.tight_layout()
    fig6.savefig(os.path.join(plot_dir, "mdot_torque_sfr_ratio.png"))

    fig7, ax7 = plt.subplots()
    sca = ax7.scatter(
        sfrs,
        mdots_torque,
        label="Torque-limited Accretion Rate vs SFR",
        c=time_colors,
    )
    ax7.set_xlabel("Star Formation Rate, $M_{\odot}/yr$")
    ax7.set_ylabel("Torque-limited Accretion Rate, $M_{\odot}/yr$")
    ax7.set_xscale("log")
    ax7.set_yscale("log")
    ax7.legend()
    plt.tight_layout()
    plt.colorbar(sca, ax=ax7)
    fig7.savefig(os.path.join(plot_dir, "torque_limited_accretion_rate_vs_sfr.png"))

    fig8, ax8 = plt.subplots()
    ax8.scatter(
        mbhs,
        mdots_torque,
        label="Torque-limited",
        facecolors="none",
        edgecolors=time_colors,
        marker="o",
    )
    sca = ax8.scatter(
        mbhs,
        mdots_coarse,
        label="Bond",
        facecolors="none",
        edgecolors=time_colors,
        marker="D",
    )
    ax8.set_xlabel("Black Hole Mass, $M_{\odot}$")
    ax8.set_ylabel("Accretion Rate, $M_{\odot}/yr$")
    ax8.set_xscale("log")
    ax8.set_yscale("log")
    ax8.legend()
    plt.tight_layout()
    plt.colorbar(sca, ax=ax8)
    fig8.savefig(os.path.join(plot_dir, "accretion_rate_vs_black_hole_mass.png"))

    arg_first_mbh_mass = np.argmin(mbhs)

    intgr8td_mdot_trq = np.zeros_like(plot_times)
    intrgr8td_bondi = np.zeros_like(plot_times)
    intgr8td_edd = np.zeros_like(plot_times)
    for istep in range(len(plot_times)):
        intgr8td_mdot_trq[istep] += np.trapz(
            mdots_torque[::-1][:istep], sim_times[:istep] * 1e6
        )
        intrgr8td_bondi[istep] += np.trapz(
            mdots_coarse[::-1][:istep], sim_times[:istep] * 1e6
        )
        intgr8td_edd[istep] += np.trapz(
            mdots_edd[::-1][:istep], sim_times[:istep] * 1e6
        )

    intrgr8td_bondi = intrgr8td_bondi[::-1]
    intgr8td_mdot_trq = intgr8td_mdot_trq[::-1]
    intgr8td_edd = intgr8td_edd[::-1]

    intrgr8td_bondi[:arg_first_mbh_mass] += mbhs[arg_first_mbh_mass]
    intgr8td_mdot_trq[:arg_first_mbh_mass] += mbhs[arg_first_mbh_mass]
    intgr8td_edd[:arg_first_mbh_mass] += mbhs[arg_first_mbh_mass]

    # intgr8td_mdot_trq = np.trapz(mdots_torque, plot_times * 1e6)
    # intgr8td_edd = np.trapz(mdots_edd, plot_times * 1e6)
    # intrgr8td_bondi = np.trapz(mdots_coarse, plot_times * 1e6)

    fig9, ax9 = plt.subplots()

    (l,) = ax9.plot(plot_times, mbhs, label="Black Hole Mass (bondi)", ls="--")
    ax9.plot(
        plot_times,
        intrgr8td_bondi,
        label="Integrated Bondi",
        c=l.get_color(),
    )
    ax9.plot(
        plot_times,
        intgr8td_mdot_trq,
        label="Integrated Torque-limited",
    )
    ax9.plot(
        plot_times,
        intgr8td_edd,
        label="Integrated Eddington",
        c="k",
        # ls="--",
    )
    ax9.set_ylabel("Mbh, $M_{\odot}$")
    ax9.set_xlabel("Time, Myr")
    ax9.legend()
    ax9.set_yscale("log")
    ax9.grid()
    plt.tight_layout()
    fig9.savefig(os.path.join(plot_dir, "integrated_accretion_rates.png"))
