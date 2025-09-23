import enum
from jplots.scatter import xvy
from zoom_analysis.catalogue.read_cat import read_cat
from zoom_analysis.halo_maker.assoc_fcts import get_gal_assoc_file

from gremlin.read_sim_params import ramses_sim

import matplotlib.pyplot as plt

import os
import numpy as np

from zoom_analysis.zoom_helpers import check_in_all_sims_vol

from zoom_analysis.visu.visu_fct import get_gal_props_snap

# from hagn.catalogues import make_super_cat

# fig, ax = sink_histories.setup_bh_plot()

# tgt_zed =  3.51559418634798084355
tgt_zed = 3.56284466148542513224

fpure = 1.0 - 1e-4


sim_dirs = [
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_coreen",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_fullresIC",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_gravSink",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_DynBondiGravSinkMass",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_kicks",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_smooth_ref",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_low_dboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_minMassDynGrav",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE_stgNHboost_stricterSF/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_vlowSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_high_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd_drag",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_frcAccrt",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_accBoost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE_lowNsink",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highSFE_lowSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE_lowSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_pdrag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nrg_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_high_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_high_thermal_eff",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE_highAGNtherm",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_noAGN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_lowerSFE_NHboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_smallICs",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_stricterSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_stricterSF_radioHeavy",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictestSF_lowSNe/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_medSFE_stgNHboost_stricterSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_highSFE_stgNHboost_strictestSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_XtremeSF_lowSNe",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_MegaSF_lowSNe",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_SuperMegaSF_midSNe",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_XtremeLowSFE_stgNHboost_strictSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_SuperLowSFE_stgNHboost_strictSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id180130_256",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id180130_model6_eps0p05",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id180130_model5",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_noHydroDragB4z4merge",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data103/jlewis/sims/lvlmax_22/mh1e12/id52380_NClike",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id112288",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288_novrel_lowerSFE_stgNHboost_strictSF/",
    "/data102/jlewis/sims/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF_high_sconstant/",
    "/data102/jlewis/sims/lvlmax_22/mh1e12/id52380_novrel_lowerSFE_stgNHboost_strictSF_Vhigh_sconstant/",
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
    "X",
    "D",
    "d",
    "|",
    "_",
]

fig_DMSM, ax_DMSM = plt.subplots(2, 2, figsize=(16, 16))
fig_DTMSM, ax_DTMSM = plt.subplots(2, 2, figsize=(16, 16))
fig_DTGSM, ax_DTGSM = plt.subplots(2, 2, figsize=(16, 16))
fig_ZSM, ax_ZSM = plt.subplots(1, 1, figsize=(8, 8))
fig_DTMtot, ax_DTMtot = plt.subplots(1, 1, figsize=(8, 8))
fig_DTGtot, ax_DTGtot = plt.subplots(1, 1, figsize=(8, 8))
fig_fdepl, ax_fdepl = plt.subplots(3, 2, figsize=(16, 16))

sim_names = []

for isim, sim_dir in enumerate(sim_dirs):

    sim = ramses_sim(sim_dir, nml="cosmo.nml")

    name = sim.name

    intID = int(sim.name.split("_")[0][2:])

    sim_names.append(intID)

    # c = l[0].get_color()
    # find last assoc_file
    found = False
    decal = 0

    z_dists = np.abs(1.0 / sim.aexps - 1.0 - tgt_zed)
    args = np.argsort(z_dists)

    while not found and decal < len(sim.snap_numbers):
        arg = args[decal]
        sim_snap = sim.snap_numbers[arg]
        sim_zed = 1.0 / sim.aexps[arg] - 1

        gfile = get_gal_assoc_file(sim_dir, sim_snap)

        found_cond = os.path.isfile(gfile) * (z_dists[arg] < 0.1)
        print(sim_snap, sim_zed, decal, gfile)

        if found_cond:
            found = True
        else:
            decal += 1

    if not found:
        continue

    gal_dict = get_gal_props_snap(sim.path, sim_snap, main_stars=False)
    vol_args, min_vol = check_in_all_sims_vol(gal_dict["pos"].T, sim, sim_dirs)

    # fig,ax = plt.subplots(1,1)

    # ax.scatter((gal_dict["pos"][0,:]-0.5)+sim.zoom_ctr[0],(gal_dict["pos"][1,:]-0.5)+sim.zoom_ctr[1])

    # fig.savefig('test_gal_coords')

    # print(gal_dict.keys())
    # print(min_vol,vol_args.sum(),len(vol_args))
    print(sim.zoom_ctr, gal_dict["pos"].mean(axis=1))

    host_purity = gal_dict["host purity"]
    central_cond = gal_dict["central"] == 1
    pure_cond = (host_purity > fpure) * central_cond * vol_args
    gal_mass = gal_dict["mass"][pure_cond]
    gal_pos = gal_dict["pos"][:, pure_cond]
    rmax = gal_dict["rmax"][pure_cond]
    halo_mass = gal_dict["host mass"][pure_cond]

    gids = gal_dict["gids"][pure_cond]

    print(f"I found {pure_cond.sum():d} galaxies with purity > {fpure:f},")
    print(f"and {len(host_purity)-pure_cond.sum()} galaxies with purity < {fpure:f}")

    gal_cat = read_cat(sim.path, sim_snap, "galaxy", "reff")

    mstel = np.zeros(len(gids))
    mdustClarge = np.zeros(len(gids))
    mdustCsmall = np.zeros(len(gids))
    mdustSilarge = np.zeros(len(gids))
    mdustSismall = np.zeros(len(gids))
    mgas = np.zeros(len(gids))
    MZgas = np.zeros(len(gids))
    Zgas = np.zeros(len(gids))
    fdeplC = np.zeros(len(gids))
    fdeplMg = np.zeros(len(gids))
    fdeplSi = np.zeros(len(gids))
    fdeplO = np.zeros(len(gids))
    fdeplFe = np.zeros(len(gids))
    DTM = np.zeros(len(gids))

    for igid, gid in enumerate(gids):

        gal_key = f"galaxy_{gid:07d}"

        mstel[igid] = np.sum(gal_cat[gal_key]["stars"]["Mstar_msun"])
        mgas[igid] = gal_cat[gal_key]["gas"]["Mgas_tot_msun"]
        Zgas[igid] = gal_cat[gal_key]["gas"]["Z_mean"]
        MZgas[igid] = gal_cat[gal_key]["gas"]["Mmetals_msun"]
        fdeplC[igid] = gal_cat[gal_key]["gas"]["fdepl_C"]
        fdeplMg[igid] = gal_cat[gal_key]["gas"]["fdepl_Mg"]
        fdeplO[igid] = gal_cat[gal_key]["gas"]["fdepl_O"]
        fdeplMg[igid] = gal_cat[gal_key]["gas"]["fdepl_Mg"]
        fdeplFe[igid] = gal_cat[gal_key]["gas"]["fdepl_Fe"]
        mdustClarge[igid] = gal_cat[gal_key]["gas"]["dust_bin1_mass_msun"]
        mdustCsmall[igid] = gal_cat[gal_key]["gas"]["dust_bin2_mass_msun"]
        mdustSilarge[igid] = gal_cat[gal_key]["gas"]["dust_bin3_mass_msun"]
        mdustSismall[igid] = gal_cat[gal_key]["gas"]["dust_bin4_mass_msun"]
        DTM[igid] = gal_cat[gal_key]["gas"]["dtm"]

    cur_marker = markers[isim]

    if len(mstel) < 2:
        print(mstel)
        continue

    xvy(
        mstel,
        mdustClarge,
        ax=ax_DMSM[0, 0],
        xlabel="$M_\star, M_\odot$",
        ylabel="$M_C \, large, M_\odot$",
        markers=cur_marker,
    )
    xvy(
        mstel,
        mdustCsmall,
        ax=ax_DMSM[0, 1],
        xlabel="$M_\star, M_\odot$",
        ylabel="$M_C \, small, M_\odot$",
        markers=cur_marker,
    )
    xvy(
        mstel,
        mdustSilarge,
        ax=ax_DMSM[1, 0],
        xlabel="$M_\star, M_\odot$",
        ylabel="$M_{Si} \, large, M_\odot$",
        markers=cur_marker,
    )
    xvy(
        mstel,
        mdustSismall,
        ax=ax_DMSM[1, 1],
        xlabel="$M_\star, M_\odot$",
        ylabel="$M_{Si} \, small, M_\odot$",
        markers=cur_marker,
    )

    xvy(
        mstel,
        mdustClarge / MZgas,
        ax=ax_DTMSM[0, 0],
        xlabel="$M_\star, M_\odot$",
        ylabel="DTM \, C \, large,M_\odot$",
        markers=cur_marker,
    )
    xvy(
        mstel,
        mdustCsmall / MZgas,
        ax=ax_DTMSM[0, 1],
        xlabel="$M_\star, M_\odot$",
        ylabel="$DTM \, C \, small, M_\odot$",
        markers=cur_marker,
    )
    xvy(
        mstel,
        mdustSilarge / MZgas,
        ax=ax_DTMSM[1, 0],
        xlabel="$M_\star, M_\odot$",
        ylabel="$DTM \, Si \, large, M_\odot$",
        markers=cur_marker,
    )
    xvy(
        mstel,
        mdustSismall / MZgas,
        ax=ax_DTMSM[1, 1],
        xlabel="$M_\star, M_\odot$",
        ylabel="$DTM \, Si \, small, M_\odot$",
        markers=cur_marker,
    )

    xvy(
        mstel,
        mdustClarge / mgas,
        ax=ax_DTGSM[0, 0],
        xlabel="$M_\star, M_\odot$",
        ylabel=r"DTG \, C \, large,M_\odot$",
        markers=cur_marker,
    )
    xvy(
        mstel,
        mdustCsmall / mgas,
        ax=ax_DTGSM[0, 1],
        xlabel="$M_\star, M_\odot$",
        ylabel=r"$DTG \, C \, small, M_\odot$",
        markers=cur_marker,
    )
    xvy(
        mstel,
        mdustSilarge / mgas,
        ax=ax_DTGSM[1, 0],
        xlabel="$M_\star, M_\odot$",
        ylabel=r"$DTG \, Si \, large, M_\odot$",
        markers=cur_marker,
    )
    xvy(
        mstel,
        mdustSismall / mgas,
        ax=ax_DTGSM[1, 1],
        xlabel="$M_\star, M_\odot$",
        ylabel=r"$DTG \, Si \, small, M_\odot$",
        markers=cur_marker,
    )

    xvy(
        mstel,
        fdeplC,
        ax=ax_fdepl[0, 0],
        xlabel="$M_\star, M_\odot$",
        ylabel="$F_{depl, Mg}$",
        markers=cur_marker,
        ylog=False,
    )
    xvy(
        mstel,
        fdeplFe,
        ax=ax_fdepl[0, 1],
        xlabel="$M_\star, M_\odot$",
        ylabel="$F_{depl, Fe}$",
        markers=cur_marker,
        ylog=False,
    )
    xvy(
        mstel,
        fdeplMg,
        ax=ax_fdepl[1, 0],
        xlabel="$M_\star, M_\odot$",
        ylabel="$F_{depl, O}$",
        markers=cur_marker,
        ylog=False,
    )
    xvy(
        mstel,
        fdeplSi,
        ax=ax_fdepl[1, 1],
        xlabel="$M_\star, M_\odot$",
        ylabel="$F_{depl, Si}$",
        markers=cur_marker,
        ylog=False,
    )
    xvy(
        mstel,
        fdeplO,
        ax=ax_fdepl[2, 0],
        xlabel="$M_\star, M_\odot$",
        ylabel="$F_{depl, C}$",
        markers=cur_marker,
        ylog=False,
    )
    ax_fdepl[2, 1].axis("off")

    xvy(
        mstel,
        Zgas,
        ax=ax_ZSM,
        xlabel="$M_\star, M_\odot$",
        ylabel="Metallicity",
        markers=cur_marker,
    )

    xvy(
        mstel,
        DTM,
        ax=ax_DTMtot,
        xlabel="$M_\star, M_\odot$",
        ylabel="DTM",
        markers=cur_marker,
    )

    xvy(
        mstel,
        (mdustClarge + mdustCsmall + mdustSilarge + mdustSismall) / mgas,
        ax=ax_DTGtot,
        xlabel="$M_\star, M_\odot$",
        ylabel="DTG",
        markers=cur_marker,
    )

    label = sim.name
    if len(sim_dirs) > 1:
        label += f", z={sim_zed:.4f}"


title_txt = ""
if len(sim_dirs) == 1:
    title_txt += f"z={sim_zed:.2f}"
    # ax.text(0.05, 0.95, f"z={sim_zed:.2f}", transform=ax.transAxes)
# also print purity threshold
# ax.text(0.05, 0.9, f"fpure > {fpure:.3f}", transform=ax.transAxes)
title_txt += f" fpure > {fpure:.3f}"

for axs in [ax_DTGtot, ax_DMSM, ax_DTGSM, ax_DTMSM, ax_DTMtot, ax_ZSM]:

    ax = np.ravel(axs)

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

    ax[0].legend(framealpha=0.0, title=title_txt)

for fig, fig_name in [
    (fig_DMSM, "zoom_dustmass"),
    (fig_DTGSM, "zoom_DTG"),
    (fig_DTGtot, "zoom_tot_DTG"),
    (fig_DTMSM, "zoom_DTM"),
    (fig_DTMtot, "zoom_tot_DTM"),
    (fig_ZSM, "zoom_mean_Z"),
    (fig_fdepl, "zoom_fdepl"),
]:

    if len(np.unique(sim_names)) == 1:
        fig_name += f"_id{sim_names[0]:d}"

    if not os.path.exists("figs/scatters"):
        os.makedirs("figs/scatters")

    if tgt_zed != None:
        fig.savefig(
            os.path.join("figs", "scatters", f"{fig_name:s}_z{tgt_zed:.2f}.png")
        )
    else:
        fig.savefig(os.path.join("figs", "scatters", f"{fig_name:s}.png"))
