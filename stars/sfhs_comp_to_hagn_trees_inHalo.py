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
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import (
    # find_zoom_tgt_gal,
    # find_zoom_tgt_halo,
    # get_central_gal_for_hid,
    get_gal_assoc_file,
    get_halo_props_snap,
    find_snaps_with_gals,
)

from zoom_analysis.zoom_helpers import find_starting_position

from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D

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
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowSFE_gravSink",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowSFE_DynBondiGravSinkMass",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_fullresIC",
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
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_vlowSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_high_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd_drag",
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
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh",  # _leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_noHydroDragB4z4merge",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
]


# setup plot
fig, ax = sfhs.setup_sfh_plot()

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


sfr_min, ssfr_min, sfh_min = np.inf, np.inf, np.inf
sfr_max, ssfr_max, sfh_max = -np.inf, -np.inf, -np.inf

last_hagn_id = -1
isim = 0

# overwrite = True
overwrite = False

zoom_ls = [
    "-",
    "--",
    ":",
    "-.",
    (0, (1, 10)),
    (0, (3, 5, 1, 5)),
    (0, (1, 2, 1, 2, 5, 3)),
] * 15
lines = []
labels = []


sfr_max = -np.inf
sfr_min = np.inf

ssfr_max = -np.inf
ssfr_min = np.inf

mstel_max = -np.inf
mstel_min = np.inf

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
    sfr_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
    time_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)

    mstel_zoom = np.zeros(nsteps, dtype=np.float32)
    sfr_zoom = np.zeros(nsteps, dtype=np.float32)
    time_zoom = np.zeros(nsteps, dtype=np.float32)

    datas_path = os.path.join(sim.path, "computed_data")
    fout_h5 = os.path.join(datas_path, f"stellar_history_inHalo.h5")
    read_existing = True

    if not os.path.exists(datas_path):
        os.makedirs(datas_path)
        read_existing = False
    else:
        if not os.path.exists(fout_h5):
            read_existing = False

    if overwrite:
        read_existing = False

    if read_existing:

        # read the data
        with h5py.File(fout_h5, "r") as f:
            saved_mstel_hagn = f["mstel_hagn"][:]
            saved_sfr_hagn = f["sfr_hagn"][:]
            saved_time_hagn = f["time_hagn"][:]

            saved_mstel_zoom = f["mstel_zoom"][:]
            saved_sfr_zoom = f["sfr_zoom"][:]
            saved_time_zoom = f["time_zoom"][:]

        if saved_time_zoom.max() >= sim_times.max():
            # if we have the same number of values
            # then just read the data
            mstel_zoom = saved_mstel_zoom
            sfr_zoom = saved_sfr_zoom
            time_zoom = saved_time_zoom

            mstel_hagn = saved_mstel_hagn
            sfr_hagn = saved_sfr_hagn
            time_hagn = saved_time_hagn
        else:  # otherwise we need to recompute because something has changed
            read_existing = False

    # print(read_existing, len(saved_mstel_zoom), len(sim_snaps))

    if not read_existing:

        # find last output with assoc files
        # assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
        # assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

        assoc_file_nbs = find_snaps_with_gals(sim_snaps, sim.path)

        avail_aexps = np.intersect1d(
            sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
        )

        if len(avail_aexps) == 0:
            print("No assoc files")
            continue

        avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

        prev_mass = -1
        # prev_pos = None

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

            # # gal_pties = get_cat_gids(super_cat, [int(hagn_tree_hids[0][istep])])

            # if len(gal_pties["gid"]) == 0:
            #     print("No galaxy")
            #     continue

            # print(gal_pties["hx"])
            tgt_pos = np.asarray(
                [gal_pties["hx"][0], gal_pties["hy"][0], gal_pties["hz"][0]]
            )
            tgt_rad = gal_pties["rvir"]

            # hagn_file = os.path.join(sim.path, f"stellar_history_{hagn_snap}.h5")

            # if not os.path.exists(hagn_file) or overwrite_hagn:

            # stars = gid_to_stars(
            #     gal_pties["gid"][0],
            #     hagn_snap,
            #     hagn_sim,
            #     ["mass", "birth_time", "metallicity"],
            # )

            stars = read_part_ball_hagn(
                hagn_sim,
                hagn_snap,
                tgt_pos,
                tgt_rad,
                ["mass", "birth_time", "metallicity"],
                fam=2,
            )

            ages = stars["age"]
            Zs = stars["metallicity"]

            masses = sfhs.correct_mass(hagn_sim, ages, stars["mass"], Zs)

            mstel_hagn[istep] = masses.sum()
            sfr_hagn[istep] = np.sum(masses[ages < 100] / 1e2)  # Msun/Myr
            # sfr_hagn[istep] = np.sum(masses[ages < 1000] / 1e3)  # Msun/Myr
            # time_hagn[istep] = hagn_tree_times[istep]

            # # print(gal_pties)
            # if len(gal_pties["mgal"]) == 0:
            #     continue

            mstel_hagn[istep] = gal_pties["mgal"][0]
            sfr_hagn[istep] = gal_pties["sfr100"][0] * 1e6
            time_hagn[istep] = time

            # # sim_snap = sim.get_closest_snap(aexp=aexp)

            # print(aexp, avail_aexps)

        hid_start, halo_dict, start_hosted_gals, found, start_aexp = (
            find_starting_position(
                sim,
                avail_aexps,
                hagn_tree_aexps,
                hagn_tree_datas,
                hagn_tree_times,
                avail_times,
            )
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

        # print(sim_tree_datas["m"])

        # print(sim_tree_hids)
        # print(sim_tree_datas)

        sim_tree_times = sim.cosmo_model.age(1.0 / sim_tree_aexps - 1).value * 1e3

        # # zoom loop
        for istep, (snap, aexp, time) in enumerate(
            zip(sim_snaps, sim_aexps, sim_times)
        ):
            # if time < sim_tree_times.min() - 5:
            # continue

            zed = 1.0 / aexp - 1.0

            if np.all(np.abs(avail_aexps - aexp) > 1e-1):
                print(f"No assoc file for at z={1./aexp-1:.1f},snap={snap:d}")
                continue

            if not os.path.exists(get_gal_assoc_file(sim.path, snap)):
                print("no assoc file")
                continue

            # assoc_file = assoc_files[assoc_file_nbs == snap]
            # if len(assoc_file) == 0:
            #     print(f"No assoc file for at z={1./aexp-1:.1f},snap={snap:d}")
            #     continue

            sim_tree_arg = np.argmin(np.abs(sim_tree_aexps - aexp))
            cur_snap_hid = sim_tree_hids[0][sim_tree_arg]
            if cur_snap_hid in [-1, 0]:
                continue

            hprops, hgals = get_halo_props_snap(sim.path, snap, hid=cur_snap_hid)

            tgt_pos = hprops["pos"]
            tgt_r = hprops["rvir"]

            stars = read_part_ball_NCdust(
                sim,
                snap,
                tgt_pos,
                tgt_r,
                tgt_fields=["mass", "birth_time", "metallicity"],
                fam=2,
            )
            # print(stars.keys())

            ages = stars["age"]
            Zs = stars["metallicity"]

            masses = sfhs.correct_mass(hagn_sim, ages, stars["mass"], Zs)
            prev_mass = masses.sum()
            # prev_mass = stars["mpart"].sum()

            # prev_mass = gal_dict["mass"]
            mstel_zoom[istep] = prev_mass
            # print(
            #     istep,
            #     snap,
            #     zed,
            #     cur_snap_hid,
            #     gid,
            #     prev_mass,
            #     gal_dict["mass"],
            #     gal_dict["r50"],
            # )
            sfr_zoom[istep] = np.sum(masses[ages < 100] / 1e2)  # Msun/Myr
            time_zoom[istep] = time

            # print(snap, 1.0 / aexp - 1, mstel_zoom[istep])

    # print(list(zip(sim_snaps, sim_times, 1.0 / sim_aexps - 1.0, mstel_zoom)))

    ssfr_hagn = sfr_hagn / mstel_hagn
    ssfr_zoom = sfr_zoom / mstel_zoom

    if np.any(mstel_hagn > 0) and np.any(mstel_zoom > 0):
        mstel_max = np.max(
            [
                mstel_max,
                # mstel_hagn[mstel_hagn > 0].max(),
                mstel_zoom[mstel_zoom > 0].max(),
            ]
        )
        mstel_min = np.min(
            [
                mstel_min,
                # mstel_hagn[mstel_hagn > 0].min(),
                mstel_zoom[mstel_zoom > 0].min(),
            ]
        )
        sfr_max = np.max(
            [
                sfr_max,
                #   sfr_hagn[mstel_hagn > 0].max(),
                sfr_zoom[mstel_zoom > 0].max(),
            ]
        )
        sfr_min = np.min(
            [
                sfr_min,
                #  sfr_hagn[mstel_hagn > 0].min(),
                sfr_zoom[mstel_zoom > 0].min(),
            ]
        )
        ssfr_max = np.max(
            [
                ssfr_max,
                # np.nanmax(ssfr_hagn[mstel_hagn > 0]),
                np.nanmax(ssfr_zoom[mstel_zoom > 0]),
            ]
        )
        ssfr_min = np.min(
            [
                ssfr_min,
                # np.nanmin(ssfr_hagn[mstel_hagn > 0]),
                np.nanmin(ssfr_zoom[mstel_zoom > 0]),
            ]
        )

    color = None
    if last_simID == intID:
        color = l[0].get_color()
        zoom_style += 1
    else:
        zoom_style = 0

    if zoom_style == 0:
        l = sfhs.plot_sf_stuff(
            ax,
            mstel_hagn,
            sfr_hagn,
            ssfr_hagn,
            time_hagn,
            0,
            hagn_sim.cosmo_model,
            ls="-",
            color=color,
            # label=sim_id.split("_")[0],
            lw=1.0,
        )

    # print(sim.name, mstel_zoom)

    # print(f"zoom style is {zoom_ls[zoom_style]}")
    l = sfhs.plot_sf_stuff(
        ax,
        mstel_zoom,
        sfr_zoom,
        ssfr_zoom,
        time_zoom,
        None,
        sim.cosmo_model,
        # label=sim.name,
        ls=zoom_ls[zoom_style],
        color=l[0].get_color(),
        lw=2.0,
    )

    lvlmax = sim.namelist["amr_params"]["levelmax"]

    labels.append(sim.name + f" {lvlmax}")
    lines.append(l)

    # print(sim.name, intID, last_simID)
    # print(zoom_style, zoom_ls[zoom_style], l[0].get_color())

    last_simID = intID

    # after plot save the data to h5py file
    with h5py.File(fout_h5, "w") as f:
        f.create_dataset("mstel_hagn", data=mstel_hagn, compression="lzf")
        f.create_dataset("sfr_hagn", data=sfr_hagn, compression="lzf")
        f.create_dataset("time_hagn", data=time_hagn, compression="lzf")

        f.create_dataset("mstel_zoom", data=mstel_zoom, compression="lzf")
        f.create_dataset("sfr_zoom", data=sfr_zoom, compression="lzf")
        f.create_dataset("time_zoom", data=time_zoom, compression="lzf")

if np.isfinite(ssfr_min) and np.isfinite(ssfr_max):
    ax[0].set_ylim(ssfr_min * 0.5, ssfr_max * 1.5)
if np.isfinite(sfr_min) and np.isfinite(sfr_max):
    ax[1].set_ylim(sfr_min * 0.5, sfr_max * 1.5)
if np.isfinite(mstel_min) and np.isfinite(mstel_max):
    ax[2].set_ylim(mstel_min * 0.5, mstel_max * 1.5)

# tlim = time_zoom[np.where(sfr_zoom == 0)[-1]]
# ax[0].set_xlim(tlim, time_zoom[-1])
# ax[1].set_xlim(tlim, time_zoom[-1])
# ax[2].set_xlim(tlim, time_zoom[-1])

# ax[0].set_xlim(
# sim_tree_times.min() / 1e3,
# sim_tree_times.max() / 1e3,
# min(hagn_tree_times.min(), sim_tree_times.min()) / 1e3,
# max(hagn_tree_times.max(), sim_tree_times.max()) / 1e3,
# )


# ax[0].set_xlim(0.5, 3.2)
# ax[0].set_xlim(0.5, 1.8)
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
    a.tick_params(
        axis="both",
        which="both",
        bottom=True,
        top=True,
        left=True,
        right=True,
        direction="in",
    )

xlim = ax[0].get_xlim()
ax[0].text(xlim[0] + 0.1, 1.1e-5, "quenched", color="k", alpha=0.33, ha="left")
# ax[0].set_ylim(4e-6, 2e-2)
ylim = ax[0].get_ylim()
ax[0].fill_between(xlim, 1e-5, ylim[0], color="k", alpha=0.33)
ax[0].set_ylim(ylim)
for a in ax:
    ax[0].set_xlim(xlim)


# sfrs_leja = sfr_ridge_leja22(tgt_zed, mstel_zoom)
# ax[0].plot(mass_bins, sfrs_leja / 10.0 / mstel_zoom, "k--")
# ax[0].annotate(
#     mass_bins[0], sfrs_leja[0] / 10.0 / mstel_zoom, "0.1xMS of Leja+22", color="k"
# )


y2 = ax[0].twiny()
y2.set_xlim(xlim)
zlabels = [f"{cosmo.age(zed).value:.1f}" for zed in ax[0].get_xticks()]
# y2.set_xticks(ax[0].get_xticks())
y2.set_xticklabels(zlabels)
y2.set_xlabel("redshift")

# plot leja+22 limits
zspan = np.arange(np.min(np.float32(zlabels)), np.max(np.float32(zlabels)), 0.05)

leja_masses = np.asarray([1e10, 1e11, 1e12])
ssfr_leja_z2_lim = [
    (
        sfr_ridge_leja22(z, leja_masses) / (leja_masses) * 1e6
        if 0.3 < z <= 2.7
        else np.full(len(leja_masses), np.nan)
    )
    for z in zspan
]

zspan_time = cosmo.age(zspan).value

for m, ssfr in zip(leja_masses, np.transpose(ssfr_leja_z2_lim)):
    # ax[0].axhline(ssfr, color="k", ls="--", alpha=0.3)
    finite = np.isfinite(ssfr)
    if finite.sum() < 2:
        continue
    ax[0].plot(zspan_time[finite], ssfr[finite], "k--", alpha=0.33)

    mid_point = np.where(finite)[0][int(0.5 * finite.sum())]

    ax[0].text(
        zspan_time[finite][mid_point],
        ssfr[finite][mid_point],
        f"Leja+22, M={m:.1e} $M_\odot$",
        color="k",
        alpha=0.33,
        ha="center",
    )
lines.append(
    [
        Line2D([0], [0], color="k", linestyle="-", lw=1),
        Line2D([0], [0], color="k", ls="-", lw=2),
    ]
)
labels.append(["HAGN", "zoom"])

ax[1].set_ylim(
    1e4,
)
ax[2].set_ylim(
    1e6,
)

ax[-1].legend(lines, labels, framealpha=0.0, ncol=2)  # , handlelength=3)

fig.savefig(f"sfhs_compHAGN_trees_inHalo.png")
