from turtle import width
import numpy as np
from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from matplotlib import patheffects as pe
import os

from hagn.utils import get_hagn_sim

from zoom_analysis.constants import *
from zoom_analysis.zoom_helpers import starting_hid_from_hagn

from zoom_analysis.halo_maker.assoc_fcts import (
    find_zoom_tgt_gal,
    get_central_gal_for_hid,
    get_gal_props_snap,
    get_halo_props_snap,
    get_halo_assoc_file,
    get_assoc_pties_in_tree,
    smooth_props,
)


# from zoom_analysis.visu.visu_fct_bckp import (
from zoom_analysis.visu.visu_fct import (
    # make_amr_img,
    # make_amr_img_smooth,
    # make_yt_img,
    basis_from_vect,
    plot_zoom_BHs,
    plot_zoom_gals,
    plot_zoom_halos,
    # plot_stars,
    plot_fields,
)

from zoom_analysis.trees.tree_reader import read_tree_file_rev
from scipy.interpolate import UnivariateSpline

vmin = vmax = None
cmap = None
cb = True

mode = "sum"
marker_color = "white"

field = "density"
cmap = "magma"
vmin = 1e-26
vmax = 1e-20

# field = "temperature"
# vmin = 1e3
# vmax = 1e7
# cmap = "plasma"

# field = "metallicity"
# vmin = "1e-5"
# vmax = "1e-1"
# cmap = "YlOrRd"

# field = "dust_bin01"
# field = "dust_bin02"
# field = "dust_bin03"
# field = "dust_bin04"

# field = "DTM"
# vmin = 1e-5
# vmax = 1
# cmap = "YlGnBu"
# mode = "mean"

# field = "stellar mass"
# cmap = "gray"
# # cmap = "viridis"
# vmin = 6e4
# vmax = 1e9
# marker_color = "r"

# field = "stellar age"
# vmin = 1e1  # Myr
# vmax = 1e3
# cmap = "Spectral_r"
# mode = "mean"

# # field = "SFR1"
# field = "SFR10"
# field = "SFR100"
# # # field = "SFR300"
# # field = "SFR500"
# # # field = "SFR1000"
# mode = "mean"
# cmap = "hot"
# vmin = 1e1
# vmax = 1e4

# field = "dm mass"
# cmap = "viridis"

# field = "pressure"
# vmin = 10  # k
# vmax = 1e5
# tree_hids, tree_datas, tree_aexps = read_tree_rev(
#     tree_path, tgt_zed, tree_type="halo", target_fields=["m", "x", "y", "z", "r"]
# )

# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_DynBondiGravSinkMass_SE"
# sim_dir = (
# "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_gravSink_SE"
# )
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd_drag"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479_old"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_old"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_eagn_T0p15"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099_inter"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowSFE_DynBondiGravSinkMass"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_minMassDynGrav"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_kicks"
# sim_dir = "/data102/jlewis/sims/lvlmax_21/mh1e12/id180130_model6_eps0p05"
# sim_dir = "/data103/jlewis/sims/lvlmax_21/mh1e12/id180130_256"
# sim_dir = "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_drag"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289"
# sim_dir = "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_frcAccrt"

sim_dirs = [
    "/data102/jlewis/sims/lvlmax_21/mh1e12/id180130_model6_eps0p05",
    "/data103/jlewis/sims/lvlmax_21/mh1e12/id180130_256",
]

delta_t = 4  # Myr
every_snap = True  # try and interpolate between tree nodes if not found

rad_fact = 5.0  # fraction of radius to use as plot window
use_r50 = True
fixed_r_ckpc = 20

max_bh_mass = 3e6
min_bh_mass = 1e5

min_gal_mass = 1e8
max_gal_mass = 2e10

zmax = 8.5

plot_win_str = str(rad_fact).replace(".", "p")
if fixed_r_ckpc > 0:
    plot_win_str = f"{fixed_r_ckpc}ckpc"

overwrite = True
clean = False  # no markers for halos/bhs
annotate = False
gal_markers = True
halo_markers = False

only_main_stars = True

nb_sims = len(sim_dirs)


zsims = []
common_zs = []
ztol = 0.01

for sim_dir in sim_dirs:

    sim = ramses_sim(sim_dir, nml="cosmo.nml")

    zsims.append(1.0 / sim.get_snap_exps() - 1.0)

for izlist, zlist in enumerate(zsims):
    for iz, z in enumerate(zlist):
        if z in common_zs:
            continue
        found = 1
        for jzlist in range(nb_sims):
            if jzlist == izlist:
                continue
            if np.any(np.abs(z - zsims[jzlist]) < ztol):
                found += 1
        if found == nb_sims:
            common_zs.append(z)

common_zs = np.unique(np.round(common_zs, decimals=2))

common_zs = np.sort(common_zs)[::-1]
common_zs = common_zs[common_zs <= zmax]

med_spacing = np.median(np.abs(np.diff(common_zs)))
max_z = np.max(common_zs)
min_z = np.min(common_zs)

done_snaps = [[] for isim in range(nb_sims)]

print(
    f"Found {len(common_zs)} common redshifts, min: {min_z:.2f}, max: {max_z:.2f}, median spacing: {med_spacing:.2f}"
)


hagn_sim = get_hagn_sim()


if sim.path[-1] == "/":
    ind = -2
else:
    ind = -1

data_path = os.path.join("/", *sim.path.split("/")[:ind], "common_plots")

rgal = "rgal"
if fixed_r_ckpc > 0:
    rgal = ""

if only_main_stars:
    main_st_str = "mainStars"
else:
    main_st_str = ""

outdir = os.path.join(
    data_path,
    "maps_own_tree",
    "halo",
    f"{plot_win_str}{rgal:s}",
    f"{main_st_str:s}",
)

all_sim_names = [ramses_sim(sim_d).name for sim_d in sim_dirs]
all_sim_names_str = "_".join(all_sim_names)


for iz, tgt_z in enumerate(common_zs):

    tgt_z_str = f"{tgt_z:.2f}".replace(".", "p")

    fout = os.path.join(
        outdir,
        f"{field.replace(' ','_')}_{all_sim_names_str:s}_{iz:03d}.png",
    )

    if os.path.isfile(fout) and not overwrite:
        continue

    fig, ax = plt.subplots(
        nb_sims, 1, figsize=(8, 16), dpi=800
    )  # , layout="constrained")
    plt.subplots_adjust(0, 0, 1, 1, wspace=0.0, hspace=0.0)

    plt.axis("tight")

    # set face color to black in figure
    fig.patch.set_facecolor("black")

    plotted = 0

    for isim, sim_dir in enumerate(sim_dirs):

        ax[isim].tick_params(
            bottom=False, left=False, labelbottom=False, labelleft=False
        )

        sim = ramses_sim(sim_dir, nml="cosmo.nml")

        name = sim.name
        intID = int(name.split("id")[-1].split("_")[0])

        zoom_ctr = sim.zoom_ctr

        if np.all(zoom_ctr == [0.5, 0.5, 0.5]):
            centre = True
            # zero_point = [0.5, 0.5, 0.5]
        else:
            centre = False
            # zero_point = [0, 0, 0]

        if "refine_params" in sim.namelist:
            if "rzoom" in sim.namelist["refine_params"]:
                zoom_r = sim.namelist["refine_params"]["rzoom"]
            else:
                zoom_r = sim.namelist["refine_params"]["azoom"]

        else:

            pass

        snaps = sim.snap_numbers
        aexps = sim.get_snap_exps(param_save=False)
        times = sim.get_snap_times(param_save=False)

        zstt = 2.0
        max_zed = np.inf

        pdf = False
        # zdist = 50  # ckpc
        # hm = "HaloMaker_DM_dust"

        sim.init_cosmo()
        l = sim.unit_l(sim.aexp_stt) / (ramses_pc * 1e6) / sim.aexp_stt

        sim_aexps = sim.get_snap_exps()  # [::-1]
        sim_times = sim.get_snap_times()  # [::-1]
        sim_snaps = sim.snap_numbers  # [::-1]
        assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
        assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

        avail_aexps = np.intersect1d(
            sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
        )
        avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

        closest_snap_to_tgt = sim.get_closest_snap(zed=tgt_z)

        print(tgt_z, np.abs(1 / aexps - 1 - tgt_z).min())

        # print(sim.name, done_snaps[isim])

        for snap, aexp, time in zip(sim_snaps, sim_aexps, sim_times):

            if snap != closest_snap_to_tgt or snap in done_snaps[isim]:
                continue

            # print(avail_aexps)

            # print(sim.name, snap, aexp, time, "running")

            # tgt_zed = max(tgt_zed, 1.0 / avail_aexps[-1] - 1.0)

            # print(tgt_zed)

            try:
                hid_start, halos, galaxies, start_aexp, found = starting_hid_from_hagn(
                    zstt, sim, hagn_sim, intID, avail_aexps, avail_times, ztgt=2.0
                )
            except IndexError:
                continue

            if not found:
                continue

            start_zed = 1.0 / start_aexp - 1.0

            directions = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

            tree_name = os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat")
            byte_file = os.path.join(sim.path, "TreeMakerDM_dust")

            tree_hids, tree_datas, tree_aexps = read_tree_file_rev(
                tree_name,
                byte_file,
                start_zed,
                [hid_start],
                # tree_type="halo",
                tgt_fields=["m", "x", "y", "z", "r"],
                debug=False,
                star=False,
            )
            tree_times = sim.cosmo_model.age(1.0 / tree_aexps - 1.0).value * 1e3  # Myr

            filt = tree_datas["x"][0] != -1

            for key in tree_datas:
                tree_datas[key] = tree_datas[key][0][filt]
            tree_hids = tree_hids[0][filt]
            tree_aexps = tree_aexps[filt]
            tree_times = tree_times[filt]

            if not os.path.exists(outdir):
                os.makedirs(outdir, exist_ok=True)

            option_str = ""
            if annotate and not clean:
                option_str += "_annotate"
            elif clean:
                option_str += "_clean"

            gal_props_tree = get_assoc_pties_in_tree(
                sim,
                tree_aexps,
                tree_hids,
                assoc_fields=["r50", "rmax", "mass", "pos", "host hid"],
            )

            r50s = gal_props_tree["r50"]
            rmaxs = gal_props_tree["rmax"]
            masses = gal_props_tree["mass"]
            poss = gal_props_tree["pos"]
            hids = gal_props_tree["host hid"]

            smooth_gal_props = smooth_props(gal_props_tree)
            if not os.path.exists(get_halo_assoc_file(sim.path, snap)):
                continue

            # zed = 1.0 / aexp - 1.0
            # if abs(zed - tgt_z) > 0.01:
            #     continue

            if fixed_r_ckpc <= 0:
                if use_r50:
                    rstr = "r50"
                else:
                    rstr = "rmax"
            else:
                rstr = ""

            l_pMpc = l * aexp

            cur_hid = tree_hids[np.argmin(np.abs(tree_aexps - aexp))]

            cur_gid, hosted_gals = get_central_gal_for_hid(
                sim, cur_hid, snap, main_stars=only_main_stars, verbose=False
            )

            if cur_gid is None:
                continue

            # _, cur_gal_props = get_gal_props_snap(
            #     sim_dir, snap, cur_gid, main_stars=only_main_stars
            # )

            aexp_arg = np.argmin(np.abs(aexp - gal_props_tree["aexps"]))

            cur_r50 = smooth_gal_props["r50"][aexp_arg]

            if not use_r50:
                cur_rad = smooth_gal_props["rmax"][aexp_arg]
            else:
                cur_rad = cur_r50

            cur_mass = masses[aexp_arg]
            # cur_pos = poss[aexp_arg]
            # cur_pos = cur_gal_props["pos"]
            cur_pos = gal_props_tree["pos"][aexp_arg]
            tgt_pos = cur_pos  # centre on the galaxy we found...
            # tgt_pos = cur_posfg

            rad_tgt = cur_rad * rad_fact

            if fixed_r_ckpc > 0:
                rad_tgt = fixed_r_ckpc / (sim.cosmo.lcMpc * 1e3)

            zdist = rad_tgt / 1 * sim.cosmo.lcMpc * 1e3

            if isim == 0:
                dist_bar = 5  # ckpc

                ax[isim].plot(
                    [-18, -18 + dist_bar],
                    [-13, -13],
                    color="white",
                    lw=3.5,
                    zorder=999,
                )
                ax[isim].text(
                    -18 + dist_bar / 2,
                    -12.33,
                    f"{dist_bar} ckpc",
                    color="white",
                    ha="center",
                    zorder=999,
                    size=15,
                )

            # print(tgt_pos, tgt_rad, rad_tgt, zdist)
            # print(tgt_pos - tgt_rad * 0.15, tgt_pos + tgt_rad * 0.15)

            # print(rad_tgt * 5, rad_tgt * 5 * sim.cosmo.lcMpc * 1e3)

            args = {
                "cb": False,
                "cmap": cmap,
                "vmin": vmin,
                "vmax": vmax,
                "mode": mode,
                "transpose": True,
                "color": marker_color,
                "hid": int(hids[aexp_arg]),
            }

            try:
                img = plot_fields(
                    field,
                    fig,
                    ax[isim],
                    aexp,
                    directions,
                    tgt_pos,
                    rad_tgt,
                    sim,
                    **args,
                )
            except AssertionError:
                continue

            ax[isim].set_ylabel("")
            ax[isim].set_xlabel("")

            # if isim == nb_sims - 1:
            #     cbar = fig.colorbar(
            #         img,
            #         ax=ax,
            #         orientation="vertical",
            #         label=field,
            #         cmap=cmap,
            #         fraction=0.08,
            #         pad=0.01,
            #     )
            #     cbar.ax.yaxis.set_tick_params(color='white')
            #     cbar.ax.yaxis.set_label_text(field, color='white')
            #     plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

            if isim == 0:
                ax[isim].text(
                    0.05,
                    0.9,
                    "z = %.2f" % (1.0 / aexp - 1),
                    color="white",
                    transform=ax[isim].transAxes,
                    path_effects=[pe.withStroke(linewidth=1, foreground="black")],
                    ha="left",
                    size=20,
                    zorder=999,
                )

            if not clean:

                circ = Circle(
                    (
                        (zoom_ctr[0] - tgt_pos[0]) * sim.cosmo.lcMpc * 1e3,
                        (zoom_ctr[1] - tgt_pos[1]) * sim.cosmo.lcMpc * 1e3,
                    ),
                    zoom_r * sim.cosmo.lcMpc * 1e3,
                    fill=False,
                    edgecolor=marker_color,
                    ls=":",
                    lw=2,
                    zorder=999,
                )
                circ = Circle(
                    (
                        (0) * sim.cosmo.lcMpc * 1e3,
                        (0) * sim.cosmo.lcMpc * 1e3,
                    ),
                    cur_r50 * sim.cosmo.lcMpc * 1e3,
                    fill=False,
                    edgecolor=marker_color,
                    ls=":",
                    lw=2,
                    zorder=999,
                )

                ax[isim].add_patch(circ)

                # plot zoom galaxies
                if gal_markers:
                    plot_zoom_gals(
                        ax[isim],
                        snap,
                        sim,
                        tgt_pos,
                        rad_tgt,
                        zdist,
                        hm="HaloMaker_stars2_dp_rec_dust",
                        gal_markers=gal_markers,
                        annotate=annotate,
                        direction=directions[0],
                        max_mass=max_gal_mass,
                        min_mass=min_gal_mass,
                        # transpose=True,
                        **args,
                    )

                # plot zoom hals
                if halo_markers:
                    plot_zoom_halos(
                        ax[isim],
                        snap,
                        sim,
                        tgt_pos,
                        rad_tgt,
                        zdist,
                        hm="HaloMaker_DM_dust",
                        gal_markers=gal_markers,
                        annotate=annotate,
                        direction=directions[0],
                        # transpose=True,
                        **args,
                    )

                # plot zoom BHs
                try:
                    plot_zoom_BHs(
                        ax[isim],
                        snap,
                        sim,
                        tgt_pos,
                        rad_tgt,
                        zdist,
                        direction=directions[0],
                        max_mass=max_bh_mass,
                        min_mass=min_bh_mass,
                        **args,
                    )
                except (ValueError, AssertionError):
                    pass

            plotted += 1
            done_snaps[isim].append(snap)

    if plotted == nb_sims:
        print(f"writing {fout}")

        fig.savefig(
            fout,
            dpi=300,
            format="png",
        )
        if pdf:
            fig.savefig(
                fout.replace(".png", ".pdf"),
                dpi=300,
                format="pdf",
            )

        plt.close()

        # break
