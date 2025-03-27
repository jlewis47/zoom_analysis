"""
functions for manipulating halo/galaxy association catalogs made with assoc.py
"""

from pydoc import text
import numpy as np
import h5py
import os

import matplotlib.pyplot as plt

from gremlin.read_sim_params import ramses_sim

# from scipy.interpolate import make_smoothing_spline, make_interp_spline, make_lsq_spline
from scipy.ndimage.filters import median_filter
from matplotlib.lines import Line2D

# from zoom_analysis.zoom_helpers import project_direction


def get_halo_assoc_file(sim_dir, snap):

    return os.path.join(sim_dir, "association", f"assoc_{snap:03d}_halo_lookup.h5")


def get_gal_assoc_file(sim_dir, snap):

    return os.path.join(sim_dir, "association", f"assoc_{snap:03d}_gal_lookup.h5")


def find_star_ctr_period(pos):
    parts_pos = np.copy(pos)

    axis_dims = np.argmin(parts_pos.shape)
    axis_parts = np.argmax(parts_pos.shape)

    part_med = np.median(parts_pos, axis=axis_parts)

    too_large = np.linalg.norm(parts_pos - part_med, axis=axis_dims) > 0.5
    if np.any(too_large):
        up = parts_pos > 0.5
        dw = parts_pos <= 0.5

        # print(parts_pos)
        # print(parts_pos[too_large])
        if axis_parts == 0:
            parts_pos[too_large * up] -= 1
            parts_pos[too_large * dw] += 1
        elif axis_parts == 1:
            parts_pos[:, too_large * up] -= 1
            parts_pos[:, too_large * dw] += 1

    part_ctr = np.mean(parts_pos, axis=axis_parts)
    part_ext = np.max(np.abs(parts_pos - part_ctr), axis=axis_parts)
    return parts_pos, part_ctr, part_ext


def find_zoom_tgt_halo(
    sim: ramses_sim,
    snap,
    pure_thresh=0.9999,
    debug=False,
    tgt_mass=None,
    tgt_ctr=None,
    tgt_rad=None,
):
    """
    find the zoom target in the association catalog
    """

    # print(tgt_mass, tgt_ctr, tgt_rad)

    # sim = ramses_sim(sim_dir, nml="cosmo.nml")

    # snap = sim.get_closest_snap(zed=ztgt)

    if tgt_ctr is None and tgt_rad is None:
        zoom_ctr = sim.zoom_ctr

        if "rzoom" in sim.namelist["refine_params"]:
            rzoom = sim.namelist["refine_params"]["rzoom"]
        else:
            rzoom = np.max(
                [
                    sim.namelist["refine_params"]["azoom"],
                    sim.namelist["refine_params"]["bzoom"],
                    sim.namelist["refine_params"]["czoom"],
                ]
            )

    else:

        zoom_ctr = tgt_ctr
        rzoom = tgt_rad

    # get assoc_halos
    hfile = get_halo_assoc_file(sim.path, snap)

    # print(snap)

    with h5py.File(hfile, "r") as f:
        fpures = f["fpure"][()]
        hids = f["hid"][()]
        mvirs = f["mvir"][()]

        is_pure = fpures > pure_thresh

        hids = hids[is_pure]

        mvirs = mvirs[is_pure]

        ok_rzooms = np.zeros_like(hids, dtype=bool)
        rzooms = np.zeros_like(hids, dtype=float)

        # if True:
        if debug:
            # print(hids, mvirs)
            print(len(list(f.keys())))

        for ihalo, (hid, mvir) in enumerate(zip(hids, mvirs)):

            hkey = f"halo_{hid:07d}"

            if hkey in f:
                hdset = f[hkey]

                pos = hdset["pos"][()]

                # check within rzoom of zoom_ctr
                rzooms[ihalo] = np.linalg.norm(pos - zoom_ctr)
                ok_rzooms[ihalo] = rzooms[ihalo] < rzoom
            else:
                ok_rzooms[ihalo] = False

            # print(
            #     hid,
            #     mvir,
            #     np.linalg.norm(pos - zoom_ctr),
            #     rzoom,
            # )
        # print(rzooms.min())

        # print(list(zip(hids, mvirs, ok_rzooms, rzooms)))

        # print(hids[ok_rzooms], mvirs[ok_rzooms], rzooms.min(), rzooms.max(), rzoom)

        if tgt_mass == None:
            # most massive halo in rzoom that passes the purity test
            tgt_hid = hids[ok_rzooms][np.argmax(mvirs[ok_rzooms])]
            found_mass = mvirs[ok_rzooms][np.argmax(mvirs[ok_rzooms])]
        else:
            # print(tgt_mass)
            tgt_hid = hids[ok_rzooms][np.argmin(np.abs(mvirs[ok_rzooms] - tgt_mass))]
            found_mass = mvirs[ok_rzooms][
                np.argmin(np.abs(mvirs[ok_rzooms] - tgt_mass))
            ]
            # print(tgt_mass, found_mass)

        print(rzooms[tgt_hid == hids] * sim.cosmo.lcMpc * 1e3, "kpc from ctr")

        out_dict = {}
        hosted_galaxies = {}
        hdset = f[f"halo_{tgt_hid:07d}"]
        for k in hdset.keys():
            if k != "galaxies":
                out_dict[k] = hdset[k][()]
            else:
                for gk in hdset[k].keys():
                    hosted_galaxies[gk] = hdset[k][gk][()]

        out_dict["mass"] = found_mass

    return tgt_hid, out_dict, hosted_galaxies


def compute_r200(H0_si, om_m, om_b, zed, massoc_kg):
    """
    M200 = massoc - > compute r200 using DM density
    """

    return np.cbrt(
        (6.67e-11 * 100.0 * massoc_kg)
        / (H0_si**2.0 * (1.0 + zed) ** 3.0 * (om_m - om_b))
    )


def get_halo_props_snap(sim_dir, snap, hid=None):

    hfile = get_halo_assoc_file(sim_dir, snap)

    out_dict = {}

    with h5py.File(hfile, "r") as f:

        hids = f["hid"][()]

        if hid != None:

            # arg = np.where(hid == hids)[0]

            out_dict = {}
            hosted_galaxies = {}
            hdset = f[f"halo_{hid:07d}"]
            # halo_arg = np.in1d(f.keys(), [f"halo_{hid:07d}"])[0]
            halo_arg = np.where(hids == hid)[0][0]

            # print(hid, hids[halo_arg])
            fpure = f["fpure"][()][halo_arg]
            mvir = f["mvir"][()][halo_arg]

            out_dict["fpure"] = fpure
            out_dict["mvir"] = mvir
            for k in hdset.keys():
                if k == "Ngals":
                    continue

                if k != "galaxies":
                    out_dict[k] = hdset[k][()]
                else:
                    for gk in hdset[k].keys():

                        hosted_galaxies[gk] = hdset[k][gk][()]
            return out_dict, hosted_galaxies

        else:  # no specified hid, take all halos, no galaxies

            out_dict = {}
            for k in f.keys():
                if "halo" in k:
                    out_dict[k] = {}
                    for field in f[k].keys():
                        if field != "galaxies":
                            out_dict[k][field] = f[k][field][()]
                else:
                    out_dict[k] = f[k][()]

            return out_dict


def get_star_grp(main_stars):

    if main_stars:
        return "main_stars"
    else:
        return "all_stars"


# def get_central_gal_for_hid(
#     sim: ramses_sim,
#     hid,
#     snap,
#     search_rad=None,
#     prev_mass=-1,
#     prev_pos=-1,
#     verbose=False,  # , prev_pos=None
#     main_stars=False,
# ):

#     gfile = get_gal_assoc_file(sim.path, snap)

#     hdict, _ = get_halo_props_snap(sim.path, snap, hid)

#     rvir = hdict["rvir"]

#     st_grp = get_star_grp(main_stars)

#     mass_thresh = 0.5
#     # mass_thresh = 0
#     # pos_max_dist = 200 / (sim.cosmo.lcMpc * 1e3)  # ckpc
#     r_thresh = 1.0
#     r_increment = 1.5

#     found = False
#     if search_rad == None:
#         search_rad = 0.2 * rvir

#     print(f"reading {gfile}")

#     with h5py.File(gfile, "r") as f:

#         host_hids = f["host hid"][()]
#         in_host = host_hids == hid
#         # in_host = np.full(len(f["mass"][()]), True)
#         if in_host.sum() == 0:
#             if verbose:
#                 print("no hosted gals")
#             return None, None
#         arg_host = np.where(in_host)[0]
#         if st_grp in f:
#             masses = f[st_grp]["mass"][()][arg_host]
#         else:
#             if st_grp == "main_stars":
#                 print(f"no main_stars group - taking all stars")
#             masses = f["mass"][()][arg_host]
#         pos = f["pos"][()].swapaxes(0, 1)[arg_host, :]
#         central_cond = f["central"][()][arg_host] == 1

#         multi_hosts = len(arg_host) > 1
#         #

#         # print(masses,in_host,multi_hosts)
#         # print(rvir)

#         # print(list(zip(f["gids"][()][arg_host], masses, central_cond)))

#         if multi_hosts or masses < (prev_mass * mass_thresh):

#             if verbose:
#                 print("main search")

#             while not found and (search_rad <= (r_thresh * rvir * r_increment)):

#                 # print(gfile)

#                 # print(gfile, hid, snap)

#                 # print(f.keys())

#                 if multi_hosts:
#                     dists = np.linalg.norm(pos - hdict["pos"][()], axis=1)
#                 else:
#                     dists = np.linalg.norm(pos - hdict["pos"][()])

#                 dist_cond = dists < search_rad

#                 # print(centrals.sum(), (host_hids == hid).sum(), dists.min(), search_rad)

#                 # print(host_hids[centrals], hid)
#                 # print(host_hids)

#                 # print(np.sum(host_hids == hid), np.sum(centrals))
#                 # print(np.sort(host_hids, hid)
#                 # print((host_hids == hid).sum(), (centrals).sum())

#                 # print(host_hids, centrals, hid)
#                 # print(len(host_hids), len(centrals))

#                 cond = dist_cond * central_cond
#                 found = np.sum(cond) > 0

#                 if not found:
#                     if verbose:
#                         print(f"no central gals in search radius {search_rad}")
#                 elif np.all(masses[cond] < (prev_mass * mass_thresh)):
#                     if verbose:
#                         print(f"prev mass {prev_mass} too high, {masses[cond]}")
#                     found = False

#                 # print(search_rad, masses[cond], found)

#                 search_rad *= r_increment

#             # print(found)
#             if not found:

#                 if verbose:
#                     print("no hosted gals in search radius")
#                 # print(f["gids"][()][arg_host], masses, dists)

#                 return None, None

#             arg_ctr = np.where(cond)[0]
#             arg = arg_host[arg_ctr]
#             # print(arg, np.sum((host_hids == hid)))

#             # if more than one then take the most massive
#             if len(arg) > 1:
#                 arg = np.argmax(masses[arg_ctr])
#             else:
#                 arg = arg[0]

#             # print(list(zip(f["gids"][()][arg_host],masses,centrals)))

#         else:

#             arg = arg_host[0]

#         # print(f["mass"][()][arg], np.max(f["mass"][()][(host_hids == hid)]))
#         # print(
#         #     f["mass"][()][(host_hids == hid)],
#         #     np.linalg.norm(
#         #         f["pos"][()].swapaxes(0, 1)[(host_hids == hid)] - hdict["pos"][()],
#         #         axis=1,
#         #     ),
#         # )

#         out_dict = {}

#         read_all_gal_assoc(st_grp, f, out_dict, arg=arg)

#         gid = f["gids"][()][arg]

#     if verbose:
#         print(f"central gal found: {gid}")

#     return gid, out_dict


def get_central_gal_for_hid(
    sim: ramses_sim,
    hid,
    snap,
    search_rad=None,
    prev_mass=-1,
    prev_pos=-1,
    prev_rad=-1,
    verbose=False,  # , prev_pos=None
    debug=False,
    main_stars=False,
    # centrals=True,
    mass_thresh=0.2,
    r_thresh=0.5,
):

    from zoom_analysis.visu.visu_fct import plot_fields, basis_from_vect

    gfile = get_gal_assoc_file(sim.path, snap)

    hdict, _ = get_halo_props_snap(sim.path, snap, hid)

    rvir = hdict["rvir"]
    host_pos = hdict["pos"][()]

    st_grp = get_star_grp(main_stars)

    # mass_thresh = 0.7
    # mass_thresh = 0
    # pos_max_dist = 200 / (sim.cosmo.lcMpc * 1e3)  # ckpc
    # r_thresh = 0.5
    r_increment = 1.2

    found = False
    if search_rad == None:
        search_rad = 0.1 * rvir

    if verbose:
        print(f"reading {gfile}")

    with h5py.File(gfile, "r") as f:

        host_hids = f["host hid"][()]
        in_host = host_hids == hid
        # in_host = np.full(len(f["mass"][()]), True)
        if in_host.sum() == 0:
            if verbose:
                print("no hosted gals")
            return None, None
        arg_host = np.where(in_host)[0]
        # print("arg_host", arg_host)
        if st_grp in f:
            masses = f[st_grp]["mass"][()][arg_host]
        else:
            if st_grp == "main_stars":
                print(f"no main_stars group - taking all stars")
            masses = f["mass"][()][arg_host]
        pos = f["pos"][()].swapaxes(0, 1)[arg_host, :]
        all_gids = f["gids"][()]
        gids = all_gids[arg_host]
        central_cond = f["central"][()][arg_host] == 1
        multi_hosts = len(arg_host) > 1

        gid = -1

        # print(f"hid {hid}, {len(arg_host)} gals")

        if multi_hosts:

            if verbose:
                print("main search")

            found = False
            arg = -1
            while not found and (search_rad <= (r_thresh * rvir * r_increment)):
                # check against prev mass
                mass_cond = np.full(len(arg_host), True)
                if prev_mass != -1:
                    # mass_cond = masses <= mass_thresh * prev_mass
                    mass_cond = masses / prev_mass >= mass_thresh * (
                        masses * 1.0 < prev_mass
                    )
                    if debug:
                        print(
                            "mass thresh",
                            masses / prev_mass,
                            mass_thresh,
                        )

                dists = np.linalg.norm(pos - host_pos, axis=1)
                pos_cond = dists < search_rad

                prev_pos_cond = np.full(len(arg_host), True)
                if np.all(prev_pos != -1):
                    prev_dists = np.linalg.norm(pos - prev_pos, axis=1)
                    if prev_rad != -1:
                        prev_pos_cond = prev_dists < prev_rad
                    else:
                        prev_pos_cond = prev_dists < search_rad
                        # print(pos, prev_pos)
                        # print(prev_dists, search_rad)

                all_cond = pos_cond * mass_cond * prev_pos_cond
                if np.sum(all_cond) > 0:
                    if verbose or debug:
                        print(f"found {np.sum(all_cond)} gals")
                    # try most massive
                    gid = gids[all_cond][np.argmax(masses[all_cond])]
                    arg = arg_host[all_cond][np.argmax(masses[all_cond])]
                    # print(list(zip(arg_host[all_cond], masses[all_cond])), arg)
                    # print(gid, all_gids[arg])
                # else:
                # if verbose:
                # print(f"no gals in search radius {search_rad}")
                # print(pos_cond.sum(), mass_cond.sum(), prev_pos_cond.sum())

                found = gid != -1
                search_rad *= r_increment

        else:

            arg = arg_host
            gid = gids[0]

            # print(f"single host, {len(arg)} gals")
            # print(list(zip(arg_host, masses)))
            # print(gids, arg)
            # print()
            found = True

        out_dict = {}

        read_all_gal_assoc(st_grp, f, out_dict, arg=arg)

    if (debug or (verbose and not found)) and multi_hosts:

        # print(gid, gids[in_host][np.argmax(masses[all_cond])])
        # gid = gids[in_host][np.argmax(masses[all_cond])]

        # print(gid, out_dict)

        if multi_hosts:
            order = np.argsort(masses)
            order_max = np.argsort(masses[all_cond])
            print("snap", snap)
            print("prev mass", np.log10(prev_mass))
            print(
                gids[all_cond][order_max],
                np.log10(masses[all_cond][order_max]),
            )
            print(np.linalg.norm(pos[all_cond] - prev_pos, axis=1), prev_rad)
            print(
                list(
                    zip(
                        gids[order],
                        np.log10(masses[order]),
                        pos_cond[order],
                        mass_cond[order],
                        np.abs(np.log10(masses) - np.log10(prev_mass))
                        / np.log10(prev_mass),
                        prev_pos_cond,
                        np.linalg.norm(pos - prev_pos, axis=1),
                    )
                )
            )

        fig, ax = plt.subplots(figsize=(8, 8))

        fig_dm, ax_dm = plt.subplots(figsize=(8, 8))

        aexp = sim.get_snap_exps(snap)[0]

        code_to_kpc = sim.cosmo.lcMpc * 1e3

        # print(search_rad * code_to_kpc)

        plot_fields(
            "stellar mass",
            fig,
            ax,
            aexp,
            directions=[0, 0, 1],
            tgt_pos=host_pos,
            tgt_rad=search_rad * 1.25,  # * code_to_kpc,
            # tgt_rad=rvir * 1,
            sim=sim,
            hid=hid,
            log=True,
            vmin=1e5,
            vmax=1e7,
            zero_ctr=False,
            units="kpc",
            transpose=False,
            lim=False,
        )
        plot_fields(
            "dm mass",
            fig_dm,
            ax_dm,
            aexp,
            directions=[0, 0, 1],
            tgt_pos=host_pos,
            tgt_rad=search_rad * 1.25,  # * code_to_kpc,
            # tgt_rad=rvir * 1,
            sim=sim,
            hid=hid,
            log=True,
            vmin=1e8,
            vmax=1e10,
            zero_ctr=False,
            units="kpc",
            transpose=False,
            lim=False,
        )

        # print(pos[mass_cond, 0] * code_to_kpc)

        # dv1, dv2, dv3 = basis_from_vect([0, 0, 1])
        # M_basis = project_direction(dv1, [0, 0, 1])

        # plot_pos = np.dot(M_basis, pos)
        # plot_prev_pos = np.dot(M_basis, prev_pos)
        # plot_halo_pos = np.dot(M_basis, host_pos)
        plot_pos = pos
        plot_prev_pos = prev_pos
        plot_halo_pos = host_pos

        if prev_mass != -1:
            ax.scatter(
                plot_pos[mass_cond, 0] * code_to_kpc,
                plot_pos[mass_cond, 1] * code_to_kpc,
                # edgecolor="blue",
                color="blue",
                marker="x",
            )

        for a in [ax, ax_dm]:
            a.scatter(
                plot_pos[pos_cond, 0] * code_to_kpc,
                plot_pos[pos_cond, 1] * code_to_kpc,
                edgecolor="red",
                facecolor="none",
                marker="o",
            )
            # plot circle around host_pos in red
            circ2 = plt.Circle(
                plot_halo_pos[:2] * code_to_kpc,
                search_rad * code_to_kpc,
                color="red",
                fill=False,
            )
            a.add_artist(circ2)

        if np.all(prev_pos != -1):
            for a in [ax, ax_dm]:
                a.scatter(
                    plot_pos[prev_pos_cond, 0] * code_to_kpc,
                    plot_pos[prev_pos_cond, 1] * code_to_kpc,
                    edgecolor="green",
                    facecolor="none",
                    marker="P",
                )
                a.scatter(
                    plot_prev_pos[0] * code_to_kpc,
                    plot_prev_pos[1] * code_to_kpc,
                    # edgecolor="green",
                    color="green",
                    marker="x",
                )
                # plot circle around prev_pos in green
                if prev_rad != -1:
                    rcirc = prev_rad
                else:
                    rcirc = search_rad
                circ = plt.Circle(
                    plot_prev_pos[:2] * code_to_kpc,
                    rcirc * code_to_kpc,
                    color="green",
                    fill=False,
                )
                a.add_artist(circ)
        ax.scatter(
            plot_pos[central_cond, 0] * code_to_kpc,
            plot_pos[central_cond, 1] * code_to_kpc,
            edgecolor="orange",
            facecolor="none",
            marker="D",
        )

        labels = ["prev mass", "pos", "prev pos", "central"]
        markers = [
            Line2D(
                [0],
                [0],
                marker="x",
                ls="none",
                markeredgecolor="blue",
                markerfacecolor="none",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                ls="none",
                markeredgecolor="red",
                markerfacecolor="none",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="P",
                ls="none",
                markeredgecolor="green",
                markerfacecolor="none",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                ls="none",
                markeredgecolor="orange",
                markerfacecolor="none",
                markersize=10,
            ),
        ]

        # white circle around chosen galaxy
        if np.sum(all_cond) > 0:
            circ3 = plt.Circle(
                plot_pos[all_cond, :2][np.argmax(masses[all_cond]), :] * code_to_kpc,
                10,
                color="white",
                fill=False,
            )
            ax.add_artist(circ3)

        ax.legend(
            markers,
            labels,
            ncol=2,
            framealpha=0.0,
            labelcolor="white",
        )

        ax.text(
            0.1,
            0.9,
            f"z={1./aexp-1:.2f}",
            color="white",
            transform=ax.transAxes,
        )

        if not os.path.exists("debug_assoc"):
            os.mkdir("debug_assoc")

        fig.savefig(f"debug_assoc/stellar_mass_debug_assoc_{snap}.png")
        fig_dm.savefig(f"debug_assoc/dm_mass_debug_assoc_{snap}.png")

    if not found and np.any(prev_pos != host_pos):
        print("no central gal found - trying to look around halo centre")
        gid, out_dict = get_central_gal_for_hid(
            sim,
            hid,
            snap,
            prev_pos=host_pos,
            prev_rad=rvir * 0.1,
            prev_mass=prev_mass,
            verbose=verbose,
        )
        found = gid != -1

    if found:

        if verbose:
            print(f"central gal found: {gid}")

        return gid, out_dict
    else:
        print("no central gal found")
        if multi_hosts and verbose:
            print(pos_cond.sum(), mass_cond.sum(), prev_pos_cond.sum())
            print(np.linalg.norm(pos - host_pos, axis=1), search_rad * 3)
        return None, None


def read_all_gal_assoc(st_grp, f, out_dict, arg=None):
    gid = f["gids"][()]

    if type(arg) in [list, np.ndarray]:
        if len(arg) == 1:
            arg = arg[0]

    if np.any(arg == None):
        arg = np.arange(len(gid))

    gid = gid[arg]

    # print(gid, arg)

    for k in f.keys():
        # out_dict[k] = f[k][arg]
        if isinstance(f[k], h5py.Dataset):
            prop = f[k]
            if len(prop.shape) > 1 and prop.shape[0] == 3:
                out_dict[k] = prop[:, arg]
            else:
                out_dict[k] = prop[arg]

    if st_grp in f:
        # out_dict[st_grp] = {}
        for k in f[st_grp].keys():
            prop = f[st_grp][k]
            if len(prop.shape) > 1 and prop.shape[0] == 3:
                # out_dict[st_grp][k] = prop[:, arg]
                out_dict[k] = prop[:, arg]
            else:
                # out_dict[st_grp][k] = prop[arg]
                out_dict[k] = prop[arg]

    else:
        if st_grp == "main_stars":
            print(f"no main_stars group - taking all stars")


def find_zoom_tgt_gal(sim: ramses_sim, ztgt, pure_thresh=0.9999, debug=False):

    hid, out_dict, hosted_gals = find_zoom_tgt_halo(
        sim, ztgt, pure_thresh=pure_thresh, debug=debug
    )

    return get_central_gal_for_hid(sim, hid, sim.get_closest_snap(zed=ztgt))


def get_gal_props_snap(sim_dir, snap, gid=None, main_stars=False):

    st_grp = get_star_grp(main_stars)

    gfile = get_gal_assoc_file(sim_dir, snap)

    out_dict = {}

    with h5py.File(gfile, "r") as f:

        # print(f.keys())

        gids = f["gids"][()]

        # print(gid, gids)

        out_dict = {}

        if gid != None:
            arg = np.where(gid == gids)[0][0]
        else:
            arg = np.arange(len(gids))

        read_all_gal_assoc(st_grp, f, out_dict, arg=arg)

    if gid != None:
        return gid, out_dict
    else:
        return out_dict


def find_snaps_with_gals(snaps, sim_dir):

    snaps_with_gals = []

    for snap in snaps:
        gfile = get_gal_assoc_file(sim_dir, snap)

        if os.path.exists(gfile):
            snaps_with_gals.append(snap)

    return snaps_with_gals


def find_snaps_with_halos(snaps, sim_dir):

    snaps_with_halos = []

    for snap in snaps:
        hfile = get_halo_assoc_file(sim_dir, snap)

        if os.path.exists(hfile):
            snaps_with_halos.append(snap)

    return snaps_with_halos


# def get_sub_gals(snap, gid):

#     gprops = get_gal_props_snap(sim.path, snap)
#     hprops


#     hosting = gprops["host hid"] == gid


def get_rfrac(st_pos, st_mass, ctr, rfrac):

    dist_ctr = np.linalg.norm(st_pos - ctr[None, :], axis=1)
    order_pos = np.argsort(dist_ctr)

    mfrac = st_mass.sum() * rfrac
    cuml_ordered_mass = np.cumsum(st_mass[order_pos])
    masses_order_pos = st_mass[order_pos]

    return dist_ctr[np.argmin(np.abs(mfrac - cuml_ordered_mass))]


def get_r50(st_pos, st_mass, ctr):
    return get_rfrac(st_pos, st_mass, ctr, 0.5)


def get_r90(st_pos, st_mass, ctr):
    return get_rfrac(st_pos, st_mass, ctr, 0.9)


def get_assoc_pties_in_tree(
    sim, sim_tree_aexps, sim_tree_hids, assoc_fields=None, verbose=False
):

    if assoc_fields == None:
        assoc_fields = ["pos", "r50", "mass"]

    out_tree = {}
    for f in assoc_fields:
        if f not in ["pos", "vel", "velocity", "position"]:
            out_tree[f] = np.zeros(len(sim_tree_aexps))
        else:
            out_tree[f] = np.zeros((len(sim_tree_aexps), 3))

    out_tree["aexps"] = np.full(len(sim_tree_aexps), -1, dtype=np.float32)
    out_tree["gids"] = np.full(len(sim_tree_aexps), -1, dtype=np.int32)
    out_tree["rvir"] = np.full(len(sim_tree_aexps), -1, dtype=np.float32)
    out_tree["mvir"] = np.full(len(sim_tree_aexps), -1, dtype=np.float32)
    out_tree["fpure"] = np.full(len(sim_tree_aexps), -1, dtype=np.float32)
    out_tree["hpos"] = np.full((len(sim_tree_aexps), 3), -1, dtype=np.float64)

    prev_mvir = -1
    prev_hpos = -1
    prev_rvir = -1

    prev_mass = -1
    prev_pos = -1
    prev_rad = -1
    prev_time = -1

    for istep, (snap, aexp, time) in enumerate(
        zip(sim.snap_numbers[::-1], sim.aexps[::-1], sim.times[::-1])
    ):

        if np.all(np.abs(sim.aexps - aexp) > 1e-1):
            print(f"No assoc file for at z={1./aexp-1:.1f},snap={snap:d}")
            continue

        if not os.path.exists(get_gal_assoc_file(sim.path, snap)):
            print(f"No assoc file for at z={1./aexp-1:.1f},snap={snap:d}")
            continue

        sim_tree_arg = np.argmin(np.abs(sim_tree_aexps - aexp))
        daexp = np.abs(sim_tree_aexps[sim_tree_arg] - aexp)
        # print(sim_tree_hids, sim_tree_arg)
        cur_snap_hid = sim_tree_hids[sim_tree_arg]


        if cur_snap_hid in [0, -1] or daexp > 1e-2:
            print(f'Closesest snapshot too far for z={1./aexp-1:.1f},snap={snap:d}')
            # print(daexp,cur_snap_hid)
            continue

        hprops, _ = get_halo_props_snap(sim.path, snap, cur_snap_hid)

        if verbose:
            print(f"halo mass: {hprops['mvir']:.1e}, prev_mass: {prev_mvir:.1e}")
            print(f"halo pos: {hprops['pos']}, prev_pos: {prev_hpos}")
            print(f"halo rvir: {hprops['rvir']:.2e}, prev_rvir: {prev_rvir:.2e}")

        dist = np.linalg.norm(hprops["pos"] - prev_hpos)
        dt = abs(time - prev_time)

        speed = (
            dist
            / dt
            * (sim.cosmo.lcMpc * 1e6 * 3.08e16 / 1e3)
            / (1e6 * 3600 * 24 * 365.0)
        )  # km/s

        if istep > 0 and verbose:
            print(
                f"snap: {snap:d}, aexp:{aexp:.4f}, speed: {speed:.2e} km/s, dist: {dist*sim.cosmo.lcMpc*1e3:.2f} kpc, dt: {dt:.2e} Myr"
            )

        prev_mvir = hprops["mvir"]
        prev_hpos = hprops["pos"]
        prev_rvir = hprops["rvir"]
        prev_time = time

        prev_rad = (
            (dt * (365.25 * 24 * 3600) * 1e6)
            * (5e6 / (3.08e16 * 1e6))
            / sim.cosmo.lcMpc
        )
        # code distance

        # print(sim.name,snap,cur_snap_hid)

        gid, gal_dict = get_central_gal_for_hid(
            sim,
            cur_snap_hid,
            snap,
            prev_mass=prev_mass,
            prev_pos=prev_pos,
            prev_rad=prev_rad,
            verbose=verbose,
        )
        if gid == None:
            if verbose:
                print(f"No central galaxy at z={1./aexp-1:.1f},snap={snap:d}")
            continue

        prev_pos = gal_dict["pos"]
        prev_mass = gal_dict["mass"]

        for f in out_tree.keys():
            if f not in ["aexps", "rvir", "hpos", "mvir", "fpure"]:
                read = gal_dict[f]
                read_sh = np.asarray(list(read.shape))
                sh = read_sh[read_sh > 1]
                out_tree[f][sim_tree_arg] = read.reshape(sh)
        out_tree["aexps"][sim_tree_arg] = aexp
        out_tree["rvir"][sim_tree_arg] = hprops["rvir"]
        out_tree["mvir"][sim_tree_arg] = hprops["mvir"]
        out_tree["fpure"][sim_tree_arg] = hprops["fpure"]
        out_tree["hpos"][sim_tree_arg, :] = hprops["pos"]
        out_tree["gids"][sim_tree_arg] = gid

        # print(out_tree["aexps"][sim_tree_arg], aexp)

    filled = out_tree["aexps"] > -1

    for f in out_tree.keys():
        out_tree[f] = out_tree[f][filled]

    return out_tree


# def smooth(aexps, y, pty_name, sim, lam=0.001, max_vel=1000.0):

#     unit_ls = np.asarray([sim.unit_l(aexp) for aexp in aexps])

#     x = sim.cosmo_model.age(1.0 / aexps - 1.0).value * 1e3  # Myr
#     x = x * (365.25 * 24 * 3600) * 1e6  # s

#     xsmooth = np.copy(x)
#     ysmooth = np.copy(y)
#     done = False

#     fig, ax = plt.subplots()
#     fig2, ax2 = plt.subplots()
#     ax2.axhline(max_vel, ls="--", color="k")
#     ax2.axhline(-max_vel, ls="--", color="k")
#     while not done:

#         if pty_name in ["r50", "rmax", "rvir", "pos"]:
#             interp = make_smoothing_spline(xsmooth, ysmooth, lam=0)
#             deriv = interp.derivative()
#             vel = deriv(xsmooth) * unit_ls / 1e5  # km/s
#             small = np.abs(vel) <= max_vel

#             ax2.plot(xsmooth, vel, ls="--")

#         elif pty_name in ["vel"]:
#             small = np.abs(ysmooth) <= max_vel
#         else:
#             done = True
#             small = np.full(len(xsmooth), True)

#         done = np.all(small)
#         first_small = np.argmin(small)
#         small = np.full(len(xsmooth), True)
#         if not done:
#             if first_small < (len(xsmooth) - 1):
#                 print(first_small, len(xsmooth), len(small))
#                 small[first_small] = False
#                 unit_ls = unit_ls[small]
#                 xsmooth = xsmooth[small]
#                 ysmooth = ysmooth[small]
#                 ax.plot(xsmooth, ysmooth, ls="--")
#             else:
#                 done = True
#                 ax.plot(xsmooth, ysmooth, ls="-")

#         print(len(xsmooth), len(x))

#     # return xsmooth, ysmooth

#     ax.plot(x, y, label="raw", c="k")

#     spl = make_smoothing_spline(xsmooth, ysmooth, lam=lam)
#     ax.plot(x, spl(x), label="spl")


#     fig.savefig(f"{pty_name}_smooth.png")
#     fig2.savefig(f"{pty_name}_vel.png")
#     return spl(x)
def smooth(
    y, size=None, fct=median_filter, fct_pties={"mode": "nearest"}
):  # ,lam=None):

    if size == None:
        size = max(int(len(y) // 5), 1)

    ysmooth = fct(y, size=size, **fct_pties)

    # fig, ax = plt.subplots()
    # ax.plot(x, y, label="raw", c="k")
    # ax.plot(x, ysmooth, label="median", c="r")

    # spl = make_smoothing_spline(x, ysmooth, lam=lam)
    # ax.plot(x, spl(x), label="spl")
    # print(spl(x),ysmooth)

    # fig.savefig(f"{pty_name}_median.png")

    return ysmooth


def smooth_props(props):  # , l_kern=15):
    """
    smooth the radii in a tree
    """

    # def gauss(x, mu, sig):
    #     return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))

    # kern = gauss(np.linspace(-1, 1, l_kern), 0, 3.0)

    smooth_props = {}

    order = np.argsort(props["aexps"])

    rev_order = np.argsort(order)

    # max_vels = {"pos": 400, "r50": 100, "rmax": 100, "rvir": 100, "vel": 100}

    for f in props:

        if f in ["aexps", "host hid", "fpure", "gids"]:
            smooth_props[f] = props[f]
            continue

        sh = props[f].shape

        if len(sh) == 1:
            # props_padded = np.pad(props[f], (l_kern // 2, l_kern // 2), mode="edge")
            # if len(props_padded) >= l_kern + sh[0]:
            #     props_padded = props_padded[
            #         : -abs(l_kern + sh[0] - len(props_padded) + 1)
            #     ]
            # # smooth_props with a gaussian filter
            # smooth_props[f] = np.convolve(props_padded, kern, mode="valid") / np.sum(
            #     kern
            # )

            smooth_props[f] = smooth(
                # aexps[order], props[f][order], f, sim, lam=lam, max_vel=max_vels[f]
                props[f][order]
            )[rev_order]

        else:
            tmp_smooth_props = np.zeros((sh[0], sh[1]))
            for id, prop_1d in enumerate(props[f].T):
                #     props_padded = np.pad(prop_1d, (l_kern // 2, l_kern // 2), mode="edge")
                #     if len(props_padded) >= l_kern + sh[0]:
                #         props_padded = props_padded[
                #             : -abs(l_kern + sh[0] - len(props_padded) + 1)
                #         ]
                #     # smooth_props with a gaussian filter
                #     tmp_smooth_props[:, id] = np.convolve(
                #         props_padded, kern, mode="valid"
                #     ) / np.sum(kern)
                # smooth_props[f] = tmp_smooth_props

                tmp_smooth_props[:, id] = smooth(
                    # aexps[order], prop_1d[order], f, sim, lam=lam, max_vel=max_vels[f]
                    prop_1d[order]
                )
            smooth_props[f] = tmp_smooth_props[rev_order]

    return smooth_props

def find_shared_gals(sim_dirs,tgt_zed=2.0,mlim=1e10,rfact=2.0):

    from scipy.spatial import KDTree
    from zoom_analysis.zoom_helpers import recentre_coordinates

    sim = ramses_sim(sim_dirs[0], nml="cosmo.nml")

    # sim_snaps = sim.snap_numbers
    tgt_snap = sim.get_closest_snap(zed=tgt_zed)
    real_start_zed = 1.0 / sim.get_snap_exps(tgt_snap)[0] - 1.0

    gprops = get_gal_props_snap(sim.path, tgt_snap)
    hprops = get_halo_props_snap(sim.path, tgt_snap)

    gmass = gprops["mass"]
    fpure = gprops["host purity"] > 0.9999
    central = gprops["central"] == 1
    massive = (gmass > mlim)*(fpure) * (central)

    hosts = gprops["host hid"][massive]
    gids = gprops["gids"][massive]

    hids = hprops["hid"]
    found_hids,hid_args,host_args= np.intersect1d(hids,hosts,return_indices=True)
    hpos = np.asarray([hprops[f"halo_{hid:07d}"]['pos'] for hid in found_hids])
    if np.all(sim.zoom_ctr!=[0.5,0.5,0.5]):
        hpos = recentre_coordinates(hpos,sim.path,sim.zoom_ctr)
    rvir = np.asarray([hprops[f"halo_{hid:07d}"]['rvir'] for hid in found_hids])
    # hpos = hprops["pos"][hosts]
    hmass = gprops["host mass"][massive]


    all_hids = np.zeros((len(sim_dirs), len(hosts)), dtype=np.int32)-1
    all_gids = np.zeros((len(sim_dirs), len(hosts)), dtype=np.int32)-1
    all_snaps = np.zeros((len(sim_dirs)), dtype=np.int32)
    all_hmasses = np.zeros((len(sim_dirs), len(hosts)), dtype=np.float32)
    all_zeds = np.zeros((len(sim_dirs)), dtype=np.float32)

    all_hids[0] = hosts
    all_gids[0] = gids
    all_snaps[0]= tgt_snap
    all_hmasses[0] = hmass
    all_zeds[0] = real_start_zed

    full_args = np.all(all_hmasses>0,axis=0)



    for isim, sdir in enumerate(sim_dirs[:]):

        if isim==0:continue

        sim = ramses_sim(sdir, nml="cosmo.nml")

        # sim_snaps = sim.snap_numbers
        tgt_snap = sim.get_closest_snap(zed=tgt_zed)
        if abs(tgt_zed-(1./sim.get_snap_exps(tgt_snap)-1))>0.1:
            continue
        all_snaps[isim] = tgt_snap
        real_start_zed = 1.0 / sim.get_snap_exps(tgt_snap)[0] - 1.0
        all_zeds[isim] = real_start_zed

        gprops = get_gal_props_snap(sim.path, tgt_snap)
        hprops = get_halo_props_snap(sim.path, tgt_snap)

        loc_hosts = gprops["host hid"]
        found_hids,hid_args,host_args= np.intersect1d(hprops['hid'],loc_hosts,return_indices=True)

        loc_gids = gprops["gids"]
        loc_hmasses = gprops["host mass"]
        loc_hpos = np.asarray([hprops[f"halo_{hid:07d}"]['pos'] for hid in found_hids])
        if np.all(sim.zoom_ctr!=[0.5,0.5,0.5]):
            loc_hpos = recentre_coordinates(loc_hpos,sim.path,sim.zoom_ctr)
        loc_fpure = gprops["host purity"]
        # loc_rvirs = gprops["host rvir"]

        fpure_args = np.where(loc_fpure > 0.9999)[0]

        central_args = np.where(gprops["central"] == 1)[0]

        htree = KDTree(loc_hpos,boxsize=1+1e-6)

        nearby =  htree.query_ball_point(hpos,rvir*rfact)

        for iarg,args in enumerate(nearby):

            if type(args)!=int:
                if len(args)==0:
                    continue

            pure_and_loc_args = np.intersect1d(np.intersect1d(fpure_args,args), central_args)

            if type(pure_and_loc_args)!=int:
                if len(pure_and_loc_args)==0:
                    continue

            # loc_arg = np.argmax(loc_hmasses[pure_and_loc_args])
            loc_arg = np.argmin(np.abs(loc_hmasses[pure_and_loc_args]-hmass[iarg]))
            all_hids[isim,iarg] = loc_hosts[pure_and_loc_args[loc_arg]]
            all_gids[isim,iarg] = loc_gids[pure_and_loc_args[loc_arg]]
            all_hmasses[isim,iarg] = loc_hmasses[pure_and_loc_args[loc_arg]]
        
    return all_hids,all_gids,all_snaps,all_hmasses,all_zeds