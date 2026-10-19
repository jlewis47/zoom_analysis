from gremlin.read_sim_params import ramses_sim
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import healpy as hp

from zoom_analysis.rascas.rascas_steps import get_directions_cart
from zoom_analysis.trees.tree_reader import read_tree_file_rev
from zoom_analysis.visu.visu_fct import plot_fields, basis_from_vect, plot_zoom_gals

# from zoom_analysis.trees.tree_reader import
from zoom_analysis.halo_maker.assoc_fcts import get_gal_props_snap, get_gal_subs_snap
from zoom_analysis.zoom_helpers import project_direction


def los_idx_2_direction(idir, ndir=12):

    return list(basis_from_vect(get_directions_cart(hp.npix2nside(ndir))[idir]))


def plot_tmplt():

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")

    for a in ax:
        a.set_xlabel("kpc")
        a.set_ylabel("kpc")
        a.tick_params(which="both", direction="in")

    return fig, ax


def plot_galaxy_stars(fig, ax, aexp, directions, hid, tgt_pos, tgt_rad, sim, **kwargs):

    plot_fields(
        "stellar mass",
        fig,
        ax,
        aexp,
        directions,
        tgt_pos,
        tgt_rad,
        sim,
        hid=hid,
        **kwargs,
    )


def plot_galaxy_gas(fig, ax, aexp, directions, hid, tgt_pos, tgt_rad, sim, **kwargs):

    plot_fields(
        "density",
        fig,
        ax,
        aexp,
        directions,
        tgt_pos,
        tgt_rad,
        sim,
        hid=hid,
        **kwargs,
    )


def plot_substr(fig, ax, central_props, sub_props, step_times, direction, scale):

    Mbasis = project_direction(direction, [0, 0, 1])

    central_pos_2d = [
        np.dot([x, y, z], Mbasis)[:2]
        for x, y, z in zip(central_props["x"], central_props["y"], central_props["z"])
    ]

    subs_pos_2d = []
    nsubs = len(sub_props["m"])
    for isub in range(nsubs):

        subs_pos_2d.append(
            [
                np.dot(Mbasis, [x, y, z])[:2]
                for x, y, z in zip(
                    sub_props["x"][isub, :],
                    sub_props["y"][isub, :],
                    sub_props["z"][isub, :],
                )
            ]
        )

    subs_pos_2d = np.asarray(subs_pos_2d)

    subs_pos_2d_ctrd = subs_pos_2d - np.asarray(central_pos_2d).swapaxes(1, 2)

    # print(subs_pos_2d_ctrd)

    ax.scatter([0], [0], marker="+", c="k")

    # ax.scatter(
    #     subs_pos_2d_ctrd[:, 0, 0] * scale,
    #     subs_pos_2d_ctrd[:, 0, 1] * scale,
    #     marker="o",
    #     edgecolors="r",
    #     facecolors="None",
    # )

    lsteps = len(subs_pos_2d_ctrd)

    normed_times = (step_times - step_times.min()) / (
        step_times.max() - step_times.min()
    )
    colors = plt.cmap("viridis")(normed_times)

    for isub in range(nsubs):

        neg = np.where(sub_props["x"][isub, :] == -1)[0]
        if len(neg) > 0:
            stop = np.argmin(neg)
        else:
            stop = -1

        for istep in range(lsteps):

            if istep == stop:
                break

            ax.plot(
                subs_pos_2d_ctrd[isub, istep : istep + 1, 0] * scale,
                subs_pos_2d_ctrd[isub, istep : istep + 1, 1] * scale,
                ls="--",
                c=colors[istep],
            )


def annotate_substr():

    pass


def annotate_central():

    pass


if __name__ == "__main__":

    ArgParse = argparse.ArgumentParser()

    ArgParse.add_argument("simdir", type=str)
    ArgParse.add_argument("snapshot", type=int)
    ArgParse.add_argument("galid", type=int)
    ArgParse.add_argument("idir_rascas", type=int, default=1)

    args = ArgParse.parse_args()

    sim = ramses_sim(args.simdir, nml="cosmo.nml")

    tree_name = os.path.join(sim.path, "TreeMakerstars2_dp_rec_dust", "tree_rev.dat")
    byte_file = os.path.join(sim.path, "TreeMakerstars2_dp_rec_dust")

    gid, central_props = get_gal_props_snap(sim.path, args.snapshot, args.galid)
    subs_props = get_gal_subs_snap(sim.path, args.snapshot, args.galid)

    not_host = np.where(args.galid != subs_props["gids"])
    for k in subs_props.keys():
        if len(subs_props[k]) != 3:
            subs_props[k] = subs_props[k][not_host]
        else:
            subs_props[k] = subs_props[k][:, not_host]

    start_aexp = sim.get_snap_exps(args.snapshot)
    start_zed = 1.0 / start_aexp - 1

    central_gids, central_datas, central_aexps = read_tree_file_rev(
        tree_name,
        byte_file,
        start_zed,
        [args.galid],
        tgt_fields=["m", "x", "y", "z", "r"],
        debug=False,
        star=True,
    )

    # print(subs_props["gids"])

    sub_gids, sub_datas, sub_aexps = read_tree_file_rev(
        tree_name,
        byte_file,
        start_zed,
        subs_props["gids"],
        tgt_fields=["m", "x", "y", "z", "r", "central"],
        debug=False,
        star=True,
    )

    sim.init_cosmo()
    sub_times = sim.cosmo_model.age(1 / sub_aexps - 1.0).value * 1e3  # Myr

    args_first_detection_subs_in_tree = []

    for igid, gid in enumerate(subs_props["gids"]):

        idx = 0
        cur_mass = sub_datas["m"][igid][idx]
        prev_mass = sub_datas["m"][igid][idx + 1]
        dmass = cur_mass - prev_mass
        prev_gid = sub_gids[igid, idx + 1]

        while prev_gid > 0 and dmass > -0.1 * cur_mass and idx < len(sub_times) - 1:

            cur_mass = sub_datas["m"][igid][idx]
            prev_mass = sub_datas["m"][igid][idx + 1]
            dmass = cur_mass - prev_mass
            prev_gid = sub_gids[igid, idx + 1]

            idx += 1

        if idx > 0:
            idx -= 1

        args_first_detection_subs_in_tree.append(idx)

    aexp_form = sub_aexps[args_first_detection_subs_in_tree]
    t_form = sub_times[args_first_detection_subs_in_tree]
    m_form = sub_datas["m"][:, args_first_detection_subs_in_tree]
    dt_form = sub_times[0] - t_form
    z_form = 1 / aexp_form - 1

    # print(subs_props)

    fig, axs = plot_tmplt()

    ax_stars, ax_gas = axs

    directions = los_idx_2_direction(args.idir_rascas)

    tgt_pos = central_props["pos"]
    tgt_r = central_props["rmax"] * 3
    hid = central_props["host hid"]

    for ax in [ax_stars, ax_gas]:
        plot_substr(
            fig,
            ax,
            central_datas,
            sub_datas,
            sub_times,
            direction=directions[0],
            scale=sim.cosmo.lcMpc * 1e3,
        )

        plot_zoom_gals(
            ax,
            args.snapshot,
            sim,
            tgt_pos,
            tgt_r,
            zdist=-1,
            hm="HaloMaker_stars2_dp_rec_dust",
            direction=directions[0],
        )

    gas_args = {
        "cmap": "plasma",
        "vmin": 1e-26,
        "vmax": 1e-20,
        "mode": "mean",
        "transpose": True,
        "lower": True,
    }
    # plot_galaxy_gas(
    #     fig, ax_gas, start_aexp, directions, hid, tgt_pos, tgt_r, sim, **gas_args
    # )

    star_args = {
        "cmap": "gray",
        "vmin": 6e4,
        "vmax": 1e9,
        "mode": "mean",
        "transpose": True,
        "lower": True,
    }
    # plot_galaxy_stars(
    #     fig, ax_stars, start_aexp, directions, hid, tgt_pos, tgt_r, sim, **star_args
    # )

    outd = os.path.join(sim.path, "clump_maps")
    if not os.path.exists(outd):
        os.makedirs(outd, exist_ok=True)

    fname = os.path.join(
        outd, f"gid{args.galid}_snap{args.snapshot}_LoS{args.idir_rascas}_gas_stars"
    )

    print(f"Writing {fname}.png")

    fig.savefig(fname)
