import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from scipy.stats import binned_statistic_2d

from zoom_analysis.rascas.rascas_steps import *

from zoom_analysis.halo_maker.read_treebricks import (
    read_zoom_brick,
    read_gal_stars,
    convert_star_units,
)

from zoom_analysis.rascas.read_rascas import read_PFS_dump

from gremlin.read_sim_params import ramses_sim


def make_PFS_img(p, ctr_gal, rgal, stars, img_size: int = 400):

    # include proj dir

    fout = os.path.join(p, "PFSDump", "PFS_img.png")
    fin = os.path.join(p, "PFSDump", "pfsdump")

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)

    try:
        ph = read_PFS_dump(fin)
        pos = ph["x_em"]
        y = np.linspace(pos[:, 1].min(), pos[:, 1].max(), img_size)
        z = np.linspace(pos[:, 2].min(), pos[:, 2].max(), img_size)

        pfs_img, _, _, _ = binned_statistic_2d(
            pos[:, 1], pos[:, 2], pos[:, 2], bins=[y, z], statistic="count"
        )

        img = ax[0].imshow(
            pfs_img.T,
            origin="lower",
            cmap="gray",
            norm=LogNorm(vmin=1),
            extent=np.asarray(
                [
                    (y[0]),
                    (y[-1]),
                    (z[0]),
                    (z[-1]),
                ]
            ),
        )

        cb = fig.colorbar(img, ax=ax)
        cb.set_label("Number of photons")

    except FileNotFoundError:
        print("something went wrong ... no pfsdump")
        pass

    ax[0].scatter(ctr_gal[1], ctr_gal[2], marker="x", c="b", s=50)
    ax[1].scatter(ctr_gal[1], ctr_gal[2], marker="x", c="b", s=50)
    # ax.scatter(ctr_gal[1], ctr_gal[0], marker="x", c="g", s=50)
    # ax.scatter(ctr_gal[1], ctr_gal[2], marker="x", c="r", s=50)

    print(ctr_gal)
    print(stars["pos"].mean(axis=0))

    circ = Circle(
        (ctr_gal[1], ctr_gal[2]),
        rgal,
        fill=False,
        edgecolor="b",
        linewidth=2,
        linestyle="--",
    )
    ax[0].add_patch(circ)
    circ = Circle(
        (ctr_gal[1], ctr_gal[2]),
        rgal,
        fill=False,
        edgecolor="b",
        linewidth=2,
        linestyle="--",
    )
    ax[1].add_patch(circ)

    # ax.set_xlim((y[0] - ctr_gal[1]), (y[-1] - ctr_gal[1]))
    # ax.set_ylim((z[0] - ctr_gal[2]), (z[-1] - ctr_gal[2]))

    ax[1].scatter(stars["pos"][:, 1], stars["pos"][:, 2], c="r", s=1)
    ax[1].set_aspect("equal")

    # print(stars["pos"].mean() / ctr_gal)

    # ax.set_xlabel(r"$y, \mathrm{ckpc/h}$")
    # ax.set_ylabel(r"$z, \mathrm{ckpc/h}$")

    # xarg, yarg = np.where(pfs_img == pfs_img.max())

    # ax.axvline(x=x[xarg[0]] * l, c="r")
    # ax.axhline(y=y[yarg[0]] * l, c="r")

    # print(x[xarg[0]], y[yarg[0]])

    fig.savefig(fout)

    plt.close(fig)


debug = True
overwrite_CDD = False
overwrite_PFS = True
# overwrite = False
# overwrite = True

sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal"

sim_names = ["id74099"]

ndir_hp = 1  # number of directions... should be one or 2^N for healpy
# delta_T = 100  # Myr spacing of rascas runs

# CENTROID & SIZE
# tgt_zed = 2.0
# pos = None
# rad = None
# if pos or rad are none we look for treebricks and setup rascas
mlim = 1e10
hm = "HaloMaker_stars2_dp_rec_dust/"
zmax = 4.0

# run_name = "ignoreDust"
run_name = None

# PFS
nphot = 500000
lamb_min = 250  # Ang
lamb_max = 25000  # Ang

# MOCK


# DUST
dust_albedo = 0.32
g_dust = 0.73
ext_law = "SMC"
dust_model = "ndust"
ignoreDust = False

# RUN
nnodes = 1
nomp = 1
ntasks = 16
wt = "02:00:00"


pfs_params = {}
pfs_params["PhotonsFromStars"] = {}
pfs_params["PhotonsFromStars"]["nPhotonPackets"] = int(nphot)
pfs_params["PhotonsFromStars"]["spec_table_lmin_Ang"] = 6.0  # angstrom
pfs_params["PhotonsFromStars"]["spec_table_lmax_Ang"] = 99995.0  # angstrom
pfs_params["ramses"] = {}
pfs_params["ramses"]["particle_families"] = "T"

rascas_params = {}
rascas_params["dust"] = {}
rascas_params["dust"]["albedo"] = dust_albedo
rascas_params["dust"]["g_dust"] = g_dust
rascas_params["dust"]["dust_model"] = dust_model
rascas_params["dust"]["ext_law"] = ext_law

rascas_params["gas_composition"] = {}
rascas_params["gas_composition"]["ignoreDust"] = ignoreDust


dir_vects = get_directions_cart(ndir_hp)
ndir = len(dir_vects)


if not overwrite_CDD and not overwrite_PFS:
    print("using existing domain and photon files where available")

for sim_name in sim_names:

    print(f"Setting up rascas for {sim_name}")

    sim = ramses_sim(os.path.join(sim_path, sim_name), nml="cosmo.nml")

    sim_aexps = sim.get_snap_exps()  # [::-1]
    sim_times = sim.get_snap_times()  # [::-1]
    sim_snaps = sim.snap_numbers  # [::-1]
    # look for treebricks
    if not hm in os.listdir(sim.path):
        print(f"Treebricks not found in {sim.path}")

    tree_brick_files = [
        f for f in os.listdir(os.path.join(sim.path, hm)) if f.startswith("tree_bricks")
    ]

    if len(tree_brick_files) == 0:
        print(f"No tree bricks found in {sim.path}/{hm}")
        continue

    tree_brick_snaps = [int(f[-3:]) for f in tree_brick_files]

    for i, (snap, aexp, time) in enumerate(zip(sim_snaps, sim_aexps, sim_times)):

        if 1.0 / aexp - 1 > zmax:
            continue

        brick = read_zoom_brick(snap, sim, hm)
        if brick == 0:
            print(f"Didn't find brickfile for snap {snap}")
            continue

        mgals = brick["hosting info"]["hmass"]
        gids = brick["hosting info"]["hid"][mgals > mlim]  # [::-1]
        pos = np.transpose(
            [
                brick["positions"]["x"],
                brick["positions"]["y"],
                brick["positions"]["z"],
            ]
        )[mgals > mlim]
        # rgals = brick["smallest ellipse"]["r"][mgals > mlim]
        rgals = brick["virial properties"]["rvir"][mgals > mlim]

        for igal, gid in enumerate(gids):

            print(f"...Setting up rascas for galaxy {gid} in snap {snap}")

            # SPEC
            spec = True
            rspec = rgals[igal]
            nspec = 250
            # IMG
            img = True
            rimg = rgals[igal]
            nimg = 1000
            # CUBE
            cube = True
            rcube = rgals[igal]
            nspec_cube = nspec
            ncube = 1000

            spec_params = {
                "do": spec,
                "nspec": nspec,
                "rspec": rspec,
                "lamb_min": lamb_min,
                "lamb_max": lamb_max,
            }

            img_params = {"do": img, "nimg": nimg, "rimg": rimg}

            cube_params = {
                "do": cube,
                "nspec": nspec_cube,
                "ncube": ncube,
                "lamb_min": lamb_min,
                "lamb_max": lamb_max,
                "rcube": rcube,
            }

            rascas_path = create_dirs(sim.path, snap, gal_id=gid, run_name=run_name)

            copy_exec(rascas_path)

            make_rascas_params(rascas_path, ndir, rascas_params)

            make_mock_params(
                rascas_path,
                pos[igal],
                dir_vects,
                spec=spec_params,
                img=img_params,
                cube=cube_params,
            )

            if (
                not os.path.exists(os.path.join(rascas_path, "params_CDD.cfg"))
                or overwrite_CDD
            ):
                make_CDD_params(
                    rascas_path, sim.path, snap, pos[igal], rgals[igal] * 1.55
                )
                # run_CDD(rascas_path)

            if (
                not os.path.exists(os.path.join(rascas_path, "params_PFS.cfg"))
                or overwrite_PFS
            ):
                make_PFS_params(
                    rascas_path,
                    sim.path,
                    snap,
                    pos[igal],
                    rgals[igal] * 1.3,
                    pfs_params,
                )
                # run_PFS(rascas_path)

                # if debug make pfs dump check image
                # if debug:
                #     stars = read_gal_stars(
                #         os.path.join(
                #             sim.path, hm, f"GAL_{snap:05d}", f"gal_stars_{gid:07d}"
                #         )
                #     )
                #     convert_star_units(stars, snap=snap, sim=sim)
                #     make_PFS_img(rascas_path, pos[igal], rgal=rgals[igal], stars=stars)

            create_sh(rascas_path, nnodes=nnodes, ntasks=ntasks, nomp=nomp, wt=wt)

        # break
