import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from scipy.stats import binned_statistic_2d

from zoom_analysis.rascas.rascas_steps import *

from zoom_analysis.zoom_helpers import starting_hid_from_hagn

from hagn.utils import get_hagn_sim


# from zoom_analysis.halo_maker.read_treebricks import (
#     read_zoom_brick,
#     read_gal_stars,
#     convert_star_units,
# )

from zoom_analysis.halo_maker.assoc_fcts import (
    # get_halo_props_snap,
    get_gal_props_snap,
    # get_central_gal_for_hid,
    find_zoom_tgt_halo,
    get_halo_props_snap,
    # get_halo_assoc_file,
)

from zoom_analysis.trees.tree_reader import read_tree_file_rev

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


def dict_conv_bools(rascas_params):
    for key in rascas_params:
        for subkey in rascas_params[key]:
            if type(rascas_params[key][subkey]) == bool:
                rascas_params[key][subkey] = "T" if rascas_params[key][subkey] else "F"


debug = True
overwrite_CDD = False
overwrite_PFS = False
overwrite = False
# overwrite = True

# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id74099"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel"
sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel"
# target_hid = 203
# zed_target = 2.0
zed_target = 3.2
deltaT = 50  # Myr
rfact = 1.1

ndir_hp = 1  # number of directions... should be one or 2^N for healpy; 1-> 12 directions in mocks
# delta_T = 100  # Myr spacing of rascas runs

# CENTROID & SIZE
# tgt_zed = 2.0
# pos = None
# rad = None
# if pos or rad are none we look for treebricks and setup rascas
# mlim = 1e10
hm = "HaloMaker_stars2_dp_rec_dust/"
zmax = 6.0
zstt = 2.0
# zmax = 4.0

tree_type = "gal_tree_hid"

# run_name = "ignoreDust"
# run_name = "zoom_tgt_bpass_kroupa100"
# run_name = "zoom_tgt_bpass_kroupa300_bpassV23"
# run_name = "zoom_tgt_bpass_kroupa100_broken_metals"
# run_name = "zoom_tgt_bpass_kroupa100_wlya"
# run_name = "zoom_tgt_bc03_chabrier100"
# run_name = "zoom_tgt_bc03_chabrier100_wlya"
run_name = "zoom_tgt_bc03_chabrier100_wlya_noDust"
# run_name = "zoom_tgt_bpass_kroupa100_wlya_noDust"
# run_name = "zoom_tgt_bpass_chabrier100_new"
# run_name = "zoom_tgt_bpass_chabrier100_new_wlya"
# run_name = "zoom_tgt_bpass_chabrier100_new_wlya_noDust"
# run_name = "zoom_tgt_bpass_chabrier100"
# run_name = "zoom_tgt_bpass_chabrier100_ignoreDust"

# make sure run on fat 128 node + use all procs (incl. first step)

# PFS
nphot = 500000
lamb_min = 250  # Ang
lamb_max = 25000  # Ang
# spec_dir = "/data101/jlewis/BPASS/BPASSV221/chabrier_100/reformatted"
# spec_dir = "/data101/jlewis/BPASS/BPASSV221/kroupa_100/reformatted"
spec_dir = "/automnt/data101/jlewis/BC03/bin/reformatted"
# spec_dir = "/data101/jlewis/BPASS/BPASSV221/kroupa_100_brokenZ/reformatted"
# spec_dir = "/data101/jlewis/BPASS/BPASSV23/kroupa_300/a00/reformatted/"
# MOCK


# nscatter = 0
nscatter = 2

# DUST
dust_albedo = 0.32
g_dust = 0.73
# ext_law = "SMC"
dust_model = "SMC"
ignoreDust = False
# ignoreDust = True

# RUN
nnodes = 1
nomp = 1
ntasks = 128
wt = "24:00:00"

cdd_params = {}
cdd_params["gas_composition"] = {}
cdd_params["gas_composition"]["ignoreDust"] = ignoreDust

pfs_params = {}
pfs_params["PhotonsFromStars"] = {}
pfs_params["PhotonsFromStars"]["spec_SSPdir"] = spec_dir
pfs_params["PhotonsFromStars"]["nPhotonPackets"] = int(nphot)
pfs_params["PhotonsFromStars"]["spec_table_lmin_Ang"] = lamb_min  # angstrom
pfs_params["PhotonsFromStars"]["spec_table_lmax_Ang"] = lamb_max  # angstrom
pfs_params["ramses"] = {}
pfs_params["ramses"]["particle_families"] = "T"

rascas_params = {}
rascas_params["master"] = {}
rascas_params["master"]["dt_backup"] = 7200  # 2h between backups

rascas_params["dust"] = {}
rascas_params["dust"]["albedo"] = dust_albedo
rascas_params["dust"]["g_dust"] = g_dust
rascas_params["dust"]["dust_model"] = dust_model
# rascas_params["dust"]["ext_law"] = ext_law

rascas_params["gas_composition"] = {}
rascas_params["gas_composition"]["ignoreDust"] = ignoreDust
rascas_params["gas_composition"]["nscatterer"] = nscatter

# go through all dict keys and convert bools to one letter strings as understood by rascas
dict_conv_bools(pfs_params)
dict_conv_bools(cdd_params)
dict_conv_bools(rascas_params)

sim = ramses_sim(sim_path, nml="cosmo.nml")

dir_vects = get_directions_cart(ndir_hp)
ndir = len(dir_vects)


tree_name = os.path.join(sim.path, "TreeMakerstars2_dp_rec_dust", "tree_rev.dat")
byte_file = os.path.join(
    sim.path, "TreeMakerstars2_dp_rec_dust"
)  # , "tree_rev_nbytes")
# tree_name = os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat")
# byte_file = os.path.join(sim.path, "TreeMakerDM_dust")  # , "tree_rev_nbytes")

# tree_name = os.path.join(sim.path, "TreeMakerTreeGal", "tree_rev.dat")
# byte_file = os.path.join(sim.path, "TreeMakerTreeGal", "tree_rev_nbytes")


# target_hid, halo_dict, galaxies = find_zoom_tgt_halo(sim, zed_target, debug=debug)

name = sim.name
intID = int(name.split("id")[1].split("_")[0])

sim_aexps = sim.get_snap_exps()  # [::-1]
sim_times = sim.get_snap_times()  # [::-1]
sim_snaps = sim.snap_numbers  # [::-1]

assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

avail_aexps = np.intersect1d(
    sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
)
avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

hagn_sim = get_hagn_sim()

hid_start, halos, galaxies, start_aexp = starting_hid_from_hagn(
    zstt, sim, hagn_sim, intID, avail_aexps, avail_times
)
target_gid = galaxies["gids"][galaxies["mass"].argmax()]

# gprops = get_gal_props_snap(sim_path, assoc_file_nbs.max())
# fpure = gprops["host purity"] > 0.9999
# max_host_mass = np.max(gprops["host mass"][fpure])
# most_massive_hid = gprops["host hid"][fpure][gprops["host mass"][fpure].argmax()]

# gals_in_hid = gprops["host hid"] == most_massive_hid
# target_gid = gprops["gids"][gals_in_hid][gprops["mass"][gals_in_hid].argmax()]

# target_gid = gprops["gids"][gprops["mass"].argmax()]


tree_gids, tree_datas, tree_aexps = read_tree_file_rev(
    tree_name,
    byte_file,
    1.0 / start_aexp - 1.0,
    [target_gid],
    tgt_fields=["m", "x", "y", "z", "r"],
    star=True,
    debug=False,
)

if not hasattr(sim, "cosmo_model"):
    sim.init_cosmo()

tree_times = sim.cosmo_model.age(1.0 / tree_aexps - 1.0).value * 1e3  # Myr

filt = tree_datas["x"][0] != -1

for key in tree_datas:
    tree_datas[key] = tree_datas[key][0][filt]

tree_gids = tree_gids[0][filt]
tree_aexps = tree_aexps[filt]
tree_times = tree_times[filt]

# tree_snaps = np.asarray([sim.get_closest_snap(aexp=a) for a in tree_aexps])

rascas_paths = []
run_pfss = []
run_cdds = []

# print(list(zip(tree_snaps, tree_gids)))

tree_snaps_gids = np.zeros((len(sim_snaps), 2), dtype=np.int64)

if not overwrite_CDD and not overwrite_PFS:
    print("using existing domain and photon files where available")

print(f"Setting up rascas for {sim_path.split('/')[-1]}")


# look for treebricks
if not np.any(hm == d for d in os.listdir(sim.path)):
    print(f"Treebricks not found in {sim.path}")

last_time = sim_times[-1] + 2 * deltaT

tgt_gid = None

for i, (snap, aexp, time) in enumerate(
    zip(sim_snaps[::-1], sim_aexps[::-1], sim_times[::-1])
):

    if not snap in assoc_file_nbs:
        continue

    # print(i, snap, aexp, time)
    if 1.0 / aexp - 1 > zmax:
        continue

    if time >= (last_time + deltaT):
        continue

    last_time = time

    if np.min(np.abs(time - tree_times)) > deltaT:
        continue

    tree_arg = np.argmin(np.abs(time - tree_times))

    cur_gid = tree_gids[tree_arg]

    tree_snaps_gids[i] = [cur_gid, snap]

    # if tgt_gid is None:
    tgt_gid = cur_gid

    _, cur_gal_props = get_gal_props_snap(sim_path, snap, cur_gid)

    # print(cur_gal_props["pos"], snap, cur_gid, cur_gal_props["mass"])
    cur_rmax = cur_gal_props["rmax"]
    # cur_r50 = cur_gal_props["r50"]
    cur_pos = cur_gal_props["pos"]

    print(f"...Setting up rascas for galaxy {cur_gid} in snap {snap}")
    print(f"Has mass {cur_gal_props['mass']} and rmax {cur_rmax}")

    # SPEC
    spec = True
    rspec = cur_rmax  # cur_r50 * rfact
    nspec = 1024
    # IMG
    img = True
    rimg = cur_rmax  # cur_r50 * rfact
    nimg = 128
    # CUBE
    cube = True
    rcube = cur_rmax  # cur_r50 * rfact
    nspec_cube = nspec
    ncube = 128

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

    # rascas_path = create_dirs(sim.path, snap, gal_id=cur_gid, run_name=run_name)
    rascas_path = create_dirs(sim.path, snap, gal_id=tgt_gid, run_name=run_name)
    print(rascas_path)
    copy_exec(rascas_path)

    make_rascas_params(rascas_path, ndir, rascas_params)

    make_mock_params(
        rascas_path,
        cur_pos,
        dir_vects,
        spec=spec_params,
        img=img_params,
        cube=cube_params,
    )

    if not os.path.exists(os.path.join(rascas_path, "params_CDD.cfg")) or overwrite_CDD:
        # make_CDD_params(rascas_path, sim.path, snap, cur_pos, cur_r50 * rfact * 1.55)
        make_CDD_params(
            rascas_path,
            sim.path,
            snap,
            cur_pos,
            cur_rmax * rfact * 1.55,
            options=cdd_params,
        )

    run_CDD = False
    if (
        not os.path.exists(os.path.join(rascas_path, "DomDump", "domain_1.dom"))
        or overwrite_CDD
    ):
        run_CDD = True
        # run_CDD(rascas_path)

    if not os.path.exists(os.path.join(rascas_path, "params_PFS.cfg")) or overwrite_PFS:
        make_PFS_params(
            rascas_path,
            sim.path,
            snap,
            cur_pos,
            cur_rmax * rfact * 1.3,
            # cur_r50 * rfact * 1.3,
            pfs_params,
        )
    run_PFS = False
    if (
        not os.path.exists(os.path.join(rascas_path, "PFSDump", "pfsdump"))
        or overwrite_PFS
    ):
        run_PFS = True

    if (
        not os.path.exists(os.path.join(rascas_path, "RASCASDump", "rascas_dump.dat"))
        or overwrite
    ):

        run_cdds.append(run_CDD)
        run_pfss.append(run_PFS)
        rascas_paths.append(rascas_path)
    # print(rascas_paths, rascas_path)
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


# print(tree_snaps_gids)
# trim zero rows in tree_snaps_gids
tree_snaps_gids = tree_snaps_gids[~np.all(tree_snaps_gids == 0, axis=1)]
tree_snaps_gids_fname = f"{tree_type:s}{intID:d}.txt"
# tree_snaps_gids_fname = f"gal_tree_most_massive.txt"

with open(
    os.path.join(sim.path, "rascas", run_name, tree_snaps_gids_fname), "w"
) as dest:
    dest.write(f"#snap gid\n")
    for i in range(tree_snaps_gids.shape[0]):
        dest.write(f"{tree_snaps_gids[i,1]:d} {tree_snaps_gids[i,0]:d}\n")


create_sh_multi(
    rascas_paths,
    nnodes=nnodes,
    ntasks=ntasks,
    nomp=nomp,
    wt=wt,
    run_pfss=run_pfss,
    run_cdds=run_cdds,
)

# break
