import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from scipy.stats import binned_statistic_2d

from zoom_analysis.zoom_helpers import starting_hid_from_hagn

# from hagn.catalogues import get_cat_gids, make_super_cat
from zoom_analysis.rascas.rascas_steps import *

# from zoom_analysis.halo_maker.read_treebricks import (
#     read_zoom_brick,
#     read_gal_stars,
#     convert_star_units,
# )

# from hagn.tree_reader import interpolate_tree_position, read_tree_rev
from hagn.utils import get_hagn_sim

from zoom_analysis.halo_maker.assoc_fcts import (
    # get_halo_props_snap,
    # get_gal_props_snap,
    # get_central_gal_for_hid,
    # find_zoom_tgt_halo,
    get_halo_assoc_file,
    get_assoc_pties_in_tree,
    smooth_props
)

# from zoom_analysis.sinks.sink_histories import get_cat_hids
from zoom_analysis.trees.tree_reader import read_tree_file_rev_correct_pos

from zoom_analysis.rascas.read_rascas import read_PFS_dump

from gremlin.read_sim_params import ramses_sim
# from zoom_analysis.zoom_helpers import decentre_coordinates


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
overwrite_CDD = True
overwrite_PFS = True
overwrite = True
overwrite = True

# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id74099"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"
sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380"
# sim_path='/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF'
# target_hid = 203
zed_target = 2.0
# zed_target = 3.6
deltaT = 15  # Myr
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

snap_targets = None
# snap_targets = [169]
snap_targets = [205]
# snap_targets =[82,90,131,159,182]
# snap_targets =[64, 105, 140, 168, 175]

# run_name = "ignoreDust"
# run_name = "zoom_tgt_bpass_kroupa100"
# run_name = "zoom_tgt_bpass_kroupa300_bpassV23"
# run_name = "zoom_tgt_bpass_kroupa100_broken_metals"
# run_name = "zoom_tgt_bpass_kroupa100_wlya"
# run_name = "zoom_tgt_bc03_chabrier100"
# run_name = "zoom_tgt_bc03_chabrier100_wlya"
# run_name = "zoom_tgt_bc03_chabrier100_wlya_ndust"
# run_name = "zoom_tgt_bc03_chabrier100_wlya_YD"
# run_name = "zoom_tgt_bc03_chabrier100_wlya_MW"
# run_name = "zoom_tgt_bc03_chabrier100_wlya_noDust"
# run_name = "zoom_tgt_bpass_kroupa100_wlya_noDust"
# run_name = "zoom_tgt_bpass_chabrier100_new"
# run_name = "zoom_tgt_bpass_chabrier100_new_wlya"
# run_name = "zoom_tgt_bpass_chabrier100_new_wlya_noDust"
# run_name = "zoom_tgt_bpass_chabrier100"
# run_name = "zoom_tgt_bpass_chabrier100_ignoreDust"

# make sure run on fat 128 node + use all procs (incl. first step)

# PFS
nphot = 500000
# obs_lamb_min = 100  # Ang obs frame
# obs_lamb_max = 55000  # Ang obs frame
obs_lamb_min = 1000  # Ang obs frame
obs_lamb_max = 55000 # Ang obs frame
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
ext_law = "SMC"
# dust_model = "scaling_Z"
# dust_model = "ndust"
# dust_model = "lookup"
# dust_model = "MW"
ignoreDust = False
# ignoreDust = True

# tgt_res = 40  # ppc
nimg = None  # 128
ncube = None  # 128


sim = ramses_sim(sim_path, nml="cosmo.nml")

print(f"Working on {sim.name:s}")

sim_aexps = sim.get_snap_exps()  # [::-1]

tgt_res = sim.cosmo.lcMpc*1e6/2**sim.levelmax #pc


# lamb_max = obs_lamb_max 
# lamb_min = obs_lamb_min

# aexp2 = 1/(2+1.)
# lamb_max = np.max([obs_lamb_max /aexp2 * sim_aexps.min(),obs_lamb_max /aexp2 * sim_aexps.max()],axis=0)
# lamb_min = np.min([obs_lamb_min /aexp2 * sim_aexps.min(),obs_lamb_min /aexp2 * sim_aexps.max()],axis=0)

# print(lamb_min, lamb_max)

# RUN
nnodes = 1
nomp = 1
ntasks = 128
wt = "24:00:00"

cdd_params = {}
cdd_params["gas_composition"] = {}
cdd_params["gas_composition"]["ignoreDust"] = ignoreDust
cdd_params["gas_composition"]["dust_model"] = dust_model
cdd_params["dust"] = {}
cdd_params["dust"]["dust_model"] = dust_model

pfs_params = {}
pfs_params["PhotonsFromStars"] = {}
pfs_params["PhotonsFromStars"]["spec_SSPdir"] = spec_dir
pfs_params["PhotonsFromStars"]["nPhotonPackets"] = int(nphot)

pfs_params["ramses"] = {}
pfs_params["ramses"]["particle_families"] = "T"

rascas_params = {}
rascas_params["master"] = {}
rascas_params["master"]["dt_backup"] = 7200  # 2h between backups

rascas_params["dust"] = {}
rascas_params["dust"]["albedo"] = dust_albedo
rascas_params["dust"]["g_dust"] = g_dust
rascas_params["dust"]["dust_model"] = dust_model
rascas_params["dust"]["ext_law"] = ext_law

rascas_params["gas_composition"] = {}
rascas_params["gas_composition"]["ignoreDust"] = ignoreDust
rascas_params["gas_composition"]["nscatterer"] = nscatter


# go through all dict keys and convert bools to one letter strings as understood by rascas
dict_conv_bools(pfs_params)
dict_conv_bools(cdd_params)
dict_conv_bools(rascas_params)

print("CDD params are:")
print(cdd_params)
assert input("Continue?") in ["y", "yes", "Y", "YES", ""], "Aborting"

print("PFS params are:")
print(pfs_params)
assert input("Continue?") in ["y", "yes", "Y", "YES", ""], "Aborting"

print("RASCAS params are:")
print(rascas_params)
assert input("Continue?") in ["y", "yes", "Y", "YES", ""], "Aborting"


dir_vects = get_directions_cart(ndir_hp)
ndir = len(dir_vects)

tree_name = os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat")
byte_file = os.path.join(sim.path, "TreeMakerDM_dust")
# tree_name = os.path.join(sim.path, "TreeMakerstars2_dp_rec_dust", "tree_rev.dat")
# byte_file = os.path.join(
#     sim.path, "TreeMakerstars2_dp_rec_dust"
# )  # , "tree_rev_nbytes")
# tree_name = os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat")
# byte_file = os.path.join(sim.path, "TreeMakerDM_dust")  # , "tree_rev_nbytes")

# tree_name = os.path.join(sim.path, "TreeMakerTreeGal", "tree_rev.dat")
# byte_file = os.path.join(sim.path, "TreeMakerTreeGal", "tree_rev_nbytes")

hagn_sim = get_hagn_sim()

name = sim_path.split("/")[-1]
intID = int(name[2:].split("_")[0])
# gid = int(make_super_cat(197, outf="/data101/jlewis/hagn/super_cats")["gid"][0])


if not hasattr(sim, "cosmo_model"):
    sim.init_cosmo()

rascas_paths = []
run_pfss = []
run_cdds = []

if not overwrite_CDD and not overwrite_PFS:
    print("using existing domain and photon files where available")

print(f"Setting up rascas for {sim_path.split('/')[-1]}")

sim_times = sim.get_snap_times()  # [::-1]
sim_snaps = sim.snap_numbers  # [::-1]
# look for treebricks
if not np.any(hm == d for d in os.listdir(sim.path)):
    print(f"Treebricks not found in {sim.path}")

last_time = sim_times[-1] + 2 * deltaT

tgt_gid = None


assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

avail_aexps = np.intersect1d(
    sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
)
avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3


hid_start, _, _, true_start_aexp,found = starting_hid_from_hagn(
    zstt, sim, hagn_sim, intID, avail_aexps, avail_times
)
# target_gid = galaxies["gids"][galaxies["mass"].argmax()]

true_start_zed = 1.0 / true_start_aexp - 1.0
true_start_snap = sim.get_closest_snap(aexp=true_start_aexp)
print(hid_start, true_start_zed, true_start_snap)

tree_hids, tree_datas, tree_aexps = read_tree_file_rev_correct_pos(
    tree_name,
    sim,
    true_start_snap,
    byte_file,
    true_start_zed,
    [hid_start],
    # tree_type="halo",
    tgt_fields=["m", "x", "y", "z", "r"],
    debug=False,
    star=False,
)

tree_hids = tree_hids[0]
# print(tree_datas["m"][0])

tree_times = sim.cosmo_model.age(1.0 / tree_aexps - 1.0).value * 1e3  # Myr

filt = tree_datas["x"][0] != -1

for key in tree_datas:
    tree_datas[key] = tree_datas[key][0][filt]
# tree_gids = tree_gids[0][filt]
# tree_gids = tree_hids[filt]
tree_aexps = tree_aexps[filt]
tree_times = tree_times[filt]

tree_snaps = np.asarray([sim.get_closest_snap(aexp=a) for a in tree_aexps])

gal_props_tree = get_assoc_pties_in_tree(
    sim, tree_aexps, tree_hids, assoc_fields=["r50", "rmax", "mass", "pos", "host hid"]
)

r50s = gal_props_tree["r50"]
rmaxs = gal_props_tree["rmax"]
masses = gal_props_tree["mass"]
poss = gal_props_tree["pos"]
hids = gal_props_tree["host hid"]

smooth_gal_props = smooth_props(gal_props_tree)


for i, (snap, aexp, time) in enumerate(
    zip(sim_snaps[::-1], sim_aexps[::-1], sim_times[::-1])
):

    if snap_targets is not None:
        if snap not in snap_targets:
            continue

    # print(i, snap, aexp, time)
    if 1.0 / aexp - 1 > zmax:
        print(f"Reached zmax at snap {snap}")
        continue

    if 1.0 / aexp - 1 < true_start_zed:
        print(f"Reached zstart at snap {snap}")
        continue

    if time >= (last_time + deltaT):
        print(f"Too close to last processed snap")
        continue

    last_time = time

    if np.min(np.abs(time - tree_times)) > deltaT:
        print(f"No tree data for snap")
        continue

    if not os.path.isfile(get_halo_assoc_file(sim_path, snap)):
        print(f"No halo associations for snap {snap}")
        continue

    tree_arg = np.argmin(np.abs(time - tree_times))
    if np.abs(time - tree_times[tree_arg]) > deltaT:
        print(f"No tree data for snap")
        continue

    # print(tree_hids[0][tree_arg])

    # hdict, hosted_gals = get_halo_props_snap(sim_path, snap, tree_hids[0][tree_arg])

    # # print(hosted_gals)
    # if not "gids" in hosted_gals:
    #     print(f"No galaxies found in snap {snap}")
    #     continue

    # cur_gid = int(hosted_gals["gids"][np.argmax(hosted_gals["mass"])])


    # _, cur_gal_props = get_gal_props_snap(sim_path, snap, cur_gid)

    aexp_arg = np.argmin(np.abs(aexp - gal_props_tree["aexps"]))

    cur_gid = gal_props_tree["gids"][aexp_arg]

    if tgt_gid is None:
       tgt_gid = cur_gid

    # print(cur_gal_props["pos"], snap, cur_gid, cur_gal_props["mass"])
    cur_rmax = gal_props_tree["rmax"][aexp_arg]
    cur_rmax_ppc = cur_rmax * sim.cosmo.lcMpc * 1e6 * aexp
    # cur_r50 = cur_gal_props["r50"]
    cur_pos = gal_props_tree["pos"][aexp_arg]



    print(f"...Setting up rascas for galaxy {cur_gid} in snap {snap}")
    print(f"Has mass {gal_props_tree['mass'][aexp_arg]} and rmax {cur_rmax}")

    # SPEC
    spec = True
    rspec = cur_rmax  # cur_r50 * rfact
    nspec = 128

    # IMG
    img = True
    rimg = cur_rmax  # cur_r50 * rfact
    if nimg is None or tgt_res is not None:
        nimg = int(np.ceil(cur_rmax_ppc / tgt_res))

    # CUBE
    cube = True
    rcube = cur_rmax  # cur_r50 * rfact
    nspec_cube = nspec
    if ncube is None or tgt_res is not None:
        ncube = int(np.ceil(cur_rmax_ppc / tgt_res))

    #get wavs that cover z=0 filters at aexp
    lamb_min = max(obs_lamb_min * aexp,50) #implicit division by aexp=1 [z=0]
    lamb_max = obs_lamb_max * aexp

    print(f"\lambda min: {lamb_min}, \lambda max: {lamb_max} \AA")

    pfs_params["PhotonsFromStars"]["spec_table_lmin_Ang"] = lamb_min  # angstrom
    pfs_params["PhotonsFromStars"]["spec_table_lmax_Ang"] = lamb_max  # angstrom

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
    start_dir = ""  # this is "" so we just write straight to the dir
    rascas_path = create_dirs(
        p=sim.path, snap=snap, gal_id=tgt_gid, run_name=run_name, start_dir=start_dir
    )

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
