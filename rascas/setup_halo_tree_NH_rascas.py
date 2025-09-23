from f90_tools.star_reader import read_part_ball_NH
from hagn.association import gid_to_stars
from hagn.catalogues import convert_cat_units, get_cat_hids, make_super_cat
from hagn.tree_reader import read_tree_rev
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from scipy.stats import binned_statistic_2d

from zoom_analysis.halo_maker.read_treebricks import read_zoom_brick
from zoom_analysis.zoom_helpers import starting_hid_from_hagn

# from hagn.catalogues import get_cat_gids, make_super_cat
from zoom_analysis.rascas.rascas_steps import *

# from zoom_analysis.halo_maker.read_treebricks import (
#     read_zoom_brick,
#     read_gal_stars,
#     convert_star_units,
# )

# from hagn.tree_reader import interpolate_tree_position, read_tree_rev
from hagn.utils import get_hagn_sim, get_nh_sim

from zoom_analysis.halo_maker.assoc_fcts import (
    # get_halo_props_snap,
    # get_gal_props_snap,
    # get_central_gal_for_hid,
    # find_zoom_tgt_halo,
    get_halo_assoc_file,
    get_assoc_pties_in_tree,
    get_r90,
    smooth_props,
)

# from zoom_analysis.sinks.sink_histories import get_cat_hids
from zoom_analysis.trees.tree_reader import read_tree_file_rev_correct_pos

from zoom_analysis.rascas.read_rascas import read_PFS_dump

# from gremlin.read_sim_params import ramses_sim

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

sim_dir = os.path.join("/data101/jlewis/sims/NH_for_rascas")

read_star_fct = read_part_ball_NH
hm_dm = "TREE_DM"
hm = "TREE_STARS_AdaptaHOP_dp_SCnew_gross"


def gal_fct(snap, sim, hm, **kwargs):
    return read_zoom_brick(
        snap, sim, hm, sim_path="/data7b/NewHorizon", galaxy=True, star=True
    )


# target_hid = 203
zed_target = 1.0
max_zed=4.0
# zed_target = 3.6
deltaT = 15  # Myr
rfact = 1.1
tgt_hid = 21742

sim = get_nh_sim()

ndir_hp = 1  # number of directions... should be one or 2^N for healpy; 1-> 12 directions in mocks
# delta_T = 100  # Myr spacing of rascas runs

# CENTROID & SIZE
# tgt_zed = 2.0
# pos = None
# rad = None
# if pos or rad are none we look for treebricks and setup rascas
# mlim = 1e10
hm = "TREE_STARS_AdaptaHOP_dp_SCnew_gross"
zmax = 7.0
zstt = 1.0
# zmax = 4.0
zed_targets = None
# zed_targets = [2.0]

# snap_targets = None
# snap_targets = np.arange(180, 200)
# snap_targets = [207]
# snap_targets = [214,210,206,202,198,194,190,186,182,178,174,170,166,162,158,154,150]
# snap_targets =[82,90,131,159,182]
# snap_targets =[64, 105, 140, 168, 175]
# snap_targets=[58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,
# 76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,
# 94,95,97,98,99,100,101,102,103,105,106,107,108,109,110,111,112,113]

# run_name = "ignoreDust"
# run_name = "zoom_tgt_bpass_kroupa100"
# run_name = "zoom_tgt_bpass_kroupa300_bpassV23"
# run_name = "zoom_tgt_bpass_kroupa100_broken_metals"
# run_name = "zoom_tgt_bpass_kroupa100_wlya"
# run_name = "zoom_tgt_bc03_chabrier100"
run_name = "zoom_tgt_bc03_chabrier100_wlya_SMC"
# run_name = "zoom_tgt_bc03_chabrier100_wlya_ndust_SMC"
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
# nphot = 500000
nphot = 1000000
# obs_lamb_min = 100# Ang obs frame
# obs_lamb_max = 55000# Ang obs frame
obs_lamb_min = 800  # Ang obs frame
obs_lamb_max = 55000  # Ang obs frame
# spec_dir = "/data101/jlewis/BPASS/BPASSV221/chabrier_100/reformatted"
# spec_dir = "/data101/jlewis/BPASS/BPASSV221/kroupa_100/reformatted"
spec_dir = "/automnt/data101/jlewis/BC03/bin/reformatted"
# spec_dir = "/data101/jlewis/BPASS/BPASSV221/kroupa_100_brokenZ/reformatted"
# spec_dir = "/data101/jlewis/BPASS/BPASSV23/kroupa_300/a00/reformatted/"
# MOCK

# nscatter = 0
nscatter = 2

# DUST
# dust_albedo = 0.32
# g_dust = 0.73
ext_law = "SMC"
dust_model = "scaling_Z"
draine_table = "./draine01_SMC.txt"
# dust_model = "ndust"
# dust_model = "lookup"
# dust_model = "MW"
ignoreDust = False
# ignoreDust = True

# tgt_res = 40  # ppc
nimg = None  # 128
ncube = None  # 128


print(f"Working on {sim.name:s}")

sim_aexps = sim.get_snap_exps(param_save=False)  # [::-1]
sim_zeds = 1.0 / sim_aexps - 1
# snap_targets= sim.snap_numbers[sim_zeds<=7.0]

# print(snap_targets)

# tgt_res = sim.cosmo.lcMpc*1e6/2**sim.levelmax #pc
# take half of the resolution of the simulation
tgt_res = sim.cosmo.lcMpc * 1e6 / 2 ** (sim.levelmax) * 2  # pc


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
cdd_params["dust"] = {}
cdd_params["dust"]["dust_model"] = dust_model
cdd_params["dust"]["ext_law"] = ext_law


pfs_params = {}
pfs_params["PhotonsFromStars"] = {}
pfs_params["PhotonsFromStars"]["spec_SSPdir"] = spec_dir
pfs_params["PhotonsFromStars"]["nPhotonPackets"] = int(nphot)

pfs_params["ramses"] = {}
pfs_params["ramses"]["particle_families"] = "F"

rascas_params = {}
rascas_params["master"] = {}
rascas_params["master"]["dt_backup"] = 7200  # 2h between backups

rascas_params["dust"] = {}
# rascas_params["dust"]["albedo"] = dust_albedo
# rascas_params["dust"]["g_dust"] = g_dust
rascas_params["dust"]["dust_model"] = dust_model
rascas_params["dust"]["ext_law"] = ext_law
rascas_params["dust"]["fname_draine_table"] = draine_table

rascas_params["gas_composition"] = {}
rascas_params["gas_composition"]["ignoreDust"] = ignoreDust
rascas_params["gas_composition"]["nscatterer"] = nscatter

# go through all dict keys and convert bools to one letter strings as understood by rascas
dict_conv_bools(pfs_params)
dict_conv_bools(cdd_params)
dict_conv_bools(rascas_params)

print("CDD params are:")
print(cdd_params)
# assert input("Continue?") in ["y", "yes", "Y", "YES", ""], "Aborting"

print("PFS params are:")
print(pfs_params)
# assert input("Continue?") in ["y", "yes", "Y", "YES", ""], "Aborting"

print("RASCAS params are:")
print(rascas_params)
# assert input("Continue?") in ["y", "yes", "Y", "YES", ""], "Aborting"


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

tgt_snap = sim.get_closest_snap(zed=zed_target)
# last_snap = sim_snaps[sim_aexps > 1.0 / (tgt_zed + 1.0)][0]

# last_snap = 251

# gal_bricks = read_zoom_brick(
#     tgt_snap, sim, hm, sim_path="/data7b/NewHorizon", star=True, galaxy=True
# )

# arg_gal = np.where(gal_bricks["hosting info"]["hid"] == tgt_hid)[0][0]

# hagn_sim = get_hagn_sim()

# tgt_pos = np.transpose(
#     [
#         gal_bricks["positions"]["x"][arg_gal],
#         gal_bricks["positions"]["y"][arg_gal],
#         gal_bricks["positions"]["z"][arg_gal],
#     ]
# )
# tgt_rad = gal_bricks["virial properties"]["rvir"][arg_gal] * 2
# tgt_mass = gal_bricks["hosting info"]["hmass"][arg_gal]


if not hasattr(sim, "cosmo_model"):
    sim.init_cosmo()

rascas_paths = []
output_dirs = []
run_pfss = []
run_cdds = []

if not overwrite_CDD and not overwrite_PFS:
    print("using existing domain and photon files where available")

print(f"Setting up rascas for NH galaxy in halo hid:{tgt_hid:d}")

sim_times = sim.get_snap_times(param_save=False)  # [::-1]
sim_snaps = sim.snap_numbers  # [::-1]
# look for treebricks
if not np.any(hm == d for d in os.listdir(sim.path)):
    print(f"Treebricks not found in {sim.path}")

last_time = sim_times[-1] + 2 * deltaT

tgt_gid = None

last_snap = 446
last_aexp = sim.get_snap_exps(446, param_save=False)[0]

nh_tree_hids, nh_tree_datas, nh_tree_aexps = read_tree_rev(
    1.0 / last_aexp - 1.0,
    [tgt_hid],
    tree_type="halo",
    # [gid],
    # tree_type="gal",
    target_fields=["m", "x", "y", "z", "r"],
    sim="nh",
    float_dtype=np.float32,
)

nh_tree_times = sim.cosmo_model.age(1.0 / nh_tree_aexps - 1.0).value * 1e3  # Myr

filt = nh_tree_datas["x"][0] != -1

for key in nh_tree_datas:
    nh_tree_datas[key] = nh_tree_datas[key][0][filt]
tree_hids = nh_tree_hids[0][filt]
tree_aexps = nh_tree_aexps[filt]
tree_times = nh_tree_times[filt]

tree_hids = tree_hids[0]
# print(tree_datas["m"][0])

tree_times = sim.cosmo_model.age(1.0 / tree_aexps - 1.0).value * 1e3  # Myr

tree_snaps = np.asarray([sim.get_closest_snap(aexp=a) for a in tree_aexps])

nsteps = len(tree_times)

reffs = np.zeros(nsteps)
rvirs = np.zeros(nsteps)
mass = np.zeros(nsteps)
pos = np.zeros((nsteps, 3))

for istep, (aexp, time) in enumerate(zip(nh_tree_aexps, nh_tree_times)):

    if nh_tree_hids[0][istep]<1:continue

    

    nh_snap = sim_snaps[np.argmin(np.abs(sim_aexps - aexp))]
    # if nh_snap >= 378:continue

    if nh_snap<200:continue

    if abs(aexp-sim_aexps[np.argmin(np.abs(sim_aexps - aexp))])>1e-3:
        continue

    if 1/aexp-1 > max_zed:
        continue

    try:
        super_cat = make_super_cat(
            nh_snap, outf="/data101/jlewis/nh/super_cats", sim="nh", overwrite=True
        )
        # print(istep, 1.0 / aexp - 1)
    except FileNotFoundError:
        print("No super cat")
        continue

    # super_cat=convert_cat_units(super_cat,sim, nh_snap)

    gal_pties = get_cat_hids(super_cat, [int(nh_tree_hids[0][istep])])

    gid = gal_pties["gid"]

    try:
        stars = gid_to_stars(gid, nh_snap, sim, ['pos','mass'])
    except (FileNotFoundError, ValueError):
        print(f'missing file for snapshot {nh_snap}... skipping')
        continue


    star_pos = np.abs(np.unwrap(stars["pos"],period=1.0,axis=1))

    # tgt_pos = np.transpose([gal_pties['x'],gal_pties['y'],gal_pties['z']])
    tgt_pos = np.median(star_pos,axis=0)
    rmax = np.max(np.linalg.norm(star_pos-tgt_pos,axis=1))

    r90 = get_r90(star_pos, stars['mass'], tgt_pos, dx=sim.cosmo.lcMpc/2**sim.levelmax)

    print(r90,rmax)

    assert rmax<0.1, "huge rmax, something probably went very wrong!"

    # gal_pties = read_zoom_brick(
    # nh_snap, sim, hm, sim_path="/data7b/NewHorizon", star=True, galaxy=True
    # )

    # print(istep,len(gal_pties['mgal']))



    if len(gal_pties["mgal"]) == 0:
        continue

    mass[istep] = gal_pties["mgal"]
    pos[istep, :] = [gal_pties["x"][0], gal_pties["y"][0], gal_pties["z"][0]]
    reffs[istep] = gal_pties["rgal"]
    rvirs[istep] = gal_pties["rvir"]

    # cur_rmax = rvirs[istep]
    cur_rmax = r90
    cur_gid = int(gal_pties['gid'][0])
    cur_pos = pos[istep,:]

    print(f"...Setting up rascas for galaxy {cur_gid} in snap {nh_snap}")
    print(f"Has mass {gal_pties["mgal"]} and rvir {cur_rmax}")

    cur_rmax_ppc = cur_rmax * sim.cosmo.lcMpc * 1e6 * aexp

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

    # get wavs that cover z=0 filters at aexp
    lamb_min = obs_lamb_min * aexp  # mult by aexp=1/(1+z) to observed frame
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
        p=sim_dir, snap=nh_snap, gal_id=cur_gid, run_name=run_name, start_dir=start_dir
    )

    print(rascas_path)
    copy_exec(rascas_path,NH=True)

    if dust_model == "scaling_Z":

        copy_draine_table(draine_table, rascas_path)

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
            sim_dir,
            nh_snap,
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
            sim_dir,
            nh_snap,
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
        output_dirs.append(f"output_{nh_snap:05d}")
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
    output_dirs=output_dirs,
)

# break
