from f90_tools.star_reader import read_part_ball_NH
from hagn.association import gid_to_stars
from zoom_analysis.halo_maker.assoc_fcts import get_gal_props_snap, get_halo_props_snap
from hagn.catalogues import convert_cat_units, get_cat_hids, make_super_cat
from hagn.tree_reader import read_tree_rev
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from scipy.stats import binned_statistic_2d

from zoom_analysis.rascas.filts.filts import read_transmissions

from zoom_analysis.halo_maker.read_treebricks import (
    convert_star_units,
    read_zoom_brick,
    read_gal_stars,
    read_zoom_stars,
)
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
overwrite_CDD = False
overwrite_PFS = False
overwrite = False

sim_dir = os.path.join("/data101/jlewis/sims/NH_for_rascas",full_check=False)

# read_star_fct = read_part_ball_NH
# hm_dm = "TREE_DM"
# hm = "TREE_STARS_AdaptaHOP_dp_SCnew_gross"


# def gal_fct(snap, sim, hm, **kwargs):
#     return read_zoom_brick(
#         snap, sim, hm, sim_path="/data7b/NewHorizon", galaxy=True, star=True
#     )


# target_hid = 203
# zed_target = 0.5
# max_zed = 7.0
# zmax=max_zed
zed_target = 0.5
# deltaT = 25  # Myr
rfact = 1.1
# tgt_hid = 21742
mlim = 5e9
# mlim=1e10 #for now ! Lower this later
# zstt = 0.5

dust_model = "SMC"
# dust_model = "noDust"

# sim = get_nh_sim()

sim = ramses_sim("/data101/jlewis/sims/NH_for_rascas", nml="cosmo.nml")

ndir_hp = 1  # number of directions... should be one or 2^N for healpy; 1-> 12 directions in mocks
# delta_T = 100  # Myr spacing of rascas runs

# CENTROID & SIZE
# tgt_zed = 2.0
# pos = None
# rad = None
# if pos or rad are none we look for treebricks and setup rascas
# mlim = 1e10
hm = "TREE_STARS_AdaptaHOP_dp_SCnew_gross"

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


# make sure run on fat 128 node + use all procs (incl. first step)

# PFS
# nphot = 500000
nphot = 1500000
# nphot = 2000000
# obs_lamb_min = 100# Ang obs frame
# obs_lamb_max = 55000# Ang obs frame
# obs_lamb_min = 800  # Ang obs frame
# obs_lamb_max = 55000  # Ang obs frame
# spec_dir = "/data101/jlewis/BPASS/BPASSV221/chabrier_100/reformatted"
# spec_dir = "/data101/jlewis/BPASS/BPASSV221/kroupa_100/reformatted"
spec_dir = "/automnt/data101/jlewis/BC03/bin/reformatted"
# spec_dir = "/data101/jlewis/BPASS/BPASSV221/kroupa_100_brokenZ/reformatted"
# spec_dir = "/data101/jlewis/BPASS/BPASSV23/kroupa_300/a00/reformatted/"
# MOCK

ns, wvs, trs = read_transmissions()

obs_lamb_min = np.min(np.concatenate(wvs)) * 0.99  # Ang obs frame
obs_lamb_max = np.max(np.concatenate(wvs)) * 1.01  # Ang obs frame

nscatter = 0
# nscatter = 2



if dust_model == "noDust":
    ignore_dust = True
else:
    ignore_dust = False

# DUST
# dust_albedo = 0.32
# g_dust = 0.73
ext_law = "SMC"
dust_model_param = "scaling_Z"
draine_table = "./draine01_SMC.txt"
# dust_model = "ndust"
# dust_model = "lookup"
# dust_model = "MW"
ignoreDust = ignore_dust
# ignoreDust = True

zmin=0
zmax=6

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
ntasks = 8
wt = "24:00:00"

cdd_params = {}
cdd_params["gas_composition"] = {}
cdd_params["gas_composition"]["ignoreDust"] = ignoreDust
cdd_params["dust"] = {}
cdd_params["dust"]["dust_model"] = dust_model_param
cdd_params["dust"]["ext_law"] = ext_law
cdd_params["dust"]["fname_draine_table"] = draine_table

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
rascas_params["dust"]["dust_model"] = dust_model_param
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

# raise ValueError

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


sim_times = sim.get_snap_times(param_save=False)  # [::-1]
sim_snaps = sim.snap_numbers  # [::-1]
# look for treebricks
if not np.any(hm == d for d in os.listdir(sim.path)):
    print(f"Treebricks not found in {sim.path}")


tgt_gid = None

last_snap = 446
last_aexp = sim.get_snap_exps(446, param_save=False)[0]


sim_aexps = sim.get_snap_exps(sim.snap_numbers)
sim_times = sim.get_snap_times(sim.snap_numbers)

nsteps = len(sim_times)

# reffs = np.zeros(nsteps)-1
# rvirs = np.zeros(nsteps)-1
mass = np.zeros(nsteps) - 1
pos = np.zeros((nsteps, 3)) - 1

dt = 50  # Myr

# snap_start = sim.get_closest_snap(zed=zmax)
# time_start = sim.get_snap_times(snap_start)[0]
# snap_end = sim.get_closest_snap(zed=zstt)
# time_end = sim.get_snap_times(snap_end)[0]
# snaps_times = np.arange(time_start, time_end, dt)

# snaps_to_run = np.asarray([sim.get_closest_snap(time=t) for t in snaps_times])

tgt_zed = 1.0

snaps_to_run = [sim.get_closest_snap(zed=tgt_zed)]

true_snap_times = np.asarray([sim.get_snap_times(s) for s in snaps_to_run])
true_snap_aexps = np.asarray([sim.get_snap_exps(s) for s in snaps_to_run])

print(snaps_to_run)

for istep, (nh_snap, aexp, time) in enumerate(
    zip(snaps_to_run, true_snap_aexps, true_snap_times)
):
    # for istep, (nh_snap, aexp, time) in enumerate(zip(sim.snap_numbers,sim_aexps, sim_times)):

    if abs(aexp - sim_aexps[np.argmin(np.abs(sim_aexps - aexp))]) > 1e-3:
        continue

    if not os.path.exists(get_halo_assoc_file(sim.path, nh_snap)):
        continue

    # gal_cat = get_gal_props_snap(sim.path, nh_snap, main_stars=True)
    gal_cat = get_gal_props_snap(sim.path, nh_snap)

    gids = gal_cat["gids"]

    masses = gal_cat["mass"]

    mgal_cond = masses > mlim

    print(
        f"redshift is {1./sim.get_snap_exps(nh_snap)[0]-1:.1e}, {np.sum(mgal_cond):d} galaxies massive enough..."
    )

    if not overwrite_CDD and not overwrite_PFS:
        print("using existing domain and photon files where available")

    # print(f"Setting up rascas for NH galaxy in halo hid:{tgt_hid:d}")

    for igal, gid in enumerate(gids[mgal_cond]):

        for fake_zed in np.arange(0.5,6,0.5):

            rascas_paths = []
            output_dirs = []
            run_pfss = []
            run_cdds = []

            zstr = f'{fake_zed:.1f}'.replace('.','p')
            run_name = f"zoom_tgt_bc03_chabrier100_wlya_ZTEST_{dust_model:s}/Z{zstr}"

            # run_name = f"zoom_tgt_bc03_chabrier100_wlya_ZTEST_{dust_model:s}"

            fake_aexp = 1/(1+fake_zed)

            # mass[istep] = gal_cat["mass"][mgal_cond][igal]
            # pos[istep, :] = gal_cat["pos"][mgal_cond, igal]
            # # reffs[istep] = gal_cat["reff"][igal]
            # # rvirs[istep] = gal_cat["rvir"][igal]

            # # cur_rmax = rvirs[istep]
            # # cur_rmax = r90
            cur_mass = gal_cat["mass"][mgal_cond][igal]

            cur_rmax = gal_cat["rmax"][mgal_cond][igal]
            cur_hid = gal_cat["host hid"][mgal_cond][igal]
            cur_pos = gal_cat["pos"][:,mgal_cond][:,igal]
            # halos = get_halo_props_snap(sim.path, nh_snap, hids=cur_hid)
            # cur_rvir = halos[f'halo_{cur_hid:07d}']['rvir']

            if cur_rmax==0:
                print('zero rmax... skipping... What happened ?')
                continue


            this_gals_subs = (gal_cat['host gid']==gid) * (gal_cat['gids']!=gid) * (gal_cat['mass']>=1e6)
            if this_gals_subs.sum()>0:
                subs_pos = gal_cat['pos'][:,this_gals_subs]
                subs_rs = gal_cat['rmax'][this_gals_subs]
                # subs_rs = all_gal_props['r90'][this_gals_subs] * 1

                dists_to_subs = np.linalg.norm(cur_pos - subs_pos.T,axis=1)
                max_extent_subs = dists_to_subs + subs_rs

                kpc50_lim = max_extent_subs< (50/(sim.cosmo.lcMpc*1e3))

                max_containinig_subs = max_extent_subs[kpc50_lim].max()

                cur_rmax = max(max_containinig_subs, gal_cat['r90'][mgal_cond][igal])

            else:

                cur_rmax = gal_cat['r90'][mgal_cond][igal]

            # cur_rmax = min(cur_rvir * 0.33, cur_rmax)

            if cur_rmax==0:
                raise ValueError
            if cur_rmax>1:
                raise ValueError
            
            cur_gid = gid
            cur_pos =  gal_cat["pos"][:,mgal_cond][:,igal]

            # cur_rmax_ppc = cur_rmax * sim.cosmo.lcMpc * 1e6 * 1 #max size corresponds to aexp=1 or z=0
            # cur_rmax_ppc = cur_rmax * sim.cosmo.lcMpc * 1e6 * fake_aexp
            cur_rmax_cpc = cur_rmax * sim.cosmo.lcMpc * 1e6 

            # SPEC
            spec = True
            rspec = cur_rmax  # cur_r50 * rfact #APERTURE!!!!
            nspec = 512

            # IMG
            img = False
            rimg = cur_rmax * 2  # cur_r50 * rfact #SIDE OF IMAGE
            if nimg is None or tgt_res is not None:
                nimg = int(np.ceil(cur_rmax_cpc / tgt_res))

            # CUBE
            cube = True
            rcube = cur_rmax * 2  # cur_r50 * rfact #SIDE OF CUBE
            nspec_cube = nspec
            if ncube is None or tgt_res is not None:
                ncube = int(np.ceil(cur_rmax_cpc / tgt_res))

            print(f"ncube is {ncube}, rmax is {cur_rmax}, cur_rmax_cpc is {cur_rmax_cpc}")
            print(ncube,cur_rmax_cpc,tgt_res)

            # get wavs that cover z=0 filters at aexp
            # lamb_min = obs_lamb_min * aexp[0]  # mult by aexp=1/(1+z) to observed frame
            # lamb_max = obs_lamb_max * aexp[0]

            lamb_min = obs_lamb_min * fake_aexp  # mult by aexp=1/(1+z) to observed frame
            lamb_max = obs_lamb_max * fake_aexp

            # lamb_min = obs_lamb_min * 1/(1+zmax)  # mult by aexp=1/(1+z) to observed frame
            # lamb_max = obs_lamb_max * 1/(1+zmin)

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
                p=sim_dir,
                snap=nh_snap,
                gal_id=cur_gid,
                run_name=run_name,
                start_dir=start_dir,
            )

            if not os.path.exists(rascas_path):
                os.makedirs(rascas_path)

            print(rascas_path)
            copy_exec(rascas_path, NH=True)

            if dust_model_param == "scaling_Z":

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

            if (
                not os.path.exists(os.path.join(rascas_path, "params_CDD.cfg"))
                or overwrite_CDD
            ):
                # make_CDD_params(rascas_path, sim.path, snap, cur_pos, cur_r50 * rfact * 1.55)
                make_CDD_params(
                    rascas_path,
                    sim_dir,
                    nh_snap,
                    cur_pos,
                    cur_rmax * rfact * 1.55,
                    options=cdd_params,
                )

                print('made CDD params')



            run_CDD = False
            if (
                not os.path.exists(os.path.join(rascas_path, "DomDump", "domain_1.dom"))
                or overwrite_CDD
            ):
                run_CDD = True
                # run_CDD(rascas_path)

            if (
                not os.path.exists(os.path.join(rascas_path, "params_PFS.cfg"))
                or overwrite_PFS
            ):
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

            if not os.path.exists(os.path.join(rascas_path, "mocks.h5")) or overwrite:

                if (
                    not os.path.exists(
                        os.path.join(rascas_path, "RASCASDump", "rascas_dump.dat")
                    )
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

            if len(rascas_paths) == 0:
                continue

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

    # break
