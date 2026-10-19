import numpy as np
import os

from zoom_analysis.rascas.filts.filts import read_transmissions

# from hagn.catalogues import get_cat_gids, make_super_cat
from zoom_analysis.rascas.rascas_steps import *

from zoom_analysis.halo_maker.assoc_fcts import (
    
    get_gal_props_snap,
    get_halo_assoc_file,

)

from zoom_analysis.rascas.read_rascas import read_PFS_dump
from zoom_analysis.rascas.rascas_plots import make_PFS_img

from gremlin.read_sim_params import ramses_sim




def dict_conv_bools(rascas_params):
    for key in rascas_params:
        for subkey in rascas_params[key]:
            if type(rascas_params[key][subkey]) == bool:
                rascas_params[key][subkey] = "T" if rascas_params[key][subkey] else "F"


debug = True
overwrite_CDD = False
overwrite_PFS = False
overwrite = False

# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id74099"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id138140"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756"
sim_path = "/data101/jlewis/sims/NH_for_rascas"
# sim_path = "/data103/jlewis/sims/lvlmax_22/mh1e12/id180130"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost"
# sim_path='/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF'
# target_hid = 203
# zed_target = 2.0
# zed_target = 3.6
deltaT = 50  # Myr
rfact = 1.0

mlim = 1e10  # galaxies must be at least this mass to be tracked

ndir_hp = 1  # number of directions... should be one or 2^N for healpy; 1-> 12 directions in mocks
# delta_T = 100  # Myr spacing of rascas runs

# CENTROID & SIZE
# tgt_zed = 2.0
# pos = None
# rad = None
# if pos or rad are none we look for treebricks and setup rascas
# mlim = 1e10
zmax = 99.0
# zstt = 2.0
# zmax = 4.0
zed_targets = None
# zed_targets = [2.0]

snap_targets = None
# snap_targets = np.arange(180, 200)
# snap_targets = [207]
# snap_targets = [214,210,206,202,198,194,190,186,182,178,174,170,166,162,158,154,150]
# snap_targets =[82,90,131,159,182]
# snap_targets =[64, 105, 140, 168, 175]
# snap_targets=[58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,
# 76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,
# 94,95,97,98,99,100,101,102,103,105,106,107,108,109,110,111,112,113]


dust_model = "SMC"
# dust_model = "noDust"
run_name = f"zoom_tgt_bc03_chabrier100_wlya_{dust_model:s}"

if dust_model == "noDust":
    ignore_dust = True
else:
    ignore_dust = False


# make sure run on fat 128 node + use all procs (incl. first step)

# PFS
# nphot = 500000
nphot = 1500000
# obs_lamb_min = 100# Ang obs frame
# obs_lamb_max = 55000# Ang obs frame

ns, wvs, trs = read_transmissions()

obs_lamb_min = np.min(np.concatenate(wvs)) * 0.95  # Ang obs frame
obs_lamb_max = np.max(np.concatenate(wvs)) * 1.05  # Ang obs frame
# obs_lamb_min = 800  # Ang obs frame
# obs_lamb_max = 55000  # Ang obs frame
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
ignoreDust = ignore_dust
# ignoreDust = True

# tgt_res = 40  # ppc
nimg = None  # 128
ncube = None  # 128


sim = ramses_sim(sim_path, nml="cosmo.nml", full_check=False)


print(f"Working on {sim.name:s}")

sim_aexps = sim.get_snap_exps()  # [::-1]
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
ntasks = 24
wt = "48:00:00"

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
pfs_params["ramses"]["particle_families"] = "T"

rascas_params = {}
rascas_params["master"] = {}
rascas_params["master"]["dt_backup"] = 7200 * 4  # 4h between backups

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

print(f"Dust model is {dust_model:s}")

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

# hagn_sim = get_hagn_sim()


name = sim_path.split("/")[-1]

if name != "NH_for_rascas":
    intID = int(name[2:].split("_")[0])
else:
    name="NewHorizon"
# gid = int(make_super_cat(197, outf="/data101/jlewis/hagn/super_cats")["gid"][0])


if not hasattr(sim, "cosmo_model"):
    sim.init_cosmo()

rascas_paths = []
output_dirs = []
run_pfss = []
run_cdds = []

if not overwrite_CDD and not overwrite_PFS:
    print("using existing domain and photon files where available")

print(f"Setting up rascas for {sim_path.split('/')[-1]}")

sim_times = sim.get_snap_times()  # [::-1]
sim_snaps = sim.snap_numbers  # [::-1]

last_time = sim_times[-1] + 2 * deltaT

tgt_gid = None


assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

avail_aexps = np.intersect1d(
    sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
)
avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

_, data_snaps = sim.get_snaps(tar_snaps=True, full_snaps=True, mini_snaps=False)
data_snap_aexps = sim.get_snap_exps(data_snaps)
close_tol = 1e-3


if zed_targets is not None:
    snap_target_args = np.asarray(
        [np.argmin(np.abs(data_snap_aexps - 1.0 / (z + 1.0))) for z in zed_targets]
    )
    snap_targets = data_snaps[snap_target_args]
    err = data_snap_aexps[snap_target_args] - 1.0 / (np.asarray(zed_targets) + 1.0)
    snap_targets = snap_targets[np.abs(err) < close_tol]

print(f"Targets are {snap_targets}")


# hid_start, _, _, true_start_aexp,found = starting_hid_from_hagn(
#     zstt, sim, hagn_sim, intID, avail_aexps, avail_times
# )
# target_gid = galaxies["gids"][galaxies["mass"].argmax()]

# true_start_zed = 1.0 / true_start_aexp - 1.0
# true_start_snap = sim.get_closest_snap(aexp=true_start_aexp)
# print(hid_start, true_start_zed, true_start_snap)


for i, (snap, aexp, time) in enumerate(
    zip(sim_snaps[::-1], sim_aexps[::-1], sim_times[::-1])
):
        
    if abs(time-last_time)<deltaT:
        print('too close to prev snapshot... skipping')
        if snap not in [446]:
            continue

    last_time=time

    if snap_targets is not None:
        if snap not in snap_targets:
            continue

    # print(i, snap, aexp, time)
    if 1.0 / aexp - 1 > zmax:
        print(f"Reached zmax at snap {snap}")
        continue

    if not os.path.isfile(get_halo_assoc_file(sim_path, snap)):
        print(f"No halo associations for snap {snap}")
        continue

    if snap not in sim.get_snaps(full_snaps=True, mini_snaps=False, tar_snaps=True)[1]:
        print(f"No output tar found for snap {snap}")
        continue

    try:
        gal_dict = get_gal_props_snap(sim.path, snap)
    except KeyError:
        print(f"No galaxy properties found for snap {snap}")
        continue

    pure = gal_dict["host purity"] > 0.9999
    central = gal_dict["central"]
    mcut = gal_dict["mass"] >= mlim

    cond_gals = pure * central * mcut

    gal_args = np.where(cond_gals)[0]

    for gal_arg in gal_args:

        rascas_paths = []
        output_dirs = []
        run_pfss = []
        run_cdds = []            

        # print(tree_hids[0][tree_arg])

        # hdict, hosted_gals = get_halo_props_snap(sim_path, snap, tree_hids[0][tree_arg])

        # # print(hosted_gals)
        # if not "gids" in hosted_gals:
        #     print(f"No galaxies found in snap {snap}")
        #     continue

        # cur_gid = int(hosted_gals["gids"][np.argmax(hosted_gals["mass"])])

        # _, cur_gal_props = get_gal_props_snap(sim_path, snap, cur_gid)

        # aexp_arg = np.argmin(np.abs(aexp - gal_props_tree["aexps"]))

        cur_gid = gal_dict["gids"][gal_arg]

        cur_rmax = gal_dict["rmax"][gal_arg]
        cur_hid = gal_dict["host hid"][gal_arg]
        cur_pos = gal_dict["pos"][:,gal_arg]
        # halos = get_halo_props_snap(sim.path, nh_snap, hids=cur_hid)
        # cur_rvir = halos[f'halo_{cur_hid:07d}']['rvir']

        if cur_rmax==0:
            print('zero rmax... skipping... What happened ?')
            continue


        this_gals_subs = (gal_dict['host gid']==cur_gid) * (gal_dict['gids']!=cur_gid) * (gal_dict['mass']>=1e6)
        if this_gals_subs.sum()>0:
            subs_pos = gal_dict['pos'][:,this_gals_subs]
            subs_rs = gal_dict['rmax'][this_gals_subs]
            # subs_rs = all_gal_props['r90'][this_gals_subs] * 1

            dists_to_subs = np.linalg.norm(cur_pos - subs_pos.T,axis=1)
            max_extent_subs = dists_to_subs + subs_rs

            kpc50_lim = max_extent_subs< (50/(sim.cosmo.lcMpc*1e3))

            if np.any(kpc50_lim):

                max_containinig_subs = max_extent_subs[kpc50_lim].max()

                cur_rmax = max(max_containinig_subs, gal_dict['r90'][gal_arg]*1.5)

            else:
            
                cur_rmax = gal_dict['r90'][gal_arg]*1.5

        else:

            cur_rmax = gal_dict['r90'][gal_arg]*1.5

        # cur_rmax = min(cur_rvir * 0.33, cur_rmax)

        if cur_rmax==0:
            raise ValueError
        if cur_rmax>1:
            raise ValueError

        

        if tgt_gid is None:
            tgt_gid = cur_gid

        # print(cur_gal_props["pos"], snap, cur_gid, cur_gal_props["mass"])
        cur_rmax = gal_dict["rmax"][gal_arg]
        cur_rmax_ppc = cur_rmax * sim.cosmo.lcMpc * 1e6 * aexp
        # cur_rmax_cpc = cur_rmax * sim.cosmo.lcMpc * 1e6
        # cur_r50 = cur_gal_props["r50"]
        cur_pos = gal_dict["pos"][:, gal_arg]

        print(f"...Setting up rascas for galaxy {cur_gid} in snap {snap}")
        print(f"Has mass {gal_dict['mass'][gal_arg]} and rmax {cur_rmax}")

        # SPEC
        spec = True
        rspec = cur_rmax  # cur_r50 * rfact
        nspec = 1024

        # IMG
        img = True
        rimg = cur_rmax * 2  # cur_r50 * rfact
        if nimg is None or tgt_res is not None:
            # nimg = int(np.ceil(cur_rmax_cpc / tgt_res))
            nimg = int(np.ceil(cur_rmax_ppc / tgt_res))

        # CUBE
        cube = True
        rcube = cur_rmax * 2  # cur_r50 * rfact
        nspec_cube = nspec
        if ncube is None or tgt_res is not None:
            # ncube = int(np.ceil(cur_rmax_cpc / tgt_res))
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
            p=sim.path,
            snap=snap,
            gal_id=cur_gid,
            run_name=run_name,
            start_dir=start_dir,
        )

        print(rascas_path)
        copy_exec(rascas_path)

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

        run_CDD = False
        run_PFS = False
        done = os.path.exists(
            os.path.join(rascas_path, "RASCASDump", "rascas_dump.dat")
        ) and os.path.exists(os.path.join(rascas_path, "mock_test.cube"))
        # print(done, overwrite)
        # input("continue")
        if not done or overwrite:
            if (
                not os.path.exists(os.path.join(rascas_path, "params_CDD.cfg"))
                or overwrite_CDD
            ):
                # make_CDD_params(rascas_path, sim.path, snap, cur_pos, cur_r50 * rfact * 1.55)
                make_CDD_params(
                    rascas_path,
                    sim.path,
                    snap,
                    cur_pos,
                    cur_rmax * rfact * 1.55,
                    options=cdd_params,
                )

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
                    sim.path,
                    snap,
                    cur_pos,
                    cur_rmax * rfact * 1.3,
                    # cur_r50 * rfact * 1.3,
                    pfs_params,
                )

            if (
                not os.path.exists(os.path.join(rascas_path, "PFSDump", "pfsdump"))
                or overwrite_PFS
            ):
                run_PFS = True

            # input("")

            run_cdds.append(run_CDD)
            run_pfss.append(run_PFS)
            rascas_paths.append(rascas_path)
            output_dirs.append(f"output_{snap:05d}")
       

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
