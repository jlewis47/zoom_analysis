
import numpy as np
import os

from zoom_analysis.rascas.filts.filts import read_transmissions

from zoom_analysis.rascas.rascas_steps import *

from zoom_analysis.halo_maker.assoc_fcts import (
    get_halo_assoc_file,
    get_gal_props_snap,
)
from gremlin.read_sim_params import ramses_sim

def dict_conv_bools(rascas_params):
    for key in rascas_params:
        for subkey in rascas_params[key]:
            if type(rascas_params[key][subkey]) == bool:
                rascas_params[key][subkey] = "T" if rascas_params[key][subkey] else "F"


debug = True
overwrite_CDD = False
overwrite_PFS = False

sim_dir = os.path.join("/data101/jlewis/sims/NH_for_rascas")

targets = {'snaps':[439,439],
           'gids':[3,2],
           } #maybe make into cli ? or read a file

rfact = 1.1

# dust_model = "SMC"
dust_model = "noDust"

# sim = get_nh_sim()

sim = ramses_sim("/data101/jlewis/sims/NH_for_rascas", nml="cosmo.nml")

ndir_hp = 1  # number of directions... should be one or 2^N for healpy; 1-> 12 directions in mocks
# delta_T = 100  # Myr spacing of rascas runs

# PFS
# nphot = 500000
nphot = 1500000
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

obs_lamb_min = np.min(np.concatenate(wvs)) * 0.95  # Ang obs frame
obs_lamb_max = np.max(np.concatenate(wvs)) * 1.05  # Ang obs frame

# nscatter = 0
nscatter = 2

run_name = f"zoom_tgt_bc03_chabrier100_wlya_{dust_model:s}"

if dust_model == "noDust":
    ignore_dust = True
else:
    ignore_dust = False

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
ntasks = 1

hper_gal = 1

ngal = len(targets['snaps'])

wt = f"{round(ngal*hper_gal,ndigits=1):02d}:00:00"

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

if not hasattr(sim, "cosmo_model"):
    sim.init_cosmo()


sim_times = sim.get_snap_times(param_save=False)  # [::-1]
sim_snaps = sim.snap_numbers  # [::-1]

nsteps = len(sim_times)


rascas_paths = []
output_dirs = []
run_pfss = []
run_cdds = []


for itgt,(nh_snap,gid) in enumerate(zip(targets['snaps'],targets['gids'])):

    aexp = sim.get_snap_exps(nh_snap)[0]

    if not os.path.exists(get_halo_assoc_file(sim.path, nh_snap)):
        continue

    gal_cat = get_gal_props_snap(sim.path, nh_snap)

    gids = gal_cat["gids"]

    cat_arg = np.where(gid==gids)[0][0]

    if not overwrite_CDD and not overwrite_PFS:
        print("using existing domain and photon files where available")


        # # cur_rmax = rvirs[istep]
        # # cur_rmax = r90
        cur_mass = gal_cat["mass"][cat_arg]
        cur_rmax = gal_cat["rmax"][cat_arg]
        cur_gid = gid
        cur_pos =  gal_cat["pos"][:,cat_arg]

        cur_rmax_ppc = cur_rmax * sim.cosmo.lcMpc * 1e6 * aexp

        # SPEC
        spec = True
        rspec = cur_rmax  # cur_r50 * rfact #APERTURE!!!!
        nspec = 1024

        # IMG
        img = True
        rimg = cur_rmax * 2  # cur_r50 * rfact #SIDE OF IMAGE
        if nimg is None or tgt_res is not None:
            nimg = int(np.ceil(cur_rmax_ppc / tgt_res))

        # CUBE
        cube = True
        rcube = cur_rmax * 2  # cur_r50 * rfact #SIDE OF CUBE
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

        if dust_model == "scaling_Z":

            copy_draine_table(draine_table, rascas_path)

        # make_rascas_params(rascas_path, ndir, rascas_params)

        # make_mock_params(
        #     rascas_path,
        #     cur_pos,
        #     dir_vects,
        #     spec=spec_params,
        #     img=img_params,
        #     cube=cube_params,
        # )

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


        run_cdds.append(run_CDD)
        run_pfss.append(run_PFS)
        rascas_paths.append(rascas_path)
        output_dirs.append(f"output_{nh_snap:05d}")
 

create_sh_multi(
    rascas_paths,
    nnodes=nnodes,
    ntasks=ntasks,
    nomp=nomp,
    wt=wt,
    run_pfss=run_pfss,
    run_cdds=run_cdds,
    output_dirs=output_dirs,
    rascas=False,
    bignode=False,
    rmv_tar=False,
)

