import numpy as np
from shutil import copyfile
import os

from time import sleep

from gremlin.read_sim_params import ramses_sim

import argparse

def create_sh(
    tmplt,
    tgt_path,
    pbs_params=None,
    nnodes=1,
    ntasks=1,
    wt=None,
    name=None,
):

    lines = []

    with open(tmplt, "r") as f:
        lines = f.readlines()

    cd_line = np.where(["cd" in l for l in lines])[0][0]

    lines[cd_line] = f"cd {tgt_path}\n"

    if pbs_params is not None:
        last_pbs_line = np.where(["#SBATCH" in l for l in lines])[0][-1]
        nb_added = 0
        for key, val in pbs_params.items():
            if key in lines:
                continue
            lines.insert(last_pbs_line + nb_added, f"#SBATCH -{key} {val}\n")
            nb_added += 1

    if nnodes is not None or wt is not None:
        if wt == None:
            wt = "48:00:00"
        if nnodes == None:
            nnodes = 1

        node_line = np.where(["nodes=" in l for l in lines])[0][0]
        lines[node_line] = f"#SBATCH --nodes={nnodes:d}\n"

        ppn_line = np.where(["ntasks-per-node=" in l for l in lines])[0][0]
        lines[ppn_line] = (
            f"#SBATCH --ntasks-per-node={int(np.ceil(float(ntasks)/nnodes)):d}\n"
        )

        time_line = np.where(["#SBATCH --time" in l for l in lines])[0][0]
        lines[time_line] = f"#SBATCH --time={wt}\n"

    if name is not None:
        name_line = np.where(["#SBATCH --job-name" in l for l in lines])[0][0]
        lines[name_line] = f"#SBATCH --job-name {name}\n"

    with open(os.path.join(tgt_path, "run_hm.sh"), "w") as f:
        f.writelines(lines)


def get_input_template(hm_type="DM"):
    cfg = {}

    fname = f"/home/jlewis/HaloMakerYD/HaloMaker_{hm_type}/input_HaloMaker.dat"
    # print(fname)

    with open(fname) as src:
        for line in src:
            if line.startswith("!"):
                continue
            else:
                useful_chars = line.split("!")[0].strip()
                key, value = useful_chars.split("=")
                cfg[key.strip()] = value.strip()

    return cfg


def configure_input_template(cfg, nsnaps, aexpi, aexpf):
    cfg["nsteps"] = str(nsnaps)
    # cfg["af"] = str(aexpf)
    # cfg["lbox"] = str(float(cfg["lbox"]) * aexpi)
    return cfg


def write_input_file_steps(fname, sim_dir, snap_nbs, ftype="Ra4"):
    with open(fname, "w") as f:
        for sn in snap_nbs:
            if sn == 0:
                continue
            outstr = f"output_{sn:05d}"
            f.write(f"'{sim_dir}/{outstr}/' {ftype} 1 {sn:05d}\n")


# sim_dir = "/automnt/data101/jlewis/sims/proposal_zooms/ExtractZoom/mh1e12/id2757_995pcnt_mstel"
# sim_dir = "/automnt/data101/jlewis/sims/proposal_zooms/ExtractZoom/mh1e12/id19782_500pcnt_mstel"
# sim_dir = "/automnt/data101/jlewis/sims/proposal_zooms/ExtractZoom/mh1e12/id13600_005pcnt_mstel"
# sim_dir = "/automnt/data101/jlewis/sims/proposal_zooms/ExtractZoom/mh1e13/TestRun1e13"

# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e11/id292074"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e11/id292074/"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479_low_nstar"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479_highnstar_nbh/"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099/"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704/lower_nstar"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099/high_nstar"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099/high_nstar_eps0p5"

sim_dirs = [
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_old/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479_old/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_lesscoarse/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_accBoost/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_gravSink_SE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_DynBondiGravSinkMass_SE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi/higher_nmax",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi/",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel/",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_high_SN/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_high_thermal_eff/",
    # "/data103/jlewis/sims/lvlmax_21/mh1e12/id180130_256",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE_lowSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_smooth_ref",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_smooth_ref",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_boostNH",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_boostNH_resimBoostFriction",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_frcAccrt",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_early_refine",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_lowerSFE_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_fullresIC",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646",  # do these
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646_nh",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_nh",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_noHydroDragB4z4merge",l
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE",
    # "/data103/jlewis/sims/lvlmax_22/mh1e12/id147479",
    # "/data103/jlewis/sims/lvlmax_22/mh1e12/id138140",
    # "/data103/jlewis/sims/lvlmax_22/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e11/id292074",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
    # # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130_nh2/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_minMassDynGrav",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_vlowSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_low_dboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE_highAGNtherm",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowMseed/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_lowerSFE/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nrg_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_pdrag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highSFE_lowSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_kicks",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_fullresIC",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id74099/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242704",
    # "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242756",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_coreen",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_lowerSFE_NHboost",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictBH",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_stricterSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242704_lower_nstar",
    # "/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242756_lower_nstar",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_noAGN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_boostgrowth",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi/id180130_meanBondi_Sconstant",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_evenlesscoarse/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_leastcoarse/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_leastcoarse/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_evenlesscoarse/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_leastcoarse",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_high_nssink",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_eagn_T0p15/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099_inter/",
    # "/data103/jlewis/sims/lvlmax_21/mh1e12/id180130_256",
    # "/data102/jlewis/sims/lvlmax_21/mh1e12/id180130_model6_eps0p05",
    # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_DynBondiGravSinkMass",
    # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_lowerSFE",
    # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_drag",
    # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_lowerSFE_drag",
]


# sim_dirs = [
# "/data102/dubois/ZoomIMAGE/Dust/BondiNoVrel",
# "/data102/dubois/ZoomIMAGE/Dust",
# "/data102/dubois/ZoomIMAGE/Dust/MeanBondi",
# "/data102/dubois/ZoomIMAGE/Dust/MeanBondiVrelNonZero",
# "/data102/dubois/ZoomIMAGE/Dust/NoAGN",
# ]

def setup_halo_maker(sim_dirs=sim_dirs,launch=True,zlim=10.,overwrite=False, out_dir=None,snap=None):

    # out_dir = "/data101/jlewis/sims/YD_accretion_tests"
    # out_dir = None
    # launch = False
    # launch = True
    # overwrite = False

    # zlim = 10.0

    # if not os.path.exists(out_dir):
    # os.makedirs(out_dir)

    hm_types = ["DM_dust", "stars2_dp_rec_dust"]
    # hm_types = ["DM_dust"]
    # hm_types = ["stars2_dp_rec_dust"]

    for hm_type in hm_types:

        for sim_dir in sim_dirs:


            if os.path.exists(os.path.join(sim_dir, "OUTPUT_DIR")):
                sim = ramses_sim(sim_dir, nml="cosmo.nml", output_path="OUTPUT_DIR")
            else:
                sim = ramses_sim(sim_dir, nml="cosmo.nml")
            print(sim.name)

            output_nbs = sim.snap_numbers
            if snap==None:
                nsnaps = len(output_nbs)
            else:
                nsnaps = 1

            print(nsnaps)

            try:
                iaexp = sim.aexp_stt
            except AttributeError:
                continue
            faexp = sim.aexp_end

            # get sim aexps
            sim.get_snap_exps(output_nbs)
            sim.init_cosmo()

            # sim_times = sim.get_snap_times(output_nbs)

            # step = 1  # Myr
            # sim_steps, sim_step_args = np.unique(
            #     (sim_times - sim_times[0]) // step, return_index=True
            # )

            # sim_step_args = np.unique(sim_step_args.tolist() + [len(output_nbs) - 1])

            # aexps = sim.aexps[sim_step_args]
            # output_nbs = sim.snap_numbers[sim_step_args]
            aexps = sim.aexps
            output_nbs = sim.snap_numbers
            
            faexp = aexps[-1]

            # print(1.0 / aexps - 1.0)

            out_dir = None

            if "star" in hm_type:
                iaexp = 1.0 / (zlim + 1.0)

                aexp_filter = aexps > iaexp

                output_nbs = output_nbs[aexp_filter]


            if out_dir == None:
                out_dir = sim_dir

            hmaker_dir = os.path.join(out_dir, "HaloMaker_" + hm_type)
            os.makedirs(hmaker_dir, exist_ok=True)

            tmplt0 = get_input_template(hm_type=hm_type)

            
            cfg = configure_input_template(tmplt0, nsnaps, iaexp, faexp)
            cfg_fname = os.path.join(hmaker_dir, "input_HaloMaker.dat")

            with open(cfg_fname, "w") as f:
                for key, value in cfg.items():
                    f.write(f"{key} = {value}\n")

            # check for existing outputs
            existing_outs = np.asarray(
                [f for f in os.listdir(hmaker_dir) if f.startswith("tree_bricks")]
            )
            existing_out_nbs = np.asarray([int(f[-3:]) for f in existing_outs])

            if not overwrite:
                # only run on not existing outputs
                new_output_nbs = np.setdiff1d(
                    output_nbs, existing_out_nbs, assume_unique=True
                )
                if len(existing_out_nbs) > 0: #skip first snaps that don't have galaxies/halos
                    new_output_nbs = new_output_nbs[existing_out_nbs.min()<new_output_nbs]
            else:
                new_output_nbs = output_nbs

            if snap is not None:
                new_output_nbs = np.intersect1d(new_output_nbs, [snap])

            if len(new_output_nbs) == 0:
                print('No new outputs to process')
                continue

            write_input_file_steps(
                os.path.join(hmaker_dir, "inputfiles_HaloMaker.dat"),
                sim.output_path,
                new_output_nbs,
            )

            # # make .sh file
            sh_name = f"{hm_type.split('_')[0]:s}_{sim.name:s}"
            create_sh(
                "./run_hm.sh",
                hmaker_dir,
                nnodes=1,
                wt="06:00:00",
                name=sh_name,
            )

            # copy executable
            src = os.path.join("/home/jlewis/HaloMakerYD/HaloMaker_" + hm_type, "HaloMaker")
            dst = os.path.join(hmaker_dir, "HaloMaker")
            copyfile(src, dst)
            # make executable
            os.system(f"chmod +x {dst}")

            print(f"Setup in {hmaker_dir}")

            working_dir = os.getcwd()
            if launch:
                os.system(f"cd {hmaker_dir}; sbatch run_hm.sh")
                print("Launched job")
                os.system(f"cd {working_dir}")

                sleep(1)

            print("Done!")


if __name__ == "__main__":

    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "-sim_dirs", type=str, help="Path to simulation directories", nargs="+", default = sim_dirs
    )
    argParser.add_argument("-zlim", type=float, default = 10.)
    argParser.add_argument("--nolaunch", action="store_true", default = False)
    argParser.add_argument("--overwrite", action="store_true", default = False)
    argParser.add_argument("--out_dir", type=str, default = None)
    argParser.add_argument("--snap", type=int, default = None)

    args = argParser.parse_args()

    setup_halo_maker(args.sim_dirs, launch=not args.nolaunch, zlim=args.zlim, overwrite=args.overwrite, out_dir=args.out_dir, snap=args.snap)