import numpy as np
from shutil import copyfile
import os

from time import sleep

from gremlin.read_sim_params import ramses_sim


def create_sh(
    tmplt,
    tgt_path,
    pbs_params=None,
    nnodes=1,
    wt=None,
    name=None,
):

    ntasks = 1
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
            wt = "10:00:00"
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

    with open(os.path.join(tgt_path, "run_tm.sh"), "w") as f:
        f.writelines(lines)


def configure_input_template(brick_files):

    cfg = ""
    cfg += f"{len(brick_files)} 0\n"

    for brick_file in brick_files:
        cfg += f"'{brick_file}'\n"
    return cfg


def detect_brick_files(sim_dir, hm_type):

    fpath = os.path.join(sim_dir, "HaloMaker_" + hm_type)

    files = os.listdir(fpath)
    brick_files = np.asarray(
        [os.path.join(fpath, f) for f in files if f.startswith("tree_bricks")]
    )

    brick_nbs = np.asarray([int(f[-3:]) for f in brick_files])

    return brick_files[np.argsort(brick_nbs)]


# def write_input_file_steps(fname, sim_dir, snap_nbs, ftype="Ra4"):
#     with open(fname, "w") as f:
#         for sn in snap_nbs:
#             if sn == 0:
#                 continue
#             outstr = f"output_{sn:05d}"
#             f.write(f"'{sim_dir}/{outstr}/' {ftype} 1 {sn:05d}\n")


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
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_gravSink_SE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowSFE_DynBondiGravSinkMass_SE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_fullresIC",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_kicks",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowMseed/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_lowerSFE/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479/",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_noAGN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_noHydroDragB4z4merge",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380ll",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130",
    # "/data103/jlewis/sims/lvlmax_21/mh1e12/id180130_256",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_accBoost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_high_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_high_thermal_eff",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_novrel_lowSFE_SE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id147479",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id138140",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_norvel_lowerSFE_highAGNtherm",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_lowerSFE_NHboost",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictSF",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_strictBH",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost_stricterSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh2",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi/id180130_meanBondi_Sconstant",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_boostgrowth",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi/id180130_meanBondi_higher_nmax",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nrg_SN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_pdrag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highSFE_lowSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag_lowerSFE_lowNsink",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_smooth_ref",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE_lowSN",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_vlowSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_low_dboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag_minMassDynGrav",
    "/data103/jlewis/sims/lvlmax_21/mh1e12/id180130_256",
    "/data102/jlewis/sims/lvlmax_21/mh1e12/id180130_model6_eps0p05",
    # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_DynBondiGravSinkMass",
    # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_lowerSFE",
    # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_drag",
    # "/data102/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_lowerSFE_drag",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_early_refine",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_boostNH",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_boostNH_resimBoostFriction",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_nosmooth_frcAccrt",
    # "/data103/jlewis/sims/lvlmax_20/mh1e12/id180130_superEdd_lowerSFE_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_sf0_noboost_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646",
    # "/data101/jlewis/sims/dust_fid/lvlmax_21/mh1e12/id26646_nh",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id26646_nh",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646_noHydroDragB4z4merge",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e11/id292074"
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130"
]

sim_dirs = np.unique(sim_dirs)

# sim_dirs = [
# "/data102/dubois/ZoomIMAGE/Dust/BondiNoVrel",
# "/data102/dubois/ZoomIMAGE/Dust",
# "/data102/dubois/ZoomIMAGE/Dust/MeanBondi",
# "/data102/dubois/ZoomIMAGE/Dust/MeanBondiVrelNonZero",
# "/data102/dubois/ZoomIMAGE/Dust/NoAGN",
# ]

# out_dir = "/data101/jlewis/sims/YD_accretion_tests"
out_dir = None
# launch = False
launch = True

# if not os.path.exists(out_dir):
# os.makedirs(out_dir)

for sim_dir in sim_dirs:

    print(sim_dir)

    if os.path.exists(os.path.join(sim_dir, "OUTPUT_DIR")):
        sim = ramses_sim(sim_dir, nml="cosmo.nml", output_path="OUTPUT_DIR")
    else:
        sim = ramses_sim(sim_dir, nml="cosmo.nml")

    output_nbs = sim.snap_numbers
    nsnaps = len(output_nbs)

    iaexp = sim.aexp_stt
    faexp = sim.aexp_end

    # get sim aexps
    sim.get_snap_exps(output_nbs)
    sim.init_cosmo()

    sim_times = sim.get_snap_times(output_nbs)

    step = 1  # Myr
    sim_steps, sim_step_args = np.unique(
        (sim_times - sim_times[0]) // step, return_index=True
    )

    sim_step_args = np.unique(sim_step_args.tolist() + [len(output_nbs) - 1])

    aexps = sim.aexps[sim_step_args]
    output_nbs = sim.snap_numbers[sim_step_args]
    nsnaps = len(output_nbs)
    faexp = aexps[-1]

    print(1.0 / aexps - 1.0)

    zlim = 10.0

    tree_types = ["Halo", "Gal"]
    hm_types = ["DM_dust", "stars2_dp_rec_dust"]
    # hm_types = ["stars2_dp_rec_dust"]

    for hm_type, tree_type in zip(hm_types, tree_types):

        out_dir = None

        if "star" in hm_type:
            iaexp = 1.0 / (zlim + 1.0)

            aexp_filter = aexps > iaexp

            output_nbs = output_nbs[aexp_filter]
            nsnaps = len(output_nbs)

        if out_dir == None:
            out_dir = sim_dir

        tmaker_dir = os.path.join(out_dir, "TreeMaker" + hm_type)
        os.makedirs(tmaker_dir, exist_ok=True)

        bricks = detect_brick_files(sim_dir, hm_type)

        cfg = configure_input_template(bricks)
        cfg_fname = os.path.join(tmaker_dir, "input_TreeMaker.dat")
        with open(cfg_fname, "w") as f:
            f.write(cfg)

        # # make .sh file
        sh_name = f"treemaker_{sim.name:s}_{hm_type:s}"
        create_sh(
            "./run_tm.sh",
            tmaker_dir,
            nnodes=1,
            wt="02:00:00",
            name=sh_name,
        )

        # copy executables
        src_path = "/home/jlewis/TreeMakerYD/Tree" + tree_type

        src = os.path.join(src_path, "TreeMaker")
        dst = os.path.join(tmaker_dir, "TreeMaker")
        copyfile(src, dst)
        # make executable
        os.system(f"chmod +x {dst}")

        src = os.path.join(src_path, "manipulate_mergertree")
        dst = os.path.join(tmaker_dir, "manipulate_mergertree")
        copyfile(
            src,
            dst,
        )
        os.system(f"chmod +x {dst}")

        print(f"Setup in {tmaker_dir}")

        working_dir = os.getcwd()
        if launch:
            os.system(f"cd {tmaker_dir}; sbatch run_tm.sh")
            print("Launched job")
            os.system(f"cd {working_dir}")

            sleep(1)

        print("Done!")
