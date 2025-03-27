from ast import Str
import os
import numpy as np
import healpy as hp
from shutil import copy


def read_params(fin):

    params = {}

    with open(fin, "r") as f:
        lines = f.readlines()

    for il, l in enumerate(lines):
        # print(l)
        if l.startswith("#") or l.startswith("!") or l.strip() == "":
            continue

        if l.startswith("["):
            cat = l.strip()[1:-1]
            params[cat] = {}
        else:
            key, val = l.split("=")
            params[cat][key.strip()] = val.strip()

    return params


def write_params(params, fout):

    with open(fout, "w") as f:
        for cat in params.keys():
            f.write(f"[{cat}]\n")
            for key in params[cat].keys():
                f.write(f"{key}={str(params[cat][key])}\n")


def create_dirs(p, snap, gal_id=None, run_name=None, start_dir=""):

    if run_name == None:
        rascas_path = os.path.join(p, "rascas", f"output_{snap:05d}", start_dir)
    else:
        rascas_path = os.path.join(
            p, "rascas", f"{run_name}", start_dir, f"output_{snap:05d}"
        )

    if gal_id != None:
        rascas_path = os.path.join(rascas_path, f"gal_{gal_id:07d}")

    os.makedirs(rascas_path, exist_ok=True)

    os.makedirs(os.path.join(rascas_path, "PFSDump"), exist_ok=True)
    os.makedirs(os.path.join(rascas_path, "RASCASDump"), exist_ok=True)
    os.makedirs(os.path.join(rascas_path, "DomDump"), exist_ok=True)

    return rascas_path


def get_directions_cart(ndir):

    if ndir > 0:

        npix = hp.nside2npix(ndir)
        print(npix)
        theta, phi = hp.pix2ang(ndir, np.arange(npix))

        s_th = np.sin(theta)

        x = s_th * np.cos(phi)
        y = s_th * np.sin(phi)
        z = np.cos(theta)

        vects = np.transpose([x, y, z])

    else:

        vects = [1, 0, 0]

    return vects


def make_PFS_params(dout, sim_dir, snap, pos, rad, options={}):

    params = read_params("./params_PFS.cfg")

    # print(params)

    for cat in params:
        for param_option in params[cat]:
            for option_cat in options:
                if param_option in options[option_cat]:
                    params[cat][param_option] = str(options[option_cat][param_option])

    if not "PhotonsFromStars" in params:
        params["PhotonsFromStars"] = {}

    params["PhotonsFromStars"]["repository"] = str(sim_dir)
    params["PhotonsFromStars"]["snapnum"] = str(snap)
    params["PhotonsFromStars"]["outputfile"] = str(
        os.path.join(dout, "PFSDump/pfsdump")
    )

    params["PhotonsFromStars"]["star_dom_pos"] = f"{pos[0]} {pos[1]} {pos[2]}"
    params["PhotonsFromStars"]["star_dom_rsp"] = str(rad * 1.1)

    # print(params)

    fout = os.path.join(dout, "params_PFS.cfg")
    write_params(params, fout)


def run_PFS(d):

    print("Running PFS")

    exec_path = os.path.join(d, "PhotonsFromStars")
    params_path = os.path.join(d, "params_PFS.cfg")

    cmd = f"{exec_path} {params_path} > {os.path.join(d, 'log_PFS')}"

    os.system(cmd)

    print("Done")


def make_CDD_params(dout, sim_dir, snap, pos, rad, options={}):

    params = read_params("./params_CDD.cfg")

    # print(params)
    # print(options)

    # print(params)
    for cat in params:
        for param_option in params[cat]:
            for option_cat in options:
                if param_option in options[option_cat]:
                    params[cat][param_option] = options[option_cat][param_option]

    params["CreateDomDump"]["DomDumpDir"] = os.path.join(dout, "DomDump")
    params["CreateDomDump"]["repository"] = sim_dir
    params["CreateDomDump"]["snapnum"] = str(snap)

    params["CreateDomDump"]["comput_dom_pos"] = f"{pos[0]} {pos[1]} {pos[2]}"
    params["CreateDomDump"]["comput_dom_rsp"] = str(rad)

    params["CreateDomDump"]["decomp_dom_xc"] = str(pos[0])
    params["CreateDomDump"]["decomp_dom_yc"] = str(pos[1])
    params["CreateDomDump"]["decomp_dom_zc"] = str(pos[2])
    params["CreateDomDump"]["decomp_dom_rsp"] = str(rad * 1.25)

    params["gas_composition"][
        "atomic_data_dir"
    ] = "/home/jlewis/rascas/ions_parameters/"

    write_params(params, os.path.join(dout, "params_CDD.cfg"))


def run_CDD(d):

    print("Running CDD")

    exec_path = os.path.join(d, "CreateDomDump")
    params_path = os.path.join(d, "params_CDD.cfg")

    cmd = f"{exec_path} {params_path} > {os.path.join(d, 'log_CDD')}"

    os.system(cmd)

    print("Done")


def make_rascas_params(dout, ndir, options={}):

    params = read_params("params_rascas.cfg")

    for cat in params:
        for param in params[cat]:
            for options_cat in options:
                if param in options[options_cat]:
                    params[cat][param] = options[options_cat][param]

    if not "mock" in params:
        params["mock"] = {}
    params["mock"]["nDirections"] = ndir

    if not "gas_composition" in params:
        params["gas_composition"] = {}
    params["gas_composition"][
        "atomic_data_dir"
    ] = "/home/jlewis/rascas/ions_parameters/"

    write_params(params, os.path.join(dout, "params_rascas.cfg"))


def make_mock_params(
    dout,
    pos,
    dir_vects,
    flux={"do": False},
    spec={"do": False},
    img={"do": False},
    cube={"do": False},
):

    param_lines = ""

    for dir in dir_vects:
        param_lines += f"{dir[0]:f} {dir[1]:f} {dir[2]:f}\n"
        param_lines += f"{pos[0]:f} {pos[1]:f} {pos[2]:f}\n"
        if flux["do"]:
            param_lines += f"{flux['flux']}\n"
        else:
            param_lines += "0.0\n"
        if spec["do"]:
            param_lines += f"{spec['nspec']:d} {spec['rspec']:f} {spec['lamb_min']:f} {spec['lamb_max']:f}\n"
        else:
            param_lines += "0 0 0 0\n"
        if img["do"]:
            param_lines += f"{img['nimg']:d} {img['rimg']:f}\n"
        else:
            param_lines += "0 0\n"
        if cube["do"]:
            param_lines += f"{cube['nspec']:d} {cube['ncube']:d} {cube['lamb_min']:f} {cube['lamb_max']:f} {cube['rcube']:f}\n"
        else:
            param_lines += "0 0 0 0 0\n"
        # param_lines += "0 0 0 0 0\n"

    fout = os.path.join(dout, "params_mock.cfg")
    with open(fout, "w") as f:
        f.write(param_lines)


def copy_exec(d):

    # src = "./rascas"
    for src in ["rascas", "CreateDomDump", "PhotonsFromStars"]:

        dst = os.path.join(d, src)

        # copy(os.path.join("/home/jlewis/rascas/f90", src), dst)
        copy(os.path.join("/home/jlewis/rascas_joe_dust/f90", src), dst)


def create_sh(
    rascas_path,
    tmplt="./run_rascas.slurm",
    pbs_params=None,
    ramses_exec=None,
    params=None,
    ntasks=8,
    nomp=128,
    nnodes=1,
    wt=None,
    name="rascas",
):

    lines = []

    with open(tmplt, "r") as f:
        lines = f.readlines()

    cd_line = np.where(["cd" in l for l in lines])[0][0]

    lines[cd_line] = f"cd {rascas_path}\n"

    if pbs_params is not None:
        last_pbs_line = np.where(["#SBATCH" in l for l in lines])[0][-1]
        nb_added = 0
        for key, val in pbs_params.items():
            if key in lines:
                continue
            lines.insert(last_pbs_line + nb_added, f"#SBATCH -{key} {val}\n")
            nb_added += 1

    exec_line = np.where([".cfg" in l for l in lines])[0][0]

    cmd_words = lines[exec_line].split(" ")
    cfg_pos = np.where([".cfg" in w for w in cmd_words])[0][0]

    if params is not None:
        cmd_words[cfg_pos] = params

    if ramses_exec is not None:
        cmd_words[cfg_pos - 1] = "./" + ramses_exec

    if ntasks is not None:
        cmd_words[cfg_pos - 2] = str(ntasks)

    # print(cmd_words)

    lines[exec_line] = " ".join(cmd_words)

    if nnodes is not None or wt is not None:
        if wt == None:
            wt = "48:00:00"
        if nnodes == None:
            nnodes = 1

        node_line = np.where(["nodes=" in l for l in lines])[0][0]
        lines[node_line] = f"#SBATCH -l nodes={nnodes},walltime={wt}\n"

        omp_line = np.where(["OMP_NUM_THREADS" in l for l in lines])[0][0]
        lines[omp_line] = f"export OMP_NUM_THREADS={nomp}\n"

    if name is not None:
        name_line = np.where(["#SBATCH -N" in l for l in lines])[0][0]
        lines[name_line] = f"#SBATCH -N {name}\n"

    # insert CCD and PFS commands before exec_line
    cdd_line = "./CreateDomDump params_CDD.cfg > log_CDD\n"
    pfs_line = "./PhotonsFromStars params_PFS.cfg > log_PFS\n"

    lines.insert(exec_line, cdd_line)
    lines.insert(exec_line + 1, pfs_line)

    with open(os.path.join(rascas_path, "run.slurm"), "w") as f:
        f.writelines(lines)


def create_sh_multi(
    rascas_paths,
    tmplt="./run_rascas.slurm",
    pbs_params=None,
    ramses_exec=None,
    params=None,
    ntasks=16,
    nomp=128,
    nnodes=1,
    wt: str = None,
    name="rascas",
    run_cdds=None,
    run_pfss=None,
):

    if run_cdds == None:
        run_cdds = [True] * len(rascas_paths)
    if run_pfss == None:
        run_pfss = [True] * len(rascas_paths)

    lines = []

    rascas_path0 = rascas_paths[0]

    with open(tmplt, "r") as f:
        lines = f.readlines()

    cd_line = np.where(["cd" in l for l in lines])[0][0]

    lines[cd_line] = f"cd {rascas_path0}\n"

    if pbs_params is not None:
        last_pbs_line = np.where(["#SBATCH" in l for l in lines])[0][-1]
        nb_added = 0
        for key, val in pbs_params.items():
            if key in lines:
                continue
            lines.insert(last_pbs_line + nb_added, f"#SBATCH -{key} {val}\n")
            nb_added += 1

    exec_line = np.where([".cfg" in l for l in lines])[0][0]

    cmd_words = lines[exec_line].split(" ")
    cfg_pos = np.where([".cfg" in w for w in cmd_words])[0][0]

    if params is not None:
        cmd_words[cfg_pos] = params

    if ramses_exec is not None:
        cmd_words[cfg_pos - 1] = "./" + ramses_exec

    if ntasks is not None:
        cmd_words[cfg_pos - 2] = str(ntasks)

    # print(cmd_words)

    lines[exec_line] = " ".join(cmd_words)

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

        omp_line = np.where(["OMP_NUM_THREADS" in l for l in lines])[0][0]
        lines[omp_line] = f"export OMP_NUM_THREADS={nomp:d}\n"

        time_line = np.where(["#SBATCH --time" in l for l in lines])[0][0]
        lines[time_line] = f"#SBATCH --time={wt}\n"

    if name is not None:
        name_line = np.where(["#SBATCH --job-name" in l for l in lines])[0][0]
        lines[name_line] = f"#SBATCH --job-name {name}\n"

    # insert CCD and PFS commands before exec_line
    cdd_line = "./CreateDomDump params_CDD.cfg &> log_CDD\n"
    pfs_line = "./PhotonsFromStars params_PFS.cfg &> log_PFS\n"

    if run_cdds[0]:
        lines.insert(exec_line, cdd_line)
    if run_pfss[0]:
        lines.insert(exec_line + 1, pfs_line)

    for rascas_path, cdd_flag, pfs_flag in zip(
        rascas_paths[1:], run_cdds[1:], run_pfss[1:]
    ):

        lines.append(f"cd {rascas_path}\n")
        if cdd_flag:
            lines.append(cdd_line)
        if pfs_flag:
            lines.append(pfs_line)
        lines.append(
            f"mpiexec -np {ntasks:d} ./rascas params_rascas.cfg &> rascas.out\n"
        )

    with open(os.path.join(rascas_path0, "run.slurm"), "w") as f:
        f.writelines(lines)

    print(".slurm setup in ", rascas_path0)
