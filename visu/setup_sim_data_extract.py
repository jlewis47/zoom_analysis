from genericpath import exists
import os
from shlex import join
import numpy as np
import subprocess
import shutil

from gremlin.read_sim_params import ramses_sim


def setup_amr2cell(sim_dir, outputs=None, deltaT=100):
    sim_params = ramses_sim(sim_dir)

    assert (
        outputs == None or type(outputs) == list
    ), "tgt_outputs must be None or a list of output numbers"

    if outputs == None:
        outputs = sim_params.snap_numbers
    else:
        outputs = np.asarray(outputs)[np.in1d(outputs, sim_params.snap_numbers)]

    times = sim_params.get_snap_times(outputs)

    umults, idxs = np.unique(times // deltaT, return_index=True)

    outputs = np.asarray(outputs)[idxs].tolist()

    for output in outputs:
        output_str = f"output_{output:05d}"

        output_dir = os.path.join(sim_dir, "amr2cell", output_str)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            if os.path.exists(os.path.join(output_dir, "out_amr2cell")):continue

        # copy amr2cell binary to dir
        shutil.copy2("/home/jlewis/ramses/utils/f90/amr2cell", output_dir)

        # input dir
        input_dir = os.path.join(sim_dir, output_str)

        output_file = os.path.join(output_dir, "out_amr2cell")

        # make bash script to run amr2cell
        with open(os.path.join(output_dir, "run_amr2cell"), "w") as f:
            f.write(f"#!/bin/bash\n")
            f.write(f"./amr2cell -inp {input_dir} -out {output_file}")
        os.chmod(os.path.join(output_dir, "run_amr2cell"), 0o0755)

    return outputs


def setup_getstarlist(sim_dir, outputs=None, deltaT=100):
    sim_params = ramses_sim(sim_dir)

    assert (
        outputs == None or type(outputs) == list
    ), "tgt_outputs must be None or a list of output numbers"

    if outputs == None:
        outputs = sim_params.snap_numbers
    else:
        outputs = np.asarray(outputs)[
            np.in1d(outputs, sim_params.snap_numbers)
        ].tolist()

    times = sim_params.get_snap_times(outputs)

    umults, idxs = np.unique(times // deltaT, return_index=True)

    outputs = np.asarray(outputs)[idxs].tolist()

    for output in outputs:
        output_str = f"output_{output:05d}"

        output_dir = os.path.join(sim_dir, "getstarlist", output_str)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            if os.path.exists(os.path.join(output_dir, "out_getstarlist")):continue

        # copy amr2cell binary to dir
        shutil.copy2("/home/jlewis/ramses/utils/f90/getstarlist", output_dir)

        # input dir
        input_dir = os.path.join(sim_dir, output_str)

        output_file = os.path.join(output_dir, "out_getstarlist")

        # make bash script to run getstarlist
        with open(os.path.join(output_dir, "run_getstarlist"), "w") as f:
            f.write(f"#!/bin/bash\n")
            f.write(f"./getstarlist -inp {input_dir} -out {output_file}")
        os.chmod(os.path.join(output_dir, "run_getstarlist"), 0o0755)

    return outputs


# gen bash scripts to run all the files
def run_all_amr2cubes(sim_dir, outputs):
    str_outputs = ["%d"%out for out in outputs]
    cmd = f"for isnap in {" ".join(str_outputs)}; do cd 'output_'$(printf '%05d' $isnap); ./run_amr2cell; cd ..; done"
    script=os.path.join(sim_dir,'amr2cell','run_amr2cells.sh')
    with open(script, 'w') as f:
        f.write(cmd)

    #chmod 755 run_amr2cells.sh
    os.chmod(script, 0o0755)
    return cmd


def run_all_getstarlist(sim_dir, outputs):
    str_outputs = ["%d"%out for out in outputs]
    cmd = f"for isnap in {" ".join(str_outputs)}; do cd 'output_'$(printf '%05d' $isnap); ./run_getstarlist; cd ..; done"
    script=os.path.join(sim_dir,'getstarlist','run_getstarlists.sh')
    with open(script, 'w') as f:
        f.write(cmd)

    #chmod 755 run_getstarlists.sh
    os.chmod(script, 0o0755)
    
    return cmd


# sim_dir = "/automnt/data101/jlewis/sims/proposal_zooms/ExtractZoom/mh1e12/id2757_995pcnt_mstel"
# sim_dir = "/automnt/data101/jlewis/sims/proposal_zooms/ExtractZoom/mh1e12/id13600_005pcnt_mstel"
# sim_dir = "/automnt/data101/jlewis/sims/proposal_zooms/ExtractZoom/mh1e12/500pcnt_mstel"
# outputs=[0, 374]

# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704"
sim_dir="/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479_highnstar_nbh/"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479"

sim = ramses_sim(sim_dir)

deltaT = 10
zmax = 10.


outputs = sim.snap_numbers
aexps = sim.get_snap_exps()
zeds = 1./aexps - 1.

times = sim.get_snap_times(outputs)

umults, idxs = np.unique(times // deltaT, return_index=True)

outputs = np.asarray(outputs)[idxs]
outputs = outputs[zeds[idxs] < zmax].tolist()

outputs = setup_amr2cell(sim_dir, deltaT=deltaT, outputs=outputs)
outputs = setup_getstarlist(sim_dir, deltaT=deltaT, outputs=outputs)

amr2cube_cmd = run_all_amr2cubes(sim_dir, outputs)
getstarlist_cmd = run_all_getstarlist(sim_dir, outputs)



print(amr2cube_cmd)
print(getstarlist_cmd)
