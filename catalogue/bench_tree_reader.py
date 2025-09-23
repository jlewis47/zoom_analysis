from gremlin.read_sim_params import ramses_sim
from zoom_analysis.halo_maker.assoc_fcts import get_halo_props_snap
from zoom_analysis.trees.tree_reader import read_tree_file_rev_correct_pos

import os
import numpy as np

def run():

    simdir="/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id112288"

    sim = ramses_sim(simdir)


    fpure_thresh = 0.9999

    ztgt=4.0

    snap = sim.get_closest_snap(zed=ztgt)
    snap_aexp = sim.get_snap_exps(snap)[0]

    halos = get_halo_props_snap(simdir, snap)

    hids = halos['hid'][halos["fpure"]>fpure_thresh]
    masses = halos['mvir'][halos["fpure"]>fpure_thresh]

    hid = hids[np.argmax(masses)]

    print(snap,hid)

    sim_tree_hids, sim_tree_datas, sim_tree_aexps = read_tree_file_rev_correct_pos(
        os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat"),
        sim,
        snap,
        os.path.join(sim.path, "TreeMakerDM_dust"),
        zstart=1./snap_aexp-1,
        tgt_ids=[hid],
        star=False,
        fpure_min=fpure_thresh,
        verbose=False
    )

    # print(sim_tree_hids)

    return sim_tree_hids

if __name__ == "main":

    run()