import argparse
import os
import numpy as np

from gremlin.read_sim_params import ramses_sim

from zoom_analysis.halo_maker.assoc_fcts import find_snaps_with_gals, find_snaps_with_halos

from compress_zoom.tar_sim import run_tar
from zoom_analysis.halo_maker.setup_halo_maker import setup_halo_maker


def untar_and_make_halos(sim_dirs=[], overwrite=False, launch=False):

    if type(sim_dirs) == str:
        sim_dirs = [sim_dirs]

    if len(sim_dirs[0])==1:
        sim_dirs="".join(sim_dirs)


    for sim_dir in sim_dirs:

        # existing_tar_files = [
        #     f for f in os.listdir(sim_dir) if f.startswith("output") and f.endswith(".tar")
        # ]
        # existing_tar_numbers = np.asarray(
        #     [int(f.split("_")[1].split(".tar")[0]) for f in existing_tar_files]
        # )

        # existing_infos = [nb for nb in existing_tar_numbers if os.path.exists(os.path.join(sim_dir,f"output_{nb:05d}",f"info_{nb:05d}.txt"))]

        # untar_txt_nbs = np.setdiff1d(existing_tar_numbers,existing_infos)

        # for untar_txt_nb in untar_txt_nbs:

        #     tar_path = os.path.join(sim_dir, f"output_{untar_txt_nb:05d}.tar")
        #     os.system(f'tar xvf {tar_path} *.txt')

        # mv_targets=[]

        # for root, dirs, files in os.walk(sim_dir, topdown=False):
        #     for name in dirs:
        #         if name.startswith ('output'):
        #          mv_targets.append(os.path.join(root, name))

        # for tgt in mv_targets:
        #     os.system(f"mv {os.path.join(sim_dir,tgt):s} {sim_dir:s}/.")

        sim = ramses_sim(sim_dir)

        print(f"Working on {sim.name:s}")

        tar_snaps, tar_snap_numbers = sim.get_snaps(full_snaps=False, mini_snaps=False, tar_snaps=True)        
        full_snaps, full_snap_numbers = sim.get_snaps(full_snaps=True, mini_snaps=False, tar_snaps=False)        

        print(tar_snap_numbers)

        FullAndTar_snap_numbers  =np.union1d(tar_snap_numbers, full_snap_numbers)

        print(FullAndTar_snap_numbers)

        compressd_files = []
        if os.path.exists(os.path.join(sim.path, "halogal_data")):
            compressd_files = os.listdir(os.path.join(sim.path, "halogal_data"))

        compressd_numbers = np.asarray(
            [int(f.split("_")[-1].split(".")[0]) for f in compressd_files]
        )

        # no_gal_snaps = np.zeros(len(full_snap_numbers), dtype=bool)
        snaps_w_halos = find_snaps_with_halos(sim.snap_numbers, sim.path)
        snaps_w_gals = find_snaps_with_gals(sim.snap_numbers, sim.path)

        print(snaps_w_gals)

        no_gal_snaps = np.setdiff1d(FullAndTar_snap_numbers,snaps_w_gals)
        no_halo_snaps = np.setdiff1d(FullAndTar_snap_numbers,snaps_w_halos)
        
        print(no_gal_snaps)

        if len(snaps_w_gals) == 0:
            first_gal_snaps = []
            first_halo_snaps = []
        else:
            first_gal_snaps = sim.snap_numbers[sim.snap_numbers < np.min(snaps_w_gals)]
            first_halo_snaps = sim.snap_numbers[sim.snap_numbers < np.min(snaps_w_halos)]

        no_gal_snaps = np.setdiff1d(no_gal_snaps,first_gal_snaps)
        no_halo_snaps = np.setdiff1d(no_halo_snaps,first_halo_snaps)

        backd_up_full_nbs = full_snap_numbers[np.in1d(full_snap_numbers, compressd_numbers)]

        existing_tar_files = [
            f for f in os.listdir(sim.path) if f.startswith("output") and f.endswith(".tar")
        ]
        existing_tar_numbers = np.asarray(
            [int(f.split("_")[1].split(".tar")[0]) for f in existing_tar_files]
        )

        target_snaps = np.intersect1d(existing_tar_numbers,np.unique([no_halo_snaps,no_gal_snaps]))

        batch_file_template = ""     

        print(target_snaps, existing_tar_files, no_gal_snaps, no_halo_snaps)   

        for isnap,target_snap in enumerate(target_snaps):

            batch_files = setup_halo_maker([sim_dir], launch=False, snap=target_snap)

            

            break


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()

    argparser.add_argument("sim_dirs",
                        nargs="+",
                        help='paths to sims')

    argparser.add_argument('--overwrite',
                        action='store_true')
    
    argparser.add_argument('--launch',
                        action='store_true')

    args =vars(argparser.parse_args())

    print(args)

    untar_and_make_halos(**args)