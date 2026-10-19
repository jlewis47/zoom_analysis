from sympy import root
from gremlin.read_sim_params import ramses_sim

import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def walk_down_dirs(root, saved_paths):

    dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

    # print(root)

    for d in dirs:

        full_d = os.path.join(root, d)
        # print(full_d)

        if d.startswith("id"):
            if not os.path.islink(full_d):
                saved_paths.append(full_d)
        else:
            saved_paths = walk_down_dirs(full_d, saved_paths)

    # print(saved_paths)

    return saved_paths


def locate_all_sims():

    sim_paths = []

    root_dirs = [
        "/data101/jlewis/sims/dust_fid/",
        "/data102/jlewis/sims/",
        "/data103/jlewis/sims/",
    ]

    for root_dir in root_dirs:
        # print(root_dir)
        sim_paths = walk_down_dirs(root_dir, sim_paths)

    sim_paths = np.unique(sim_paths)

    return sim_paths


all_sim_paths = locate_all_sims()

# for every sim, work out :
#  - number of total snapshots
#  - number of snapshots with associated galaxies
#  - are trees up-to-date
#  - number of compressed snapshots
#  - number snapshots that are tarred and compressed
#  - number of snapshots that have catalogs

hmaker = "HaloMaker_DM_dust"
gmaker = "HaloMaker_stars2_dp_rec_dust"

htree = "TreeMakerDM_dust"
gtree = "TreeMakerstars2_dp_rec_dust"
# Initialize arrays to store data for each simulation
lvlmax = []
droot = []
name = []
total_snaps = []
full_snaps = []
mini_snaps = []
tarred_snaps_count = []
last_snap_has_gal = []
last_snap_has_halo = []
galaxy_snaps_count = []
halo_snaps_count = []
associated_snaps_count = []
compressed_snaps_count = []
catalog_snaps_count = []
tree_ok_gal = []
tree_ok_halo = []
zmax = []

for d_sim in all_sim_paths:

    try:
        sim = ramses_sim(d_sim, nml="cosmo.nml")
    except:
        all_sim_paths = np.delete(all_sim_paths, np.where(all_sim_paths == d_sim))
        continue

    nbsnaps_all = sim.get_snaps(full_snaps=False, mini_snaps=False)[1]

    if len(nbsnaps_all) == 0:
        all_sim_paths = np.delete(all_sim_paths, np.where(all_sim_paths == d_sim))
        continue

    nbsnaps_full = sim.get_snaps(full_snaps=True, mini_snaps=False)[1]
    nbsnaps_mini = sim.get_snaps(full_snaps=False, mini_snaps=True)[1]

    print(d_sim)
    print(f"Working on {sim.name}")

    try:
        print(f"lvlmin:{sim.levelmin:d}, lvlmax:{sim.levelmax:d}")
    except:
        all_sim_paths = np.delete(all_sim_paths, np.where(all_sim_paths == d_sim))
        continue

    compressed_snaps = np.zeros(len(nbsnaps_all), dtype=bool)
    tarred_snaps = np.zeros(len(nbsnaps_all), dtype=bool)
    catalog_snaps = np.zeros(len(nbsnaps_all), dtype=bool)
    assoc_snaps = np.zeros(len(nbsnaps_all), dtype=bool)
    gal_snaps = np.zeros(len(nbsnaps_all), dtype=bool)
    halo_snaps = np.zeros(len(nbsnaps_all), dtype=bool)
    tree_OK_gal = False
    tree_OK_halo = False

    for isnap, snap in enumerate(nbsnaps_all):

        assoc_snaps[isnap] = os.path.exists(
            os.path.join(sim.path, "association", f"assoc_{snap:03d}_halo_lookup.h5")
        )

        compressed_snaps[isnap] = os.path.exists(
            os.path.join(sim.path, "halogal_data", f"halo_data_{snap:d}.h5")
        )

        tarred_snaps[isnap] = os.path.exists(
            os.path.join(sim.path, f"output_{snap:05d}.tar")
        )

        catalog_snaps[isnap] = os.path.exists(
            os.path.join(sim.path, "catalogues", f"galaxy_2p0Xr50_{snap:03d}.hdf5")
        )

        gal_snaps[isnap] = os.path.exists(
            os.path.join(sim.path, gmaker, f"tree_bricks{snap:03d}")
        )
        halo_snaps[isnap] = os.path.exists(
            os.path.join(sim.path, hmaker, f"tree_bricks{snap:03d}")
        )


    if np.any(gal_snaps):
        last_snap_has_gal.append(nbsnaps_all[gal_snaps].max()==nbsnaps_all.max())
    else:
        last_snap_has_gal.append(False)
    if np.any(halo_snaps):
        last_snap_has_halo.append(nbsnaps_all[halo_snaps].max()==nbsnaps_all.max())
    else:
        last_snap_has_halo.append(False)

    tree_OK_gal = os.path.isdir(os.path.join(sim.path, gtree))
    if tree_OK_gal:
        gal_ntree_steps = len(
            [
                f
                for f in os.listdir(os.path.join(sim.path, gtree))
                if f.startswith("bytes_step_")
            ]
        )
        tree_OK_gal = np.sum(gal_snaps) == gal_ntree_steps

    tree_OK_halo = os.path.isdir(os.path.join(sim.path, htree))
    if tree_OK_halo:
        halo_ntree_steps = len(
            [
                f
                for f in os.listdir(os.path.join(sim.path, htree))
                if f.startswith("bytes_step_")
            ]
        )
        tree_OK_halo = np.sum(halo_snaps) == halo_ntree_steps

    # print(f"Total snaps: {len(nbsnaps_all)}")
    # print(f"Full snaps: {len(nbsnaps_full)}")
    # print(f"Mini snaps: {len(nbsnaps_mini)}")
    # print(f"Tarred snaps: {np.sum(tarred_snaps)}")
    # print(f"Galaxy snaps: {np.sum(gal_snaps)}")
    # print(f"Halo snaps: {np.sum(halo_snaps)}")
    # print(f"Associated snaps: {np.sum(assoc_snaps)}")
    # print(f"Compressed snaps: {np.sum(compressed_snaps)}")
    # print(f"Catalog snaps: {np.sum(catalog_snaps)}")
    # print(f"Tree OK for galaxies: {tree_OK_gal}")
    # print(f"Tree OK for halos: {tree_OK_halo}")

    total_snaps.append(len(nbsnaps_all))
    full_snaps.append(len(nbsnaps_full))
    mini_snaps.append(len(nbsnaps_mini))
    tarred_snaps_count.append(np.sum(tarred_snaps))
    galaxy_snaps_count.append(np.sum(gal_snaps))
    halo_snaps_count.append(np.sum(halo_snaps))
    associated_snaps_count.append(np.sum(assoc_snaps))
    compressed_snaps_count.append(np.sum(compressed_snaps))
    catalog_snaps_count.append(np.sum(catalog_snaps))
    tree_ok_gal.append(tree_OK_gal)
    tree_ok_halo.append(tree_OK_halo)

    droot.append(d_sim.split("/")[1])
    name.append(sim.name)
    lvlmax.append(sim.levelmax)
    zmax.append(np.round(1.0 / sim.get_snap_exps(nbsnaps_all[-1])[0] - 1, decimals=2))


# make a nice csv where the rows are different simulations and the columns are the different counts
# save this csv in the same directory as the script
import pandas as pd

data = {
    "Root on infinity": droot,
    "Simulation": name,
    "max level": lvlmax,
    "Last redshift": zmax,
    "Total snaps": total_snaps,
    "Full snaps": full_snaps,
    "Mini snaps": mini_snaps,
    "Tarred snaps": tarred_snaps_count,
    "Galaxy snaps": galaxy_snaps_count,
    "Last snap has galaxy": last_snap_has_gal,
    "Halo snaps": halo_snaps_count,
    "Last snap has halo": last_snap_has_halo,
    "Associated snaps": associated_snaps_count,
    "Compressed snaps": compressed_snaps_count,
    "Catalog snaps": catalog_snaps_count,
    "Tree OK for galaxies": tree_ok_gal,
    "Tree OK for halos": tree_ok_halo,
}

df = pd.DataFrame(data)

df.to_csv("sim_snap_counts.csv", index=False)
