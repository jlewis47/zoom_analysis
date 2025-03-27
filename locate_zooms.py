from textwrap import fill
from gremlin.read_sim_params import ramses_sim

import numpy as np
import os
import argparse

from matplotlib.patches import Circle, Ellipse, Rectangle
import matplotlib.pyplot as plt

from zoom_analysis.zoom_helpers import decentre_coordinates, get_old_ctr

# from shutil import rmtree


if __name__ == "__main__":

    # get directory from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sim_dirs", nargs="+", help="Directory of the simulation(s) to map"
    )
    # get prune law from command line

    args = parser.parse_args()

    sim_dirs = args.sim_dirs

    is_sim_dir = False

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")

    # print(fig, ax)

    if len(sim_dirs) == 1:
        try:
            ramses_sim(sim_dirs[0])
            is_sim_dir = True
        except:
            pass

        if not is_sim_dir:

            sim_dirs = [d for d in os.listdir(sim_dirs[0]) if os.path.isdir(d)]
            # print(sim_dirs)
            # now check there is an output folder in each of the directories

    sim = None

    # print(sim_dirs)

    cmap = plt.get_cmap("tab20")
    obelisk_coords = (
        np.asarray([0.312611265, 0.519358132901, 0.558668440387]) * 100 / 0.704
    )
    nh_stuff = np.asarray([0.18776, 0.42237, 0.27435, 0.07]) * 100 / 0.704

    for isim, d in enumerate(sim_dirs):

        if not os.path.isdir(d):
            continue
        try:
            sim = ramses_sim(d, nml="cosmo.nml")
        except FileNotFoundError:
            print(f"Skipping {d}... there was no .nml")
            continue

        zoom_ctr = sim.zoom_ctr
        azoom, bzoom, czoom = (
            sim.namelist["refine_params"]["azoom"],
            sim.namelist["refine_params"]["bzoom"],
            sim.namelist["refine_params"]["czoom"],
        )
        if hasattr(sim, "aexp_stt"):  # false if no snaps yet !!!
            cMpc = sim.cosmo.lcMpc
        else:
            cMpc = 1.0

        if np.all(zoom_ctr == [0.5, 0.5, 0.5]):
            print("converting to from centered coordinates")
            zoom_ctr = get_old_ctr(sim.path)
            # zoom_ctr = decentre_coordinates(zoom_ctr, sim.path)
        # ax.scatter(zoom_ctr[0] * cMpc, zoom_ctr[1] * cMpc, s=1, label=sim.name)
        color = cmap(isim % 5)

        coord1s = [zoom_ctr[0], zoom_ctr[1], zoom_ctr[0]]
        coord2s = [zoom_ctr[1], zoom_ctr[2], zoom_ctr[2]]
        size1s = [azoom, bzoom, azoom]
        size2s = [bzoom, czoom, czoom]

        for iax in range(0, 3):

            # print(isim, iax)

            coord1 = coord1s[iax]
            coord2 = coord2s[iax]
            size1 = size1s[iax]
            size2 = size2s[iax]

            # print(isim, iax, coord1 * cMpc, coord2 * cMpc, size1 * cMpc, size2 * cMpc)

            if iax == 0:
                label = sim.name
            else:
                label = ""

            if "zoom_shape" in sim.namelist["refine_params"]:
                shape = sim.namelist["refine_params"]["zoom_shape"]

                if shape == "ellipsoid":

                    p = Ellipse(
                        (coord1 * cMpc, coord2 * cMpc),
                        size1 * 2 * cMpc,
                        size2 * 2 * cMpc,
                        label=label,
                        fill=False,
                        color=color,
                    )
                    print("ellip")

                elif shape == "rectangle":

                    p = Rectangle(
                        (
                            coord1 * cMpc - size1 * cMpc,
                            coord2 * cMpc - size2 * cMpc,
                        ),
                        size1 * 2 * cMpc,
                        size2 * 2 * cMpc,
                        label=label,
                        fill=False,
                        color=color,
                    )
                    print("rect")

            else:

                p = Circle(
                    (coord1 * cMpc, coord2 * cMpc),
                    size1 * cMpc,
                    label=label,
                    fill=False,
                    color=color,
                )
                print("circ")

            # print(iax, coord1, coord2, p)

            ax[iax].add_patch(p)

            if isim == len(sim_dirs) - 1:
                if iax == 0:
                    label = "HAGN"
                else:
                    label = ""
                # print(isim, iax, "lims")
                ax[iax].set_xlim(-0.05, cMpc + 0.05)
                ax[iax].set_ylim(-0.05, cMpc + 0.05)
                box = Rectangle(
                    (0, 0), cMpc, cMpc, fill=False, color="k", ls="--", label=label
                )
                # print(isim, iax, "box")
                ax[iax].add_patch(box)
                # print(iax, ax[iax])
                ax[iax].set_aspect("equal")

    if sim != None:
        # if hasattr(sim, "aexp_stt"):
        ax[0].scatter(obelisk_coords[0], obelisk_coords[1], c="k", label="Obelisk")
        ax[1].scatter(obelisk_coords[1], obelisk_coords[2], c="k")
        ax[2].scatter(obelisk_coords[0], obelisk_coords[2], c="k")
        #

        # ax[0].scatter(nh_stuff[0], nh_stuff[1], c="r", label="NH")
        # ax[1].scatter(nh_stuff[1], nh_stuff[2], c="r")
        # ax[2].scatter(nh_stuff[0], nh_stuff[2], c="r")

        circ1 = Circle(
            (nh_stuff[0], nh_stuff[1]),
            nh_stuff[3],
            label="NH",
            fill=False,
            color="r",
        )
        ax[0].add_patch(circ1)

        circ2 = Circle(
            (nh_stuff[1], nh_stuff[2]),
            nh_stuff[3],
            fill=False,
            color="r",
        )
        ax[1].add_patch(circ2)

        circ3 = Circle(
            (nh_stuff[0], nh_stuff[2]),
            nh_stuff[3],
            fill=False,
            color="r",
        )
        ax[2].add_patch(circ3)

        ax[0].legend(ncols=2, framealpha=0.0, prop={"size": 9})

    # make it pretty by plotting HAGN DM density info?

    fig.savefig("locate_zooms.png")
