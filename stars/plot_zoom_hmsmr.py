import numpy as np
import matplotlib.pyplot as plt

# from zoom_analysis.halo_maker.read_treebricks import (
#     read_zoom_brick,
#     read_zoom_stars,
# )

from zoom_analysis.halo_maker.assoc_fcts import find_zoom_tgt_halo

from gremlin.read_sim_params import ramses_sim

sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099"


fig, ax = plt.subplots(1, 1, figsize=(8, 8))


sim = ramses_sim(sim_dir, nml="cosmo.nml")

snaps = sim.snap_numbers
aexps = sim.aexps

# aexps to colors
colors = plt.cm.viridis(np.linspace(0, 1, len(aexps)))


def size_fct(mass):

    max_mass = 1e12
    min_mass = 1e9
    max_size = 50
    min_size = 1

    return (mass - min_mass) / (max_mass - min_mass) * (max_size - min_size) + min_size


for snap, aexp, c in zip(snaps, aexps, colors):

    try:
        hid, halo_dict, gals = find_zoom_tgt_halo(sim, snap, debug=False)
    except FileNotFoundError:
        continue

    mgals = gals["mass"]

    # print(list(halo_dict.keys()))

    mhalo = halo_dict["mass"]

    # print(hid, mgals / mhalo)

    print(1.0 / aexp - 1)

    ax.scatter(
        [mhalo] * len(mgals), mgals, c=c, s=size_fct(mhalo)
    )  # , label=f"z={1.0/aexp-1:.2f}")

# colorbar from aexps
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r)
sm.set_array(1.0 / aexps[::-1] - 1.0)

# create cb axis on top of main axis
cax = fig.add_axes([0.2, 0.95, 0.6, 0.03])
cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
cbar.set_label("Redshift")

ax.set_yscale("log")
ax.set_xscale("log")

ax.grid()

ax.set_xlabel("Halo Mass")
ax.set_ylabel("Galaxy Mass")


fig.savefig(f"zoom_{sim.name}_hmsmr.png")
