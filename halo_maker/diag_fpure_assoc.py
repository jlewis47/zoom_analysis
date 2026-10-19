import matplotlib.pyplot as plt
import numpy as np

from zoom_analysis.halo_maker.assoc_fcts import get_halo_props_snap, get_gal_props_snap

from gremlin.read_sim_params import ramses_sim

sim_dir_root = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel"

snap = 109

color_bins = np.linspace(0, 1, 10)
color_scale = plt.cm.viridis(color_bins)
mass_bins = np.logspace(8, 12, 8)


hprops = get_halo_props_snap(sim_dir_root, snap)
gprops = get_gal_props_snap(sim_dir_root, snap)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)


hpure = hprops["fpure"]
hpos = np.zeros((len(hpure), 3), dtype=np.float64)
rvir = np.zeros(len(hpure), dtype=np.float64)
mvir = hprops["mvir"]

hids = hprops["hid"]

for id, hid in enumerate(hids):
    hkey = f"halo_{hid:07d}"
    hpos[id, :] = hprops[hkey]["pos"]
    rvir[id] = hprops[hkey]["rvir"]


hcolors = color_scale[
    np.min(
        [
            np.digitize(hpure, color_bins),
            np.full(len(hpure), len(color_bins) - 1, dtype=int),
        ],
        axis=0,
    )
]

gpos = gprops["pos"].T
gpure = gprops["host purity"]
gmass = gprops["mass"]
gcolors = color_scale[
    np.min(
        [
            np.digitize(gpure, color_bins),
            np.full(len(gpure), len(color_bins) - 1, dtype=int),
        ],
        axis=0,
    )
]
gsize = np.digitize(gmass, mass_bins) * 5

massive_gals = gmass > 1e8
massive_halos = mvir > 1e11

ax.scatter(
    gpos[massive_gals, 0],
    gpos[massive_gals, 1],
    c=gcolors[massive_gals, :],
    s=gsize[massive_gals],
    alpha=0.5,
)

massive_args = np.where(massive_halos)[0]

for ih in range(massive_halos.sum()):
    massive_arg = massive_args[ih]
    circle = plt.Circle(
        hpos[massive_arg, :2], rvir[massive_arg], color=hcolors[massive_arg], fill=False
    )
    ax.add_artist(circle)

sim = ramses_sim(sim_dir_root, nml="cosmo.nml")

ax.set_aspect("equal")


fig.savefig("test_fpure.png")


ghids = gprops["host hid"]
_, arg_h, arg_g = np.intersect1d(hids, ghids, return_indices=True)

print(list(zip(hpure[arg_h], gpure[arg_g])))

print(hids[arg_h] - ghids[arg_g])
