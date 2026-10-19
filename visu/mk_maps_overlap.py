import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

from scipy.spatial import KDTree
from scipy.stats import binned_statistic_2d

from gremlin.read_sim_params import ramses_sim

from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.read_treebricks import *

hm = "HaloMaker_stars2_dp_rec_dust/"

sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704"
sim = ramses_sim(sim_dir)

box_size = sim.cosmo["unit_l"] / (ramses_pc * 1e6) / sim.aexp_stt  # cMpc

zoom_ctr = np.asarray(
    [
        sim.namelist["refine_params"]["xzoom"],
        sim.namelist["refine_params"]["yzoom"],
        sim.namelist["refine_params"]["zzoom"],
    ]
)

rzoom = sim.namelist["refine_params"]["rzoom"]

zoom_size = rzoom * box_size

img_size = 0.05 * zoom_size

res = 0.001  # kpc


slice_width_cMpc = 1.00
slice_width_box = slice_width_cMpc / box_size

# img = np.zeros((ldx, ldx), dtype=np.float32)

# snap_tgt = -1
snap_tgt = 177

if snap_tgt == -1:
    snap_tgt = sim.snap_numbers[-1]


brick_file = read_brickfile(os.path.join(sim.path, hm, f"tree_bricks{snap_tgt:03d}"))
convert_brick_units(brick_file, sim)
# get massive central galaxy
ms = brick_file["hosting info"]["hmass"]
xgal = brick_file["positions"]["x"]
ygal = brick_file["positions"]["y"]
zgal = brick_file["positions"]["z"]
pos_gal = np.array([xgal, ygal, zgal])

tree = KDTree(pos_gal.T, boxsize=1.0 + 1e-6)

dists, args = tree.query(
    zoom_ctr, k=len(ms), distance_upper_bound=0.33 * zoom_size / box_size
)

args = args[np.isfinite(dists)]
dists = dists[np.isfinite(dists)]

dist_score = np.argsort(dists)
mass_score = np.argsort(ms[args])

# score = dist_score + mass_score**2  # mass is more important - hand wavy scaling
# score = mass_score

# central_arg = args[np.argmin(score)]
central_arg = args[mass_score[-1]]

central_mstel = ms[central_arg]
central_pos = pos_gal[:, central_arg]
print("%e" % central_mstel, central_pos)

# central_pos = zoom_ctr


zfilt = np.abs(pos_gal[2, :] - central_pos[2]) < slice_width_box

ids = brick_file["hosting info"]["hid"][zfilt]

fig, ax = plt.subplots(1, 1)

for gal_id in ids:
    stars = get_gal_stars(snap_tgt, gal_id, sim, fields=["mpart", "pos"])

    dx = stars["pos"][:, 0].max() - stars["pos"][:, 0].min()
    ldx = int(dx / (res / box_size))

    print(dx, ldx)

    xbins = np.linspace(
        stars["pos"][:, 0].min(),
        stars["pos"][:, 0].max(),
        ldx + 1,
    )
    ybins = np.linspace(
        stars["pos"][:, 1].min(),
        stars["pos"][:, 1].max(),
        ldx + 1,
    )

    img = binned_statistic_2d(
        stars["pos"][:, 0],
        stars["pos"][:, 1],
        stars["mpart"],
        "sum",
        bins=[xbins, ybins],
    )[0]

    ax.imshow(
        img,
        extent=np.asarray([xbins[0], xbins[-1], ybins[0], ybins[-1]]) * box_size,
        origin="lower",
        cmap="magma",
        aspect="auto",
        norm=LogNorm(),
    )


# ax.scatter(zoom_ctr[0] * box_size, zoom_ctr[1] * box_size, c="r", s=100, marker="x")
# circle for zoom_region

circle = plt.Circle(
    (zoom_ctr[0] * box_size, zoom_ctr[1] * box_size),
    rzoom * box_size,
    color="r",
    fill=False,
    lw=3,
)
ax.add_artist(circle)

# bad background color = black
ax.set_facecolor("black")

fig.savefig("st_map.png")
