from math import fabs
from types import DynamicClassAttribute
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
import os

from scipy.spatial import KDTree
from scipy.stats import binned_statistic_2d

# from distutils import fancy_getopt

from gremlin.read_sim_params import ramses_sim
from zoom_analysis.constants import ramses_pc

from zoom_analysis.halo_maker.read_treebricks import *

from hagn.tree_reader import read_tree_rev
from hagn.utils import get_hagn_sim


hm = "HaloMaker_stars2_dp_rec_dust/"

sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704"
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479"
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

# img_size = 0.2 * zoom_size
img_size = 0.33 * zoom_size
# ldx = 2048
# ldx = 4096
# ldx = 8192 * 2

res = 5e-4  # Mpc


slice_width_cMpc = 0.5
slice_width_box = slice_width_cMpc / box_size

# img = np.zeros((ldx, ldx), dtype=np.float32)

# snap_tgt = -1
# snap_tgt = 177
zed_tgt = 5.5
# zed_tgt = 5.0
# zed_tgt = 4.5
# zed_tgt = 4.0

brick_path = os.path.join(sim.path, hm)
brick_snaps = np.asarray([f[-3:] for f in os.listdir(brick_path)])

_, sim_args, brick_args = np.intersect1d(
    sim.snap_numbers, brick_snaps, return_indices=True
)  # make sure the snapshot we pick has a brick file

sim_zeds = 1.0 / sim.get_snap_exps(sim.snap_numbers[sim_args]) - 1.0
snap_tgt = sim.snap_numbers[sim_args][np.argmin(np.abs(sim_zeds - zed_tgt))]
aexp_tgt = sim.get_snap_exps(snap_tgt)


if snap_tgt == -1:
    snap_tgt = sim.snap_numbers[-1]

hagn_sim = get_hagn_sim()
hagn_tree_hids, hagn_tree_datas, hagn_tree_aexps = read_tree_rev(
    2.0,
    [int(sim_dir.split("/")[-1][2:])],
    target_fields=["m", "x", "y", "z", "r"],
    tree_type="halo",
)

tree_arg = np.argmin(np.abs(aexp_tgt - hagn_tree_aexps))

hagn_ctr = np.asarray(
    [
        hagn_tree_datas["x"][0][tree_arg],
        hagn_tree_datas["y"][0][tree_arg],
        hagn_tree_datas["z"][0][tree_arg],
    ]
)

hagn_ctr += 0.5 * (hagn_sim.cosmo.lcMpc * hagn_tree_aexps[tree_arg])
hagn_ctr /= hagn_sim.cosmo.lcMpc * hagn_tree_aexps[tree_arg]

hagn_r = hagn_tree_datas["r"][0][tree_arg]
hagn_r /= hagn_sim.cosmo.lcMpc * hagn_tree_aexps[tree_arg]


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
    hagn_ctr,
    k=len(ms),
    distance_upper_bound=img_size / box_size,
    # zoom_ctr, k=len(ms), distance_upper_bound=0.33 * zoom_size / box_size
)

args_mass = tree.query_ball_point(hagn_ctr, r=hagn_r)

print(
    f"z={1./aexp_tgt[0]-1.}: Found {ms[args_mass].sum():.1e} solar mass in tree bricks"
)

args = args[np.isfinite(dists)]
dists = dists[np.isfinite(dists)]

dist_score = np.argsort(dists)
mass_score = np.argsort(ms[args])

# score = dist_score + mass_score**2  # mass is more important - hand wavy scaling
# score = mass_score

# central_arg = args[np.argmin(score)]
# central_arg = args[mass_score[-1]]

# central_mstel = ms[central_arg]
# central_pos = pos_gal[:, central_arg]
# print("%e" % central_mstel, central_pos)

# central_pos = zoom_ctr


zfilt = np.abs(pos_gal[2, args] - hagn_ctr[2]) < slice_width_box

ids = brick_file["hosting info"]["hid"][args][zfilt]


fig, ax = plt.subplots(1, 1, dpi=300, figsize=(10, 10))

for gal_id in ids:
    # print(f"mapping stars from galaxy {gal_id:d}...")
    stars = get_gal_stars(snap_tgt, gal_id, sim, fields=["mpart", "pos"])

    x = stars["pos"][:, 0]
    y = stars["pos"][:, 1]

    # find smallest box contaning all data galaxy
    maxx = x.max()
    minx = x.min()
    maxy = y.max()
    miny = y.min()

    img_size_x = int((maxx - minx) * box_size / res)
    img_size_y = int((maxy - miny) * box_size / res)

    xbins = np.linspace(
        minx,
        maxx,
        img_size_x + 1,
    )
    ybins = np.linspace(
        miny,
        maxy,
        img_size_y + 1,
    )

    dx = xbins[1] - xbins[0]

    x0 = np.digitize(minx, xbins) - 1
    x1 = np.digitize(maxx, xbins) + 1
    y0 = np.digitize(miny, ybins) - 1
    y1 = np.digitize(maxy, ybins) + 1

    dx = np.abs(x - (np.int32(x) + 0.5))
    dy = np.abs(y - (np.int32(y) + 0.5))

    # print((maxx - minx) * box_size * 1e3, img_size_x)
    # print((maxy - miny) * box_size * 1e3, img_size_y)

    img = np.zeros((img_size_x, img_size_y), dtype=np.float32)

    if np.any([x0 < 0, y0 < 0]):
        # print("skipping galaxy")
        continue

    img += binned_statistic_2d(
        x,
        y,
        stars["mpart"] * (1 - dx) * (1 - dy),
        "sum",
        bins=[xbins, ybins],
    )[0]

    dx = np.abs(x - (np.int32(x) + 0.5 + 1))
    dy = np.abs(y - (np.int32(y) + 0.5))

    img += binned_statistic_2d(
        x + 1.0,
        y,
        stars["mpart"] * (dx) * (1 - dy),
        "sum",
        bins=[xbins, ybins],
    )[0]

    dx = np.abs(x - (np.int32(x) + 0.5 - 1))
    dy = np.abs(y - (np.int32(y) + 0.5))

    img += binned_statistic_2d(
        x - 1.0,
        y,
        stars["mpart"] * (dx) * (1 - dy),
        "sum",
        bins=[xbins, ybins],
    )[0]

    dx = np.abs(x - (np.int32(x) + 0.5))
    dy = np.abs(y - (np.int32(y) + 0.5 + 1))

    img += binned_statistic_2d(
        x + 1.0,
        y,
        stars["mpart"] * (1 - dx) * (dy),
        "sum",
        bins=[xbins, ybins],
    )[0]

    dx = np.abs(x - (np.int32(x) + 0.5))
    dy = np.abs(y - (np.int32(y) + 0.5 - 1))

    img += binned_statistic_2d(
        x - 1.0,
        y,
        stars["mpart"] * (1 - dx) * (dy),
        "sum",
        bins=[xbins, ybins],
    )[0]

    # ax.imshow(
    #     img.T,
    #     extent=np.asarray([xbins[0], xbins[-1], ybins[0], ybins[-1]]) * box_size,
    #     origin="lower",
    #     cmap="magma",
    #     interpolation="none",
    #     # aspect="equal",
    #     norm=LogNorm(vmin=1),
    # )

    ax.pcolormesh(
        xbins * box_size,
        ybins * box_size,
        img.T,
        cmap="magma",
        norm=LogNorm(),
    )
# print("plotting image")


# ax.imshow(
#     img.T,
#     extent=np.asarray([xbins[0], xbins[-1], ybins[0], ybins[-1]]) * box_size,
#     origin="lower",
#     cmap="magma",
#     # aspect="equal",
#     norm=LogNorm(vmin=1),
# )

# filled = img > 0
# X, Y = np.where(filled)
# filled_img = img[X, Y]
# ax.pcolormesh(X, Y, img, cmap="magma", norm=LogNorm())

print("plotting galaxy info")

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

ax.text(0.1, 0.9, f"z = {zed_tgt}", transform=ax.transAxes, color="white")

ax.scatter(
    zoom_ctr[0] * box_size, zoom_ctr[1] * box_size, c="r", s=100, marker="+", lw=1
)

ax.scatter(
    hagn_ctr[0] * box_size, hagn_ctr[1] * box_size, c="b", s=100, marker="+", lw=1
)
# rvir circle from hagn
circ = Circle(
    (hagn_ctr[0] * box_size, hagn_ctr[1] * box_size),
    hagn_r * box_size,
    color="b",
    fill=False,
    lw=1.5,
)
# annotate with rvir
ax.text(
    hagn_ctr[0] * box_size,
    hagn_ctr[1] * box_size - hagn_r * box_size * 1.1,
    f"HAGN rvir={hagn_r*box_size*1e3:.2e} kpc",
    ha="center",
    va="center",
    color="b",
    fontsize=6,
)

ax.add_patch(circ)

smin = 25
smax = 200

gm_max = ms.max()
gm_min = ms.min()

sizes = (ms - gm_min) / (gm_max - gm_min) * (smax - smin) + smin

ax.scatter(
    xgal[args][zfilt] * box_size,
    ygal[args][zfilt] * box_size,
    edgecolor="g",
    facecolor="none",
    s=sizes[args][zfilt],
    marker="o",
    lw=0.5,
    alpha=0.5,
)

# plot path of hagn halo
halo_ctrs = np.array(
    [
        hagn_tree_datas["x"][0],
        hagn_tree_datas["y"][0],
        hagn_tree_datas["z"][0],
    ]
)

halo_ctrs += 0.5 * (hagn_sim.cosmo.lcMpc * hagn_tree_aexps)
halo_ctrs /= hagn_sim.cosmo.lcMpc * hagn_tree_aexps

halo_ctrs *= box_size


halo_steps = np.diff(halo_ctrs.T, axis=0)


for istep in range(1, len(halo_steps)):

    xdir = np.sign(halo_steps[istep, 0] - halo_steps[istep - 1, 0])
    ydir = np.sign(halo_steps[istep, 1] - halo_steps[istep - 1, 1])

    ax.arrow(
        halo_ctrs[0, istep - 1],  # + xdir * halo_steps[istep - 1, 0] * 0.1,
        halo_ctrs[1, istep - 1],  # + ydir * halo_steps[istep - 1, 1] * 0.1,
        halo_steps[istep - 1, 0],
        halo_steps[istep - 1, 1],
        color="b",
        lw=0.25,
        length_includes_head=True,
        width=5e-3,
    )
    # ax.plot(
    #     [halo_ctrs[0, istep - 1], halo_ctrs[0, istep]],
    #     [halo_ctrs[1, istep - 1], halo_ctrs[1, istep]],
    #     color="b",
    #     lw=1,
    # )

ax.scatter(halo_ctrs[0, 0], halo_ctrs[1, 0], c="b", s=100, marker="x")


outdir = os.path.join(sim_dir, "maps")
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

# ax.set_xlim(xbins[0] * box_size, xbins[-1] * box_size)
# ax.set_ylim(ybins[0] * box_size, ybins[-1] * box_size)
ax.set_xlim(hagn_ctr[0] * box_size - img_size, hagn_ctr[0] * box_size + img_size)
ax.set_ylim(hagn_ctr[1] * box_size - img_size, hagn_ctr[1] * box_size + img_size)

fig.savefig(os.path.join(outdir, f"st_map_{snap_tgt:d}.png"))
fig.savefig(os.path.join(outdir, f"st_map_{snap_tgt:d}.pdf"))
