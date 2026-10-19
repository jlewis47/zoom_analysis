from ast import Pass
from cgitb import text
from tempfile import tempdir
from webbrowser import get
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe
from matplotlib.tri import Triangulation
from matplotlib.patches import Circle, Rectangle
import numpy as np
import os
from scipy.spatial import KDTree
from scipy.stats import binned_statistic_2d
from zoom_analysis.constants import *

from zoom_analysis.rascas.rascas_steps import get_directions_cart

from itertools import combinations

# from scipy.stats import binned_statistic_2d

import yt

from gremlin.read_sim_params import ramses_sim

from zoom_analysis.halo_maker.read_treebricks import (
    convert_brick_units,
    read_brickfile,
    read_zoom_brick,
)
from zoom_analysis.halo_maker.assoc_fcts import get_gal_props_snap
from zoom_analysis.visu.read_amr2cell import *

from f90_tools.hilbert import get_files

from zoom_analysis.sinks.sink_reader import (
    read_sink_bin,
    snap_to_coarse_step,
    convert_sink_units,
)


def check_prev_boxes(X, Y, prev_boxes):

    mask = np.empty_like(X, dtype=bool)
    mask[:] = True

    # intersect = False

    for prev_box in prev_boxes:

        prev_xmin, prev_xmax = prev_box[0], prev_box[2]
        prev_ymin, prev_ymax = prev_box[1], prev_box[3]

        mask_check = (
            (X >= prev_xmin) & (X <= prev_xmax) & (Y >= prev_ymin) & (Y <= prev_ymax)
        )

        if mask.any():

            mask[mask_check] = False

    return mask


# def indexes_to_slices(indexes, data):
#     ##assume sorted!!!

#     def multislice(a, sls):

#         for sl in sls:
#             yield a[sl]

#     if len(indexes) == 0:
#         return []

#     slice_inds = (
#         [indexes[0]]
#         + indexes[np.where(np.diff(indexes) > 1)[0]].tolist()
#         + [indexes[-1]]
#     )
#     # sl = slice(tuple(slice_inds))

#     slices = [slice(sl, slp1) for sl, slp1 in zip(slice_inds[:-1], slice_inds[1:])]

#     # print(indexes, slice_inds, np.where(np.diff(indexes) > 1))

#     # return data[sl]

#     return np.concatenate(list(multislice(data, slices)))


def segment_2d_point_cloud(x, y, res):

    # take median and expand radius around it until we don't collect any more points
    # do this until we have no more points to collect
    # boxy segmentation fast and perfect for what we want: making images
    # returns a list of bounding box coordinates

    imgs = []

    x_tosplit = x.copy()
    y_tosplit = y.copy()

    while len(x_tosplit) > 0:

        # print(len(x_tosplit))

        medx = np.median(x_tosplit)
        medy = np.median(y_tosplit)

        # print(medx, medy)

        looking = True
        rad = res

        found = 0

        while looking:

            # tree = KDTree(np.vstack((x_tosplit, y_tosplit)).T)

            # args = tree.query_ball_point([medx, medy], r=rad, p=2)

            dists = np.linalg.norm([x_tosplit - medx, y_tosplit - medy], axis=0, ord=2)

            # print(dists)
            # print(rad)

            cur_found = dists < rad

            nb_found = cur_found.sum()
            # nb_found = len(args)
            looking = nb_found > found or nb_found == 0
            rad += res
            found = nb_found

        args = np.where(cur_found)[0]
        # print(args)
        cur_x = x_tosplit[args]
        cur_y = y_tosplit[args]

        x_tosplit = np.delete(x_tosplit, args)
        y_tosplit = np.delete(y_tosplit, args)

        img = [np.min(cur_x), np.min(cur_y), np.max(cur_x), np.max(cur_y)]
        imgs.append(img)

    return imgs


def fill_pixels(
    vals, xbins, ybins, img, flat_ximgs, flat_yimgs, balls, op=np.sum, weights="count"
):
    if len(balls) == 0:
        return 0
    # sizes_caught = np.asarray(list(map(len, balls)))
    # caught = sizes_caught > 0

    caught = [len(elem) > 0 for elem in balls]

    use_weights = weights is not None

    # inds = np.arange(len(balls))

    balls = balls[caught]
    useful_vals = vals[caught]
    # print(useful_vals, vals)
    if use_weights:
        if np.any(weights == "count"):
            useful_weights = np.ones_like(useful_vals)
        else:
            useful_weights = weights[caught]
    # inds = inds[caught]

    # print(useful_weights)

    (uniqs, invs) = np.unique(balls, return_inverse=True)

    # reverse_inds = first_inds[invs]

    x0 = int(xbins[0])
    y0 = int(ybins[0])

    if use_weights:
        img_weights = np.zeros_like(img)

    # print(len(uniqs), uniqs.min(), uniqs.max(), len(flat_ximgs), len(flat_yimgs))

    for i, cur_inds in enumerate(uniqs):

        # if cur_inds == []:
        #     continue

        # ximg = flat_ximgs[cur_inds] - x0
        # yimg = flat_yimgs[cur_inds] - y0

        # print(inds, invs == uniq_i, uniq_i)
        # i = np.where(reverse_inds == uniq_i)
        # inds = i == invs
        inds = np.where(i == invs)[0]

        # if len(inds) == 0:
        # continue
        # print(i)

        xargs = flat_ximgs[cur_inds] - x0
        yargs = flat_yimgs[cur_inds] - y0

        # img[xarg, yargs] += op(indexes_to_slices(inds, useful_vals))
        # img_weights[xarg, yargs] += op(indexes_to_slices(inds, useful_weights))

        # print(len(inds), len(useful_vals), len(xargs), len(yargs))

        if not use_weights:
            img[xargs, yargs] += op(useful_vals[inds])
        else:
            img[xargs, yargs] += op(
                useful_vals[inds] * useful_weights[inds]
            )  # checked and this IS the
            # same as doing a loop
            img_weights[xargs, yargs] += op(useful_weights[inds])

            # print(
            #     op(useful_vals[inds] * useful_weights[inds]),
            #     op(useful_weights[inds]),
            #     op(useful_vals[inds]),
            # )
        # img[ximg, yimg] += np.sum(useful_vals[inds] * useful_weights[inds])

    if use_weights:
        non_zero = img_weights > 0
        img[non_zero] /= img_weights[non_zero]

    return 1


def make_yt_img(
    fig,
    ax,
    snap,
    sim: ramses_sim,
    pos,
    rvir,
    direction="x",
    perp_dist=-1,
    vmin=None,
    vmax=None,
    hfields=None,
    **kwargs,
):

    field_list = [("ramses", hfield) for hfield in hfields]
    # plot_fields

    # print(field_list, hfields)

    # load_fields = field_list.copy()

    if "weight_field" in kwargs:
        hfields += [kwargs["weight_field"]]

    bbox = [pos - rvir, pos + rvir]

    if perp_dist != -1:
        bbox[0][2] = pos[2] - perp_dist
        bbox[1][2] = pos[2] + perp_dist

    ds = yt.load(
        os.path.join(sim.output_path, f"output_{snap:05d}"),
        bbox=bbox,
        fields=hfields,
    )

    # print(list(ds.fields.gas))

    for field in field_list:

        # print(field)

        # print(ds, direction, field, pos, rvir, kwargs)

        if perp_dist != -1:

            img = yt.ProjectionPlot(
                ds=ds,
                normal=direction,
                fields=field,
                center=(pos, "kpc"),
                width=(rvir, "kpc"),
                **kwargs,
            )

        else:

            img = yt.SlicePlot(
                ds=ds,
                normal=direction,
                fields=field,
                center=(pos, "kpc"),
                width=(rvir, "kpc"),
                **kwargs,
            )

        # img.save("test.png")

        img.render()
        plot = img.plots[field]
        plot.axes = ax
        plot.figure = fig
        # plot.cax = grid.cbar_axes[i]


# def make_amr_img(
#     ax,
#     snap,
#     sim: ramses_sim,
#     pos,
#     rvir,
#     zdist=-1,
#     vmin=None,
#     vmax=None,
#     field="density",
#     debug=False,
# ):

#     lvlmin = sim.namelist["amr_params"]["levelmin"]
#     lvlmax = sim.namelist["amr_params"]["levelmax"]

#     lbox_kpc = sim.cosmo.unit_l / (ramses_pc * 1e6) / sim.aexp_stt * 1e3  # ckpc

#     sim_dir = sim.path

#     outdir = os.path.join(sim_dir, "amr2cell", "main_branch_cutouts", field)
#     if not os.path.exists(outdir):
#         os.makedirs(outdir, exist_ok=True)

#     print("reading amr2cell files...")

#     # cmap to bad pixels are black
#     plt.rcParams["image.cmap"] = "magma"
#     cmap = plt.get_cmap()
#     cmap.set_bad(color=(0, 0, 0, 0))

#     # zdist = 100
#     r = rvir
#     # r = 0.1 * r
#     x, y, z = pos
#     # zdist_cm = zdist * ramses_pc*1e3  # cm
#     # vmin, vmax = 1e-23, 1e-17
#     # vmin, vmax = 1e-22, 1e-17

#     for ihalo, (snap) in enumerate([snap]):
#         print(f"Reading amr2cell for snap {snap}...")

#         # could speed up by segmenting the regions where we plot...

#         aexp = sim.get_snap_exps(snap)

#         # x, y, z = (props["x"][ihalo], props["y"][ihalo], props["z"][ihalo])
#         # amrdata = read_amr2cell_output(
#         #     snap, sim_dir, fields=["density", "x", "y", "z", "ilevel"]
#         # )

#         mins, maxs = pos - r, pos + r

#         if zdist != -1:
#             mins[2] = pos[2] - zdist / sim.cosmo.lcMpc / 1e3
#             maxs[2] = pos[2] + zdist / sim.cosmo.lcMpc / 1e3

#         # find index of requested fields:

#         idx = [i for i in range(1, len(sim.hydro) + 1) if sim.hydro[str(i)] == field]
#         if len(idx) == 0:
#             if "temperature" == field and 1 not in idx:
#                 print("Temperature requested loading density and pressure")
#                 idx.extend([1, 5])
#             else:
#                 print(f"Didn't find requested field: {field}")

#             # print(f"Fields found: {sim.hydro[idx]}")
#             # print(f"Available fields: {sim.hydro}")
#         # print(i)
#         # print(idx)

#         amrdata = read_amrcells(sim, snap, np.ravel([mins, maxs]), idx)

#         # print(amrdata)

#         # convert units
#         # for field in fields:
#         # print(field, idx)
#         if field == "density":
#             amrdata[field] = amrdata[field] * sim.cosmo.unit_d
#         elif field == "temperature":
#             amrdata[field] = (
#                 amrdata["pressure"] / amrdata["density"]
#             ) / sim.cosmo.unit_d

#             print(amrdata[field].min(), amrdata[field].max())

#         all_levels = amrdata["ilevel"]
#         loaded_lvlmin = np.min(all_levels)

#         if zdist == -1:
#             # clamp z to nearest lvlmax cell
#             # print(z)
#             z = (np.round(z * 2**loaded_lvlmin) + 0.5) / 2**loaded_lvlmin
#             zdist = 1.0 / 2**loaded_lvlmin * lbox_kpc

#         zcells = amrdata["z"]
#         zfilt = np.abs(zcells - z) < (zdist / lbox_kpc)

#         xcells, ycells = (
#             amrdata["x"][zfilt],
#             amrdata["y"][zfilt],
#         )

#         tree = KDTree(np.vstack((xcells, ycells)).T, boxsize=1 + 1e-6)

#         inds = tree.query_ball_point([x, y], r=r, p=np.inf)

#         # print(x, y, r)

#         densities = amrdata[field][zfilt][inds]
#         levels = all_levels[zfilt][inds]

#         if len(inds) == 0:
#             print(f"No cells found for halo {ihalo} at snap {snap}")
#             continue

#         u_lvls = np.unique(levels)
#         xcells, ycells = xcells[inds], ycells[inds]

#         zorder = 0

#         for lvl in np.sort(u_lvls[:]):
#             lvl_below_args = levels <= lvl
#             lvl_above_args = levels > lvl

#             this_lvl_args = levels == lvl

#             this_lvl_x = xcells[this_lvl_args]
#             this_lvl_y = ycells[this_lvl_args]

#             lvl_below_x = xcells[lvl_below_args]
#             lvl_below_y = ycells[lvl_below_args]
#             lvls_below = levels[lvl_below_args]
#             lvl_below_dens = densities[lvl_below_args]

#             lvl_above_x = xcells[lvl_above_args]
#             lvl_above_y = ycells[lvl_above_args]
#             lvls_above = levels[lvl_above_args]
#             lvl_above_dens = densities[lvl_above_args]

#             dx = 1.0 / 2**lvl
#             dxs_below = 1.0 / 2**lvls_below
#             dxs_above = 1.0 / 2**lvls_above

#             # segment the cells into boxes... avoid high res voids
#             boxes = segment_2d_point_cloud(this_lvl_x, this_lvl_y, dx)
#             for box in boxes:
#                 # for i in range(1):

#                 # smallest bounding box
#                 # xmin, xmax = np.min(this_lvl_x), np.max(this_lvl_x)
#                 # ymin, ymax = np.min(this_lvl_y), np.max(this_lvl_y)

#                 xmin, xmax = box[0], box[2]
#                 ymin, ymax = box[1], box[3]

#                 # if lvl == lvlmin:
#                 #     xmin, xmax = 0, 1
#                 #     ymin, ymax = 0, 1

#                 # nx = int(np.ceil((xmax - xmin) / dx))
#                 # xbins = np.linspace(xmin, xmax, nx + 1)
#                 # xbins = np.arange(xmin, xmax + dx, dx)
#                 # ny = int(np.ceil((ymax - ymin) / dx))
#                 # ybins = np.linspace(ymin, ymax, ny + 1)
#                 # ybins = np.arange(ymin, ymax + dx, dx)

#                 xmin_cells = xmin * 2**lvl
#                 xmax_cells = xmax * 2**lvl
#                 ymin_cells = ymin * 2**lvl
#                 ymax_cells = ymax * 2**lvl

#                 xbins = np.arange(max(xmin_cells - 0.5 - 5, 0), xmax_cells + 0.5 + 6, 1)
#                 ybins = np.arange(max(ymin_cells - 0.5 - 5, 0), ymax_cells + 0.5 + 6, 1)
#                 # xbins = np.arange(max(xmin_cells - 0.5, 0), xmax_cells + 0.5, 1)
#                 # ybins = np.arange(max(ymin_cells - 0.5, 0), ymax_cells + 0.5, 1)

#                 print(
#                     f"lvl = {lvl}, dx = {dx*lbox_kpc}, xlen = {len(xbins)}, ylen = {len(ybins)}"
#                 )

#                 vol_fact_below = dxs_below * 1
#                 vol_fact_above = dxs_above * 1
#                 # print(vol_fact / np.nansum(vol_fact))

#                 img = np.zeros((len(xbins) - 1, len(ybins) - 1))
#                 # swap out bins for kdtree? KDTree for every image pixel position
#                 ximgs, yimgs = np.meshgrid(xbins[:-1], ybins[:-1])

#                 flat_ximgs = np.int32(np.ravel(ximgs))
#                 flat_yimgs = np.int32(np.ravel(yimgs))

#                 xcoords = (flat_ximgs + 0.5) / 2**lvl
#                 ycoords = (flat_yimgs + 0.5) / 2**lvl

#                 # print(xcoords.min(), xcoords.max(), ycoords.min(), ycoords.max())

#                 img_tree = KDTree(
#                     (np.transpose([xcoords, ycoords])),
#                     boxsize=1 + 1e-6,
#                 )

#                 balls_below = img_tree.query_ball_point(
#                     np.transpose([lvl_below_x, lvl_below_y]),
#                     r=dxs_below,
#                     p=np.inf,
#                     return_sorted=False,
#                 )

#                 fill_pixels(
#                     lvl_below_dens,
#                     xbins,
#                     ybins,
#                     img,
#                     flat_ximgs,
#                     flat_yimgs,
#                     balls_below,
#                     op=np.sum,
#                     weights=vol_fact_below,
#                 )

#                 balls_above = img_tree.query_ball_point(
#                     np.transpose([lvl_above_x, lvl_above_y]),
#                     r=dx,
#                     p=np.inf,
#                     return_sorted=False,
#                 )

#                 fill_pixels(
#                     lvl_above_dens,  # * dxs_above**3,
#                     xbins,
#                     ybins,
#                     img,
#                     flat_ximgs,
#                     flat_yimgs,
#                     balls_above,
#                     op=np.sum,
#                     weights=vol_fact_above,
#                 )

#                 # if vmin == None:
#                 #     lvl_diff = 2 ** (lvlmax - lvl)
#                 #     vmin = np.min(img[img > 0] / lvl_diff)
#                 #     print(f"Assigning vmin: {vmin:.1e}")
#                 # elif vmax == None:
#                 #     vmax = np.max(img[img > 0] * 0.75)
#                 #     print(f"Assigning vmax: {vmax:.1e}")
#                 if vmin == None or vmax == None:
#                     cdf, cdf_bins = np.histogram(np.log10(img[img > 0]), bins=50)
#                     cdf = np.cumsum(cdf)
#                     cdf = cdf / cdf[-1]
#                     if vmax == None:
#                         vmax = 10 ** (cdf_bins[np.where(cdf > 0.95)[0][0]])
#                         print(f"set vmax = {vmax}")

#                     if vmin == None:
#                         vmin = 10 ** (cdf_bins[np.where(cdf > 0.05)[0][0]])
#                         print(f"set vmin = {vmin}")

#                 # very slow but functional
#                 # img = np.zeros((len(xbins) - 1, len(ybins) - 1))
#                 # for ix, img_xbin in enumerate((xbins[:-1] + 0.5) / 2**lvl):
#                 #     for iy, img_ybin in enumerate((ybins[:-1] + 0.5) / 2**lvl):
#                 #         # img_xbin = xbin + 0.5
#                 #         # img_ybin = ybin + 0.5
#                 #         # img_xbin = xbin / 2**lvl
#                 #         # img_ybin = ybin / 2**lvl

#                 #         dists_below = np.linalg.norm(
#                 #             np.asarray([lvl_below_x - img_xbin, lvl_below_y - img_ybin]),
#                 #             axis=0,
#                 #             ord=1,
#                 #         )
#                 #         ball_below = dists_below < (dxs_below * 2)
#                 #         dists_above = np.linalg.norm(
#                 #             np.asarray([lvl_above_x - img_xbin, lvl_above_y - img_ybin]),
#                 #             axis=0,
#                 #             ord=1,
#                 #         )
#                 #         ball_above = dists_above < (dx * 2)

#                 #         img[ix, iy] = np.sum(
#                 #             lvl_below_dens[ball_below] * vol_fact_below[ball_below]
#                 #         ) + np.sum(+lvl_above_dens[ball_above] * vol_fact_above[ball_above])

#                 ax.imshow(
#                     img.T,
#                     origin="lower",
#                     extent=np.asarray(
#                         (
#                             [
#                                 xbins[0] / 2**lvl - x,
#                                 xbins[-1] / 2**lvl - x,
#                                 ybins[0] / 2**lvl - y,
#                                 ybins[-1] / 2**lvl - y,
#                             ]
#                         )
#                     )
#                     * lbox_kpc,
#                     # extent=np.asarray(([xmin, xmax, ymin, ymax])) * lbox_kpc,
#                     cmap=cmap,
#                     interpolation="none",
#                     norm=LogNorm(vmin=vmin, vmax=vmax),
#                     zorder=zorder,
#                 )

#                 # x_coords = np.asarray(list(zip(lvl_x, lvl_x + dx)))
#                 # y_coords = np.asarray(list(zip(lvl_y, lvl_y + dx)))

#                 # mesh_x, mesh_y = np.meshgrid(lvl_x + dx * 0.5, lvl_y + dx * 0.5)

#                 # axs.pcolor(
#                 #     mesh_x,
#                 #     mesh_y,
#                 #     lvl_dens,
#                 #     cmap=cmap,
#                 #     norm=LogNorm(vmin=vmin, vmax=vmax),
#                 #     zorder=zorder,
#                 # )

#                 zorder += 1

#         ax.set_xlabel("x, [kpc]")
#         ax.set_ylabel("y, [kpc]")

#         zed = 1 / aexp - 1

#         # print(x,y,r)

#         ax.set_xlim(x - r * lbox_kpc, x + r * lbox_kpc)
#         ax.set_ylim(y - r * lbox_kpc, y + r * lbox_kpc)

#         ax.text(
#             0.05,
#             0.9,
#             "z = %.2f" % zed,
#             color="white",
#             transform=ax.transAxes,
#             path_effects=[pe.withStroke(linewidth=1, foreground="black")],
#             ha="left",
#             size=20,
#             zorder=999,
#         )

#         # axs.fill(
#         #     x - r * lbox_kpc,
#         #     y - r * lbox_kpc,
#         #     x + r * lbox_kpc,
#         #     y + r * lbox_kpc,
#         #     color="k",
#         #     zorder=-999,
#         # )

#         # set facecolor to cmap color at vmin
#         ax.set_facecolor(cmap(0))

#         # img_size = len(xbins)
#         # axs.imshow(
#         #     np.full((img_size, img_size), vmin),
#         #     cmap=cmap,
#         #     norm=LogNorm(vmin=vmin, vmax=vmax),
#         #     zorder=-9999,
#         # )

#         # axs.set_xlim(-0.5 * r * lbox_kpc, 0.5 * r * lbox_kpc)
#         # axs.set_ylim(-0.5 * r * lbox_kpc, 0.5 * r * lbox_kpc)
#         # axs.set_xlim((x - r) * lbox_kpc, (x + r) * lbox_kpc)
#         # axs.set_ylim((y - r) * lbox_kpc, (y + r) * lbox_kpc)

#         # plt.savefig(os.path.join(outdir, fname))
#         # plt.savefig(os.path.join(outdir, fname.replace(".png", ".pdf")), format="pdf")
#         # plt.close()


def scatter_decomp(d3_points, axs, **kwargs):

    axs[0].scatter(d3_points[:, 0], d3_points[:, 1], **kwargs)
    axs[1].scatter(d3_points[:, 0], d3_points[:, 2], **kwargs)
    axs[2].scatter(d3_points[:, 1], d3_points[:, 2], **kwargs)


def make_amr_img_smooth(
    fig,
    ax,
    snap,
    sim: ramses_sim,
    pos,
    rvir,
    zdist=-1,
    vmin=None,
    vmax=None,
    mins=None,
    maxs=None,
    field="density",
    debug=False,
    weights="density",
    op=np.sum,
    cb=False,
    # direction=[0, 0, 1],  # project along z axis: [0,0,1]
    direction=[0, 0, 1],
):

    #
    # direction = get_directions_cart(12)[0]
    # print(direction)

    dv1 = np.array(direction)
    dv1, dv2, dv3 = basis_from_vect(dv1)

    cartesian = True

    if not np.all(
        [np.sum(dv1) == 1, np.sum(dv2) == 1, np.sum(dv3) == 1]
    ):  # not cartesian x,y,z basis... treat cells as spheres
        # dist_norm = 2
        dist_norm = np.inf
        dx_boost = np.sqrt(3)
        cartesian = False
    else:
        dist_norm = np.inf
        dx_boost = 1

    # ndir = len(dir_vects)

    # dv1, dv2, dv3 = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    # dv1, dv2, dv3 = [0, 0, 1], [0, 1, 0], [1, 0, 0]
    # dv1, dv2, dv3 = (
    #     [0, 1, 0],
    #     [0, 0, 1],
    #     [1, 0, 0],
    # )

    # print(dv1, dv2, dv3)

    # lvlmin = sim.namelist["amr_params"]["levelmin"]
    # lvlmax = sim.namelist["amr_params"]["levelmax"]

    lbox_kpc = sim.unit_l(sim.aexp_stt) / (ramses_pc * 1e6) / sim.aexp_stt * 1e3  # ckpc

    # sim_dir = sim.path

    print("reading amr2cell files...")

    # cmap to bad pixels are black
    plt.rcParams["image.cmap"] = "magma"
    cmap = plt.get_cmap()
    cmap.set_bad(color=(0, 0, 0, 0))

    # zdist = 100
    r = rvir
    # r = 0.1 * r

    # dfig, dax = plt.subplots(1, 3, sharex=True, sharey=True)

    # dax[0].set_xlim(-1, 2)
    # dax[0].set_ylim(-1, 2)

    # for da in dax:
    #     da.set_aspect("equal")
    #     da.plot([0, 0], [0, 1], "k")
    #     da.plot([0, 1], [1, 1], "k")
    #     da.plot([1, 1], [0, 1], "k")
    #     da.plot([0, 1], [0, 0], "k")

    M_basis = [dv3, dv2, dv1]
    # print(pos)

    # print(dv1, dv2, dv3)
    # print(M_basis)

    # dax.scatter(*pos, c="b")
    # scatter_decomp(np.asarray([pos]), dax, c="b", marker="x")

    pos = np.dot(pos, M_basis)

    # dax.scatter(*pos, c="r")
    # scatter_decomp(np.asarray([pos]), dax, c="r", marker="x")
    if np.any(mins == None):
        mins = pos - r
    if np.any(maxs == None):
        maxs = pos + r

    if zdist != -1:
        # print(pos[2], zdist)
        mins[2] = pos[2] - zdist / sim.cosmo.lcMpc / 1e3
        maxs[2] = pos[2] + zdist / sim.cosmo.lcMpc / 1e3

    # print(pos, mins, maxs)

    pos[pos < 0] += 1
    pos[pos > 1] -= 1

    # print("pos new basis")
    # print(pos)
    # # dax.scatter(*pos, c="g")
    # scatter_decomp(np.asarray([pos]), dax, c="g", marker="x")

    # print(np.dot(M_basis, [1, 0, 0]))
    # print(np.dot(M_basis, [0, 1, 0]))
    # print(np.dot(M_basis, [0, 0, 1]))

    # print(M_basis)

    M_basis_inv = np.linalg.inv(M_basis)

    # print(M_basis, M_basis_inv)

    # transform pos back into original basis
    # print(np.dot(M_basis, pos))
    # print(pos)

    x, y, z = pos

    # zdist_cm = zdist * ramses_pc*1e3  # cm
    # vmin, vmax = 1e-23, 1e-17
    # vmin, vmax = 1e-22, 1e-17

    for ihalo, (snap) in enumerate([snap]):
        print(f"Reading amr2cell for snap {snap}...")

        # could speed up by segmenting the regions where we plot...

        aexp = sim.get_snap_exps(snap)

        # x, y, z = (props["x"][ihalo], props["y"][ihalo], props["z"][ihalo])
        # amrdata = read_amr2cell_output(
        #     snap, sim_dir, fields=["density", "x", "y", "z", "ilevel"]
        # )

        # print("mins/maxs new basis")
        # print(mins)
        # print(maxs)
        # scatter_decomp(np.asarray([mins]), dax, c="r", marker="P")
        # scatter_decomp(np.asarray([maxs]), dax, c="r", marker="P")

        # mins[mins < 0] += 1
        # maxs[maxs < 0] += 1
        # mins[mins > 1] -= 1
        # maxs[maxs > 1] -= 1

        # convert mins,maxs back into original basis
        # scatter_decomp(np.asarray([mins]), dax, c="g", marker="P")
        # scatter_decomp(np.asarray([maxs]), dax, c="g", marker="P")
        mins = np.dot(mins, M_basis_inv)
        maxs = np.dot(maxs, M_basis_inv)

        mins[mins < 0] += 1
        maxs[maxs < 0] += 1
        mins[mins > 1] -= 1
        maxs[maxs > 1] -= 1

        # scatter_decomp(np.asarray([mins]), dax, c="b", marker="P")
        # scatter_decomp(np.asarray([maxs]), dax, c="b", marker="P")
        # print(mins > maxs, mins[mins > maxs], maxs[mins > maxs])

        # print(mins)
        # print(maxs)
        ext_bool = mins > maxs
        a = np.copy(mins[ext_bool])
        b = np.copy(maxs[ext_bool])
        mins[ext_bool] = b
        maxs[ext_bool] = a
        # print("mins/maxs")
        # print(mins)
        # print(maxs)

        if not cartesian:
            # messy and requires more data but works
            # if you're at a weird angle you need to expand your
            # data range to fill the projected image square
            mins = mins * 0.99
            maxs = maxs * 1.01

        # scatter_decomp(np.asarray([mins]), dax, c="k", marker="P")
        # scatter_decomp(np.asarray([maxs]), dax, c="k", marker="P")
        # dfig.savefig("debug_transform")

        # find index of requested fields:
        # print(sim.hydro)

        idx = [i for i in range(1, len(sim.hydro) + 1) if sim.hydro[str(i)] == field]
        # if len(idx) == 0:
        if "temperature" == field and (5 not in idx):
            print("Temperature requested loading density and pressure")
            idx.extend([1, 5])
        elif "metallicity" == field and (6 not in idx):
            print("Metal density requested loading density and metallicity")
            # idx.extend([6])
            idx.extend([1, 6])
        elif "dust_bin01" == field and (16 not in idx):
            print(
                "small carbon fraction requested loading density and small carbon grains"
            )
            # idx.extend([16])
            idx.extend([1, 16])
        elif "dust_bin02" == field and (17 not in idx):
            print(
                "large carbon fraction requested loading density and large carbon grains"
            )
            # idx.extend([17])
            idx.extend([1, 17])
        elif "dust_bin03" == field and (17 not in idx):
            print(
                "small silicate fraction requested loading density and small silicate grains"
            )
            # idx.extend([18])
            idx.extend([1, 18])
        elif "dust_bin04" == field and (17 not in idx):
            print(
                "large silicate fraction requested loading density and large silicate grains"
            )
            # idx.extend([19])
            idx.extend([1, 19])

        elif field != "density":
            print("No rule for Didn't find requested field: {field}")

            # print(f"Fields found: {sim.hydro[idx]}")
            # print(f"Available fields: {sim.hydro}")
        # print(i)
        # print(idx)

        idx = np.unique(idx)

        # print(mins, maxs)
        # print(np.asarray([mins, maxs]) * sim.cosmo.lcMpc * 1e3)

        print(mins, maxs, idx)

        amrdata = read_amrcells(sim, snap, np.ravel([mins, maxs]), idx)

        # print(amrdata.keys())
        # print(field)

        # print(amrdata)
        if len(amrdata["ilevel"]) == 0:
            print("No cells found")
            return

        # convert units
        # for field in fields:

        # print(field)

        if field == "density":
            amrdata[field] = amrdata[field] * sim.unit_d(aexp)
        elif field == "temperature":
            unit_T = sim.unit_T(aexp)
            # scale_l = sim.unit_l(aexp)
            # scale_t = sim.unit_t(aexp)
            # scale_l = aexp * 1.0 * (ramses_pc * 1e6) / (sim.cosmo["H0"] / 100.0)
            # scale_t = aexp**2 / (sim.cosmo["H0"] * 1e5 / (ramses_pc * 1e6))
            # scale_T2 = mass_H_cgs / Bmann_cgs * (scale_l / scale_t) ** 2

            # amrdata[field] = (amrdata["pressure"] / amrdata["density"]) * scale_T2
            amrdata[field] = (amrdata["pressure"] / amrdata["density"]) * unit_T

            # print(amrdata[field].min(), amrdata[field].max())

        # elif field in [
        #     "metallicty",
        #     "large carbon density",
        #     "small carbon density",
        #     "large silicate density",
        #     "small silicate density",
        # ]:
        #     # amrdata[field] = amrdata["density"] * amrdata[field] * sim.unit_d(aexp)
        #     amrdata[field] = amrdata[field]

        # else:
        #     amrdata[field] = amrdata[field]

        # if debug:
        # print(field)
        # print(amrdata[field].min(), amrdata[field].max())

        all_levels = amrdata["ilevel"]
        loaded_lvlmin = np.min(all_levels)

        # print(amrdata["x"].min(), amrdata["x"].max())
        # print(amrdata["y"].min(), amrdata["y"].max())
        # print(amrdata["z"].min(), amrdata["z"].max())

        if zdist == -1:
            # clamp z to nearest lvlmax cell
            # print(z)
            z = (np.round(z * 2**loaded_lvlmin) + 0.5) / 2**loaded_lvlmin
            zdist = 1.0 / 2**loaded_lvlmin * lbox_kpc

        # print(zdist, lbox_kpc)

        cell_pos = np.asarray([amrdata["x"], amrdata["y"], amrdata["z"]]).T

        # print(np.median(cell_pos, axis=0))

        cell_pos = np.dot(M_basis, cell_pos.T)

        # print(np.median(cell_pos, axis=1))

        for idim in range(3):
            if np.all(cell_pos[idim, :] < 0):
                # cell_pos[idim,:] = np.abs(cell_pos[idim,:])
                cell_pos[idim, :] += 1
            if np.all(cell_pos[idim, :] > 1):
                # cell_pos[idim,:] = np.abs(cell_pos[idim,:])
                cell_pos[idim, :] -= 1
        # print(np.median(cell_pos, axis=1))

        # zcells = np.dot(dv1, amrdata["z"])

        # xcells, ycells = (
        #     np.dot(dv2, amrdata["x"][zfilt]),
        #     np.dot(dv3, amrdata["y"][zfilt]),
        # )

        # print(xcells)
        # print(ycells)

        xcells, ycells, zcells = cell_pos
        zfilt = np.abs(zcells - z) < (zdist / lbox_kpc)
        # zfilt = np.full_like(zcells, True, dtype=bool)

        # zfilt = ((zcells - z) < zdist) * (((-zcells + z) < zdist))

        # print(zcells, z, zfilt)

        # print(zfilt.sum())
        # print(np.abs(zcells - z), (zdist / lbox_kpc))

        xcells = xcells[zfilt]
        ycells = ycells[zfilt]
        zcells = zcells[zfilt]

        # print(np.min([xcells, ycells]))

        tree = KDTree(np.vstack((xcells, ycells)).T, boxsize=1 + 1e-6)

        inds = tree.query_ball_point([x, y], r=r, p=dist_norm)

        # print(x, y, r)

        # print(amrdata.keys())

        values = amrdata[field][zfilt][inds]
        if weights == "density":
            densities = amrdata["density"][zfilt][inds]
        levels = all_levels[zfilt][inds]

        if len(inds) == 0:
            print(f"No cells found for halo {ihalo} at snap {snap}")
            continue

        u_lvls = np.unique(levels)
        xcells, ycells = xcells[inds], ycells[inds]

        # zorder = 0

        lvl_maps = []

        for lvl in np.sort(u_lvls[:][::-1]):  # desc order

            this_lvl_args = levels == lvl

            this_lvl_x = xcells[this_lvl_args]
            this_lvl_y = ycells[this_lvl_args]

            dx = 1.0 / 2**lvl * dx_boost
            # dxs_below = 1.0 / 2**lvls_below
            # dxs_above = 1.0 / 2**lvls_above

            # segment the cells into boxes... avoid high res voids
            boxes = segment_2d_point_cloud(this_lvl_x, this_lvl_y, dx)

            lvl_maps.append([lvl, boxes])

        # maps = []

        for ilvl, (lvl, lvl_boxes) in enumerate(lvl_maps):

            lvl_imgs = []

            for box in lvl_boxes:

                lvl_dx = 1.0 / 2**lvl * dx_boost

                xmin, xmax = box[0], box[2]
                ymin, ymax = box[1], box[3]

                # print(xmin, xmax)
                # print(ymin, ymax)

                xmin_cells = xmin * 2**lvl
                xmax_cells = xmax * 2**lvl
                ymin_cells = ymin * 2**lvl
                ymax_cells = ymax * 2**lvl

                xbins = np.arange(max(xmin_cells - 0.5 - 1, 0), xmax_cells + 0.5 + 2, 1)
                ybins = np.arange(max(ymin_cells - 0.5 - 1, 0), ymax_cells + 0.5 + 2, 1)

                print(
                    f"lvl = {lvl}, dx = {lvl_dx*lbox_kpc}, xlen = {len(xbins)}, ylen = {len(ybins)}"
                )

                ximgs, yimgs = np.meshgrid(xbins[:-1], ybins[:-1])

                flat_ximgs = np.int32(np.ravel(ximgs))
                flat_yimgs = np.int32(np.ravel(yimgs))

                xcoords = (flat_ximgs + 0.5) / 2**lvl
                ycoords = (flat_yimgs + 0.5) / 2**lvl

                img = np.zeros((len(xbins) - 1, len(ybins) - 1))
                # print(xcoords.max(), ycoords.max())  # some things outside of box
                tree = KDTree(np.transpose([xcoords, ycoords]), boxsize=1 + 1e-3)

                cond = (
                    (xcells >= xmin - lvl_dx * 6)
                    * (xcells <= xmax + lvl_dx * 6)
                    * (ycells >= ymin - lvl_dx * 6)
                    * (ycells <= ymax + lvl_dx * 6)
                )

                # cond = levels <= lvl

                # if cond.sum() == 0:
                #     continue

                box_lvls = levels[cond]
                box_values = values[cond]
                box_dxs = 1.0 / 2**box_lvls
                box_xcells = xcells[cond]
                box_ycells = ycells[cond]
                # box_lvls = levels
                # box_values = densities
                # box_dxs = 1.0 / 2**box_lvls
                # box_xcells = xcells
                # box_ycells = ycells
                if weights == "volume":
                    box_weights = box_dxs**3
                elif weights == "density":
                    box_weights = densities[cond]
                else:
                    box_weights = "counts"

                balls = tree.query_ball_point(
                    np.transpose([box_xcells, box_ycells]),
                    r=box_dxs * 0.5 * dx_boost,
                    p=dist_norm,
                    return_sorted=False,
                )

                fill_pixels(
                    box_values,
                    xbins,
                    ybins,
                    img,
                    flat_ximgs,
                    flat_yimgs,
                    balls,
                    op=op,
                    # weights=None,
                    weights=box_weights,
                )

                # print(img[img > 0].min(), img.max())

                lvl_imgs.append(img)
                # print(img.shape)

            # print(len(lvl_imgs))
            lvl_maps[ilvl].append(lvl_imgs)

        # print(len(lvl_maps))

        all_maps = np.concatenate(
            [
                np.ravel(elem3)
                for elem in lvl_maps
                for elem2 in elem[-1]
                for elem3 in elem2
            ]
        )

        # print(all_maps)
        all_maps = all_maps[np.isfinite(all_maps) * (all_maps > 0)]

        # for elem in lvl_maps:
        #     print(len(elem))
        #     for elem2 in elem[-1]:
        #         print(len(elem2))
        #         for elem3 in elem2:
        #             print(len(elem3))

        # print(len(all_maps))
        # print(all_maps[0], all_maps.min(), all_maps.max())
        # print(len(np.ravel(all_maps)))

        if vmin == None or vmax == None:
            # print(all_maps)
            cdf, cdf_bins = np.histogram(np.log10(all_maps), bins=100)
            cdf = np.cumsum(cdf)
            cdf = cdf / cdf[-1]
            if vmax == None:
                vmax = 10 ** (cdf_bins[np.where(cdf > 0.99)[0][0]])
                print(f"set vmax = {vmax}")

            if vmin == None:
                vmin = 10 ** (cdf_bins[np.where(cdf > 0.01)[0][0]])
                print(f"set vmin = {vmin}")

        # values = []
        # xcoords = []
        # ycoords = []
        # plot_lvls = []

        # prev_boxes = []

        for ilvl, (lvl, boxes, imgs) in enumerate(lvl_maps):

            # print(lvl, boxes, imgs)
            # if lvl < 17:
            #     continue

            for box, img in zip(boxes, imgs):

                # print(box, img)

                xmin, xmax = box[0], box[2]
                ymin, ymax = box[1], box[3]

                xmin_cells = xmin * 2**lvl
                xmax_cells = xmax * 2**lvl
                ymin_cells = ymin * 2**lvl
                ymax_cells = ymax * 2**lvl

                xbins = np.arange(max(xmin_cells - 0.5 - 1, 0), xmax_cells + 0.5 + 2, 1)
                ybins = np.arange(max(ymin_cells - 0.5 - 1, 0), ymax_cells + 0.5 + 2, 1)

                # rect = Rectangle(
                #     (xbins[0] / 2**lvl - x, ybins[0] / 2**lvl - y),
                #     (xbins[-1] - xbins[0]),
                #     (ybins[-1] - ybins[0]),
                #     fill=None,
                #     edgecolor="r",
                # )
                # ax.add_patch(rect)
                # print(
                #     f"lvl = {lvl}, dx = {dx*lbox_kpc}, xlen = {len(xbins)}, ylen = {len(ybins)}"
                # )

                # ximgs, yimgs = np.meshgrid(xbins[:-1], ybins[:-1])

                # mask = check_prev_boxes(ximgs.T, yimgs.T, prev_boxes)

                # ax.scatter(
                #     (ximgs.T[mask] + 0.5) / 2**lvl,
                #     (yimgs.T[mask] + 0.5) / 2**lvl,
                #     c=img[mask],
                #     s=1 / 2**lvl,
                #     zorder=999,
                # )

                img = ax.imshow(
                    img.T,
                    origin="lower",
                    extent=np.asarray(
                        (
                            [
                                (xbins[0] + 0.5) / 2**lvl - x,
                                (xbins[-1] + 0.5) / 2**lvl - x,
                                (ybins[0] + 0.5) / 2**lvl - y,
                                (ybins[-1] + 0.5) / 2**lvl - y,
                            ]
                        )
                    )
                    * lbox_kpc,
                    interpolation="none",
                    norm=LogNorm(vmin=vmin, vmax=vmax),
                    cmap=cmap,
                    # zorder=ilvl,
                )

                # print(mask)

                # print(ximgs.shape, img.shape, mask.shape)

                # print(img[img > 0].min(), img.max())
                # print(img[mask.T][img[mask.T] > 0].min(), img[mask.T].max())

                # values.append(np.ravel(img[mask]))
                # xcoords.append(np.ravel(ximgs.T[mask]) / 2**lvl)
                # ycoords.append(np.ravel(yimgs.T[mask]) / 2**lvl)
                # plot_lvls.append(np.ones_like(np.ravel(ximgs.T[mask])) * lvl)

                # prev_boxes.append(box)

        if cb:
            # colorbar at right of imshow, fitted nicely to width of imshow
            cbar = fig.colorbar(
                img,
                ax=ax,
                orientation="vertical",
                location="right",
                fraction=0.046,
                pad=0.04,
            )
            cbar.set_label(field)

        # values = np.concatenate(values)
        # xcoords = np.concatenate(xcoords)
        # ycoords = np.concatenate(ycoords)
        # plot_lvls = np.concatenate(plot_lvls)

        # plot_dxs = 1.0 / 2**plot_lvls

        # norm = np.max([xcoords, ycoords])

        # xcoords /= norm
        # ycoords /= norm

        # print(np.unique(values))
        # print(values[values != 0].min(), values.max()) d
        # values[values <= 0] = vmin
        # print(values[values != 0].min(), values.max())

        # triang = Triangulation(xcoords, ycoords)

        # print(triang)
        # print(xcoords, ycoords)

        # something funky happened...

        # print(list(zip(xcoords, ycoords)))

        # ax.tricontourf(
        #     xcoords,
        #     ycoords,
        #     # triangulation=triang,
        #     values,  # / values.min(),
        #     # levels=50,
        #     cmap=cmap,
        #     # vmin=vmin,
        #     # vmax=vmax,
        #     norm=LogNorm(vmin=vmin, vmax=vmax),
        # )

        # ax.scatter(
        #     xcoords + 0.5 * plot_dxs,
        #     ycoords + 0.5 * plot_dxs,
        #     # c=values,
        #     c=plot_lvls,
        #     s=plot_dxs**2,
        #     marker="s",
        #     cmap=cmap,
        #     # norm=LogNorm(vmin=vmin, vmax=vmax),
        # )

        # ax.pcolor(
        #     [xcoords, ycoords], values, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax)
        # )

        ax.set_xlabel("x, [kpc]")
        ax.set_ylabel("y, [kpc]")

        zed = 1 / aexp - 1

        # print(x,y,r)

        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)

        ax.set_xlim(x - r * lbox_kpc, x + r * lbox_kpc)
        ax.set_ylim(y - r * lbox_kpc, y + r * lbox_kpc)

        ax.text(
            0.05,
            0.9,
            "z = %.2f" % zed,
            color="white",
            transform=ax.transAxes,
            path_effects=[pe.withStroke(linewidth=1, foreground="black")],
            ha="left",
            size=20,
            zorder=999,
        )

        ax.set_facecolor(cmap(0))


def plot_zoom_BHs(
    ax,
    snap,
    sim,
    tgt_pos,
    tgt_rad,
    zdist,
    sink_read_fct=read_sink_bin,
    direction=[0, 0, 1],
):

    sink_files = np.asarray(os.listdir(sim.sink_path))
    sink_fnbs = np.asarray([int(f.split("_")[-1].split(".")[0]) for f in sink_files])

    step = snap_to_coarse_step(snap, sim)
    sink_f = sink_files[step == sink_fnbs][0]

    bhs = sink_read_fct(os.path.join(sim.sink_path, sink_f))
    aexp = sim.get_snap_exps(snap, param_save=False)
    convert_sink_units(bhs, aexp, sim)

    dv1 = np.array(direction)
    dv1, dv2, dv3 = basis_from_vect(dv1)
    M_basis = [dv3, dv2, dv1]

    pos = np.dot(bhs["position"], M_basis).T

    # print(pos.T)

    rot_tgt_pos = np.dot(tgt_pos, M_basis).T

    bhxs, bhys, bhzs = pos
    bhms = bhs["mass"]

    # in frame
    bhxs = (bhxs - rot_tgt_pos[0]) * sim.cosmo.lcMpc * 1e3
    bhys = (bhys - rot_tgt_pos[1]) * sim.cosmo.lcMpc * 1e3
    bhzs = (bhzs - rot_tgt_pos[2]) * sim.cosmo.lcMpc * 1e3

    in_frame = (
        np.linalg.norm(np.asarray([bhxs, bhys]), axis=0)
        < (tgt_rad) * sim.cosmo.lcMpc * 1e3 * 2**0.5
    ) * (np.abs(bhzs) < zdist)

    if not in_frame.any():
        return 0

    max_mbh = bhms[in_frame].max()
    min_mbh = bhms[in_frame].min()
    max_size = 2e2

    sizes = (
        (np.log10(bhms[in_frame]) - np.log10(min_mbh))
        / (np.log10(max_mbh) - np.log10(min_mbh))
        * max_size
    )

    sizes[sizes < 0.1 * max_size] = 0.1 * max_size

    ax.scatter(
        bhxs[in_frame],
        bhys[in_frame],
        s=sizes,
        edgecolors="k",
        facecolors="none",
        alpha=0.5,
        label="BHs",
        zorder=999,
        lw=0.5,
    )

    # make a legend for sizes
    if np.abs(max_mbh - sim.namelist["smbh_params"]["mseed"]) < 1:
        biggest = max_size
        # med = max_size * ((np.median(bhms)) / (max_mbh))
        smallest = max_size * 0.1
        med = max_size * 0.5
    else:
        biggest = smallest = med = max_size
    # get transform for legend
    # go from fraction of axes to data
    # redshift text is at 0.1,0.1... convert to data
    # tr = ax.transAxes + ax.transData.inverted()
    # get data coords
    # data_coords = tr.transform([0.1, 0.1])

    # ax.scatter(
    #     # [data_coords[0]-50, data_coords[0], data_coords[0] + 50],
    #     [data_coords[0] - 50, data_coords[0] + 50],
    #     # [data_coords[1] - 50] * 3,
    #     [data_coords[1] - 100] * 2,
    #     s=[biggest, smallest],
    #     # s=[biggest, med, smallest],
    #     edgecolors="k",
    #     facecolors="none",
    #     alpha=0.5,
    #     lw=1,
    #     zorder=999,
    # )

    markers = []
    for size in [biggest, med, smallest]:
        markers.append(
            ax.scatter(
                [],
                [],
                s=size,
                edgecolors="w",
                facecolors="none",
                alpha=0.5,
                lw=0.5,
            )
        )

    # print(biggest, med, smallest)
    # print(max_mbh, )
    bh_leg = ax.legend(
        markers,
        [
            f"{max_mbh:.1e} M$_\odot$",
            f"{med*max_mbh/max_size:.1e} M$_\odot$",
            f"{smallest*max_mbh/max_size:.1e} M$_\odot$",
        ],
        loc="lower left",
        # fontsize=12,
        title="BH mass scale",
        # title_fontsize="16",
        framealpha=0.0,
        prop=dict(
            size=12,
        ),
        title_fontproperties=dict(
            size=16,
        ),
        labelcolor="white",
    )
    plt.setp(bh_leg.get_title(), color="white")

    # ax.annotate(
    #     f"{max_mbh:.1e} M$_\odot$",
    #     (data_coords[0] - 50, data_coords[1] - 100),
    #     arrowprops=dict(arrowstyle="->"),
    #     fontsize=12,
    # )

    # ax.annotate(
    #     f"{np.min(bhms):.1e} M$_\odot$",
    #     (data_coords[0] + 50, data_coords[1] - 100),
    #     arrowprops=dict(arrowstyle="->"),
    #     fontsize=12,
    # )

    # ax.text(
    #     data_coords[0],
    #     data_coords[1] - 50,
    #     "BH mass scale",
    #     fontsize=12,
    #     verticalalignment="center",
    # )

    return 1


def plot_zoom_gals(
    ax,
    snap,
    sim: ramses_sim,
    tgt_pos,
    tgt_rad,
    zdist,
    hm=None,
    brick_fct=read_zoom_brick,
    gal_markers=True,  # scatter plots of galaxies
    annotate=True,  # plot gids next to makers
    direction=[0, 0, 1],
):
    if hm:
        bricks = brick_fct(snap, sim, hm)
    else:
        bricks = brick_fct(snap, sim)

    if bricks == 0:
        return 0

    # print(bricks)

    mains = bricks["hosting info"]["hlvl"] == 1
    subs = bricks["hosting info"]["hlvl"] == 2

    stmass = bricks["hosting info"]["hmass"]

    gids = bricks["hosting info"]["hid"]
    x = bricks["positions"]["x"]
    y = bricks["positions"]["y"]
    z = bricks["positions"]["z"]

    gal_pos = np.asarray([x, y, z]).T

    # direction = [1, 0, 0]
    # direction = [0, 1, 0]

    dv1 = np.array(direction)
    dv1, dv2, dv3 = basis_from_vect(dv1)
    M_basis = [dv3, dv2, dv1]

    # print(M_basis)

    # print(direction)

    # print(np.dot([1, 0, 0], M_basis))
    # print(np.dot([0, 1, 0], M_basis))
    # print(np.dot([0, 0, 1], M_basis))

    gal_pos = np.dot(gal_pos, M_basis).T

    x, y, z = gal_pos

    rot_tgt_pos = np.dot(tgt_pos, M_basis).T

    in_plane = np.abs(z - rot_tgt_pos[2]) * sim.cosmo.lcMpc * 1e3 < zdist

    if not in_plane.any():
        return 0

    max_mass_size = 3e2
    max_mass = stmass[in_plane].max()
    min_mass = stmass[in_plane].min()
    med_mass = 10 ** (0.5 * (np.log10(max_mass) + np.log10(min_mass)))

    sizes = (
        (np.log10(stmass) - np.log10(min_mass))
        / (np.log10(max_mass) - np.log10(min_mass))
        * max_mass_size
    )

    sizes[sizes < 0.1 * max_mass_size] = 0.1 * max_mass_size

    mains_to_plot = mains * in_plane
    subs_to_plot = subs * in_plane

    # print(mains.sum(), in_plane.sum())
    # print((mains_to_plot).sum())

    # print(sizes[in_plane].max())

    # print((x[mains_to_plot] - tgt_pos[0]) * sim.cosmo.lcMpc * 1e3)

    if gal_markers:
        # plot mains
        ax.scatter(
            (x[mains_to_plot] - rot_tgt_pos[0]) * sim.cosmo.lcMpc * 1e3,
            (y[mains_to_plot] - rot_tgt_pos[1]) * sim.cosmo.lcMpc * 1e3,
            s=sizes[mains * in_plane],
            c="k",
            marker="x",
            alpha=0.5,
            lw=0.5,
            zorder=999,
        )

        # plot subs
        ax.scatter(
            (x[subs_to_plot] - rot_tgt_pos[0]) * sim.cosmo.lcMpc * 1e3,
            (y[subs_to_plot] - rot_tgt_pos[1]) * sim.cosmo.lcMpc * 1e3,
            s=sizes[subs_to_plot],
            edgecolors="k",
            facecolors="none",
            marker="v",
            alpha=0.5,
            lw=0.5,
            zorder=999,
        )

        markers = ["x", "v"]
        markers_sizes = [
            max_mass_size,
            # np.log10(med_mass) / np.log10(max_mass) * max_mass_size,
            # np.log10(min_mass) / np.log10(max_mass) * max_mass_size,
            0.5 * max_mass_size,
            0.1 * max_mass_size,
        ]
        marker_sizes = []

        for marker_size in markers_sizes:
            marker_sizes.append(
                ax.scatter([], [], s=marker_size, c="w", marker="x", alpha=0.5, lw=0.5)
            )

        # legend for the mass of galaxies
        gal_mass_leg = ax.legend(
            marker_sizes,
            [
                f"{max_mass:.1e} M$_\odot$",
                f"{med_mass:.1e} M$_\odot$",
                f"{min_mass:.1e} M$_\odot$",
            ],
            loc="lower center",
            # fontsize=12,
            title="Galaxy mass scale",
            # title_fontsize="16",
            framealpha=0.0,
            prop=dict(
                size=12,
            ),
            title_fontproperties=dict(
                size=16,
            ),
            labelcolor="white",
        )
        plt.setp(gal_mass_leg.get_title(), color="white")
        ax.add_artist(gal_mass_leg)

        type_markers = []
        for marker in markers:
            type_markers.append(
                ax.scatter(
                    [],
                    [],
                    s=max_mass_size,
                    edgecolors="white",
                    facecolors="none",
                    marker=marker,
                    alpha=0.5,
                    lw=0.5,
                )
            )

        # legend for type of galaxy markers
        gal_lvl_leg = ax.legend(
            type_markers,
            ["Central", "Subhalo"],
            loc="lower right",
            # fontsize=12,,
            # title_fontsize="16",
            framealpha=0.0,
            prop=dict(
                size=12,
            ),
            labelcolor="white",
        )
        ax.add_artist(gal_lvl_leg)

    if annotate:
        # annotate makers with ids
        for i, (gid, x, y) in enumerate(
            zip(gids[mains_to_plot], x[mains_to_plot], y[mains_to_plot])
        ):

            try:
                _, gal_dict = get_gal_props_snap(sim.path, snap, gid)

                print(gid, gal_dict["sfr1000"] / gal_dict["mass"])

            except FileNotFoundError:
                pass

            annotate_txt = f"{gid}"

            ax.annotate(
                annotate_txt,
                (
                    (x - rot_tgt_pos[0]) * sim.cosmo.lcMpc * 1e3,
                    (y - rot_tgt_pos[1]) * sim.cosmo.lcMpc * 1e3,
                ),
                fontsize=12,
                zorder=999,
                color="w",
                path_effects=[pe.withStroke(linewidth=1, foreground="black")],
            )
        # for i, (gid, x, y) in enumerate(
        #     zip(gids[subs_to_plot], x[subs_to_plot], y[subs_to_plot])
        # ):
        #     ax.annotate(
        #         f"{gid}",
        #         (
        #             (x - tgt_pos[0]) * sim.cosmo.lcMpc * 1e3,
        #             (y - tgt_pos[1]) * sim.cosmo.lcMpc * 1e3,
        #         ),
        #         fontsize=8,
        #         zorder=999,
        #         color="w",
        #         path_effects=[pe.withStroke(linewidth=1, foreground="black")],
        #     )

    return 1


def basis_from_vect(u1):

    # old complicated
    # #grahm_schmidt
    # #https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process

    # v1 = np.asarray([1, 0, 0])
    # v2 = np.asarray([0, 1, 0])
    # v3 = np.asarray([0, 0, 1])

    # u2 = v2 - np.dot(u1, v2) * u1 / np.linalg.norm(u1) ** 2

    # u3 = (
    #     v3
    #     - np.dot(u1, v3) * u1 / np.linalg.norm(u1) ** 2
    #     - np.dot(u2, v3) * u2 / np.linalg.norm(u2) ** 2
    # )

    # return u1, u2, u3

    if np.sum(u1) != 1:
        u1 = u1 / np.linalg.norm(u1)

    # new simple

    # if np.all(np.abs(np.abs(u1) - np.asarray([0, 0, 1])) < 1e-3):  # kobs=x

    #     u3 = np.asarray([0.0, 0.0, 1.0]) * np.sign(u1)
    #     u2 = np.asarray([0.0, 1.0, 0.0])
    #     u1 = np.asarray([1.0, 0.0, 0.0])

    if np.sum(np.abs(u1) < 1e-3) == 2:  # if two null axes

        # print(u1)

        # find non nul axis
        nn_ax = np.where(np.abs(u1) > 1e-3)[0][0]
        u1 = np.zeros(3)
        u1[nn_ax] = 1.0

        u2 = np.zeros(3)
        u2[(nn_ax + 1) % 3] = 1.0

        u3 = np.zeros(3)
        u3[(nn_ax + 2) % 3] = 1.0

        # print(u1, u2, u3)

    else:

        u2 = np.cross(u1, np.asarray([0.0, 0.0, 1.0]))
        if np.all(u2 == np.zeros(3)):
            u2 += 1.0 / 3
        # print(u2)
        u2 = u2 / np.linalg.norm(u2)

        u3 = np.cross(u1, u2)
        if np.all(u3 == np.zeros(3)):
            u3 += 1.0 / 3
        # print(u3)
        u3 = u3 / np.linalg.norm(u3)

    return u1, u2, u3


def CIC_parts_2D(
    part_pos, part_weights, nb_img_bins=None, xbins=None, ybins=None, direction=2
):
    """
    if no bins are given then they are created to span the whole range of positions
    with nb_img_bins bins
    """

    assert nb_img_bins != None or (np.all(xbins != None) and np.all(ybins != None))

    part_pos = part_pos.copy()

    if type(direction) == int:
        if direction == 1:
            dim1 = 0
            dim2 = 2
        elif direction == 0:
            dim1 = 1
            dim2 = 2
        elif direction == 2:
            dim1 = 0
            dim2 = 1
        else:
            raise ValueError("direction must be 0, 1 or 2")
    else:  # projection

        assert len(direction) == 3, "direction vector must be of length 3"

        dv1, dv2, dv3 = basis_from_vect(direction)

        part_pos = np.asarray(
            [
                np.dot(dv2, part_pos.T),
                np.dot(dv3, part_pos.T),
            ]
        ).T

        dim1 = 0
        dim2 = 1

    if np.any(xbins == None):
        x0 = part_pos[:, dim1].min()
        x1 = part_pos[:, dim1].max()
        xbins = np.linspace(x0, x1, nb_img_bins + 1)
    if np.any(ybins == None):
        y0 = part_pos[:, dim2].min()
        y1 = part_pos[:, dim2].max()
        ybins = np.linspace(y0, y1, nb_img_bins + 1)
    else:
        nb_img_bins = len(ybins) - 1

    img = np.zeros((nb_img_bins, nb_img_bins), np.float32)

    xdist_to_ctr = part_pos[:, dim1] // 1 + 0.5 - part_pos[:, dim1]
    ydist_to_ctr = part_pos[:, dim2] // 1 + 0.5 - part_pos[:, dim2]

    int_arg = 0

    for xdelta in [-1, 0, 1]:
        for ydelta in [-1, 0, 1]:

            cur_x = part_pos[:, dim1] + xdelta
            cur_y = part_pos[:, dim2] + ydelta

            cur_xdist = np.abs(xdelta + xdist_to_ctr)
            cur_ydist = np.abs(ydelta + ydist_to_ctr)

            dx = 1 - cur_xdist
            dy = 1 - cur_ydist

            ok = (cur_xdist < 1.0) * (
                cur_ydist < 1.0
            )  # ignore cell centers more than 1 element away
            # from points

            if not ok.any():
                continue

            weights = part_weights * dx * dy

            img += binned_statistic_2d(
                cur_x[ok], cur_y[ok], weights[ok], bins=[xbins, ybins], statistic="sum"
            )[0]

    return (img, xbins, ybins)
