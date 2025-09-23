from pickle import NONE
from astropy.utils import data
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection

import numpy as np
import os
from scipy.spatial import KDTree
from scipy.stats import binned_statistic_2d, f
from f90_tools.star_reader import read_part_ball_NCdust
from hagn.tree_reader import map_tree_steps_bytes
from zoom_analysis.constants import *

from zoom_analysis.dust.gas_reader import code_to_cgs

# from zoom_analysis.rascas.rascas_steps import get_directions_cart

from zoom_analysis.read.read_data import read_data_ball

from itertools import combinations

from scipy.interpolate import CubicSpline


import yt

from gremlin.read_sim_params import ramses_sim

from zoom_analysis.halo_maker.read_treebricks import (
    convert_brick_units,
    read_brickfile,
    read_zoom_brick,
)
from zoom_analysis.halo_maker.assoc_fcts import (
    get_central_gal_for_hid,
    get_gal_props_snap,
    get_halo_props_snap,
)
from zoom_analysis.stars.sfhs import correct_mass, get_sf_stuff
from zoom_analysis.visu.read_amr2cell import *

from f90_tools.hilbert import get_files

from zoom_analysis.sinks.sink_reader import (
    read_sink_bin,
    snap_to_coarse_step,
    convert_sink_units,
)

from compress_zoom.read_compressd import read_compressed_target
from zoom_analysis.zoom_helpers import project_direction


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


def scatter_decomp(d3_points, axs, **kwargs):

    axs[0].scatter(d3_points[:, 0], d3_points[:, 1], **kwargs)
    axs[1].scatter(d3_points[:, 0], d3_points[:, 2], **kwargs)
    axs[2].scatter(d3_points[:, 1], d3_points[:, 2], **kwargs)


def lookup_hydro_idx_for_field(sim, field):

    if "nvar" not in list(sim.hydro.keys()):
        idx = [i for i in range(1, len(sim.hydro) + 1) if sim.hydro[str(i)] == field]
    else:  # NH like
        idx = [i for i in range(1, len(sim.hydro)) if sim.hydro[f"{i:d}"] == field]
    # if len(idx) == 0:
    # print(idx)
    if "temperature" == field and (5 not in idx):
        print("Temperature requested loading density and pressure")
        idx.extend([1, 5])
    elif "DTM" in ["DTM", "DTMC", "DTMSi", "DTMCs", "DTMCl", "DTMSis", "DTMSil"] and (
        1 not in idx
        or 6 not in idx
        or 16 not in idx
        or 17 not in idx
        or 18 not in idx
        or 19 not in idx
    ):
        print("Dust to metal requested loading density and dust bins")
        idx.extend([1, 6, 16, 17, 18, 19])
    elif "density" != field:
        idx.append(1)

    elif "metallicity" == field and (6 not in idx or 1 not in idx):
        print("Metal density requested loading density and metallicity")
        # idx.extend([6])
        idx.extend([1, 6])
    # elif "dust_bin01" == field and (16 not in idx):
    #     print("small carbon fraction requested loading density and small carbon grains")
    #     # idx.extend([16])
    #     idx.extend([1, 16])
    # elif "dust_bin02" == field and (17 not in idx):
    #     print("large carbon fraction requested loading density and large carbon grains")
    #     # idx.extend([17])
    #     idx.extend([1, 17])
    # elif "dust_bin03" == field and (17 not in idx):
    #     print(
    #         "small silicate fraction requested loading density and small silicate grains"
    #     )
    #     # idx.extend([18])
    #     idx.extend([1, 18])
    # elif "dust_bin04" == field and (17 not in idx):
    #     print(
    #         "large silicate fraction requested loading density and large silicate grains"
    #     )
    #     idx.extend([1, 19])
    # elif field != "density":
    #     print(f"No rule for requested field: {field}")

    return np.unique(idx)


def plot_amr_data(
    amrdata,
    zdist,
    aexp,
    lbox_kpc,
    M_basis,
    dx_boost,
    fig,
    ax,
    x,
    y,
    z,
    r,
    field,
    dist_norm,
    **kwargs,
):

    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    debug = kwargs.get("debug", False)
    weights = kwargs.get("weights", "density")
    op = kwargs.get("op", np.sum)
    cb = kwargs.get("cb", False)
    cmap = kwargs.get("cmap", "magma")
    transpose = kwargs.get("transpose", True)
    log = kwargs.get("log", True)
    plot_text = kwargs.get("plot_text", True)
    mode = kwargs.get("mode", "sum")

    cb_args = kwargs.get("cb_args", {})
    cb_loc = cb_args.get("location", "right")
    cb_orientation = cb_args.get("orientation", "vertical")

    cmap = plt.get_cmap(cmap)

    # print(vmin, vmax)
    # print(amrdata)

    all_levels = amrdata["ilevel"]
    loaded_lvlmin = np.min(all_levels)

    if zdist == -1:
        # clamp z to nearest lvlmax cell
        # print(z)
        z = (np.round(z * 2**loaded_lvlmin) + 0.5) / 2**loaded_lvlmin
        zdist = 1.0 / 2**loaded_lvlmin * lbox_kpc

    # zdist = max(zdist, lbox_kpc / 2**loaded_lvlmin)

    cell_pos = np.asarray([amrdata["x"], amrdata["y"], amrdata["z"]]).T

    cell_pos = np.dot(cell_pos.T, M_basis)

    # print(np.median(cell_pos, axis=1))

    for idim in range(3):
        if np.all(cell_pos[idim, :] < 0):
            # cell_pos[idim,:] = np.abs(cell_pos[idim,:])
            cell_pos[idim, :] += 1
        if np.all(cell_pos[idim, :] > 1):
            # cell_pos[idim,:] = np.abs(cell_pos[idim,:])
            cell_pos[idim, :] -= 1

    if not transpose:
        ycells, xcells, zcells = cell_pos

        a = y.copy()
        y = x.copy()
        x = a

    else:

        xcells, ycells, zcells = cell_pos

    zfilt = np.abs(zcells - z) < (zdist / lbox_kpc)

    # print(zdist, zdist / lbox_kpc)
    # print("zfilt", zfilt.sum())

    xcells = xcells[zfilt]
    ycells = ycells[zfilt]
    zcells = zcells[zfilt]

    # print(np.min([xcells, ycells]))

    tree = KDTree(np.vstack((xcells, ycells)).T, boxsize=1 + 1e-6)

    inds = tree.query_ball_point([x, y], r=r, p=dist_norm)

    values = np.float64(
        amrdata[field][zfilt][inds]
    )  #! attention aux float 64/32 putain ici c'est IMPORTANT
    if weights == "density":
        densities = np.float64(
            amrdata["density"][zfilt][inds]
        )  #! attention aux float 64/32 putain ici c'est IMPORTANT
    levels = all_levels[zfilt][inds]

    u_lvls = np.unique(levels)
    xcells, ycells = xcells[inds], ycells[inds]

    # zorder = 0
    if len(u_lvls) == 0:
        print("No cells found")
        return

    lvl_maps = []

    for lvl in np.sort(u_lvls[:][::-1]):  # desc order

        this_lvl_args = levels == lvl

        this_lvl_x = xcells[this_lvl_args]
        this_lvl_y = ycells[this_lvl_args]

        dx = 1.0 / 2**lvl * dx_boost

        # segment the cells into boxes... avoid high res voids
        boxes = segment_2d_point_cloud(this_lvl_x, this_lvl_y, dx)

        lvl_maps.append([lvl, boxes])

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

            box_lvls = levels[cond]
            box_values = values[cond]
            box_dxs = 1.0 / 2**box_lvls
            box_xcells = xcells[cond]
            box_ycells = ycells[cond]

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

            lvl_imgs.append(img)

        lvl_maps[ilvl].append(lvl_imgs)

    # print(lvl_maps)

    all_maps = np.concatenate(
        [np.ravel(elem3) for elem in lvl_maps for elem2 in elem[-1] for elem3 in elem2]
    )

    all_maps = all_maps[np.isfinite(all_maps) * (all_maps > 0)]

    if vmin == None or vmax == None:
        cdf, cdf_bins = np.histogram(np.log10(all_maps), bins=100)
        cdf = np.cumsum(cdf)
        cdf = cdf / cdf[-1]
        if vmax == None:
            vmax = 10 ** (cdf_bins[np.where(cdf > 0.99)[0][0]])
            print(f"set vmax = {vmax}")

        if vmin == None:
            vmin = 10 ** (cdf_bins[np.where(cdf > 0.01)[0][0]])
            print(f"set vmin = {vmin}")

    for ilvl, (lvl, boxes, imgs) in enumerate(lvl_maps):

        for box, img in zip(boxes, imgs):

            xmin, xmax = box[0], box[2]
            ymin, ymax = box[1], box[3]

            xmin_cells = xmin * 2**lvl
            xmax_cells = xmax * 2**lvl
            ymin_cells = ymin * 2**lvl
            ymax_cells = ymax * 2**lvl

            xbins = np.arange(max(xmin_cells - 0.5 - 1, 0), xmax_cells + 0.5 + 2, 1)
            ybins = np.arange(max(ymin_cells - 0.5 - 1, 0), ymax_cells + 0.5 + 2, 1)

            # print(img.mean())

            img[img < vmin] = vmin
            # img[img > vmax] = vmax
            img[np.isfinite(img) == False] = vmin

            # print(np.sum(np.isnan(img)), np.sum(np.isinf(img)))
            # print(vmin, vmax, img.min(), img.max())

            if log:
                if vmin > 0:
                    norm = LogNorm(vmin=vmin, vmax=vmax)
                else:
                    norm = SymLogNorm(vmin=vmin, vmax=vmax, linthresh=vmax / 1e2)
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
                    norm=norm,
                    # vmin=vmin,
                    # vmax=vmax,
                    cmap=cmap,
                    # zorder=ilvl,
                )
            else:
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
                    vmin=vmin,
                    vmax=vmax,
                    # vmin=vmin,
                    # vmax=vmax,
                    cmap=cmap,
                    # zorder=ilvl,
                )

    if cb:

        if mode == "sum":
            len_dim = "$cm^{-2}$"
        else:
            len_dim = "$cm^{-3}$"

        if "label" in cb_args:
            cb_label = cb_args["label"]
        elif field == "density":
            cb_label = r"Density, " + len_dim
        elif field == "temperature":
            cb_label = f"Temperature, $[K]$"
        elif field == "DTM":
            cb_label = f"Dust to Metal ratio"
        elif field == "DTMC":
            cb_label = f"Carbon Dust to Metal ratio"
        elif field == "DTMCs":
            cb_label = f"Small carbon Dust to Metal ratio"
        elif field == "DTMCl":
            cb_label = f"Large carbon Dust to Metal ratio"
        elif field == "DTMSi":
            cb_label = f"Sillicate Dust to Metal ratio"
        elif field == "DTMSis":
            cb_label = f"Small sillicate Dust to Metal ratio"
        elif field == "DTMSil":
            cb_label = f"Large sillicate Dust to Metal ratio"
        elif field == "velocity":
            cb_label = f"Velocity, $[km/s]$"
        else:
            cb_label = field

        # colorbar at right of imshow, fitted nicely to width of imshow
        cbar = fig.colorbar(
            img,
            ax=ax,
            orientation=cb_orientation,
            location=cb_loc,
            # orientation="vertical",
            # location="right",
            fraction=0.046,
            pad=0.04,
        )
        cbar.set_label(cb_label)

    ax.set_xlabel("x, [ckpc]")
    ax.set_ylabel("y, [ckpc]")

    zed = 1 / aexp - 1

    ax.set_xlim(x - r * lbox_kpc, x + r * lbox_kpc)
    ax.set_ylim(y - r * lbox_kpc, y + r * lbox_kpc)

    if plot_text:
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

    return lvl_maps


def make_amr_img_smooth(
    fig,
    ax,
    field,
    snap,
    sim: ramses_sim,
    pos,
    rvir,
    zdist=-1,
    direction=[0, 0, 1],
    NH_read=False,
    **kwargs,
):

    mins = kwargs.get("mins", None)
    maxs = kwargs.get("maxs", None)

    hid = kwargs.get("hid", None)

    lbox_kpc = sim.unit_l(sim.aexp_stt) / (ramses_pc * 1e6) / sim.aexp_stt * 1e3  # ckpc

    # zdist = 100
    r = rvir
    mins = pos - r
    maxs = pos + r

    mins[2] = pos[2] - zdist
    maxs[2] = pos[2] + zdist
    # r = 0.1 * r

    # print(pos)

    (
        pos_proj,
        mins_proj,
        maxs_proj,
        cartesian,
        M_basis,
        M_basis_inv,
        dx_boost,
        dist_norm,
    ) = proj_math(sim, pos, zdist, mins, maxs, direction, r)

    x, y, z = pos
    x_proj, y_proj, z_proj = pos_proj

    for snap in [snap]:
        print(f"Reading amr2cell for snap {snap}...")

        # could speed up by segmenting the regions where we plot...
        aexp = sim.get_snap_exps(snap)
        mins = np.dot(mins, M_basis_inv)
        maxs = np.dot(maxs, M_basis_inv)

        mins[mins < 0] += 1
        maxs[maxs < 0] += 1
        mins[mins > 1] -= 1
        maxs[maxs > 1] -= 1

        ext_bool = mins > maxs
        a = np.copy(mins[ext_bool])
        b = np.copy(maxs[ext_bool])
        mins[ext_bool] = b
        maxs[ext_bool] = a

        if not cartesian:
            # messy and requires more data but works
            # if you're at a weird angle you need to expand your
            # data range to fill the projected image square
            mins = mins * 0.99
            maxs = maxs * 1.01

        if NH_read:
            # find index of requested fields, add density for projection if necessary
            idx = lookup_hydro_idx_for_field(sim, field)
            # print(idx)
            amrdata = read_amrcells(sim, snap, np.ravel([mins, maxs]), idx)

            if field not in amrdata:
                amrdata[field] = []

            if len(amrdata["ilevel"]) == 0:
                print("No cells found")
                return

            # convert units
            amrdata = code_to_cgs(sim, aexp, amrdata)

            # print(tgt_fields)
        else:

            amrdata = read_data_ball(
                sim,
                snap,
                pos,
                (r + 1.0 / 2**sim.levelmin) * np.sqrt(3),
                hid,
                data_types=["gas"],
                tgt_fields=field,
                minmax=[mins, maxs],
            )

        subtract_mean = kwargs.get("subtract_mean", False)
        # project data if needed
        # e.g. 3D velocity becomes 2D norm in img plane directions
        # possibly with bulk vel removed
        if "velocity" in field:
            tmp_vel = (
                np.transpose(
                    [
                        amrdata["velocity_x"],
                        amrdata["velocity_y"],
                        amrdata["velocity_z"],
                    ]
                )
                / 1e5
            )  # km/s
            cell_pos = np.transpose([amrdata["x"], amrdata["y"], amrdata["z"]])

            vel_proj = np.dot(tmp_vel.T, M_basis)
            cell_pos_proj = np.dot(cell_pos.T, M_basis)

            vel_plane = vel_proj[:2, :]
            cell_pos_plane = cell_pos_proj[:2, :]

            vel_norm = np.linalg.norm(vel_plane, axis=0)

            if subtract_mean:
                vel_plane = vel_plane - np.mean(vel_plane, axis=1)[:, None]

                # #proj direction is always last axis
                # vel_norm = np.linalg.norm(vel_plane, axis=0)

                # #if sign of position and velocity is the same, then it's an ouflow
                # inflow = np.all(np.sign(vel_plane)==np.sign(cell_pos_plane),axis=0)
                # outflow = np.all(-np.sign(vel_plane)==np.sign(cell_pos_plane),axis=0)
                # #outflow is positive, inflow is negative

                # sign = np.zeros_like(vel_norm)
                # sign[inflow] = -1
                # sign[outflow] = 1

                directions = (cell_pos_plane - pos_proj[:2, None]) / np.linalg.norm(
                    cell_pos_plane - pos_proj[:2, None], axis=0
                )

                dot_prod = np.sum(vel_plane * directions, axis=0)

                # vel_norm *= sign
                vel_norm = dot_prod

            # print(f"vel_norm : {vel_norm.min()} {vel_norm.max()}")

            amrdata["velocity"] = vel_norm

        else:
            if subtract_mean:
                amrdata[field] = amrdata[field] - np.mean(amrdata[field])

        img = plot_amr_data(
            amrdata,
            zdist,
            aexp,
            lbox_kpc,
            M_basis,
            dx_boost,
            fig,
            ax,
            x_proj,
            y_proj,
            z_proj,
            r,
            field,
            dist_norm,
            **kwargs,
        )


def make_amr_img_halogal(
    fig,
    ax,
    field,
    snap,
    sim: ramses_sim,
    rfact,
    hid,
    gid=None,
    zdist=-1,
    direction=[0, 0, 1],
    **kwargs,
):
    """
    if gid is None, load halo data, otherwise load galaxy data
    """

    mins = None
    maxs = None

    if mins in kwargs:
        mins = kwargs["mins"]
    if maxs in kwargs:
        maxs = kwargs["maxs"]

    lbox_kpc = sim.unit_l(sim.aexp_stt) / (ramses_pc * 1e6) / sim.aexp_stt * 1e3  # ckpc

    # sim_dir = sim.path

    print("reading amr2cell files...")

    # cmap to bad pixels are black
    plt.rcParams["image.cmap"] = "magma"
    cmap = plt.get_cmap()
    cmap.set_bad(color=(0, 0, 0, 0))

    # zdist = 100
    r = rfact
    # r = 0.1 * r

    if gid is None:
        catalog_assoc = get_halo_props_snap(sim.path, snap, hid)[0]
    else:
        catalog_assoc = get_gal_props_snap(sim.path, snap, gid)[1]
        if r > catalog_assoc["rmax"]:
            r = catalog_assoc["rmax"]
            print(f"clipping r to rmax : {r:f}")
    pos = catalog_assoc["pos"]

    zdist = r / 1 * sim.cosmo.lcMpc * 1e3

    pos, mins, maxs, cartesian, M_basis, M_basis_inv, dx_boost, dist_norm = proj_math(
        sim, pos, zdist, mins, maxs, direction, r
    )

    x, y, z = pos

    for snap in [snap]:
        print(f"Reading amr2cell for snap {snap}...")

        # could speed up by segmenting the regions where we plot...
        aexp = sim.get_snap_exps(snap)
        mins = np.dot(mins, M_basis_inv)
        maxs = np.dot(maxs, M_basis_inv)

        mins[mins < 0] += 1
        maxs[maxs < 0] += 1
        mins[mins > 1] -= 1
        maxs[maxs > 1] -= 1

        ext_bool = mins > maxs
        a = np.copy(mins[ext_bool])
        b = np.copy(maxs[ext_bool])
        mins[ext_bool] = b
        maxs[ext_bool] = a

        if not cartesian:
            # messy and requires more data but works
            # if you're at a weird angle you need to expand your
            # data range to fill the projected image square
            mins = mins * 0.99
            maxs = maxs * 1.01

        amrdata = read_compressed_target(sim, snap, gid=gid, hid=hid, data_type="gas")

        # print(np.unique(amrdata["ilevel"]))

        if len(amrdata["ilevel"]) == 0:
            print("No cells found")
            return

        plot_amr_data(
            amrdata,
            zdist,
            aexp,
            lbox_kpc,
            M_basis,
            dx_boost,
            fig,
            ax,
            x,
            y,
            z,
            r,
            field,
            dist_norm,
            **kwargs,
        )


def proj_math(sim, pos, zdist, mins, maxs, direction, r):
    dv1 = np.array(direction)
    dv1, dv2, dv3 = basis_from_vect(dv1)

    cartesian = True

    if not np.all(
        [np.sum(dv1) == 1, np.sum(dv2) == 1, np.sum(dv3) == 1]
    ):  # not cartesian x,y,z basis... treat cells as spheres
        # dist_norm = 2
        dist_norm = np.inf
        dx_boost = np.sqrt(3) * 1.5
        cartesian = False
    else:
        dist_norm = np.inf
        dx_boost = 1

    # M_basis = [dv3, dv2, dv1]
    M_basis = project_direction(dv1, [0, 0, 1])
    M_basis_inv = project_direction([0, 0, 1], dv1)

    pos = np.dot(pos, M_basis)

    if np.any(mins == None):
        mins = pos - r
    if np.any(maxs == None):
        maxs = pos + r

    if zdist != -1:
        # print(pos[2], zdist)
        mins[2] = pos[2] - zdist / sim.cosmo.lcMpc / 1e3
        maxs[2] = pos[2] + zdist / sim.cosmo.lcMpc / 1e3

    pos[pos < 0] += 1
    pos[pos > 1] -= 1

    # M_basis_inv = np.linalg.inv(M_basis)

    return pos, mins, maxs, cartesian, M_basis, M_basis_inv, dx_boost, dist_norm


def plot_zoom_BHs(
    ax,
    snap,
    sim,
    tgt_pos,
    tgt_rad,
    zdist,
    # legend=True,
    # color="grey",
    **kwargs,
):

    legend = kwargs.get("legend", True)
    color = kwargs.get("color", "white")
    txt_color = kwargs.get("color", "white")
    direction = kwargs.get("direction", [0, 0, 1])
    sink_read_fct = kwargs.get("sink_read_fct", read_sink_bin)
    transpose = kwargs.get("transpose", True)
    annotate = kwargs.get("annotate", False)

    max_mbh = kwargs.get("max_mass", None)
    min_mbh = kwargs.get("min_mass", None)

    # cmap = plt.get_cmap()
    # cmap.set_bad(color=(0, 0, 0, 0))

    sink_files = np.asarray(os.listdir(sim.sink_path))
    sink_fnbs = np.asarray([int(f.split("_")[-1].split(".")[0]) for f in sink_files])

    step, found = snap_to_coarse_step(snap, sim)
    if not found:
        return 0
    sink_f = sink_files[step == sink_fnbs][0]

    bhs = sink_read_fct(os.path.join(sim.sink_path, sink_f))
    aexp = sim.get_snap_exps(snap, param_save=False)
    convert_sink_units(bhs, aexp, sim)

    bh_pos = np.asarray(bhs["position"], dtype=np.float64)

    dv1 = np.array(direction)
    dv1, dv2, dv3 = basis_from_vect(dv1)
    # M_basis = [dv3, dv2, dv1]

    M_basis = project_direction(dv1, [0, 0, 1])

    pos = np.asarray([np.dot(bhp - tgt_pos, M_basis) for bhp in bh_pos])

    # print(pos.T)

    # rot_tgt_pos = np.dot(tgt_pos, M_basis).T

    bhxs, bhys, bhzs = pos.T
    bhms = bhs["mass"]

    sids = bhs["identity"]

    # print(bhms.min(), bhms.mean(), bhms.max())

    # print(list(zip(sids, np.log10(bhms))))

    # in frame
    bhxs = (bhxs) * sim.cosmo.lcMpc * 1e3
    bhys = (bhys) * sim.cosmo.lcMpc * 1e3
    bhzs = (bhzs) * sim.cosmo.lcMpc * 1e3

    in_frame = (
        np.linalg.norm(np.asarray([bhxs, bhys]), axis=0)
        < (tgt_rad) * sim.cosmo.lcMpc * 1e3 * 2**0.5
    ) * (np.abs(bhzs) < zdist)

    if not in_frame.any():
        print("No BHs in frame")
        return 0

    if "smbh_params" in sim.namelist:
        mseed = sim.namelist["smbh_params"]["mseed"]
    elif "physics_params" in sim.namelist:
        mseed = sim.namelist["physics_params"]["mseed"]

    if max_mbh == None:
        max_mbh = bhms[in_frame].max()
    if min_mbh == None:
        min_mbh = bhms[in_frame].min()
    max_size = 2e2

    if max_mbh > min_mbh:
        sizes = (
            (np.log10(bhms[in_frame]) - np.log10(min_mbh))
            / (np.log10(max_mbh) - np.log10(min_mbh))
            * max_size
        )
    else:
        sizes = np.full(in_frame.sum(), max_size)

    mseed_size = (
        (np.log10(mseed) - np.log10(min_mbh))
        / (np.log10(max_mbh) - np.log10(min_mbh))
        * max_size
    )

    # sizes[sizes < 0.1 * max_size] = 0.1 * max_size
    sizes[sizes < mseed_size] = mseed_size

    if not transpose:
        bhxs, bhys = bhys, bhxs

    ax.scatter(
        bhxs[in_frame],
        bhys[in_frame],
        s=sizes,
        edgecolors=color,
        facecolors="none",
        alpha=1.0,
        label="BHs",
        zorder=999,
        lw=1.0,
    )

    # make a legend for sizes
    if np.abs(max_mbh - mseed) / mseed > 0.1:
        biggest = max_size
        # med = max_size * ((np.median(bhms)) / (max_mbh))
        smallest = max(mseed_size, max_size * 0.1)
        med = max(max_size * 0.5, (biggest - smallest) / 2 + smallest)
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
                edgecolors=color,
                facecolors="none",
                alpha=1.0,
                lw=1.0,
            )
        )

    # print(biggest, med, smallest)
    # print(max_mbh, )
    if legend:
        bh_leg = ax.legend(
            markers,
            [
                f"{max_mbh:.1e} M$_\odot$",
                f"{max(med*max_mbh/max_size,mseed):.1e} M$_\odot$",
                f"{max(smallest*max_mbh/max_size,mseed):.1e} M$_\odot$",
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
            labelcolor=txt_color,
        )
        plt.setp(bh_leg.get_title(), color=txt_color)

    if annotate:

        for i, (sid, x, y, m) in enumerate(
            zip(sids[in_frame], bhxs[in_frame], bhys[in_frame], bhms[in_frame])
        ):
            ax.annotate(
                f"{sid:d}",
                (x, y),
                fontsize=12,
                zorder=999,
                color=txt_color,
                path_effects=[pe.withStroke(linewidth=1, foreground="black")],
            )
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


def plot_brick_scatter(
    ax,
    snap,
    sim,
    tgt_pos,
    zdist,
    bricks,
    obj_type,
    markers,
    leg_pos,
    props_fct,
    **kwargs,
):

    color = kwargs.get("color", "white")
    txt_color = kwargs.get("color", color)
    legend = kwargs.get("legend", True)
    annotate = kwargs.get("annotate", True)
    direction = kwargs.get("direction", [0, 0, 1])
    gal_markers = kwargs.get("gal_markers", True)
    transpose = kwargs.get("transpose", True)

    max_mass = kwargs.get("max_mass", None)
    min_mass = kwargs.get("min_mass", None)

    mains = bricks["hosting info"]["hlvl"] == 1
    subs = bricks["hosting info"]["hlvl"] == 2

    stmass = bricks["hosting info"]["hmass"]

    gids = bricks["hosting info"]["hid"]
    x = bricks["positions"]["x"]
    y = bricks["positions"]["y"]
    z = bricks["positions"]["z"]

    gal_pos = np.asarray([x, y, z], dtype=np.float64).T

    # print("gal_pos",np.max(gal_pos, axis=0), np.min(gal_pos, axis=0), np.median(gal_pos, axis=0))

    # direction = [1, 0, 0]
    # direction = [0, 1, 0]

    dv1 = np.array(direction)
    dv1, dv2, dv3 = basis_from_vect(dv1)
    # M_basis = [dv3, dv2, dv1]
    # print(dv1, [0, 0, 1])
    M_basis = project_direction(dv1, [0, 0, 1])

    # print(direction,M_basis)

    # print(direction)

    # print(M_basis, gal_pos, tgt_pos)

    # print(np.dot([1, 0, 0], M_basis))
    # print(np.dot([0, 1, 0], M_basis))
    # print(np.dot([0, 0, 1], M_basis))

    # gal_pos = np.dot(M_basis, gal_pos - tgt_pos).T

    gal_pos = np.asarray([np.dot(gp - tgt_pos, M_basis) for gp in gal_pos])

    x, y, z = gal_pos.T

    # rot_tgt_pos = np.dot(tgt_pos, M_basis).T
    # print(z.min(), z.max(), np.mean(z), np.median(z))
    # print(
    #     np.max(z * sim.cosmo.lcMpc * 1e3),
    #     np.min(z * sim.cosmo.lcMpc * 1e3),
    #     zdist,
    # )

    # in_plane = (np.abs(z) * sim.cosmo.lcMpc * 1e3) < 100
    # in_plane = (np.abs(z) * sim.cosmo.lcMpc * 1e3) < zdist
    in_plane = (
        # np.linalg.norm(np.asarray([x, y]), axis=0)
        # < (tgt_rad) * sim.cosmo.lcMpc * 1e3 * 2**0.5*(
        (np.abs(z) * sim.cosmo.lcMpc * 1e3 * 2**0.5)
        < zdist
    )

    if not in_plane.any():
        return 0

    max_mass_size = 3e2
    if max_mass == None:
        max_mass = stmass[in_plane].max()
    if min_mass == None:
        min_mass = stmass[in_plane].min()
    med_mass = 10 ** (0.5 * (np.log10(max_mass) + np.log10(min_mass)))

    if max_mass > min_mass:
        sizes = (
            (np.log10(stmass) - np.log10(min_mass))
            / (np.log10(max_mass) - np.log10(min_mass))
            * max_mass_size
        )
        markers_sizes = [
            max_mass_size,
            # np.log10(med_mass) / np.log10(max_mass) * max_mass_size,
            # np.log10(min_mass) / np.log10(max_mass) * max_mass_size,
            0.5 * max_mass_size,
            0.1 * max_mass_size,
        ]
    else:
        sizes = np.full(len(stmass), max_mass_size)
        markers_sizes = [max_mass_size, max_mass_size, max_mass_size]

    # sizes[sizes < 0.1 * max_mass_size] = 0.1 * max_mass_size

    mains_to_plot = mains * in_plane
    subs_to_plot = subs * in_plane

    if not transpose:
        y, x = x, y

    if gal_markers:
        # plot mains

        if markers[0] in ["x"]:
            color_args = {"color": color}
        else:
            color_args = {"edgecolors": color, "facecolors": "none"}

        ax.scatter(
            (x[mains_to_plot]) * sim.cosmo.lcMpc * 1e3,
            (y[mains_to_plot]) * sim.cosmo.lcMpc * 1e3,
            s=sizes[mains * in_plane],
            **color_args,
            marker=markers[0],
            alpha=1.0,
            lw=1.0,
            zorder=999,
        )

        if markers[1] in ["x"]:
            color_args = {"color": color}
        else:
            color_args = {"edgecolors": color, "facecolors": "none"}

        # plot subs
        ax.scatter(
            (x[subs_to_plot]) * sim.cosmo.lcMpc * 1e3,
            (y[subs_to_plot]) * sim.cosmo.lcMpc * 1e3,
            s=sizes[subs_to_plot * in_plane],
            **color_args,
            marker=markers[1],
            alpha=1.0,
            lw=0.5,
            zorder=999,
        )

        marker_sizes = []

        if markers[0] in ["x"]:
            color_args = {"color": color}
        else:
            color_args = {"edgecolors": color, "facecolors": "none"}

        for marker_size in markers_sizes:
            marker_sizes.append(
                ax.scatter(
                    [],
                    [],
                    s=marker_size,
                    **color_args,
                    marker=markers[0],
                    alpha=1.0,
                    lw=0.5,
                )
            )

        # legend for the mass of galaxies
        if legend:
            gal_mass_leg = ax.legend(
                marker_sizes,
                [
                    f"{max_mass:.1e} M$_\odot$",
                    f"{med_mass:.1e} M$_\odot$",
                    f"{min_mass:.1e} M$_\odot$",
                ],
                loc=leg_pos,
                # fontsize=12,
                title=f"{obj_type} mass scale",
                # title_fontsize="16",
                framealpha=0.0,
                prop=dict(
                    size=12,
                ),
                title_fontproperties=dict(
                    size=16,
                ),
                labelcolor=txt_color,
            )
            plt.setp(gal_mass_leg.get_title(), color=txt_color)
            ax.add_artist(gal_mass_leg)

            type_markers = []
            for marker in markers:

                if marker in ["x"]:
                    color_args = {"color": color}
                else:
                    color_args = {"edgecolors": color, "facecolors": "none"}

                if marker != markers[0]:
                    type_markers.append(
                        ax.scatter(
                            [],
                            [],
                            s=max_mass_size,
                            **color_args,
                            marker=marker,
                            alpha=1.0,
                            lw=1.0,
                        )
                    )
                else:
                    type_markers.append(
                        ax.scatter(
                            [],
                            [],
                            s=max_mass_size,
                            **color_args,
                            marker=marker,
                            alpha=1.0,
                            lw=1.0,
                        )
                    )

            # legend for type of galaxy markers
            if "center" in leg_pos:

                type_leg_pos = leg_pos.replace("center", "right")
                gal_lvl_leg = ax.legend(
                    type_markers,
                    ["Central", "Sub"],
                    loc=type_leg_pos,
                    # fontsize=12,,
                    # title_fontsize="16",
                    framealpha=0.0,
                    prop=dict(
                        size=12,
                    ),
                    labelcolor=txt_color,
                )
                ax.add_artist(gal_lvl_leg)

    if annotate:
        # annotate makers with ids
        for i, (gid, x, y) in enumerate(
            zip(gids[mains_to_plot], x[mains_to_plot], y[mains_to_plot])
        ):

            try:
                gal_dict = props_fct(sim.path, snap, gid)

            except FileNotFoundError:
                pass

            annotate_txt = f"{gid}"

            ax.annotate(
                annotate_txt,
                (
                    (x) * sim.cosmo.lcMpc * 1e3,
                    (y) * sim.cosmo.lcMpc * 1e3,
                ),
                fontsize=12,
                zorder=999,
                color=txt_color,
                path_effects=[pe.withStroke(linewidth=1, foreground="black")],
            )


def plot_zoom_gals(
    ax,
    snap,
    sim: ramses_sim,
    tgt_pos,
    tgt_rad,
    zdist,
    **kwargs,
):

    brick_fct = kwargs.get("brick_fct", read_zoom_brick)
    hm = kwargs.get("hm", None)

    if hm:
        bricks = brick_fct(snap, sim, hm)
    else:
        bricks = brick_fct(snap, sim)

    if bricks == 0:
        return 0

    markers = ["x", "v"]
    obj_type = "Galaxy"
    leg_pos = "lower center"
    props_fct = get_gal_props_snap

    # print(bricks)

    # print(kwargs)

    plot_brick_scatter(
        ax,
        snap,
        sim,
        tgt_pos,
        zdist,
        bricks,
        obj_type,
        markers,
        leg_pos,
        props_fct,
        **kwargs,
    )

    return 1


def plot_zoom_halos(
    ax,
    snap,
    sim: ramses_sim,
    tgt_pos,
    tgt_rad,
    zdist,
    **kwargs,
):

    brick_fct = kwargs.get("brick_fct", read_zoom_brick)
    hm = kwargs.get("hm", None)

    if hm:
        bricks = brick_fct(snap, sim, hm, galaxy=False, star=False)
    else:
        bricks = brick_fct(snap, sim, galaxy=False, star=False)

    if bricks == 0:
        return 0

    markers = ["s", "D"]
    obj_type = "Halo"
    leg_pos = "upper center"
    props_fct = get_halo_props_snap

    # print(bricks)

    plot_brick_scatter(
        ax,
        snap,
        sim,
        tgt_pos,
        zdist,
        # np.inf,
        bricks,
        obj_type,
        markers,
        leg_pos,
        props_fct,
        **kwargs,
    )

    return 1


# def plot_zoom_halos(ax, snap, sim: ramses_sim, tgt_pos, tgt_rad, zdist, **kwargs):

#     hm = kwargs.get("hm", None)
#     brick_fct = kwargs.get("brick_fct", read_zoom_brick)
#     halo_markers = kwargs.get("halo_markers", True)
#     annotate = kwargs.get("annotate", True)
#     direction = kwargs.get("direction", [0, 0, 1])
#     legend = kwargs.get("legend", True)
#     color = kwargs.get("color", "white")
#     txt_color = kwargs.get("color", color)
#     transpose = kwargs.get("transpose", True)


#     if hm:
#         bricks = brick_fct(snap, sim, hm, star=True, galaxy=True)
#     else:
#         bricks = brick_fct(snap, sim, star=False, galaxy=False)

#     if bricks == 0:
#         return 0

#     # print(bricks)

#     mains = bricks["hosting info"]["hlvl"] == 1
#     subs = bricks["hosting info"]["hlvl"] == 2

#     stmass = bricks["hosting info"]["hmass"]

#     hids = bricks["hosting info"]["hid"]
#     x = bricks["positions"]["x"]
#     y = bricks["positions"]["y"]
#     z = bricks["positions"]["z"]

#     halo_pos = np.asarray([x, y, z],dtype=np.float64).T

#     # print("halo_pos",np.max(halo_pos, axis=0), np.min(halo_pos, axis=0), np.median(halo_pos, axis=0))

#     # direction = [1, 0, 0]
#     # direction = [0, 1, 0]

#     dv1 = np.array(direction)
#     dv1, dv2, dv3 = basis_from_vect(dv1)
#     M_basis = [dv3, dv2, dv1]

#     # print(M_basis)

#     # print(direction)

#     # print(np.dot([1, 0, 0], M_basis))
#     # print(np.dot([0, 1, 0], M_basis))
#     # print(np.dot([0, 0, 1], M_basis))

#     halo_pos = np.dot(halo_pos-tgt_pos, M_basis).T

#     x, y, z = halo_pos

#     # rot_tgt_pos = np.dot(tgt_pos, M_basis).T

#     in_plane = (np.abs(z) * sim.cosmo.lcMpc * 1e3) < zdist

#     if not in_plane.any():
#         return 0

#     max_mass_size = 3e2
#     max_mass = stmass[in_plane].max()
#     min_mass = stmass[in_plane].min()
#     med_mass = 10 ** (0.5 * (np.log10(max_mass) + np.log10(min_mass)))

#     if max_mass > min_mass:
#         sizes = (
#             (np.log10(stmass) - np.log10(min_mass))
#             / (np.log10(max_mass) - np.log10(min_mass))
#             * max_mass_size
#         )
#         markers_sizes = [
#             max_mass_size,
#             # np.log10(med_mass) / np.log10(max_mass) * max_mass_size,
#             # np.log10(min_mass) / np.log10(max_mass) * max_mass_size,
#             0.5 * max_mass_size,
#             0.1 * max_mass_size,
#         ]
#     else:
#         sizes = np.full(len(stmass), max_mass_size)
#         markers_sizes = [max_mass_size, max_mass_size, max_mass_size]

#     sizes[sizes < 0.1 * max_mass_size] = 0.1 * max_mass_size

#     mains_to_plot = mains * in_plane
#     subs_to_plot = subs * in_plane

#     if not transpose:
#         y, x = x, y
#         #rot_tgt_pos[0], rot_tgt_pos[1] = rot_tgt_pos[1], rot_tgt_pos[0]

#     if halo_markers:
#         # plot mains
#         ax.scatter(
#             (x[mains_to_plot]) * sim.cosmo.lcMpc * 1e3,
#             (y[mains_to_plot]) * sim.cosmo.lcMpc * 1e3,
#             s=sizes[mains * in_plane],
#             edgecolors=color,
#             facecolors="none",
#             marker="s",
#             alpha=1.0,
#             lw=1.0,
#             zorder=999,
#         )

#         # plot subs
#         ax.scatter(
#             (x[subs_to_plot]) * sim.cosmo.lcMpc * 1e3,
#             (y[subs_to_plot]) * sim.cosmo.lcMpc * 1e3,
#             s=sizes[subs_to_plot* in_plane],
#             edgecolors=color,
#             facecolors="none",
#             marker="D",
#             alpha=1.0,
#             lw=0.5,
#             zorder=999,
#         )

#         markers = ["s", "D"]

#         marker_sizes = []

#         for marker_size in markers_sizes:
#             marker_sizes.append(
#                 ax.scatter([], [], s=marker_size, c="w", marker="x", alpha=1.0, lw=0.5)
#             )

#         # legend for the mass of galaxies
#         if legend:
#             gal_mass_leg = ax.legend(
#                 marker_sizes,
#                 [
#                     f"{max_mass:.1e} M$_\odot$",
#                     f"{med_mass:.1e} M$_\odot$",
#                     f"{min_mass:.1e} M$_\odot$",
#                 ],
#                 loc="upper center",
#                 # fontsize=12,
#                 title="Halo mass scale",
#                 # title_fontsize="16",
#                 framealpha=0.0,
#                 prop=dict(
#                     size=12,
#                 ),
#                 title_fontproperties=dict(
#                     size=16,
#                 ),
#                 labelcolor=txt_color,
#             )
#             plt.setp(gal_mass_leg.get_title(), color=txt_color)
#             ax.add_artist(gal_mass_leg)

#             type_markers = []
#             for marker in markers:
#                 if marker != "x":
#                     type_markers.append(
#                         ax.scatter(
#                             [],
#                             [],
#                             s=max_mass_size,
#                             edgecolors=color,
#                             facecolors="none",
#                             marker=marker,
#                             alpha=1.0,
#                             lw=1.0,
#                         )
#                     )
#                 else:
#                     type_markers.append(
#                         ax.scatter(
#                             [],
#                             [],
#                             s=max_mass_size,
#                             edgecolors=color,
#                             facecolors="none",
#                             marker=marker,
#                             alpha=1.0,
#                             lw=1.0,
#                         )
#                     )

#             # legend for type of galaxy markers
#             gal_lvl_leg = ax.legend(
#                 type_markers,
#                 ["Halo", "Subhalo"],
#                 loc="upper right",
#                 # fontsize=12,,
#                 # title_fontsize="16",
#                 framealpha=0.0,
#                 prop=dict(
#                     size=12,
#                 ),
#                 labelcolor=txt_color,
#             )
#             ax.add_artist(gal_lvl_leg)

#     if annotate:
#         # annotate makers with ids
#         for i, (hid, x, y) in enumerate(
#             zip(hids, x, y)
#             # zip(hids[mains_to_plot], x[mains_to_plot], y[mains_to_plot])
#         ):

#             try:
#                 _, halo_dict = get_halo_props_snap(sim.path, snap, hid)

#             except FileNotFoundError:
#                 pass

#             annotate_txt = f"{hid}"

#             ax.annotate(
#                 annotate_txt,
#                 (
#                     (x - rot_tgt_pos[0]) * sim.cosmo.lcMpc * 1e3,
#                     (y - rot_tgt_pos[1]) * sim.cosmo.lcMpc * 1e3,
#                 ),
#                 fontsize=12,
#                 zorder=999,
#                 color=txt_color,
#                 path_effects=[pe.withStroke(linewidth=1, foreground="black")],
#             )
#         # for i, (gid, x, y) in enumerate(
#         #     zip(gids[subs_to_plot], x[subs_to_plot], y[subs_to_plot])
#         # ):
#         #     ax.annotate(
#         #         f"{gid}",
#         #         (
#         #             (x - tgt_pos[0]) * sim.cosmo.lcMpc * 1e3,
#         #             (y - tgt_pos[1]) * sim.cosmo.lcMpc * 1e3,
#         #         ),
#         #         fontsize=8,
#         #         zorder=999,
#         #         color="w",
#         #         path_effects=[pe.withStroke(linewidth=1, foreground="black")],
#         #     )

#     return 1


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

        # u2 = np.cross(np.asarray([1.0, 0.0, 0.0]), u1) #seems wrong formalism vs rascas?
        # u2 = np.cross(
        #     u1, np.asarray([0.0, 0.0, 1.0])
        # )  # seems wrong formalism vs rascas?

        # u2 = [
        # u1[0] * 0.0 - u1[1] * 1.0,
        # u1[1] * 0.0 - u1[2] * 0.0,
        # u1[2] * 1.0 - u1[0] * 0.0,
        # ]

        u2 = [0, u1[2], -u1[1]]  # copied from rascas

        if np.all(u2 == np.zeros(3)):
            u2 += 1.0 / 3
        # print(u2)
        u2 = u2 / np.linalg.norm(u2)

        # u3 = np.cross(u1, u2)
        u3 = [
            u1[1] * u2[2] - u1[2] * u2[1],
            u1[2] * u2[0] - u1[0] * u2[2],
            u1[0] * u2[1] - u1[1] * u2[0],
        ]
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


def plot_stars(
    fig,
    ax,
    sim,
    aexp,
    directions,
    bins,
    stmass,
    ctr_st_pos,
    ctr_tgt,
    rad_tgt,
    binning="cic",
    **kwargs,
):

    # print(*kwargs)

    mode = kwargs.get("mode", "sum")
    if mode not in kwargs:
        kwargs["mode"] = mode
    cb = kwargs.get("cb", True)
    cmap = kwargs.get("cmap", "gray")
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    label = kwargs.get("label", None)
    log = kwargs.get("log", False)
    transpose = kwargs.get("transpose", True)
    lower = kwargs.get("lower", True)
    if lower:
        lower_str = "lower"
    else:
        lower_str = "upper"
    zero_ctr = kwargs.get("zero_ctr", True)
    units = kwargs.get("units", "kpc")
    lim = kwargs.get("lim", True)

    cb_args = kwargs.get("cb_args", {})
    cb_args["location"] = cb_args.get("location", "right")
    cb_args["orientation"] = cb_args.get("orientation", "vertical")

    if type(bins) == int or type(bins) == float:
        nb_img_bins = int(bins)
        planx_bins = None
        plany_bins = None
    elif type(bins) == list or type(bins) == np.ndarray:
        nb_img_bins = len(bins)
        planx_bins = bins[0]
        plany_bins = bins[1]

    # print(rad_tgt)

    # dv1, dv2, dv3 = directions
    # print(directions)
    if type(directions[0]) in [float, int, np.float64, np.int64, np.float32, np.int32]:
        dv1, dv2, dv3 = basis_from_vect(directions)
    elif len(np.shape(directions)) == 2:
        dv1, dv2, dv3 = directions
    else:
        raise ValueError("directions must be a list of 3 vectors or a single vector")

    M_basis = project_direction(dv1, [0, 0, 1])
    # M_basis = [dv3, dv2, dv1]
    # vis_dir_ctr_st_pos = rot.apply(ctr_st_pos)
    # vis_dir_ctr_st_pos = np.asarray(
    #     [
    #         np.dot(dv3, np.transpose(ctr_st_pos - ctr_tgt)),
    #         np.dot(dv2, np.transpose(ctr_st_pos - ctr_tgt)),
    #     ]
    # ).T

    if zero_ctr:
        vis_dir_ctr_st_pos = np.asarray(
            [
                np.dot(
                    stp - ctr_tgt,
                    M_basis,
                )
                for stp in ctr_st_pos
            ]
        )
    else:
        vis_dir_ctr_st_pos = np.asarray(
            [
                np.dot(
                    stp,
                    M_basis,
                )
                for stp in ctr_st_pos
            ]
        )
        ctr_tgt = np.dot(
            ctr_tgt,
            M_basis,
        )

    # vis_dir_ctr_tgt = np.asarray(
    #     [
    #         np.dot(dv2, ctr_tgt),
    #         np.dot(dv3, ctr_tgt),
    #     ]
    # )

    # print(
    #     len(vis_dir_ctr_st_pos),
    #     vis_dir_ctr_st_pos.max(axis=0),
    #     vis_dir_ctr_st_pos.min(axis=0),
    # )

    if planx_bins == None:
        # planx_bins = np.linspace(-rad_tgt, rad_tgt, nb_img_bins + 1)
        # planx_bins = np.linspace(
        #     vis_dir_ctr_st_pos[:, 0].min(),
        #     vis_dir_ctr_st_pos[:, 0].max(),
        #     nb_img_bins + 1,
        # )
        if zero_ctr:
            planx_bins = np.linspace(
                -rad_tgt / (sim.cosmo.lcMpc * 1e3),
                +rad_tgt / (sim.cosmo.lcMpc * 1e3),
                nb_img_bins + 1,
            )
        else:
            planx_bins = np.linspace(
                ctr_tgt[0] - rad_tgt / (sim.cosmo.lcMpc * 1e3),
                ctr_tgt[0] + rad_tgt / (sim.cosmo.lcMpc * 1e3),
                nb_img_bins + 1,
            )

    if plany_bins == None:
        # plany_bins = np.linspace(-rad_tgt, rad_tgt, nb_img_bins + 1)
        # plany_bins = np.linspace(
        #     vis_dir_ctr_st_pos[:, 1].min(),
        #     vis_dir_ctr_st_pos[:, 1].max(),
        #     nb_img_bins + 1,
        # )
        if zero_ctr:
            plany_bins = np.linspace(
                -rad_tgt / (sim.cosmo.lcMpc * 1e3),
                +rad_tgt / (sim.cosmo.lcMpc * 1e3),
                nb_img_bins + 1,
            )
        else:
            plany_bins = np.linspace(
                ctr_tgt[1] - rad_tgt / (sim.cosmo.lcMpc * 1e3),
                ctr_tgt[1] + rad_tgt / (sim.cosmo.lcMpc * 1e3),
                nb_img_bins + 1,
            )

    if zero_ctr:
        extent = np.asarray([-rad_tgt, rad_tgt, -rad_tgt, rad_tgt]) / (
            sim.cosmo.lcMpc * 1e3
        )
    else:
        extent = np.asarray(
            [
                ctr_tgt[0] - rad_tgt,
                ctr_tgt[0] + rad_tgt,
                ctr_tgt[1] - rad_tgt,
                ctr_tgt[1] + rad_tgt,
            ]
        ) / (sim.cosmo.lcMpc * 1e3)

    # print(vis_dir_ctr_st_pos.max(axis=0), vis_dir_ctr_st_pos.min(axis=0))
    # print(planx_bins.min(), planx_bins.max(), plany_bins.min(), plany_bins.max())
    # print(rad_tgt, extent, planx_bins[0], planx_bins[-1])

    ctr_img = [0, 0]

    if binning == "cic":

        img = make_img_cic(
            vis_dir_ctr_st_pos[:, 0],
            vis_dir_ctr_st_pos[:, 1],
            stmass,
            ctr_img,
            nb_img_bins,
            extent,
            mode=mode,
        )

    elif binning == "simple":
        img = make_img_hist(
            vis_dir_ctr_st_pos[:, 0],
            vis_dir_ctr_st_pos[:, 1],
            stmass,
            ctr_img,
            nb_img_bins,
            extent,
            mode=mode,
        )

    if np.all((img == 0 + (np.isfinite(img) == False)) > 0):
        return img

    # print(img.max(), img[img > 0].min())

    if vmin == None:
        vmin = np.nanpercentile(img, 1)
    if vmax == None:
        vmax = np.nanpercentile(img, 99)

    if vmax / vmin > 25 and vmin > 0:
        log = True
        # if vmin == 0:
        #     np.nanpercentile(img[img > 0], 1)

    if vmax < vmin:
        vmax = vmin + 1

    print("guessed vmin, vmax", vmin, vmax)

    # print(img.max(), rad_tgt)
    # print(vis_dir_ctr_st_pos.max(axis=0), vis_dir_ctr_st_pos.min(axis=0))
    # print(planx_bins.min(), planx_bins.max(), plany_bins.min(), plany_bins.max())
    # print(np.nanmin(img), np.nanmax(img))

    if transpose:
        plot_img = img.T
        extent_plot = extent
    else:
        plot_img = img
        extent_plot = extent[[2, 3, 0, 1]]

    if units == "kpc":
        extent_plot *= sim.cosmo.lcMpc * 1e3
    elif units == "Mpc":
        extent_plot *= sim.cosmo.lcMpc
    elif units == "code":
        pass
    else:
        raise ValueError("units must be kpc, Mpc or code")

    if log:
        plimg = ax.imshow(
            plot_img,
            origin=lower_str,
            extent=extent_plot,
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
    else:
        plimg = ax.imshow(
            plot_img,
            origin=lower_str,
            extent=extent_plot,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    if cb:

        if label != None:
            if kwargs["mode"] == "mean":
                label = f"Mean {label}"

        cb = plt.colorbar(
            plimg,
            ax=ax,
            # fraction=0.046,
            pad=0.0,
            **cb_args,
        )
        cb.set_label(label)

    if transpose:
        ax.set_ylabel("y [ckpc]")
        ax.set_xlabel("x [ckpc]")
    else:
        ax.set_xlabel("y [ckpc]")
        ax.set_ylabel("x [ckpc]")

    if lim:
        ax.set_xlim(extent_plot[:2])
        ax.set_ylim(extent_plot[2:])

    # # plot hagn galaxies
    # plot_zoom_gals(
    #     ax,
    #     snap,
    #     sim,
    #     gal_pos,
    #     rgal,
    #     zdist,
    #     hm=hm,
    #     brick_fct=read_zoom_brick,
    # )

    # plot hagn BHs
    # plot_zoom_BHs(ax, snap, sim, gal_pos, rgal, zdist)

    ax.set_facecolor("black")

    return img


def make_img_cic(x, y, z, ctr, ncells, extent_code, mode="sum"):

    # x, y, extent_code, and ctr in between 0 and 1
    # ncells number of image pixels

    """
    value in cell and neighboring cells scales with distance from particle
    """

    # cell_size = abs(extent_code[1] - extent_code[0]) / ncells

    bins = np.arange(0, ncells + 1)

    xcells_code = np.linspace(
        ctr[0] + extent_code[0], ctr[0] + extent_code[1], ncells + 1
    )
    ycells_code = np.linspace(
        ctr[1] + extent_code[2], ctr[1] + extent_code[3], ncells + 1
    )

    dx = xcells_code[1] - xcells_code[0]
    xcells_code = xcells_code + dx * 0.5
    dy = ycells_code[1] - ycells_code[0]
    ycells_code = ycells_code + dy * 0.5

    print(x.min(), x.max(), dx, xcells_code)

    print(y.min(), y.max(), dy, ycells_code)

    x_in_cells = np.clip(np.digitize(x, xcells_code), 0, ncells)
    y_in_cells = np.clip(np.digitize(y, ycells_code), 0, ncells)

    # x_in_cells = np.full(len(x), ncells)

    # print(x_in_cells, y_in_cells)

    dist_to_x = np.abs(x - xcells_code[x_in_cells])
    dist_to_y = np.abs(y - ycells_code[y_in_cells])

    # dist_to_x[(x < xcells_code[0]) * (x > xcells_code[-1])] = np.inf
    # dist_to_y[(y < ycells_code[0]) * (y > ycells_code[-1])] = np.inf

    # print(np.max([x_in_cells + 1, np.full(len(x), ncells)], axis=0))

    dist_to_xp1 = np.abs(
        xcells_code[np.min([x_in_cells + 1, np.full(len(x), ncells)], axis=0)] - x
    )
    dist_to_xm1 = np.abs(
        xcells_code[np.max([x_in_cells - 1, np.full(len(x), 0)], axis=0)] - x
    )

    dist_to_yp1 = np.abs(
        ycells_code[np.min([y_in_cells + 1, np.full(len(x), ncells)], axis=0)] - y
    )
    dist_to_ym1 = np.abs(
        ycells_code[np.max([y_in_cells - 1, np.full(len(x), 0)], axis=0)] - y
    )

    img = np.zeros((ncells, ncells))
    if mode == "mean":
        img_count = np.zeros((ncells, ncells), dtype=np.float32)

    for dx, xdist in zip([-1, 0, 1], [dist_to_xm1, dist_to_x, dist_to_xp1]):
        for dy, ydist in zip([-1, 0, 1], [dist_to_ym1, dist_to_y, dist_to_yp1]):

            fact = 1 - np.sqrt(xdist**2 + ydist**2)

            fact[np.any([xdist == np.inf, ydist == np.inf])] = 0
            fact[fact < 0] = 0
            fact[fact > 1] = 1

            in_frame = (
                (x_in_cells + dx >= 0)
                * (x_in_cells + dx < ncells + 1)
                * (y_in_cells + dy >= 0)
                * (y_in_cells + dy < ncells + 1)
            )

            print(len(z), len(x_in_cells))

            img += binned_statistic_2d(
                x_in_cells + dx,
                y_in_cells + dy,
                values=fact * in_frame * z,
                bins=[bins[:], bins[:]],
                statistic="sum",
            )[0]

            if mode == "mean":

                img_count += binned_statistic_2d(
                    x_in_cells + dx,
                    y_in_cells + dy,
                    fact * in_frame,
                    bins=[bins[:], bins[:]],
                    statistic="sum",
                )[0]

            # print(img.max())

    if mode == "mean":
        img = img / img_count

    return img


def make_img_hist(x, y, z, ctr, ncells, extent_code, mode="sum"):

    # x, y, extent_code, and ctr in between 0 and 1
    # ncells number of image pixels
    # mode can be anything understood by scipy.stats.binned_statistic

    """
    value in cell
    """

    xcells_code = np.linspace(
        ctr[0] + extent_code[0], ctr[0] + extent_code[1], ncells + 1
    )
    ycells_code = np.linspace(
        ctr[1] + extent_code[2], ctr[1] + extent_code[3], ncells + 1
    )

    img = np.zeros((ncells, ncells), dtype=np.float32)
    img += binned_statistic_2d(
        x,
        y,
        z,
        bins=[xcells_code[:], ycells_code[:]],
        statistic=mode,
    )[0]

    return img


def plot_trail(ax, coords, time, cur_time, time_lim, **plot_args):
    """use coords to draw interpolated trail within time_lim Myr, using cmap colormap"""

    # if "cmap" in plot_args:
    #     cmap = plt.get_cmap(plot_args["cmap"])
    # else:
    #     cmap = plt.get_cmap("viridis")

    if "c" not in plot_args and "color" not in plot_args:
        plot_args["color"] = "tab:red"

    ok_time = (time < cur_time) * (time > (cur_time - time_lim))

    ok_coords = coords[ok_time]

    # colors = cmap(np.linspace(0, 1, len(ok_coords)))
    # colors_hex = [matplotlib.colors.to_hex(c) for c in colors]

    # print(colors_hex)

    # print(ok_coords[:, 0].shape, colors.shape)

    ax.plot(ok_coords[:, 0], ok_coords[:, 1], **plot_args)
    # colored_line(ok_coords[:, 0], ok_coords[:, 1], colors, ax, **plot_args)


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    taken from https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    # if "array" in lc_kwargs:
    #     warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


def make_img_stars(
    fig,
    ax,
    plot_field,
    sim: ramses_sim,
    aexp,
    directions,
    nbins,
    tgt_pos,
    tgt_rad,
    **kwargs,
):

    read_ball_fct = kwargs.get("read_ball_fct", read_data_ball)

    hid = kwargs.get("hid", None)

    debug = kwargs.get("debug", False)

    snap = sim.get_closest_snap(aexp=aexp)

    tgt_fields = ["mass", "pos", "birth_time", "metallicity", "age"]

    if "velocity" in plot_field:
        tgt_fields.append("vel")

    # print(read_ball_fct, sim, snap, tgt_pos, tgt_rad * np.sqrt(2), tgt_fields)
    # stars = read_ball_fct(
    #     sim, snap, tgt_pos, tgt_rad * np.sqrt(2), tgt_fields, fam=2, debug=debug
    # )

    # print(hid)

    stars = read_data_ball(
        sim,
        snap,
        tgt_pos,
        tgt_rad * np.sqrt(2),
        hid,
        data_types=["stars"],
        tgt_fields=tgt_fields,
    )

    # print(stars)

    if len(stars["mass"]) == 0:
        print("no stars found")
        print(tgt_pos, tgt_rad)
        return 0

    # print(stars["mass"].sum(), stars["mass"].min())

    # stars["mass"] = correct_mass(
    #     sim, stars["age"], stars["mass"], stars["metallicity"], aexp
    # )

    if plot_field == "stellar mass":

        data_to_plot = stars["mass"]
        # if not "vmin" in kwargs or kwargs["vmin"] == None:
        #     kwargs["vmin"] = stars["mass"].min() * 3
        #     print(f"Setting vmin to {kwargs['vmin']}")
        label = "Stellar mass [M$_\odot$]"

    elif "SFR" in plot_field:
        msr_dt = float(plot_field[3:])

        # print(stars.keys())

        # if not hasattr(sim, "cosmo_model"):
        # sim.init_cosmo()
        # cur_time = sim.cosmo_model.age(1.0 / aexp - 1).value * 1e3  # Myr
        yng = stars["age"] <= msr_dt
        data_to_plot = (stars["mass"] * yng) / msr_dt  # Msun/Myr

        if not "vmin" in kwargs or kwargs["vmin"] == None:
            # print(data_to_plot.min(), data_to_plot.max())
            kwargs["vmin"] = data_to_plot[data_to_plot > 0].min()
            # print(kwargs["vmin"])

        if int(msr_dt) == msr_dt:
            str_msr_dt = str(int(msr_dt))
        else:
            str_msr_dt = "%.1f" % msr_dt

        label = f"SFR [{str_msr_dt} Myr]" + ", [M$_\odot.\mathrm{Myr^{-1}}$]"

        # print(
        #     cur_time,
        #     msr_dt,
        #     stars["age"].min(),
        #     stars["age"].max(),
        # )

        # print(yng.sum())

        # print(data_to_plot.min(), data_to_plot.max(), np.unique(data_to_plot))

    elif plot_field == "stellar age":

        data_to_plot = stars["age"]

        label = "Stellar age [Myr]"
    elif plot_field == "stellar metallicity":

        data_to_plot = stars["metallicity"]

        label = "Stellar metallicity [Z$_\odot$]"
    elif plot_field == "stellar velocity":

        data_to_plot = stars["vel"] / 1e5

        label = "Stellar velocity [km/s]"

    else:
        print(f"field {plot_field} not recognized")

    # print(data_to_plot.min(), data_to_plot.max(), np.median(data_to_plot))

    img = plot_stars(
        fig,
        ax,
        sim,
        aexp,
        directions,
        nbins,
        data_to_plot,
        stars["pos"],
        tgt_pos,
        tgt_rad * (sim.cosmo.lcMpc * 1e3) * np.sqrt(2),
        label=label,
        **kwargs,
    )

    return img


def make_img_dm(
    fig, ax, plot_field, sim, aexp, directions, nbins, tgt_pos, tgt_rad, **kwargs
):

    snap = sim.get_closest_snap(aexp=aexp)

    hid = kwargs.get("hid", None)

    tgt_fields = ["mass", "pos"]

    if "velocity" in plot_field:
        tgt_fields.append("vel")

    # read_ball_fct = kwargs.get("read_ball_fct", read_part_ball_NCdust)

    # dm = read_ball_fct(sim, snap, tgt_pos, tgt_rad * np.sqrt(2), tgt_fields, fam=1)
    dm = read_data_ball(
        sim,
        snap,
        tgt_pos,
        tgt_rad * np.sqrt(2),
        hid,
        data_types=["dm"],
        tgt_fields=tgt_fields,
    )

    # print(dm.keys())

    # print(dm["mass"].min(), dm["mass"].max())

    # print(dm["pos"].mean(axis=0))

    if plot_field == "dm mass":

        data_to_plot = dm["mass"]
        # if not "vmin" in kwargs or kwargs["vmin"] == None:
        #     kwargs["vmin"] = dm["mass"].min() * 1

        label = "DM mass [M$_\odot$]"

    elif plot_field == "dm velocity":

        data_to_plot = dm["vel"]
        label = "DM velocity [km/s]"

    else:
        print(f"field {plot_field} not recognized")

    img = plot_stars(
        fig,
        ax,
        sim,
        aexp,
        directions,
        nbins,
        data_to_plot,
        dm["pos"],
        tgt_pos,
        tgt_rad * np.sqrt(2) * (sim.cosmo.lcMpc * 1e3),
        label=label,
        **kwargs,
    )

    # print(img.min(), img.max())

    return img


def plot_fields(
    field_name,
    fig,
    ax,
    aexp,
    directions,
    tgt_pos,
    tgt_rad,
    sim: ramses_sim,
    **kwargs,
):

    read_ball_fct = kwargs.get("read_ball_fct", read_data_ball)
    if not "read_ball_fct" in kwargs:
        kwargs["read_ball_fct"] = read_ball_fct

    stellar_fields = [
        "stellar mass",
        "stellar age",
        "stellar metallicity",
        "SFR1",
        "SFR5",
        "SFR10",
        "SFR100",
        "SFR300",
        "SFR500",
        "SFR1000",
        "stellar velocity",
        # "stellar radial velocity",
        # "stellar circular velocity",
    ]
    dm_fields = [
        "dm mass",
        "dm velocity",
    ]  # , "dm radial velocity", "dm circular velocity"]
    gas_fields = [
        "ilevel",
        "density",
        "temperature",
        "DTM",
        "DTMC",
        "DTMCs",
        "DTMCl",
        "DTMSi",
        "DTMSis",
        "DTMSil",
        "metallicity",
        "velocity",
        # "radial velocity",
        # "circular velocity",
        "pressure",
        "dust_bin01",
        "dust_bin02",
        "dust_bin03",
        "dust_bin04",
        "alpha_vir",
        "mach",
    ]

    if field_name in stellar_fields:

        if "nbins" in kwargs:
            nbins = kwargs["nbins"]
        else:
            nbins = guess_nbin(tgt_rad, sim, aexp)

        img = make_img_stars(
            fig,
            ax,
            field_name,
            sim,
            aexp,
            directions,
            nbins,
            tgt_pos,
            tgt_rad,
            **kwargs,
        )

    elif field_name in dm_fields:

        if "nbins" in kwargs:
            nbins = kwargs["nbins"]
        else:
            nbins = int(guess_nbin(tgt_rad, sim, aexp) * 0.1)

        img = make_img_dm(
            fig,
            ax,
            field_name,
            sim,
            aexp,
            directions,
            nbins,
            tgt_pos,
            tgt_rad,
            **kwargs,
        )

    elif field_name in gas_fields:

        if "zdist" not in kwargs:
            zdist = tgt_rad * (sim.cosmo.lcMpc * 1e3)
        else:
            zdist = kwargs["zdist"]

        snap = sim.get_closest_snap(aexp=aexp)

        # print(tgt_pos, tgt_rad, zdist)
        if type(directions[0]) not in [list, np.ndarray]:
            directions_img = directions
        else:
            directions_img = directions[2]

        img = make_amr_img_smooth(
            fig,
            ax,
            field_name,
            snap,
            sim,
            tgt_pos,
            tgt_rad,
            zdist,
            direction=directions_img,
            **kwargs,
        )
    else:
        print(f"field {field_name} not recognized")
        raise ValueError

    return img


def guess_nbin(tgt_rad, sim, aexp):

    lvls, zeds, reds = sim.compute_aexps_lvlchange()

    aexps = 1.0 / (1.0 + zeds)

    lvl = lvls[np.digitize(aexp, aexps)]

    sim_max_res = 1.0 / 2**lvl  # sim.namelist["amr_params"]["levelmax"]
    nbins = int(tgt_rad / sim_max_res * 0.33 / aexp)
    print(f"guessing appropriate nbins at 0.33 max resolution: {nbins}")
    return nbins
