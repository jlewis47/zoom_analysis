import numpy as np
import os
from scipy.spatial import cKDTree

from f90_tools.IO import read_record, read_tgt_fields
from f90_tools.hilbert import get_files
from gremlin.read_sim_params import ramses_sim

from zoom_analysis.halo_maker.read_treebricks import convert_star_time

# import yt


def read_star(fname: str, tgt_pos=None, tgt_r=None, tgt_fields: list = None):

    names = [
        "pos",
        "vel",
        "mass",
        "ids",
        "lvl",
        "family",
        "tag",
        "birth_time",
        "metallicity",
        "mass_init",
    ]

    # extra chem fields ???
    # SD birth gas density dump

    ndims = [3, 3, 1, 1, 1, 1, 1, 1, 1, 1]

    dtypes = [
        np.dtype("f8"),
        np.dtype("f8"),
        np.dtype("f8"),
        np.dtype("i8"),
        np.dtype("i4"),
        np.dtype("i1"),
        np.dtype("i1"),
        np.dtype("f8"),
        np.dtype("f8"),
        np.dtype("f8"),
    ]

    steps = np.arange(0, len(names))

    if tgt_fields is None:
        tgt_fields = names

    out = {}

    # find order of tgt_fields in names
    tgt_field_order = [idx for idx, field in zip(steps, names) if field in tgt_fields]
    if len(tgt_field_order) < len(tgt_fields):
        print("Not all fields match requested fields")
        print("Available fields are: ", names)
        print("Requested fields are: ", tgt_fields)
        return out

    tgt_fields = [names[i] for i in tgt_field_order]

    with open(fname, "rb") as f:

        ncpu = read_record(f, 1, "i4")
        # ncpu = np.fromfile(f, dtype="i4", count=3)[1]
        ndim = read_record(f, 1, "i4")
        nstars = read_record(f, 1, "i4")

        # print(ncpu, ndim, nstars)

        tot_nstars = read_record(f, 1, "i8")
        tot_mstars = read_record(f, 1, "f8")
        lost_mstars = read_record(f, 1, "f8")
        nsink = read_record(f, 1, "i4")

        # pos = read_record(f, int(nstars * ndim), "f8").reshape(ndim, -1)
        pos = read_record(f, int(nstars * ndim), "f8").reshape(ndim, -1)

        # print(pos.mean(axis=0))

        # pos_filt = False
        if tgt_pos is not None and tgt_r is not None:

            xmax, xmin = pos[0].max(), pos[0].min()
            cond = xmax >= tgt_pos[0] - tgt_r and xmin <= tgt_pos[0] + tgt_r
            if not cond:
                return out
            ymax, ymin = pos[1].max(), pos[1].min()
            cond *= ymax >= tgt_pos[1] - tgt_r and ymin <= tgt_pos[1] + tgt_r
            if not cond:
                return out
            zmax, zmin = pos[2].max(), pos[2].min()
            cond *= zmax >= tgt_pos[2] - tgt_r and zmin <= tgt_pos[2] + tgt_r
            if not cond:
                return out

            # print(cond, xmax, xmin, ymax, ymin, zmax, zmin)

            pos_tree = cKDTree(pos.T, boxsize=1.0 + 1e-10)

            tgt_in_file = pos_tree.query_ball_point(tgt_pos, tgt_r)

            if len(tgt_in_file) == 0:
                return out

            pos = pos[:, tgt_in_file].T

        if "pos" in tgt_fields:
            out["pos"] = pos

        read_tgt_fields(
            out,
            tgt_fields,
            list(zip(names[1:], ndims[1:], dtypes[1:])),
            f,
            nstars,
            args=tgt_in_file,
            debug=False,
        )

        return out


def read_part_ball(
    sim: ramses_sim,
    snap,
    tgt_pos=None,
    tgt_r=None,
    tgt_fields: list = None,
    fam=None,
    debug=False,
):
    """
    fam=2 for stars
    """

    under = np.any(tgt_pos - tgt_r < 0)
    over = np.any(tgt_pos + tgt_r > 1)
    if not np.any(under + over):

        cpus_to_read = get_files(
            sim, snap, tgt_pos - tgt_r, tgt_pos + tgt_r, debug=debug
        )

    else:

        cpus_to_read = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    cpus_to_read += get_files(
                        sim,
                        snap,
                        tgt_pos - tgt_r + np.array([i, j, k]),
                        tgt_pos + tgt_r + np.array([i, j, k]),
                        debug=debug,
                    )  #

        cpus_to_read = np.unique(cpus_to_read)

    part_dir = os.path.join(sim.output_path, "output_{:05d}".format(snap))

    files = np.asarray(
        [
            os.path.join(part_dir, f"part_{snap:05d}.out{icpu:05d}")
            for icpu in cpus_to_read
        ]
    )

    npart_tots = 0

    for f in files:
        with open(f, "rb") as src:
            ndim, nparts = read_hdr(src)
            # print(f, nparts)
        npart_tots += nparts

    if debug:
        print(f"Found {npart_tots:d} particles to read")

    names = [
        "pos",
        "vel",
        "mass",
        "ids",
        "lvl",
        "family",
        "tag",
        "birth_time",
        "metallicity",
        "mass_init",
    ]

    # extra chem fields ???
    # SD birth gas density dump

    ndims = [3, 3, 1, 1, 1, 1, 1, 1, 1, 1]

    dtypes = [
        np.dtype("f8"),
        np.dtype("f8"),
        np.dtype("f8"),
        np.dtype("i4"),
        np.dtype("i4"),
        np.dtype("i1"),
        np.dtype("i1"),
        np.dtype("f8"),
        np.dtype("f8"),
        np.dtype("f8"),
    ]

    steps = np.arange(0, len(names))

    if tgt_fields is None:
        tgt_fields = names

    if "family" not in tgt_fields:
        tgt_fields.append("family")

    out = {}

    # find order of tgt_fields in names
    tgt_field_order = [idx for idx, field in zip(steps, names) if field in tgt_fields]
    if len(tgt_field_order) < len(tgt_fields):
        print("Not all fields match requested fields")
        print("Available fields are: ", names)
        print("Requested fields are: ", tgt_fields)
        return out

    tgt_fields = [names[i] for i in tgt_field_order if names[i] in tgt_fields]

    # initialize out dict
    for field in tgt_fields:
        dt = dtypes[names.index(field)]
        dim = ndims[names.index(field)]
        if dim == 1:
            out[field] = np.zeros(npart_tots, dtype=dt)
        else:
            out[field] = np.zeros((npart_tots, dim), dtype=dt)

    nread = 0
    for fname in files:

        with open(fname, "rb") as f:

            # print(fname)

            ndim, nparts = read_hdr(f)

            if debug:
                print(nparts)

            # pos = read_record(f, int(nstars * ndim), "f8").reshape(ndim, -1)
            pos = read_record(f, int(nparts * ndim), "f8", debug=False).reshape(
                ndim, -1
            )

            if tgt_pos is not None and tgt_r is not None:

                xmax, xmin = pos[0].max(), pos[0].min()
                cond = xmax >= tgt_pos[0] - tgt_r and xmin <= tgt_pos[0] + tgt_r
                if not cond:
                    continue
                ymax, ymin = pos[1].max(), pos[1].min()
                cond *= ymax >= tgt_pos[1] - tgt_r and ymin <= tgt_pos[1] + tgt_r
                if not cond:
                    continue
                zmax, zmin = pos[2].max(), pos[2].min()
                cond *= zmax >= tgt_pos[2] - tgt_r and zmin <= tgt_pos[2] + tgt_r
                if not cond:
                    continue

                pos_tree = cKDTree(pos.T, boxsize=1.0 + 1e-10)

                tgt_in_file = pos_tree.query_ball_point(tgt_pos, tgt_r)

                if len(tgt_in_file) == 0:
                    continue

                # pos = pos[:, tgt_in_file].T

            loc_out = {}

            if "pos" in tgt_fields:
                loc_out["pos"] = pos[:, tgt_in_file].T

            read_tgt_fields(
                loc_out,
                tgt_fields,
                list(zip(names[1:], ndims[1:], dtypes[1:])),
                f,
                nparts,
                args=tgt_in_file,
                debug=debug,
            )

            # keep only one family
            # print(fam, len(loc_out["family"]), np.sum(loc_out["family"] == 2))
            if fam is not None and "family" in loc_out:
                fam_idx = loc_out["family"] == fam
                for k in loc_out:
                    loc_out[k] = loc_out[k][fam_idx]

            # fill main dict)
            for k in loc_out:
                loc_data = loc_out[k]
                out[k][nread : nread + len(loc_data)] = loc_data
            nread += len(loc_data)

    # cut out the rest of the particles
    if nread < npart_tots:
        for k in out:
            out[k] = out[k][:nread]

    if np.any(np.in1d(["birth_time", "age"], tgt_fields)):
        out["age"] = convert_star_time(out["birth_time"], sim, sim.get_snap_exps(snap))
        # del out["birth_time"]

        # print(out)

        if fam == 2 and "family" not in out:
            filt = out["age"] > 0
            for k in out:
                out[k] = out[k][filt]
            if debug:
                print(filt.sum(), len(filt))

        if debug:
            if len(out["age"]) > 0:
                print("age info")
                print(out["birth_time"].min(), out["birth_time"].max())
                print(out["age"].min(), out["age"].max())

    # convert_mass
    cur_aexp = sim.get_snap_exps(snap)

    if "mass" in tgt_fields:
        unit_m = sim.unit_d(cur_aexp) * sim.unit_l(cur_aexp) ** 3 / 2.0e33
        out["mass"] *= unit_m

    return out


def read_hdr(f):
    ncpu = read_record(f, 1, "i4")
    # ncpu = np.fromfile(f, dtype="i4", count=3)[1]
    ndim = read_record(f, 1, "i4")
    nparts = read_record(f, 1, "i4")

    # print(ncpu, ndim, nparts)

    tot_nparts = read_record(f, 1, "i8")
    tot_mparts = read_record(f, 1, "f8")
    lost_mparts = read_record(f, 1, "f8")
    nsink = read_record(f, 1, "i8")
    tot_msink = read_record(f, 1, "f8")
    # print(lost_mparts)
    # print(nsink)
    # print(ncpu, ndim, nparts, tot_nparts, tot_mparts, lost_mparts, nsink)
    return ndim, nparts


# def yt_read_star_ball(sim: ramses_sim, snap, ctr, r):

#     out_dir = sim.path

#     bbox = [ctr - r, ctr + r]

#     ds = yt.load(
#         os.path.join(out_dir, f"output_{snap:05d}"),
#         bbox=bbox,
#     )  # bbox returns a subset of the data - might not be cubic due to hilbert decomp

#     # print(ds.fields)

#     # box = ds.box(
#     # left_edge=bbox[0], right_edge=bbox[1]
#     # )  # box cuts off the data outside of bbox
#     # load a portion of the data
#     # data = box["Density"]
#     # data.convert_to_units("g/cm**3")
#     # print([f for f in dir(ds.fields.star) if "age" in f or "birth" in f])

#     # # print(box[("io", "star_age")].min(), box[("io", "star_age")].max())
#     # # io -> stars/debris; sink; nbody -> dark matter
#     # # negative ages are not stars

#     # Create a r radius sphere, centered ctr
#     sp = ds.sphere((ctr, "code_length"), (r, "code_length"))

#     # Use the total_quantity derived quantity to sum up the
#     # values of the mass and particle_mass fields
#     # within the sphere.
#     # particle_mass = sp.quantities.total_quantity(
#     #     [("gas", "mass"), ("io", "particle_mass")]
#     # )
#     # return sp

#     # def minmax(x):
#     #     return x.min(), x.max()

#     # print(minmax(sp[("io", "particle_birth_time")].to('Myr')),
#     #     minmax(sp[("io", "age")].to('Myr')),
#     #          minmax(sp[("io","age")].to('Myr')),
#     #          minmax(sp[("io","star_age")].to('Myr')),
#     # )

#     stars = sp[("star", "star_age")].to("Myr") > 0
#     # stars = np.ones_like(sp[("io", "particle_birth_time")])==1

#     # print(minmax(sp[("io", "particle_mass")][stars].to('Msun')), minmax(sp[("io", "particle_mass")][stars==False].to('Msun')),)
#     # stellar_mass = sp[("io", "particle_mass")][stars].sum()
#     # stellar_mass.convert_to_units("msun")

#     # print(np.sum(stars),len(stars),sp[("io", "star_age")])

#     # keys = sp.derived_field_list
#     yt_names = [
#         "particle_mass",
#         "star_age",
#         "particle_position",
#         "particle_metallicity",
#     ]
#     names = ["mass", "age", "position", "metallicity"]
#     units = ["Msun", "Myr", "code_length", "code_metallicity"]
#     st_dict = {}
#     # convert units
#     for k, yt_k, u in zip(names, yt_names, units):
#         st_dict[k] = (
#             sp[("star", yt_k)].to(u).value[stars]
#         )  # mimmick the output from my reading functions

#     print(f"found {stars.sum()} stellar particles")
#     # print(f"stellar mass is {sp.quantities.total_quantity([("star", "particle_mass")]).to('Msun'):.1e}")

#     return st_dict
