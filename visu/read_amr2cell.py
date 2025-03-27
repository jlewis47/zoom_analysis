from tkinter import N
import numpy as np
import os

from gremlin.read_sim_params import ramses_sim

from f90_tools.IO import read_all_record_sizes, read_record, skip_record, read_records
from f90_tools.hilbert import get_files

# import time

# from amr2cell.f90:                     write(20,999)x(i,1),x(i,2),x(i,3),dx,icpu,ilevel,&
#                          & (var(i,ind,ivar),ivar=1,nvarh)


def read_amr2cell_file(filename, hvars):
    with open(filename, "r") as f:
        dt = np.dtype(
            [
                ("buf1", np.int32),
                ("x", np.float64),
                ("y", np.float64),
                ("z", np.float64),
                ("dx", np.float64),
                ("icpu", np.int32),
                ("ilevel", np.int32),
            ]
            + [(hvar, np.float64) for hvar in hvars]
            + [("buf2", np.int32)]
        )

        # print(list(dt.fields.keys()))

        # usefull_keys = list(dt.fields.keys())[1:-1]

        # print(dt[usefull_keys])

        data = np.fromfile(f, dtype=dt)

    return data


def read_amr2cell_redshift(zed, sim_dir, fields=None):
    sim_params = ramses_sim(sim_dir)

    aexps = sim_params.get_snap_exps()

    zeds = 1.0 / aexps - 1.0

    closest_arg = np.argmin(np.abs(zeds - zed))

    hvars = sim_params.hydro
    hvars.pop("nvar")
    hvars = list(hvars.values())

    print(f"Asked for redshift {zed:.2f}")
    print(f"Closest redshift is {zeds[closest_arg]:.2f}")

    output_str = f"output_{sim_params.snap_numbers[closest_arg]:05d}"

    if fields is None:
        return read_amr2cell_file(
            os.path.join(sim_dir, "amr2cell", output_str, "out_amr2cell"), hvars
        )
    else:
        data_dict = {}
        data = read_amr2cell_file(
            os.path.join(sim_dir, "amr2cell", output_str, "out_amr2cell"), hvars
        )[fields]
        for field in fields:
            data_dict[field] = data[field]
        return data_dict


def read_amr2cell_output(out_nb, sim_dir, fields=None):
    sim_params = ramses_sim(sim_dir)

    hvars = sim_params.hydro

    if "nvar" in hvars:
        hvars.pop("nvar")
    hvars = list(hvars.values())
    # print(hvars)

    output_str = f"output_{out_nb:05d}"

    print(f"Asked for {output_str}")

    # print(hvars)

    if fields is None:
        return read_amr2cell_file(
            os.path.join(sim_dir, "amr2cell", output_str, "out_amr2cell"), hvars
        )
    else:
        data_dict = {}
        data = read_amr2cell_file(
            os.path.join(sim_dir, "amr2cell", output_str, "out_amr2cell"), hvars
        )[fields]
        for field in fields:
            data_dict[field] = data[field]
        return data_dict


# moche tout ca tu pourrais refactorer...

def amr2cell_units(cell_dict, sim_dir):
    sim_params = ramses_sim(sim_dir)

    unit_d = sim_params.cosmo.unit_d
    unit_l = sim_params.cosmo.unit_l
    unit_t = sim_params.cosmo.unit_t

    dist_units = unit_l  # cMpc/h?? -check these
    vel_units = unit_l / unit_t * 1e6  # ckm/s
    dens_units = unit_d  # g/ccm^3

    cell_dict["x"] *= dist_units
    cell_dict["y"] *= dist_units
    cell_dict["z"] *= dist_units

    cell_dict["dx"] *= dist_units

    cell_dict["velocity_x"] *= vel_units
    cell_dict["velocity_y"] *= vel_units
    cell_dict["velocity_z"] *= vel_units

    cell_dict["density"] *= dens_units

    return cell_dict


# sim_dir = "/automnt/data101/jlewis/sims/proposal_zooms/ExtractZoom/TestRun1e12"
# outputs=[374]

# # print(read_amr2cell_output(outputs[0], sim_dir))
# print(read_amr2cell_output(outputs[0], sim_dir, ['density', 'passive_scalar_1']))


def read_amrcells(
    sim: ramses_sim, snap, bounds=None, nvars=None, fnbs=None, debug=False
):

    if bounds is None:
        bounds = [0, 0, 0, 1, 1, 1]

    # xmin, ymin, zmin, xmax, ymax, zmax = bounds
    # print(bounds)

    xc = np.zeros((8, 3), dtype=np.float64)

    # data = {}

    with open(
        os.path.join(sim.output_path, f"output_{snap:05d}", f"amr_{snap:05d}.out00001"),
        "rb",
    ) as src:
        ncpu = read_record(src, 1, np.int32)
        ndim = read_record(src, 1, np.int32)
        nx, ny, nz = read_record(src, 3, np.int32)
        nlevelmax = read_record(src, 1, np.int32)
        ngridmax = read_record(src, 1, np.int32)
        nboundary = read_record(src, 1, np.int32)
        ngrid_current = read_record(src, 1, np.int32)
        boxlen = read_record(src, 1, np.float32)

        ngridlevel = np.empty((ncpu, nlevelmax), dtype=np.int32)
        if nboundary == 0:
            ngridfile = np.empty((ncpu, nlevelmax), dtype=np.int32)
        else:
            ngridfile = np.empty((ncpu + nboundary, nlevelmax), dtype=np.int32)

    xbound = np.asarray([float(nx // 2), float(ny // 2), float(nz // 2)])

    levelmin = sim.levelmin

    if fnbs == None:
        files_to_read = get_files(sim, snap, bounds[:3], bounds[3:])

        if debug:
            print(files_to_read)

        # input("")

    else:
        files_to_read = fnbs

    with open(
        os.path.join(
            sim.output_path, f"output_{snap:05d}", f"hydro_{snap:05d}.out00001"
        ),
        "rb",
    ) as src_hydro:

        skip_record(src_hydro, 1, debug=False)
        nvarh = read_record(src_hydro, 1, np.int32)
        skip_record(src_hydro, 4, debug=False)

        # input("")

        # time.sleep(1)

        if nvars is None:
            nvars = np.arange(1, nvarh + 1)

        else:
            nvars = np.sort(nvars)

    full_vars = np.empty((0, len(nvars)), dtype=np.float64)
    full_lvl = np.empty((0), dtype=np.float64)
    # full_inds = np.empty((0), dtype=np.int32)
    full_pos = np.empty((0, 3), dtype=np.float64)

    # print(sim.hydro, nvars)
    # print(list(sim.hydro.keys())[nvars[0]])
    requested_fields = [sim.hydro[str(i)] for i in nvars]
    # requested_fields = [sim.hydro[list(sim.hydro.keys())[i]] for i in nvars]

    # print(requested_fields)

    tot_found = 0

    for fnb in files_to_read:

        # if debug:

        # f = open(f"debug_{fnb}.log", "w")

        fname_amr = os.path.join(
            sim.output_path, f"output_{snap:05d}", f"amr_{snap:05d}.out{fnb:05d}"
        )  #

        if debug:
            print(f"reading {fname_amr}")

        # nrec = 0
        with open(fname_amr, "rb") as src:

            skip_record(src, 21, debug=debug)
            # nrec += 21

            read_ngrid = read_record(src, ncpu * nlevelmax, np.int32)
            # nrec += 1

            ngridfile[:ncpu, :] = read_ngrid.reshape((ncpu, nlevelmax), order="F")

            skip_record(src, 1, debug=debug)
            # nrec += 1

            if nboundary > 0:
                skip_record(src, 2)
                # nrec += 2

                # print(nboundary, nlevelmax)
                read_ngrid = read_record(
                    src, nboundary * nlevelmax, np.int32, debug=False
                )

                # nrec += 1
                # print("ngridbound", read_ngrid)

                ngridfile[ncpu:, :] = read_ngrid.reshape(
                    (nboundary, nlevelmax), order="F"
                )

            skip_record(src, 6, debug=debug)
            # nrec += 6

            fname_hydro = os.path.join(
                sim.output_path, f"output_{snap:05d}", f"hydro_{snap:05d}.out{fnb:05d}"
            )

            # with open(fname_hydro, "rb") as src:

            #     rec_sizes = read_all_record_sizes(src)

            # for s in rec_sizes:
            #     print(s)
            #     input("")

            with open(fname_hydro, "rb") as src_hydro:

                skip_record(src_hydro, 1, debug=False)
                nvarh = read_record(src_hydro, 1, np.int32)
                skip_record(src_hydro, 4, debug=False)

                # input("")

                # time.sleep(1)

                for ilvl in range(1, nlevelmax + 1):

                    dx = 0.5**ilvl
                    dx2 = 0.5 * dx

                    # nx_full = 2**ilvl
                    # ny_full = 2**ilvl
                    # nz_full = 2**ilvl

                    inds = np.arange(1, 2**ndim + 1)
                    iz = (inds - 1) // 4
                    iy = (inds - 1 - 4 * iz) // 2
                    ix = inds - 1 - 4 * iz - 2 * iy
                    xc[inds - 1, 0] = (ix - 0.5) * dx
                    xc[inds - 1, 1] = (iy - 0.5) * dx
                    xc[inds - 1, 2] = (iz - 0.5) * dx

                    # print(xc[:, 0])
                    # print(xc[:, 1])
                    # print(xc[:, 2])

                    ngrida = ngridfile[fnb - 1, ilvl - 1]
                    # print(ngridfile.min(), ngridfile.max(), fnb, ilvl)
                    # print(f"{nrec:d} records so far")
                    if debug:
                        print(fnb, ilvl, "level:", ilvl, "ngrida:", ngrida)

                    xg = np.zeros((ngrida, ndim), dtype=np.float64)
                    son = np.zeros((ngrida, int(2**ndim)), dtype=np.int32)
                    var = np.zeros((ngrida, int(2**ndim), len(nvars)), dtype=np.float64)

                    for j in range(1, nboundary + ncpu + 1):

                        # input("")
                        # time.sleep(0.1)

                        if ngridfile[j - 1, ilvl - 1] > 0:
                            # print(j, ilvl, ngridfile[j - 1, ilvl - 1])
                            # skip_record(src, 3, debug=debug)
                            # skip_record(src, 3, debug=False)
                            # print(read_record(src, 1, np.int32, debug=False))
                            # print(read_record(src, 1, np.int32, debug=False))
                            # print(read_record(src, 1, np.int32, debug=False))
                            skip_record(src, 3, debug=debug)
                            # nrec += 3
                            # print(inds)

                            if j == fnb:
                                for idim in range(ndim):
                                    xg[:, idim] = read_record(
                                        src, ngrida, np.float64, debug=debug
                                    )

                            else:

                                skip_record(src, ndim, debug=debug)

                            # skip father
                            # skip_record(src, 1, debug=debug)
                            # father + nbor
                            skip_record(src, int(1 + 2 * ndim), debug=debug)
                            # nrec += 2 * ndim
                            # skip sons
                            # print("sons")
                            if j == fnb:
                                for ind in range(int(2**ndim)):
                                    # print(
                                    # "son",
                                    son[:, ind] = read_record(
                                        src, ngrida, np.int32, debug=debug
                                    )
                                    # skip_record(src, ngrida, debug=debug)
                                    # nrec += 1
                            else:
                                skip_record(src, int(2**ndim), debug=debug)
                                # nrec += 1
                            # skpi cpu map
                            # print("cpu_map")
                            # skip_record(src, int(2**ndim), debug=debug)
                            # nrec += 2**ndim
                            # skip refinement map + cpu map
                            # print("ref_map")
                            skip_record(src, int(2 * 2**ndim), debug=debug)
                            # nrec += 2**ndim

                            # something wrong here...
                            # debugging thanks to /home/jlewis/ramses-yomp/utils/f90/test_log

                            # print(src.tell(), os.path.getsize(fname_amr))

                            # return 0
                        # read hydro data
                        skip_record(src_hydro, 2, debug=False)
                        # print(read_records(src, 2, np.int32))

                        # print(j, fnb, ngrida, ngridfile[j - 1, ilvl - 1])

                        # print(xg.shape)
                        # print(var.shape)

                        if ngridfile[j - 1, ilvl - 1] > 0:
                            for ind in range(int(2**ndim)):
                                read_vars = 0
                                for ivar in range(1, nvarh + 1):
                                    if j == fnb and ivar in nvars:
                                        var[:, ind, read_vars] = read_record(
                                            src_hydro, ngrida, np.float64
                                        )
                                        # print(ind, ivar, read_vars)
                                        # input("")
                                        read_vars += 1
                                    else:
                                        skip_record(src_hydro, 1, debug=False)
                                        # print(read_records(src, 1, np.int32))

                    if ngrida > 0:

                        # print(xc.shape, xg.shape, xbound.shape, var.shape)

                        # print(xg.max(), xg.min())
                        # print(bounds)
                        # print(xbound)

                        for ind in range(2**ndim):

                            x = xg + xc[ind, :] - xbound

                            ref = (son[:, ind] > 0) * (ilvl < nlevelmax)

                            cond = np.all(
                                ((x[:, :] + dx2) >= bounds[:3])
                                * ((x[:, :] - dx2) <= bounds[3:])
                                * (ref[:, np.newaxis] == False),
                                axis=1,
                            )

                            # print(x.max(), x.min())
                            # print(xc[ind, :].max(), xc[ind, :].min())

                            # input("")

                            # print(bounds, (x + dx2).min(axis=0), (x - dx2).max(axis=0))
                            # print(ref.sum())

                            # print((x[:, :] + dx2 > bounds[:3]).sum())

                            # print((x[:, :] - dx2 < bounds[3:]).sum())

                            nsave = cond.sum()

                            if debug:
                                tot_found += nsave
                                print(f"found {nsave:d} cells to output")

                            if nsave == 0:
                                continue
                            # print(np.where(cond))
                            # print(cond.shape, var.shape, full_vars.shape)

                            load_var = var[cond, ind, :]
                            load_pos = np.transpose(
                                [x[cond, 0], x[cond, 1], x[cond, 2]]
                            )
                            load_lvl = np.full(nsave, ilvl)

                            # print(full_vars.shape)
                            # print(nsave)
                            # print(full_vars.shape, full_pos.shape, full_lvl.shape)

                            # print(load_var.shape)
                            # print(var.shape)

                            # load_inds = np.arange(nsave)

                            full_vars = np.concatenate([full_vars, load_var])
                            full_pos = np.concatenate([full_pos, load_pos])
                            full_lvl = np.concatenate([full_lvl, load_lvl])
                            # print(full_inds)
                            # full_inds = np.concatenate([full_inds, load_inds])

                            # print(full_inds[-nsave:], load_inds)
                            # input("")

                            # if debug:
                            # # if ilvl > 10:
                            # for i in range(nsave):
                            #     # write x,y,z,dx,fnb,ilvl,var(i,ind,ivar),ivar=1,nvarh
                            #     # f.write(

                            #     print(
                            #         f"{load_pos[i, 0]:.6e} {load_pos[i, 1]:.6e} {load_pos[i, 2]:.6e} {dx:.6e} {fnb} {ilvl} {load_var[i,:]}"
                            #     )
                            #     input("")

    keys = ["ilevel", "x", "y", "z"] + requested_fields
    vals = [full_lvl, full_pos[:, 0], full_pos[:, 1], full_pos[:, 2]] + [
        full_vars[:, i] for i in range(full_vars.shape[1])
    ]

    if debug:
        print(f"found {tot_found:d} cells")

    amrcells = dict(zip(keys, vals))

    if 1 in nvars and 5 in nvars:  # probably want temperature
        amrcells["temperature"] = []

    # print(full_inds)
    return amrcells
