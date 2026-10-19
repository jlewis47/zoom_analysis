from matplotlib.pylab import f
import numpy as np
import os

# from gremlin.read_sim_params import ramses_sim
from f90_tools.IO import read_record, skip_record

import h5py

from zoom_analysis.halo_maker.assoc_fcts import get_halo_props_snap, get_halo_assoc_file, find_snaps_with_halos

# from hagn.utils import get_hagn_sim


def convert_adaptahop_pos(hagn_l_pMpc, tgt_pos):
    tgt_pos += 0.5 * hagn_l_pMpc
    # print(tgt_pos)
    tgt_pos /= hagn_l_pMpc  # in code units or /comoving box size
    # print(tgt_pos)
    tgt_pos[tgt_pos < 0] += 1
    tgt_pos[tgt_pos > 1] -= 1


def map_tree_rev_steps_bytes(fname, out_path, star=False, debug=False,doubles=False):

    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    dt_switch = np.float32
    if doubles:
        dt_switch=np.float64

    # pass over whole tree and print list of byte numbers that correspond to the start of each step...
    with open(fname, "rb") as src:

        nsteps = read_record(src, 1, np.int32)
        # print(nsteps)
        # print(read_record(src, 1, np.int32))
        # print(read_record(src, 1, np.int32))
        nb_halos = read_record(src, nsteps * 2, np.int32, debug=False)
        nb_halos, nb_shalos = nb_halos[:nsteps], nb_halos[nsteps:]
        # print(nb_halos)
        tree_aexps = read_record(src, nsteps, dt_switch)
        # tree_omega_t = read_record(src, nsteps, np.float32)
        # tree_age_univ = read_record(src, nsteps, np.float32)
        skip_record(src, 1, debug)
        skip_record(src, 1, debug)

        # print(nb_halos, nb_shalos, tree_aexps, tree_omega_t, tree_age_univ)

        byte_positions = np.empty(nsteps, dtype=np.int64)

        for istep in range(nsteps):

            ntot = nb_halos[istep] + nb_shalos[istep]

            # print(ntot)

            ids = np.empty(ntot, dtype=np.int32)
            nbytes = np.empty(ntot, dtype=np.int64)

            print(f"istep: {istep:d} has {ntot:d} halos+subhalos")
            print(f"redshift was {1.0 / tree_aexps[istep] - 1.0:.4f}")

            byte_pos = src.tell()

            # print("byte position is ", byte_pos)

            byte_positions[istep] = byte_pos

            for iobj in range(ntot):

                # print(iobj)

                # skip_record(src, 27, dtype=np.int32)
                # nb_fathers = read_record(src, 1, dtype=np.int32)
                nbytes[iobj] = src.tell()
                # nb_fathers = np.fromfile(src, dtype=np.int32, count=28 + 13 * 2)
                ids[iobj] = read_record(src, 1, np.int32, debug=debug)
                # print(ids[iobj])

                skip_record(src, 11, debug)

                # nb_fathers = nb_fathers[-2]
                nb_fathers = read_record(src, 1, np.int32,debug=debug)
                # print(nb_fathers)
                if nb_fathers > 0:
                    skip_record(src, 1, debug)
                    skip_record(src, 1, debug)

                nb_sons = read_record(src, 1, np.int32,debug=debug)
                # print(nb_sons)

                if nb_sons > 0:
                    skip_record(src, 1, debug)

                skip_record(src, 1, debug)
                skip_record(src, 1, debug)
                if star:
                    skip_record(src, 1, debug)

            with h5py.File(
                os.path.join(out_path, f"bytes_step_{istep:d}.h5"), "w"
            ) as out:
                # out.write(f"{byte_positions[istep]}")
                out.create_dataset(
                    "step_nbytes",
                    data=byte_positions[istep],
                    dtype=np.int64,
                )
                out.create_dataset(
                    "obj_ids", data=ids, dtype=np.int32, compression="lzf"
                )
                out.create_dataset(
                    "obj_nbytes", data=nbytes, dtype=np.int64, compression="lzf"
                )

                # out.write(f"{ids[iobj]:d},{nbytes[iobj]:d}")


def istep_to_nbyte(fpath, istep):

    # with open(os.path.join(fpath, f"bytes_step_{istep:d}.txt"), "r") as src:
    # lines = src.readlines()

    # return int(lines[0])

    with h5py.File(os.path.join(fpath, f"bytes_step_{istep:d}.h5"), "r") as src:
        return int(src["step_nbytes"][()])


def iobj_to_nbyte(fpath, istep, obj_id):

    # with open(os.path.join(fpath, f"bytes_step_{istep:d}.txt"), "r") as src:
    # lines = src.readlines()

    # byte_lines = lines[1:]
    # ids, nbytes = np.asarray([np.int64(line.split(",")) for line in byte_lines]).T

    # find_line = np.searchsorted(ids, obj_id)

    # print(ids, nbytes, find_line, obj_id)
    # print(ids[find_line])

    # return int(nbytes[find_line])

    with h5py.File(os.path.join(fpath, f"bytes_step_{istep:d}.h5"), "r") as src:
        # ids = src["obj_ids"][()]
        # find_line = np.searchsorted(ids, obj_id)
        return src["obj_nbytes"][obj_id - 1]

def iobjs_to_nbytes(fpath, istep, obj_ids):

    with h5py.File(os.path.join(fpath, f"bytes_step_{istep:d}.h5"), "r") as src:
        # ids = src["obj_ids"][()]
        # find_line = np.searchsorted(ids, obj_id)
        return src["obj_nbytes"][obj_ids - 1]


def read_tree_file_rev(
    fname,
    fbytes,
    zstart,
    tgt_ids,
    star=True,
    debug=False,
    tgt_fields=["m", "x", "y", "z", "r"],
):

    # using the offset files to jump to the right byte position,
    # follow the main branch from zstart to the highest possible redshift
    # in the tree for all tgt_ids

    if star:
        dt = np.float64
    else:
        dt = np.float32

    with open(fname, "rb") as src:

        nsteps = read_record(src, 1, np.int32)
        # print(nsteps)
        # print(read_record(src, 1, np.int32))
        # print(read_record(src, 1, np.int32))
        nb_halos = read_record(src, nsteps * 2, np.int32, debug=debug)
        # nb_halos = np.fromfile(src, dtype=np.int32, count=2 * nsteps + 2)[1:-1]
        nb_halos, nb_shalos = nb_halos[:nsteps], nb_halos[nsteps:]
        # print(nb_halos, nb_shalos)
        tree_aexps = read_record(src, nsteps, np.float32)
        # tree_omega_t = read_record(src, nsteps, np.float32)
        # tree_age_univ = read_record(src, nsteps, np.float32)
        skip_record(src, 1, debug)
        skip_record(src, 1, debug)

        # print(nb_halos, nb_shalos, tree_aexps, tree_omega_t, tree_age_univ)

        # print(1./tree_aexps-1, zstart)

        # skip = tree_aexps > (1.0 / (1.0 + zstart))
        skip = np.argmin(np.abs(tree_aexps - (1.0 / (1.0 + zstart))))

        found_ids = np.full((len(tgt_ids), nsteps - skip), -1, dtype=np.int32)
        # found_masses = np.full((len(tgt_ids), nsteps - skip), -1, dtype=np.float32)

        found_fields = {}
        for tgt_f in tgt_fields:
            found_fields[tgt_f] = np.full(
                (len(tgt_ids), nsteps - skip), -1, dtype=np.float32
            )

        found_ids[:, 0] = np.sort(tgt_ids)

        # print(nb_halos, nb_shalos, tree_aexps, tree_omega_t, tree_age_univ)

        # print(nsteps - skip)

        for istep in range(skip, nsteps):

            nyte_skip = istep_to_nbyte(fbytes, istep)
            src.seek(nyte_skip)
            # ntot = nb_halos[istep] + nb_shalos[istep]
            # print(f"istep: {istep:d} has {ntot:d} halos+subhalos")
            if istep > skip and np.all(found_ids[:, istep - skip - 1] == -1):
                print("stopping tree... no fathers found in previous step")
                break

            # if z_end != None:
            #     if (1.0 / tree_aexps[istep] - 1) > z_end:
            #         print("stopping tree... reached end redshift")
            #         break

            if debug:
                print(f"step {istep}, z={1.0 / tree_aexps[istep] - 1.0:.4f}")

            nfound_this_step = 0

            # for iobj in range(ntot):

            for iobj in np.sort(found_ids[:, istep - skip]):

                obj_bytes = iobj_to_nbyte(fbytes, istep, iobj)

                src.seek(obj_bytes)

                # print("iobj:", iobj)

                mynumber = read_record(src, 1, np.int32)
                # print(mynumber)
                bushID = read_record(src, 1, np.int32)
                # print(bushID)
                mystep = read_record(src, 1, np.int32) - 1  # py indexing
                # print(mystep)
                level, hosthalo, hostsub, nbsub, nextsub = read_record(
                    src, 5, np.int32, debug=debug
                )

                # np.fromfile(src, dtype=np.int32, count=7 + 3 * 2)

                # print(level, hosthalo, hostsub, nbsub, nextsub)
                m = read_record(src, 1, np.float32)
                # print("%e" % (m * 1e11))
                macc = read_record(src, 1, dt)
                # print(macc)
                px, py, pz = read_record(src, 3, dt)
                # print(px, py, pz)
                vx, vy, vz = read_record(src, 3, np.float32)
                # print(vx, vy, vz)
                Lx, Ly, Lz = read_record(src, 3, np.float32)
                # # print(Lx, Ly, Lz)
                r, ra, rb, rc = read_record(src, 4, np.float32)
                # # print(r, ra, rb, rc)
                ek, ep, et = read_record(src, 3, np.float32)
                # # print(ek, ep, et)
                spin = read_record(src, 1, np.float32)
                # # print(spin)

                # np.fromfile(src, dtype=np.int32, count=18 + 7 * 2)

                nb_fathers = read_record(src, 1, np.int32)
                # print(nb_fathers)

                if nb_fathers > 0:
                    id_fathers = read_record(src, nb_fathers, np.int32, debug=debug)
                    m_fathers = read_record(src, nb_fathers, np.float32)

                    # print(m_fathers)
                    # print(id_fathers.shape, m_fathers.shape)

                    # print(
                    #     mynumber,
                    #     found_ids[:, istep - skip],
                    # )

                    # assert (
                    #     mynumber in found_ids[:, istep - skip]
                    # ), "Error... didn't find expected id at coordinates... byte map file likely wrong or mistmatched"

                    if mynumber in found_ids[:, istep - skip]:

                        found_arg = np.where(found_ids[:, istep - skip] == mynumber)[0]

                        if "m" in tgt_fields:
                            found_fields["m"][found_arg, istep - skip] = m * 1e11
                        if "macc" in tgt_fields:
                            found_fields["macc"][found_arg, istep - skip] = m * 1e11
                        if "level" in tgt_fields:
                            found_fields["level"][found_arg, istep - skip] = level
                        if "hosthalo" in tgt_fields:
                            found_fields["hosthalo"][found_arg, istep - skip] = hosthalo
                        if "hostsub" in tgt_fields:
                            found_fields["hostsub"][found_arg, istep - skip] = hostsub
                        if "nbsub" in tgt_fields:
                            found_fields["nbsub"][found_arg, istep - skip] = nbsub
                        if "nextsub" in tgt_fields:
                            found_fields["nextsub"][found_arg, istep - skip] = nextsub
                        if "x" in tgt_fields:
                            found_fields["x"][found_arg, istep - skip] = px
                        if "y" in tgt_fields:
                            found_fields["y"][found_arg, istep - skip] = py
                        if "z" in tgt_fields:
                            found_fields["z"][found_arg, istep - skip] = pz
                        if "vx" in tgt_fields:
                            found_fields["vx"][found_arg, istep - skip] = vx
                        if "vy" in tgt_fields:
                            found_fields["vy"][found_arg, istep - skip] = vy
                        if "vz" in tgt_fields:
                            found_fields["vz"][found_arg, istep - skip] = vz
                        if "Lx" in tgt_fields:
                            found_fields["Lx"][found_arg, istep - skip] = Lx
                        if "Ly" in tgt_fields:
                            found_fields["Ly"][found_arg, istep - skip] = Ly
                        if "Lz" in tgt_fields:
                            found_fields["Lz"][found_arg, istep - skip] = Lz
                        if "r" in tgt_fields:
                            found_fields["r"][found_arg, istep - skip] = r
                        if "ra" in tgt_fields:
                            found_fields["ra"][found_arg, istep - skip] = ra
                        if "rb" in tgt_fields:
                            found_fields["rb"][found_arg, istep - skip] = rb
                        if "rc" in tgt_fields:
                            found_fields["rc"][found_arg, istep - skip] = rc
                        if "ek" in tgt_fields:
                            found_fields["ek"][found_arg, istep - skip] = ek
                        if "ep" in tgt_fields:
                            found_fields["ep"][found_arg, istep - skip] = ep
                        if "et" in tgt_fields:
                            found_fields["et"][found_arg, istep - skip] = et
                        if "spin" in tgt_fields:
                            found_fields["spin"][found_arg, istep - skip] = spin

                        if nb_fathers > 1:  # if several follow main branch
                            massive_father = np.argmax(m_fathers)

                            main_id = id_fathers[massive_father]
                            # main_m = m_fathers[massive_father]

                            this_m_father = m_fathers[massive_father]

                            # print(list(zip(id_fathers, m_fathers * 1e11))s)
                        else:

                            this_m_father = 1.0
                            main_id = id_fathers
                            # main_m = m_fathers
                            # print(id_fathers, m_fathers * 1e11)

                        if "m_father" in tgt_fields:
                            found_fields["m_father"][
                                found_arg, istep - skip
                            ] = this_m_father  # * 1e11

                        # print(found_arg, np.sum(found_ids[:, istep - skip] == mynumber))

                        if istep < nsteps - 1:
                            found_ids[found_arg, istep - skip + 1] = main_id
                        # found_masses[found_arg, istep - skip + 1] = main_m * 1e11

                        # print(
                        # mynumber,
                        # found_ids[found_arg, istep - skip + 1],
                        # found_masses[found_arg, istep - skip + 1],
                        # )

                        nfound_this_step += 1

                        if nfound_this_step == len(
                            tgt_ids
                        ):  # if I found all the ids at this step... skip to next
                            # skip_bytes = zed_to_nbyte(
                            #     fbytes, 1.0 / tree_aexps[istep - skip + 1] - 1.0
                            # )
                            # src.seek(skip_bytes)
                            break  # get out of object id

                # nb_sons = read_record(src, 1, np.int32)

                # # print(nb_sons)

                # if nb_sons > 0:
                # id_sons = read_record(src, nb_sons, np.int32)
                # skip_record(src, nb_sons, np.int32)

                #     # print(id_sons)

                # # print(mynumber, id_fathers, id_sons)

                # skip_record(src, 1, np.int32)
                # skip_record(src, 1, np.int32)
                # if star:
                #     skip_record(src, 1, np.int32)

    return found_ids, found_fields, tree_aexps[skip:]


def read_tree_file_rev_correct_pos(
    fname,
    sim,
    snap,
    fbytes,
    zstart,
    tgt_ids,
    star=True,
    tgt_fields=["m", "x", "y", "z", "r"],
    fpure_min=1 - 1e-4,
    speed_lim=5e4,
    verbose=False,
    overwrite=False
):

    # using the offset files to jump to the right byte position,
    # follow the main branch from zstart to the highest possible redshift
    # in the tree for all tgt_ids

    if not hasattr(sim, "times"):
        sim.get_snap_times()

    tgt_ids = np.asarray(np.unique(np.sort(np.asarray(tgt_ids, dtype=np.int32))), dtype=np.int32)

    out_path = os.path.join(sim.path, "saved_trees")
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    out_name = os.path.join(
        out_path,
        f"tree_rev_ids_start_tgt_snap_{snap:d}_z{zstart:.2f}_fpure{fpure_min:.1e}_speed{speed_lim:.1e}.h5",
    )

    if overwrite or (not os.path.exists(out_name)):
        ids_to_process = tgt_ids
        write_mode = "w"
    else:
        with h5py.File(out_name, "r") as h5src:
            saved_ids = np.asarray(
                [int(k) for k in h5src.keys() if k != "aexps"], dtype=np.int32
            )
        ids_to_process = np.setdiff1d(tgt_ids, saved_ids, assume_unique=True)
        write_mode = "a"

    if len(ids_to_process) == 0 and os.path.exists(out_name):
        with h5py.File(out_name, "r") as h5src:
            tree_aexps = np.asarray(h5src["aexps"][...], dtype=np.float32)
            found_ids = np.full((len(tgt_ids), len(tree_aexps)), -1, dtype=np.int32)
            found_fields = {}
            for tgt_f in tgt_fields:
                found_fields[tgt_f] = np.full(
                    (len(tgt_ids), len(tree_aexps)), -1, dtype=np.float32
                )
            for i, tid in enumerate(tgt_ids):
                tid_key = str(int(tid))
                if tid_key in h5src:
                    tid_group = h5src[tid_key]
                    if "ids" in tid_group:
                        found_ids[i] = tid_group["ids"][...]
                    for tgt_f in tgt_fields:
                        if tgt_f in tid_group:
                            found_fields[tgt_f][i] = tid_group[tgt_f][...]
        return found_ids, found_fields, tree_aexps

    assoc_snaps = find_snaps_with_halos(sim.snap_numbers, sim.path)

    if star:
        dt = np.float64
    else:
        dt = np.float32

    with open(fname, "rb") as src:

        nsteps = read_record(src, 1, np.int32, debug=verbose)
        nb_halos = read_record(src, nsteps * 2, np.int32, debug=verbose)
        nb_halos, nb_shalos = nb_halos[:nsteps], nb_halos[nsteps:]
        tree_aexps = read_record(src, nsteps, np.float32, debug=verbose)
        skip_record(src, 1, verbose)
        skip_record(src, 1, verbose)

        skip = np.argmin(np.abs(tree_aexps - (1.0 / (1.0 + zstart))))

        found_ids = np.full((len(ids_to_process), nsteps - skip), -1, dtype=np.int32)
        found_fields = {}
        for tgt_f in tgt_fields:
            found_fields[tgt_f] = np.full(
                (len(ids_to_process), nsteps - skip), -1, dtype=np.float32
            )

        found_ids[:, 0] = np.sort(ids_to_process)

        for istep in range(skip, nsteps):

            if istep > skip and np.all(found_ids[:, istep - skip - 1] == -1):
                print("stopping tree... no fathers found in previous step")
                break

            nyte_skip = istep_to_nbyte(fbytes, istep)
            src.seek(nyte_skip)

            if verbose:
                print(f"step {istep}, z={1.0 / tree_aexps[istep] - 1.0:.4f}")

            nfound_this_step = 0

            cur_aexp = tree_aexps[istep]
            cur_snap = sim.get_closest_snap(aexp=cur_aexp)
            cur_assoc_exists = snap in assoc_snaps

            if np.any(sim.snap_numbers < cur_snap) and cur_assoc_exists:

                prev_snap = np.intersect1d(
                    assoc_snaps, sim.snap_numbers[sim.snap_numbers < cur_snap]
                )
                if len(prev_snap) >= 1:
                    prev_snap = prev_snap[-1]
                else:
                    print("stopping tree... no galaxies at previous step")
                    continue

                cur_time, prev_time = sim.times[
                    np.in1d(sim.snap_numbers, [cur_snap, prev_snap])
                ]
                prev_aexp = sim.aexps[np.where(prev_snap == sim.snap_numbers)[0][0]]
                dt_step = abs(cur_time - prev_time)
                dt_steps = dt_step * (3600 * 25 * 365 * 1e6)

                vect_iobjs = np.sort(found_ids[:, istep - skip])
                vect_obj_bytes = iobjs_to_nbytes(fbytes, istep, vect_iobjs)

                for iobj, obj_bytes in enumerate(vect_obj_bytes):
                    src.seek(obj_bytes)

                    mynumber = read_record(src, 1, np.int32, debug=verbose)
                    bushID = read_record(src, 1, np.int32, debug=verbose)
                    mystep = read_record(src, 1, np.int32, debug=verbose) - 1
                    level, hosthalo, hostsub, nbsub, nextsub = read_record(
                        src, 5, np.int32, debug=verbose
                    )
                    m = read_record(src, 1, np.float32)
                    macc = read_record(src, 1, dt)
                    px, py, pz = read_record(src, 3, dt)
                    vx, vy, vz = read_record(src, 3, np.float32)
                    Lx, Ly, Lz = read_record(src, 3, np.float32)
                    r, ra, rb, rc = read_record(src, 4, np.float32)
                    ek, ep, et = read_record(src, 3, np.float32)
                    spin = read_record(src, 1, np.float32)
                    nb_fathers = read_record(src, 1, np.int32)

                    if nb_fathers > 0:
                        id_fathers = read_record(src, nb_fathers, np.int32, debug=verbose)
                        m_fathers = read_record(src, nb_fathers, np.float32)

                        if mynumber in found_ids[:, istep - skip]:
                            found_arg = np.where(found_ids[:, istep - skip] == mynumber)[0]

                            if "m" in tgt_fields:
                                found_fields["m"][found_arg, istep - skip] = m * 1e11
                            if "macc" in tgt_fields:
                                found_fields["macc"][found_arg, istep - skip] = m * 1e11
                            if "level" in tgt_fields:
                                found_fields["level"][found_arg, istep - skip] = level
                            if "hosthalo" in tgt_fields:
                                found_fields["hosthalo"][found_arg, istep - skip] = hosthalo
                            if "hostsub" in tgt_fields:
                                found_fields["hostsub"][found_arg, istep - skip] = hostsub
                            if "nbsub" in tgt_fields:
                                found_fields["nbsub"][found_arg, istep - skip] = nbsub
                            if "nextsub" in tgt_fields:
                                found_fields["nextsub"][found_arg, istep - skip] = nextsub
                            if "x" in tgt_fields:
                                found_fields["x"][found_arg, istep - skip] = px
                            if "y" in tgt_fields:
                                found_fields["y"][found_arg, istep - skip] = py
                            if "z" in tgt_fields:
                                found_fields["z"][found_arg, istep - skip] = pz
                            if "vx" in tgt_fields:
                                found_fields["vx"][found_arg, istep - skip] = vx
                            if "vy" in tgt_fields:
                                found_fields["vy"][found_arg, istep - skip] = vy
                            if "vz" in tgt_fields:
                                found_fields["vz"][found_arg, istep - skip] = vz
                            if "Lx" in tgt_fields:
                                found_fields["Lx"][found_arg, istep - skip] = Lx
                            if "Ly" in tgt_fields:
                                found_fields["Ly"][found_arg, istep - skip] = Ly
                            if "Lz" in tgt_fields:
                                found_fields["Lz"][found_arg, istep - skip] = Lz
                            if "r" in tgt_fields:
                                found_fields["r"][found_arg, istep - skip] = r
                            if "ra" in tgt_fields:
                                found_fields["ra"][found_arg, istep - skip] = ra
                            if "rb" in tgt_fields:
                                found_fields["rb"][found_arg, istep - skip] = rb
                            if "rc" in tgt_fields:
                                found_fields["rc"][found_arg, istep - skip] = rc
                            if "ek" in tgt_fields:
                                found_fields["ek"][found_arg, istep - skip] = ek
                            if "ep" in tgt_fields:
                                found_fields["ep"][found_arg, istep - skip] = ep
                            if "et" in tgt_fields:
                                found_fields["et"][found_arg, istep - skip] = et
                            if "spin" in tgt_fields:
                                found_fields["spin"][found_arg, istep - skip] = spin

                            if nb_fathers > 1:
                                cur_pos = np.array([px, py, pz])
                                convert_adaptahop_pos(sim.cosmo.lcMpc * cur_aexp, cur_pos)

                                cur_m_fathers = m_fathers[id_fathers > 0]
                                cur_id_fathers = id_fathers[id_fathers > 0]

                                if prev_aexp == tree_aexps[istep + 1]:
                                    hprops_prev = {}
                                    prev_hids = np.zeros(
                                        len(cur_id_fathers), dtype=np.int32
                                    )
                                    prev_fpures = np.zeros(
                                        len(cur_id_fathers), dtype=np.float32
                                    )
                                    prev_rvirs = np.zeros(
                                        len(cur_id_fathers), dtype=np.float32
                                    )
                                    hprops_prev["pos"] = np.zeros((len(cur_id_fathers), 3))
                                    cur_fathers_hprops_prev = get_halo_props_snap(
                                        sim.path, prev_snap, cur_id_fathers
                                    )

                                    for i, id_father in enumerate(cur_id_fathers):
                                        halo_tag = f"halo_{id_father:07d}"
                                        hprops_prev[halo_tag] = {}
                                        cur_hprops_prev = cur_fathers_hprops_prev[halo_tag]
                                        for k in cur_hprops_prev.keys():
                                            hprops_prev[halo_tag][k] = cur_hprops_prev[k]
                                        prev_hids[i] = id_father
                                        prev_fpures[i] = cur_hprops_prev["fpure"]
                                        prev_rvirs[i] = cur_hprops_prev["rvir"]

                                    fpure_cond = prev_fpures > fpure_min

                                    if not np.any(fpure_cond) and verbose:
                                        print(
                                            f"Error... no fathers purer than {fpure_min:.5f}"
                                        )

                                    pos_father = np.zeros((len(cur_id_fathers), 3))
                                    for ihid, prev_hid in enumerate(prev_hids):
                                        cur_father_props = hprops_prev[f"halo_{prev_hid:07d}"]
                                        pos_father[ihid, :] = cur_father_props["pos"]

                                    distances = np.linalg.norm(pos_father - cur_pos, axis=1)
                                    dists_km = distances * (
                                        sim.cosmo.lcMpc * 1e6 * 3.08e16 / 1e3
                                    )
                                    speed_cond = (dists_km / dt_steps) < speed_lim

                                    if not np.any(speed_cond) and verbose:
                                        print(
                                            f"Error... no fathers within speed limit of {speed_lim:.2f} km/s"
                                        )

                                    if np.sum(speed_cond * fpure_cond) < 1:
                                        all_cond = fpure_cond
                                    else:
                                        all_cond = speed_cond * fpure_cond
                                        if all_cond.sum() < fpure_cond.sum() and verbose:
                                            print(
                                                "Warning... not all fathers are close to the current position"
                                            )

                                    if all_cond.sum() < len(all_cond):
                                        cur_m_fathers = cur_m_fathers[all_cond]
                                        cur_id_fathers = cur_id_fathers[all_cond]
                                        diff = np.setdiff1d(id_fathers, prev_hids)
                                        diff_arg = np.where(np.in1d(id_fathers, diff))[0]
                                        m_fathers = np.concatenate(
                                            [cur_m_fathers, m_fathers[diff_arg]]
                                        )
                                        id_fathers = np.concatenate(
                                            [cur_id_fathers, id_fathers[diff_arg]]
                                        )

                            if len(m_fathers) > 0:
                                massive_father = np.argmax(m_fathers)
                                main_id = id_fathers[massive_father]
                                this_m_father = m_fathers[massive_father]
                            else:
                                this_m_father = -1
                                main_id = -1

                            if "m_father" in tgt_fields:
                                found_fields["m_father"][found_arg, istep - skip] = this_m_father

                            if istep < nsteps - 1:
                                found_ids[found_arg, istep - skip + 1] = main_id

                            nfound_this_step += 1

                            if nfound_this_step == len(ids_to_process):
                                break

    if overwrite or (not os.path.exists(out_name)):
        mode = "w"
    else:
        mode = "a"
    with h5py.File(out_name, mode) as h5out:
        if "aexps" not in h5out or overwrite:
            if "aexps" in h5out:
                del h5out["aexps"]
            h5out.create_dataset("aexps", data=tree_aexps[skip:], dtype=np.float32)
        for i, tid in enumerate(ids_to_process):
            tid_key = str(int(tid))
            if tid_key in h5out:
                del h5out[tid_key]
            tid_group = h5out.require_group(tid_key)
            tid_group.attrs["target_id"] = int(tid)
            tid_group.create_dataset("ids", data=found_ids[i], dtype=np.int32)
            for tgt_f in tgt_fields:
                if tgt_f in found_fields:
                    tid_group.create_dataset(
                        tgt_f, data=found_fields[tgt_f][i], dtype=np.float32
                    )

    with h5py.File(out_name, "r") as h5src:
        tree_aexps = np.asarray(h5src["aexps"][...], dtype=np.float32)
        found_ids = np.full((len(tgt_ids), len(tree_aexps)), -1, dtype=np.int32)
        found_fields = {}
        for tgt_f in tgt_fields:
            found_fields[tgt_f] = np.full(
                (len(tgt_ids), len(tree_aexps)), -1, dtype=np.float32
            )
        for i, tid in enumerate(tgt_ids):
            tid_key = str(int(tid))
            if tid_key in h5src:
                tid_group = h5src[tid_key]
                if "ids" in tid_group:
                    found_ids[i] = tid_group["ids"][...]
                for tgt_f in tgt_fields:
                    if tgt_f in tid_group:
                        found_fields[tgt_f][i] = tid_group[tgt_f][...]

    return found_ids, found_fields, tree_aexps


# def revert_tree(fname, fbytes, star=True):

# not yet implemented... using YDs f90 file to revert the trees
# its very fast etc... so no need to do it in python

#     # using the offset files to jump to the right byte position,
#     # follow the main branch from zstart to the highest possible redshift
#     # in the tree for all tgt_ids

#     with open(fname, "rb") as src:

#         nsteps = read_record(src, 1, np.int32)
#         # print(nsteps)
#         # print(read_record(src, 1, np.int32))
#         # print(read_record(src, 1, np.int32))
#         nb_halos = read_record(src, nsteps * 2, np.int32, debug=False)
#         # nb_halos = np.fromfile(src, dtype=np.int32, count=2 * nsteps + 2)[1:-1]
#         nb_halos, nb_shalos = nb_halos[:nsteps], nb_halos[nsteps:]
#         # print(nb_halos, nb_shalos)
#         tree_aexps = read_record(src, nsteps, np.float32)
#         tree_omega_t = read_record(src, nsteps, np.float32)
#         tree_age_univ = read_record(src, nsteps, np.float32)

#         # print(nb_halos, nb_shalos, tree_aexps, tree_omega_t, tree_age_univ)
#         found_ids = np.full((len(tgt_ids), nsteps - skip), -1, dtype=np.int32)
#         found_masses = np.full((len(tgt_ids), nsteps - skip), -1, dtype=np.float32)

#         found_ids[:, 0] = tgt_ids

#         # print(nb_halos, nb_shalos, tree_aexps, tree_omega_t, tree_age_univ)

#         # print(nsteps - skip)

#         for istep in range(skip, nsteps):

#             # ntot = nb_halos[istep] + nb_shalos[istep]
#             # print(f"istep: {istep:d} has {ntot:d} halos+subhalos")
#             print(f"step {istep}, z={1.0 / tree_aexps[istep] - 1.0:.4f}")

#             # print("byte position is ", src.tell())
#             # we aren't at the first step, and didn't find any ids in the previous step... stop the tree
#             if istep > skip and np.all(found_ids[:, istep - skip - 1] == -1):
#                 break

#             nfound_this_step = 0

#             # for iobj in range(ntot):

#             for iobj in np.sort(found_ids[:, istep - skip]):

#                 obj_bytes = iobj_to_nbyte(fbytes, istep, iobj)

#                 src.seek(obj_bytes)

#                 # print("iobj:", iobj)

#                 mynumber = read_record(src, 1, np.int32)
#                 # # print(mynumber)
#                 # bushID = read_record(src, 1, np.int32)
#                 # # # print(bushID)
#                 # mystep = read_record(src, 1, np.int32) - 1  # py indexing
#                 # # # print(mystep)
#                 # level, hosthalo, hostsub, nbsub, nextsub = read_record(
#                 #     src, 5, np.int32, debug=False
#                 # )

#                 np.fromfile(src, dtype=np.int32, count=7 + 3 * 2)

#                 # print(level, hosthalo, hostsub, nbsub, nextsub)
#                 m = read_record(src, 1, np.float32)
#                 # print("%e" % (m * 1e11))
#                 # macc = read_record(src, 1, np.float32)
#                 # # print(macc)
#                 # px, py, pz = read_record(src, 3, np.float32)
#                 # # print(px, py, pz)
#                 # vx, vy, vz = read_record(src, 3, np.float32)
#                 # # print(vx, vy, vz)
#                 # Lx, Ly, Lz = read_record(src, 3, np.float32)
#                 # # print(Lx, Ly, Lz)
#                 # r, ra, rb, rc = read_record(src, 4, np.float32)
#                 # # print(r, ra, rb, rc)
#                 # ek, ep, et = read_record(src, 3, np.float32)
#                 # # print(ek, ep, et)
#                 # spin = read_record(src, 1, np.float32)
#                 # # print(spin)

#                 np.fromfile(src, dtype=np.int32, count=18 + 7 * 2)

#                 nb_fathers = read_record(src, 1, np.int32)
#                 # print(nb_fathers)

#                 if nb_fathers > 0:
#                     id_fathers = read_record(src, nb_fathers, np.int32, debug=False)
#                     m_fathers = read_record(src, nb_fathers, np.float32)

#                     # print(m_fathers)
#                     # print(id_fathers.shape, m_fathers.shape)

#                     # print(
#                     #     mynumber,
#                     #     found_ids[:, istep - skip],
#                     # )

#                     # assert (
#                     #     mynumber in found_ids[:, istep - skip]
#                     # ), "Error... didn't find expected id at coordinates... byte map file likely wrong or mistmatched"

#                     if mynumber in found_ids[:, istep - skip]:

#                         found_arg = np.where(found_ids[:, istep - skip] == mynumber)[0]

#                         found_masses[found_arg, istep - skip] = m * 1e11

#                         if nb_fathers > 1:  # if several follow main branch
#                             massive_father = np.argmax(m_fathers)

#                             # print(massive_father)

#                             main_id = id_fathers[massive_father]
#                             # main_m = m_fathers[massive_father]

#                             # print(list(zip(id_fathers, m_fathers * 1e11)))
#                         else:

#                             main_id = id_fathers
#                             # main_m = m_fathers
#                             # print(id_fathers, m_fathers * 1e11)

#                         # print(found_arg, np.sum(found_ids[:, istep - skip] == mynumber))

#                         if istep < nsteps - 1:
#                             found_ids[found_arg, istep - skip + 1] = main_id
#                         # found_masses[found_arg, istep - skip + 1] = main_m * 1e11

#                         # print(
#                         # mynumber,
#                         # found_ids[found_arg, istep - skip + 1],
#                         # found_masses[found_arg, istep - skip + 1],
#                         # )

#                         nfound_this_step += 1

#                         if nfound_this_step == len(
#                             tgt_ids
#                         ):  # if I found all the ids at this step... skip to next
#                             # skip_bytes = zed_to_nbyte(
#                             #     fbytes, 1.0 / tree_aexps[istep - skip + 1] - 1.0
#                             # )
#                             # src.seek(skip_bytes)
#                             break  # get out of object id

#                 # nb_sons = read_record(src, 1, np.int32)

#                 # # print(nb_sons)

#                 # if nb_sons > 0:
#                 # id_sons = read_record(src, nb_sons, np.int32)
#                 # skip_record(src, nb_sons, np.int32)

#                 #     # print(id_sons)

#                 # # print(mynumber, id_fathers, id_sons)

#                 # skip_record(src, 1, np.int32)
#                 # skip_record(src, 1, np.int32)
#                 # if star:
#                 #     skip_record(src, 1, np.int32)

#     return found_ids, found_masses, tree_aexps[skip:]


def smooth_dm_tree(sim, tree_aexps, tree_ids, tree_datas):

    tree_pos = tree_datas["pos"][0]
    tree_mass = tree_datas["m"][0]

    # start at end of tree

    # go through tree, checking that :
    #   -next step is pure!
    #   -mass is smaller
    #   -next step is nearby
    #   -next step has most of sinks
    # if none of these is satisfied, check dm part ids by hand...
