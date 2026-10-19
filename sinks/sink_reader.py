import numpy as np
import os
from gremlin.read_sim_params import ramses_sim
from f90_tools.IO import read_tgt_fields, read_record
from zoom_analysis.halo_maker.assoc_fcts import (
    find_zoom_tgt_halo,
    get_central_gal_for_hid,
    get_gal_props_snap,
    get_halo_props_snap
)
import functools
from scipy.spatial import KDTree

def get_sink_info_path(p, snap):
    return os.path.join(p, f"output_{snap:05d}", f"sink_{snap:05d}.info")


def get_sink_csv_path(p, snap):
    return os.path.join(p, f"output_{snap:05d}", f"sink_{snap:05d}.csv")


def get_nsink(p, snap):

    fname = get_sink_info_path(p, snap)

    with open(fname, "r") as f:
        for line in f:
            return int(line.split("=")[-1].strip())

    return


def read_sink_info(p, snap):

    fname = get_sink_info_path(p, snap)

    with open(fname, "r") as f:
        lines = f.readlines()

    sinks = {}

    keys = [elem for elem in lines[2].split(" ") if elem != "" and elem != "\n"]

    datas = np.zeros((int(lines[0].split("=")[-1].strip()), len(keys)))

    # print(keys, datas.shape)

    for iline, line in enumerate(lines[4:]):

        if line.__contains__("=="):
            continue

        split_line = [elem for elem in line.split(" ") if elem != "" and elem != "\n"]
        # print(line, split_line)

        datas[iline, :] = np.array(np.float32(split_line))

    for ikey, key in enumerate(keys):

        sinks[key] = datas[:, ikey]

    return sinks


def get_sink_info(sid, p, snap):

    sinks = read_sink_info(p, snap)
    # print(sinks)
    id_arg = np.where(sinks["Id"] == sid)[0]

    sink = {}

    for key in sinks.keys():
        sink[key] = sinks[key][id_arg][0]

    return sink


def read_sink_csv(p, snap):

    fname = get_sink_csv_path(p, snap)

    keys = ["Id", "Mass(Msol)", "x", "y", "z", "vx", "vy", "vz", "tsink", "mdot_BH"]

    nsink = get_nsink(p, snap)

    sinks = np.zeros((nsink, len(keys)))
    sinks = np.genfromtxt(fname, delimiter=",")

    if len(sinks) == 0:
        return {}

    # convert units
    sim = ramses_sim(p)

    unit_l = sim.cosmo["unit_l"]
    unit_d = sim.cosmo["unit_d"]
    unit_t = sim.cosmo["unit_t"]

    myr = 3600 * 24 * 365 * 1e6  # s
    unit_m = unit_d * unit_l**3 / 2e33

    sinks[:, 5:8] *= unit_l / unit_t * 1e-6  # km/s

    sinks[:, 1] *= unit_m  # Msun

    sinks[:, 8] *= unit_t / myr  # Myr

    sinks[:, 9] *= unit_m / unit_t * myr  # Msun/Myr

    sink_dict = {}

    for ikey, key in enumerate(keys):
        sink_dict[key] = sinks[:, ikey]

    return sink_dict


def get_sink_csv(sid, p, snap):

    sinks = read_sink_csv(p, snap)

    id_arg = np.where(sinks["Id"] == sid)[0]

    sink = {}

    for key in sinks.keys():
        sink[key] = sinks[key][id_arg][0]

    return sink


def find_closest_sink(pos, p, snap, rmax=None):

    sinks = read_sink_csv(p, snap)

    dists = np.linalg.norm(
        pos[:, None] - np.asarray([sinks["x"], sinks["y"], sinks["z"]]), axis=1
    )

    if rmax is None:
        print("No rmax given, returning closest sink")
        close_arg = np.argmin(dists)

    else:
        print(f"Returning sinks within {rmax} of {pos}")
        close_arg = np.where(dists < rmax)[0]

    # get sink info
    sink = {}
    for key in sinks.keys():
        sink[key] = sinks[key][close_arg]

    return sink

@functools.cache
def snap_to_coarse_step(snap, sim, max_iter=200, delta_aexp=1e-4, **kwargs):

    aexp = sim.get_snap_exps(snap)[0]

    sink_path = sim.sink_path

    sink_files = np.asarray([f for f in os.listdir(sink_path) if ".dat" in f])

    fnbs = np.array([int(f.split("_")[-1].split(".")[0]) for f in sink_files])
    sort = np.argsort(fnbs)
    fnbs = fnbs[sort]
    sink_files = sink_files[sort]

    nfiles = len(fnbs)
    # inspect = int(0.5 * nfiles)

    found = False
    coarse_nb = -1

    # print(fnbs.max(), nfiles, len(fnbs))

    # print(snap,aexp)

    # print(len(fnbs))

    # last_inspect = 0
    istep = 0

    bot = 0
    mid = int(0.5 * nfiles)
    top = nfiles - 1
    keep_top=False
    keep_bot=False

    while not found and istep < max_iter:

        
        if not keep_top:top_aexp = read_sink_bin(
            os.path.join(sink_path, sink_files[top]), [], **kwargs
        )["aexp"]
        mid_aexp = read_sink_bin(
            os.path.join(sink_path, sink_files[mid]), [], **kwargs
        )["aexp"]
        if not keep_bot:bot_aexp = read_sink_bin(
            os.path.join(sink_path, sink_files[bot]), [], **kwargs
        )["aexp"]
        

        if abs(top_aexp - aexp) < delta_aexp:
            found = True
            coarse_nb = fnbs[top]
        elif abs(bot_aexp - aexp) < delta_aexp:
            found = True
            coarse_nb = fnbs[bot]
        elif mid_aexp > aexp > bot_aexp:
            keep_bot=True
            keep_top=True
            top_aexp = mid_aexp
            top = mid
            mid = int(0.5 * (mid + bot))
        elif top_aexp > aexp > mid_aexp:
            keep_bot=True
            keep_top=True
            bot_aexp = mid_aexp
            bot = mid
            mid = int(0.5 * (mid + top))

        # print(bot_aexp, mid_aexp, top_aexp, "->", aexp, found)

        if mid == top:
            found = True
            coarse_nb = fnbs[mid]
        elif mid == bot:
            found = True
            coarse_nb = fnbs[bot]

        istep += 1

    # print(istep, max_iter)
    # assert istep < max_iter, "Failed to find coarse step"

    return coarse_nb, found


def coarse_step_to_snap(coarse_nb, sim, **kwargs):

    coarse_aexp = read_sink_bin(
        os.path.join(sim.sink_path, f"sink_{coarse_nb:05d}.dat"), [], **kwargs
    )["aexp"]

    sim_aexps = sim.get_snap_exps()

    snap = np.argmin(np.abs(sim_aexps - coarse_aexp))

    return snap


def read_sink_bin(fname, tgt_fields=None, sid=None, **kwargs):

    data = {}

    hagn = False
    if "hagn" in kwargs.keys():
        hagn = kwargs["hagn"]
    hdr_only = kwargs.get("hdr_only", False)
    debug = kwargs.get("debug", False)

    fields = [
        #     ("identity", 1, "i4"),
        #     ("mass", 1, "f8"),
        #     ("position", 3, "f8"),
        #     ("velocity", 3, "f8"),
        #     ("birth_time", 1, "f8"),
        #     ("dMsmbh", 1, "f8"),
        #     ("dMBH_coarse", 1, "f8"),
        #     ("dMEd_coarse", 1, "f8"),
        #     ("Esave", 1, "f8"),
        #     ("jsinks", 3, "f8"),
        #     ("spins", 3, "f8"),
        #     ("spin_magnitude", 1, "f8"),
        #     ("eps_sink", 1, "f8"),
        ("identity", 1, "i4"),
        ("mass", 1, "f8"),
        ("position", 3, "f8"),
        ("velocity", 3, "f8"),
        ("angular_mom", 3, "f8"),
        ("birth_time", 1, "f8"),
        ("dMBH_coarse", 1, "f8"),
        ("dMEd_coarse", 1, "f8"),
        ("dMsmbh", 1, "f8"),
        ("dens", 1, "f8"),
        ("csound", 1, "f8"),
        ("vrel", 1, "f8"),
        ("Esave", 1, "f8"),
        ("j_sink", 3, "f8"),
        ("spin_magnitude", 1, "f8"),
        ("eps_sink", 1, "f8"),
    ]  # what is sink stat ? ndim*2+1... loop on levels
    # if RT... LAGN_coarse

    # print(hagn)

    if hagn:  # accomodate hagn format

        fields = (np.asarray(fields)[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]]).tolist()
        # hagn doesnt have birth time or spin stuff
        fields = [
            (f[0], int(f[1]), f[2]) for f in fields
        ]  # need to remake tuples avec this horrible trick, integers where strings

    if tgt_fields is None:
        tgt_fields = fields

    with open(fname, "rb") as src:

        data["nsink"] = read_record(src, 1, np.int32)
        data["ndim"] = read_record(src, 1, np.int32)
        data["aexp"] = read_record(src, 1, np.float64)
        data["unit_l"] = read_record(src, 1, np.float64)
        data["unit_d"] = read_record(src, 1, np.float64)
        data["unit_t"] = read_record(src, 1, np.float64)

        if not hdr_only:

            if debug:
                print(data["nsink"], data["ndim"])
                print(data["aexp"], data["unit_l"], data["unit_d"], data["unit_t"])

            if tgt_fields != []:

                if sid == None:

                    read_tgt_fields(
                        data,
                        tgt_fields,
                        fields,
                        src,
                        data["nsink"],
                        debug=debug
                    )

                else:

                    ids = read_record(src, data["nsink"], fields[0][2])

                    # print(fname,ids,sid)

                    id_arg = np.where(ids == sid)[0]

                    if len(id_arg) == 0:
                        return {}

                    if type(ids) in [list, np.ndarray]:
                        data["identity"] = ids[id_arg]
                    else:
                        data["identity"] = ids

                    read_tgt_fields(
                        data,
                        tgt_fields[1:],
                        fields[1:],
                        src,
                        data["nsink"],
                        args=id_arg,
                        debug=debug,
                    )

    return data



def read_sink_bin_NH(fname, tgt_fields=None, sid=None, **kwargs):

    data = {}

    hagn = False
    if "hagn" in kwargs.keys():
        hagn = kwargs["hagn"]
    hdr_only = kwargs.get("hdr_only", False)

    fields = [
        #     ("identity", 1, "i4"),
        #     ("mass", 1, "f8"),
        #     ("position", 3, "f8"),
        #     ("velocity", 3, "f8"),
        #     ("birth_time", 1, "f8"),
        #     ("dMsmbh", 1, "f8"),
        #     ("dMBH_coarse", 1, "f8"),
        #     ("dMEd_coarse", 1, "f8"),
        #     ("Esave", 1, "f8"),
        #     ("jsinks", 3, "f8"),
        #     ("spins", 3, "f8"),
        #     ("spin_magnitude", 1, "f8"),
        #     ("eps_sink", 1, "f8"),
        ("identity", 1, "i4"),
        ("mass", 1, "f8"),
        ("position", 3, "f8"),
        ("velocity", 3, "f8"),
        ("angular_mom", 3, "f8"),
        # ("birth_time", 1, "f8"),
        ("dMBH_coarse", 1, "f8"),
        ("dMEd_coarse", 1, "f8"),
        ("dMsmbh", 1, "f8"),
        ("dens", 1, "f8"),
        ("csound", 1, "f8"),
        ("vrel", 1, "f8"),        
        ("Esave", 1, "f8"),
        ("j_sink", 3, "f8"),
        # ("spins", 3, "f8"),
        ("spin_magnitude", 1, "f8"),
        ("eps_sink", 1, "f8"),
        # ("sink_stat", 1, "f8"),

    ]  # what is sink stat ? ndim*2+1... loop on levels
    # if RT... LAGN_coarse

    # print(hagn)

    if hagn:  # accomodate hagn format

        fields = (np.asarray(fields)[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]]).tolist()
        # hagn doesnt have birth time or spin stuff
        fields = [
            (f[0], int(f[1]), f[2]) for f in fields
        ]  # need to remake tuples avec this horrible trick, integers where strings

    if tgt_fields is None:
        tgt_fields = fields

    with open(fname, "rb") as src:

        data["nsink"] = read_record(src, 1, np.int32)
        data["ndim"] = read_record(src, 1, np.int32)
        data["aexp"] = read_record(src, 1, np.float64)
        data["unit_l"] = read_record(src, 1, np.float64)
        data["unit_d"] = read_record(src, 1, np.float64)
        data["unit_t"] = read_record(src, 1, np.float64)

        if not hdr_only:

            # print(data["nsink"], data["ndim"], data["aexp"])
            # print(data["aexp"], data["unit_l"], data["unit_d"], data["unit_t"])

            if tgt_fields != []:

                if sid == None:

                    read_tgt_fields(
                        data,
                        tgt_fields,
                        fields,
                        src,
                        data["nsink"],
                    )

                else:

                    ids = read_record(src, data["nsink"], fields[0][2])

                    # print(fname,ids,sid)

                    id_arg = np.where(ids == sid)[0]

                    if len(id_arg) == 0:
                        return {}

                    if type(ids) in [list, np.ndarray]:
                        data["identity"] = ids[id_arg]
                    else:
                        data["identity"] = ids

                    read_tgt_fields(
                        data,
                        tgt_fields[1:],
                        fields[1:],
                        src,
                        data["nsink"],
                        args=id_arg,
                        debug=False,
                    )

    return data

def read_sink_snap(sim:ramses_sim, snap,**kwargs):

    coarse_nb, found = snap_to_coarse_step(snap,sim,**kwargs)

    if found:

        sinks = read_sink_bin(os.path.join(sim.sink_path,f'sink_{coarse_nb:05d}.dat'),**kwargs)

        if sinks is not None and sinks!={}:
            convert_sink_units(sinks, sinks['aexp'], sim)

        return sinks

    else:

        return {}

def read_sink_output(fname, fdesc, sid=None, **kwargs):

    data = {}

    hagn = False
    if "hagn" in kwargs.keys():
        hagn = kwargs["hagn"]
    hdr_only = kwargs.get("hdr_only", False)
    debug = kwargs.get("debug", False)
    tgt_fields = kwargs.get("tgt_fields", None)

    desc = np.genfromtxt(fdesc,comments='#',delimiter=',',dtype=np.dtype(('i4,U15,U2')))

    fields=[]

    type_dict={
        'i':'i4',
        'd':'f8'
    }

    for field in desc:

        fields.append((field[1].strip(),1,type_dict[field[2].strip()]))
    # print(hagn)

    if tgt_fields is None:
        tgt_fields = fields

    print(desc,fields)

    with open(fname, "rb") as src:

        data["nsink"] = read_record(src, 1, np.int32)
        data["nindsink"] = read_record(src, 1, np.int32)

        if not hdr_only:

            if debug:
                print(data["nsink"])

            if tgt_fields != []:

                if sid == None:

                    read_tgt_fields(
                        data,
                        tgt_fields,
                        fields,
                        src,
                        data["nsink"],
                        debug=debug
                    )

                else:

                    ids = read_record(src, data["nsink"], fields[0][2])

                    # print(fname,ids,sid)

                    id_arg = np.where(ids == sid)[0]

                    if len(id_arg) == 0:
                        return {}

                    if type(ids) in [list, np.ndarray]:
                        data["identity"] = ids[id_arg]
                    else:
                        data["identity"] = ids

                    read_tgt_fields(
                        data,
                        tgt_fields[1:],
                        fields[1:],
                        src,
                        data["nsink"],
                        args=id_arg,
                        debug=False,
                    )

    return data



def convert_sink_units(sinks, aexp, sim: ramses_sim, coarse_info=None):

    # print(sinks, aexp, sim)

    unit_l = sim.unit_l(aexp)
    unit_d = sim.unit_d(aexp)
    unit_t = sim.unit_t(aexp)
    unit_m = unit_d * unit_l**3 / 2e33

    if "mass" in sinks:
        sinks["mass"] *= unit_m

    if "dens" in sinks:
        sinks["dens"] *= unit_d / 1.67262192e-24 * 0.76  # Hpcc

    if "vrel" in sinks:
        sinks["vrel"] *= unit_l / unit_t * 1e-5  # to km/s
    
    if "velocity" in sinks:
        sinks["velocity"] *= unit_l / unit_t * 1e-5  # to km/s

    if "csound" in sinks:
        sinks["csound"] *= unit_l / unit_t * 1e-5  # to km/s

    # print(
    #     list(
    #         map(
    #             lambda x: (sinks[x].min(), sinks[x].max(), sinks[x].mean()),
    #             ["dMsmbh", "dMBH_coarse", "dMEd_coarse"],
    #         )
    #     )
    # )




    if "dMBH_coarse" in sinks:
        sinks["dMBH_coarse"] *= unit_m / (unit_t / (3600.0 * 24 * 365))  # msun/yr

    found = True

    if "dMsmbh" in sinks:
        sinks["dMsmbh"] *= unit_m  # msun
        if coarse_info == None:
            coarse_steps, coarse_zeds, coarse_times = get_coarse_dts(sim)
        else:
            coarse_steps, coarse_zeds, coarse_times = coarse_info

        # print(coarse_steps, coarse_zeds, coarse_times)

        # print(aexp)

        cur_coarse_step, found = snap_to_coarse_step(sim.get_closest_snap(aexp), sim)

        if not found:
            print("Failed to find coarse step")
            return

        if cur_coarse_step > coarse_steps.min():
            cur_arg = np.where(coarse_steps == cur_coarse_step)[0]
            cur_t = coarse_times[cur_arg][0]
            prev_t = coarse_times[cur_arg - 1][0]
            sinks["dMsmbhdt_coarse"] = sinks["dMsmbh"] / (cur_t - prev_t)  # msun/yr

    if "dMEd_coarse" in sinks:
        sinks["dMEd_coarse"] *= unit_m / (unit_t / (3600.0 * 24 * 365))  # msun/yr

    return found


def find_massive_sink(
    pos, snap, sim: ramses_sim, rmax=None, all_sinks=False, verbose=False, coarse_step=None, **kwargs
):

    # sink_files = np.asarray(os.listdir(sim.sink_path))
    # sink_fnbs = np.asarray([int(f.split("_")[-1].split(".")[0]) for f in sink_files])

    if "tgt_fields" not in kwargs.keys():
        kwargs["tgt_fields"] = ["identity", "mass", "position"]

    if coarse_step is None:
        coarse_step, found = snap_to_coarse_step(snap, sim)
    else:
        found=True

    if found == False or coarse_step == -1:
        print("Failed to find coarse step")
        return {},None

    # print(snap, coarse_step)
    # sink_f = sink_files[coarse_step == sink_fnbs][0]
    # print(sink_files, coarse_step, sink_f)
    # sinks = read_sink_bin(os.path.join(sim.sink_path, sink_f))
    sinks = read_sink_bin(
        os.path.join(sim.sink_path, f"sink_{coarse_step:05d}.dat"),
        **kwargs,
    )

    aexp = sim.get_snap_exps(snap, param_save=False)[0]

    convert_sink_units(sinks, aexp, sim)

    if len(sinks) == 0:
        return {},None
    elif type(sinks["mass"]) in [list,np.ndarray]:
        if len(sinks["mass"]) == 1:
            return sinks,coarse_step
        elif len(sinks["mass"])==0:
            return {},None

    masses = sinks["mass"]

    # print(masses, pos)
    # print(pos)
    # print(sinks["position"])

    if rmax is None:
        if verbose:
            print("No rmax given, returning most massive sink")
        close_arg = np.argmax(masses)

    else:
        if verbose:
            print(f"Returning sinks within {rmax} of {pos}")
        # print(sinks["position"], pos)
        dists = np.linalg.norm(pos[None, :] - sinks["position"], axis=1)
        # print(
        # dists.max(), dists.min(), rmax, pos[:], sinks["position"][np.argmin(dists)]
        # )
        close_arg = np.where(dists < rmax)[0]
        # print(close_arg, dists, dists[close_arg], dists.min(), rmax)
        if not all_sinks and len(close_arg) > 0:
            close_arg = close_arg[np.argmax(masses[close_arg])]

    # get sink info
    sink = {}

    # if close_arg in [int,np.int16,np.int32,np.int64]:


    # if len(close_arg) > 0:
    for key in sinks.keys():
        if type(sinks[key]) == np.ndarray:
            sink[key] = sinks[key][close_arg]

    # print(sink,close_arg,len(close_arg))

    return sink,coarse_step


def get_sink(sid, snap, sim: ramses_sim, **kwargs):

    # sink_files = np.asarray(os.listdir(sim.sink_path))
    # sink_fnbs = np.asarray([int(f.split("_")[-1].split(".")[0]) for f in sink_files])

    coarse_step, found = snap_to_coarse_step(snap, sim)
    # print(snap, coarse_step)
    # sink_f = sink_files[coarse_step == sink_fnbs][0]
    # print(sink_files, coarse_step, sink_f)
    # sinks = read_sink_bin(os.path.join(sim.sink_path, sink_f))
    sinks = read_sink_bin(
        os.path.join(sim.sink_path, f"sink_{coarse_step:05d}.dat"),
        tgt_fields=["identity", "mass", "position",'dMBH_coarse','dMEd_coarse'],
        **kwargs,
    )

    aexp = sim.get_snap_exps(snap, param_save=False)[0]

    convert_sink_units(sinks, aexp, sim)

    out_dict = {}

    found_sid = sinks["identity"] == sid
    if found_sid.sum() == 0:
        print("Didn't find sid")
        return {}
    sid_filter = np.where(found_sid)[0]

    for k in sinks.keys():
        if type(sinks[k]) in [np.ndarray, list]:
            out_dict[k] = sinks[k][sid_filter]
        else:
            out_dict[k] = sinks[k]

    return out_dict


def get_sink_mhistory(
    sid, snap, sim: ramses_sim, out_keys=None, debug=False, max_z=np.inf, skip=1, **kwargs
):

    hagn = False
    if "hagn" in kwargs.keys():
        hagn = kwargs["hagn"]

    # p = sim.path

    snaps = sim.snaps
    snap_nbs = sim.snap_numbers

    start_coarse, found = snap_to_coarse_step(snap, sim)

    coarse_info = get_coarse_dts(sim)

    order_snaps = np.argsort(snap_nbs)[::-1]

    snap_nbs = np.array(snap_nbs)[order_snaps]
    snaps = np.array(snaps)[order_snaps]

    # snaps_have_bhs = [os.path.isfile(get_sink_info_path(p, sn)) for sn in snap_nbs]

    # load first sink.dat and get aexp
    sink_files = np.asarray(os.listdir(sim.sink_path))
    sink_fnbs = np.asarray([int(f.split("_")[-1].split(".")[0]) for f in sink_files])

    if debug:
        print(start_coarse, sink_fnbs)

    sink_files = sink_files[sink_fnbs <= start_coarse]
    sink_fnbs = sink_fnbs[sink_fnbs <= start_coarse]
    sort = np.argsort(sink_fnbs)[::-1]
    sink_fnbs = sink_fnbs[sort]
    sink_files = np.array(sink_files)[sort]

    if debug:
        print(sink_files)

    if out_keys == None:
        out_keys = [
            ("zeds", 1),
            ("coarse_step", 1),
            ("mass", 1),
            # ("position",3),
            # ("velocity",3),
            ("birth_time", 1),
            ("dMBH_coarse", 1),
            ("dMEd_coarse", 1),
            ("dMsmbh", 1),
            ("vrel", 1),
            ("csound", 1),
        ]

        if hagn:
            out_keys = [
                ("zeds", 1),
                ("coarse_step", 1),
                ("mass", 1),
                # ("position",3),
                # ("velocity",3),
                ("dMBH_coarse", 1),
                ("dMEd_coarse", 1),
                ("dMsmbh", 1),
                ("vrel", 1),
                ("csound", 1),
            ]

    else:
        out_keys = [("zeds", 1), ("coarse_step", 1)] + out_keys

    bh_data = {}

    if debug:
        print(sink_fnbs, out_keys)

    for k in out_keys:
        bh_data[k[0]] = np.zeros((len(sink_fnbs//skip), k[1]))

    out_keys_for_read = [k[0] for k in out_keys]

    if debug:
        print(bh_data, out_keys_for_read)

    # print(sink_fnbs, sink_files)
    # print(len(sink_fnbs), len(sink_files))

    prev_sink={}

    lvls,zed_lvls,res_lvls = sim.compute_aexps_lvlchange()    

    for ifill, (sink_fnb, sink_file) in enumerate(zip(sink_fnbs[::skip], sink_files[::skip])):

        # sink = get_sink_csv(sid, p, snap_nb)

        sink = read_sink_bin(
            os.path.join(sim.sink_path, sink_file),
            tgt_fields=out_keys_for_read + ["identity"],
            sid=sid,
            debug=debug,
            **kwargs,
        )

    

        if debug:
            print(sink_fnb, sink_file, sink)
            # print(sink)

        if sink == {}:
            if debug:print('Failed to find sink in previous file',sink_file)
            continue
            # if prev_sink != {}:

            #     if debug:print('Looking for merger')
            #     all_sinks =  read_sink_bin(
            #     os.path.join(sim.sink_path, sink_file),
            #     tgt_fields=out_keys_for_read + ["identity"],
            #     debug=debug,
            #     **kwargs,
            #     )

            #     convert_sink_units(all_sinks,all_sinks['aexp'],sim)

            #     if all_sinks != {}:

            #         lookup_pos = all_sinks['position']

            #         if len(lookup_pos.shape)<2:
            #             lookup_pos=[lookup_pos]
            #         sink_tree = KDTree(lookup_pos,boxsize=1.0+1e-6)

            #         # prev_sink_file = sink_files[::skip][ifill+1]
            #         # prev_sink =         sink = read_sink_bin(
            #         #     os.path.join(sim.sink_path, prev_sink_file),
            #         #     tgt_fields=out_keys_for_read + ["identity"],
            #         #     sid=sid,
            #         #     debug=debug,
            #         #     **kwargs,
            #         # )

            #         # print(sink_file,prev_sink,sink)

            #         dx_min_pc = res_lvls[np.argmin(np.abs(zed_lvls-1./all_sinks["aexp"]-1.))]
            #         # dx_min_code = dx_min_pc / (1e6) / sim.cosmo.lcMpc
            #         # ok_args_dist = sink_tree.query_ball_point(prev_sink['position'],r=10*dx_min_code)
            #         dx_min_code = 10e-3 / sim.cosmo.lcMpc
            #         ok_args_dist = sink_tree.query_ball_point(prev_sink['position'],r=1*dx_min_code)
            #         ok_mass_args = np.where(prev_sink['mass']*1.01>=all_sinks['mass'])
            #         ok_args = np.intersect1d(ok_args_dist,ok_mass_args)


            #         largs = len(ok_args)

            #         if largs==0:
            #             # break
            #             # print(ok_args,ok_args_dist,ok_mass_args,all_sinks['aexp'],prev_sink['aexp'],(lookup_pos),(prev_sink['position']))
            #             # print(all_sinks['mass'],prev_sink['mass'])
            #             # print(ok_args,dx_min_code,dx_min_pc)
            #             # print(np.linalg.norm(prev_sink['position']-lookup_pos,axis=1))
            #             # print(np.linalg.norm(prev_sink['position']-lookup_pos,axis=1)*sim.cosmo.lcMpc*1e3)

            #             continue
            #         elif largs==1:
            #             sid = all_sinks['identity'][ok_args]
            #             print(sid)
            #             sink = read_sink_bin(
            #                 os.path.join(sim.sink_path, sink_file),
            #                 tgt_fields=out_keys_for_read + ["identity"],
            #                 sid=sid,
            #                 debug=debug,
            #                 **kwargs,
            #             )
            #             convert_sink_units(sink,sink['aexp'],sim)
            #         else:
            #             print("Several possible merger candidates... not implemented")
            #             continue
            #             # break

                    # print(sid,sink)
                    # input('continue?')

            # else:
            #     continue

        else:
            found_step = convert_sink_units(sink, sink['aexp'], sim)    


        prev_sink=sink

        aexp = sink["aexp"]

        if not found_step:
            continue

        if (1.0 / aexp - 1.0) > max_z:
            for k in bh_data.keys():
                bh_data[k] = bh_data[k][:ifill, :]
            break

        if ifill % 100 == 0:
            print(f"...{ifill:d}, {aexp:.3f}, {1.0/aexp-1.0:.3f}")

        # ids = sink["identity"]
        # id_args = np.where(ids == sid)[0]

        # if debug:
        #     print(sid, ids, id_args)

        if sink == {}:
            print(f"Didn't find sink {sid} in {sink_file}, end of sink tree")
            break

        id_datas = {
            key: sink[key]
            for key in sink.keys()
            if key != "Id"
            # and type(sink[key]) == np.ndarray
            and key in out_keys_for_read
        }

        # print(sid)
        # print(sink,id_datas)
        bh_data["zeds"][ifill] = 1.0 / aexp - 1.0
        bh_data["coarse_step"][ifill] = sink_fnb

        for key in id_datas.keys():
            # print(key,ifill,np.shape(bh_data[key]),np.shape(id_datas[key]))
            bh_data[key][ifill] = id_datas[key]


            # convert_sink_units(bh_data[key][ifill], aexp, sim)
    #
    # bh_data[ifill, 2:] = id_datas

    # bh_data[ifill, 0] = 1.0 / aexp - 1.0
    # bh_data[ifill, 1] = ifill

    if ifill != len(sink_fnbs) - 1:
        for k in bh_data.keys():
            bh_data[k] = bh_data[k][:ifill, :]
        # bh_data = bh_data[:ifill, :]

    # bh_dict = {}

    for ikey, key in enumerate(out_keys):
        bh_data[key[0]] = np.squeeze(bh_data[key[0]])

    # for ikey, key in enumerate(out_keys):
    #     bh_dict[key] = bh_data[:, ikey]

    return bh_data


def check_if_superEdd(sim):
    """
    returns true is eddington_limit .eq. false
    """
    super_edd = False
    if "smbh_params" in sim.namelist.keys():
        if "eddington_limit" in sim.namelist["smbh_params"]:
            super_edd = sim.namelist["smbh_params"]["eddington_limit"] == False
    return super_edd


def find_zoom_massive_central_sink(sim: ramses_sim, aexp, ctr=None, rlim=None):

    zoom_hid, zoom_hprops, _ = find_zoom_tgt_halo(sim, 1.0 / aexp - 1.0)

    snap = sim.get_closest_snap(aexp)

    print(snap, aexp)

    if ctr is None:
        ctr = zoom_hprops["pos"]
    if rlim is None:
        rlim = zoom_hprops["rvir"] * 0.2

    return find_massive_sink(ctr, snap, sim, rlim)["identity"]


def gid_to_sid(sim: ramses_sim, gid, sim_snap, gdict=None, hdict=None, step=None):

    if gdict == None:
        gdict = get_gal_props_snap(sim.path, sim_snap, gid=gid)

    
    if type(gdict["rmax"]) in [float,np.float32,np.float64] and gdict['gids']==gid:
        # rmax = gdict["rmax"]*3
        if hdict == None:
            host_hid = gdict['host hid']
            hdict = get_halo_props_snap(sim.path, sim_snap, hids=host_hid)
        rmax = hdict[f"halo_{host_hid:07d}"]["rvir"]
        r50 = gdict["r50"]
        gpos = gdict["pos"][:]
    else:
        garg = np.where(gdict['gids']==gid)[0][0]
        if hdict == None:
            host_hid = gdict['host hid'][garg]
            hdict = get_halo_props_snap(sim.path, sim_snap, hids=host_hid)

        rmax = hdict[f"halo_{host_hid:07d}"]["rvir"]
        # rmax = gdict["rmax"][garg]*3
        r50 = gdict["r50"][garg]
        gpos = gdict["pos"][:,garg]


    # hctr = hprops["pos"]
    # hrvir = hprops["rvir"]

    # hagn_massive_sid = find_massive_sink(pos, hagn_snap, hagn_sim, rmax=rvir * 2)[
    found = False
    rsearch = 3*r50

    sim_massive_sid = None

    while (not found) and (rsearch < rmax):
        # print(found, rsearch, rmax)
        # try:
        massive_sink,step = find_massive_sink(
            gpos,
            sim_snap,
            sim,
            rmax=rsearch,
            coarse_step=step
            # sim_hagn_ctr,
            # sim_snap,
            # sim,
            # rmax=rmax,
        )

        if massive_sink != {} and 'mass' in massive_sink.keys():
            # print(massive_sink)
            if np.sum(massive_sink['mass'])>0: #check not empty arrays in dict

                sim_massive_sid = massive_sink["identity"]

                # print(sim_massive_sid)
                if type(sim_massive_sid) in [np.ndarray, list]:
                    if len(sim_massive_sid) > 0:
                        found = True
                elif type(sim_massive_sid) in [
                    int,
                    float,
                    np.int32,
                    np.int64,
                    np.float32,
                    np.float64,
                ]:
                    found = True
        # print(sim_massive_sid, found)
        # print(type(sim_massive_sid))
        # print(type(sim_massive_sid) in [np.ndarray, list])
        # print(sim_massive_sid, rsearch, rmax)
        rsearch *= 1.5
    # except ValueError:
    #     rsearch *= 1.5

    return sim_massive_sid, found, step


def hid_to_sid(sim: ramses_sim, hid, sim_snap, debug=False, step=None):

    gid, gdict = get_central_gal_for_hid(sim, hid, sim_snap)
    if debug:
        print(gid, gdict)
    if gid == None:
        return None, False, None

    sim_massive_sid, found, step = gid_to_sid(sim, gid, sim_snap, gdict=gdict, step = step)

    if debug and found:
        print(sim_massive_sid, found)
        print(f"found sink {sim_massive_sid:d} in halo {hid:d}")
    return sim_massive_sid, found, step


def sid_to_hid(sim: ramses_sim, sid, snap, hprops=None, sprops=None, fpure=0.9999):

    if hprops is None:
        hprops = get_halo_props_snap(sim.path,snap)

    if sprops is None:
        sink = read_sink_snap(sim, snap, sid=sid)


    pure = hprops['fpure']>fpure
    pure_hids = hprops['hid'][pure]
    hpos = [hprops[f"halo_{hid:07d}"]['pos'] for hid in pure_hids]
    # rvirs = [hprops[f"halo_{hid:07d}"]['rvir'] for hid in pure_hids]
    # hpos = hprops['pos'][:,pure]


    tree = KDTree(hpos)

    rquery = 1/(sim.cosmo.lcMpc*1e3) #1ckpc in code units
    nfound = 0
    cur_mmax = -1
    last_mmax = -1
    iter=0

    while ((iter==0) or (cur_mmax != last_mmax)) and rquery<10/(sim.cosmo.lcMpc*1e3):

        # print(iter,cur_mmax,last_mmax,rquery * sim.cosmo.lcMpc*1e3)

        last_mmax = np.copy(cur_mmax)

        args = tree.query_ball_point(sink['position'],r=rquery)

        if len(args)>0:
            masses = hprops['mvir'][args]

            arg_max = np.argmax(masses)
            cur_mmax = masses[arg_max]

        rquery*=1.5

    if rquery/1.5<=10/(sim.cosmo.lcMpc*1e3):

        return pure_hids[args][arg_max]    
    else:
        return None
    

def get_coarse_dts(sim: ramses_sim):

    sink_files = np.asarray(os.listdir(sim.sink_path))
    sink_fnbs = np.asarray([int(f.split("_")[-1].split(".")[0]) for f in sink_files])

    sort = np.argsort(sink_fnbs)[::-1]
    sink_fnbs = sink_fnbs[sort]
    sink_files = np.array(sink_files)[sort]

    aexps = np.zeros(len(sink_fnbs))

    for ifill, (sink_fnb, sink_file) in enumerate(zip(sink_fnbs, sink_files)):

        sink = read_sink_bin(os.path.join(sim.sink_path, sink_file), hdr_only=True)
        aexps[ifill] = sink["aexp"]

    sim.init_cosmo()

    zeds = 1.0 / aexps - 1.0

    step_times = sim.cosmo_model.age(zeds).value * 1e9  # yr

    return sink_fnbs, zeds, step_times
