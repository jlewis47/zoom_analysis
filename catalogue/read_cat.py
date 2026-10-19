import argparse
import numpy as np
import os
import h5py
from h5tools.h5py_tools import read_lvl_h5
from zoom_analysis.catalogue.make_cat import make_cat
from zoom_analysis.catalogue.cat_tools import args2catname
from gremlin.read_sim_params import ramses_sim


def read_cat(
    simdir, snap, cat_type, rtype, rfact=1.0, id=None, tgt_keys=None, central=False
):

    fdir = os.path.join(simdir, "catalogues")
    cat_name = args2catname(
        snap,
        cat_type,
        rtype,
        rfact=rfact,
        subs=central == False,
    )
    fname = os.path.join(fdir, cat_name)

    cat = {}

    # print(fname)

    with h5py.File(fname, "r") as f:

        # fill cat with attrs
        for k, v in f.attrs.items():
            cat[k] = v

        # print(f.keys())

        # fill cat with data, copying path of h5py file
        # if id is None, read all TODO
        # else read only id
        # if cat_type == "galaxy":
        #     arg = f["galaxy ids"] == id
        # elif cat_type == "halo":
        #     arg = f["halo ids"] == id

        if id is not None:
            key = f"{cat_type:s}_{int(id):07d}"
            if np.all(tgt_keys == None):
                read_lvl_h5(f[key], cat)
            else:
                for ktype in f[key].keys():
                    if ktype not in cat:
                        cat[ktype] = {}
                    for k in tgt_keys:
                        if k in f[key][ktype]:
                            # print("found ", k)
                            cat[ktype][k] = f[key][ktype][k][()]
        else:

            if np.all(tgt_keys == None):
                read_lvl_h5(f, cat)
            else:
                for obj in f.keys():
                    cat[obj] = {}
                    for k in tgt_keys:
                        cat[obj][k] = f[obj][k][()]

    return cat


def obj_based_to_table(cat: dict, key_path):

    gids = cat["galaxy_ids"]

    vals = np.zeros_like(gids, dtype=np.float32)

    keys = key_path.split('/')
    

    for igal, gid in enumerate(gids):

        found=True
        gid_key=f"galaxy_{gid:07d}"
        if gid_key in cat.keys():
            read = cat[gid_key]
            found=True
            for k in keys:
                if k in read.keys():
                    read = read[k]
                else:
                    print(f"didn't find {k}")
                    found=False
                    break
            if found:
                vals[igal] = read

    return vals


def read_or_make_cat(
    sim: ramses_sim,
    snap,
    cat_type,
    rtype,
    rfact=1.0,
    id=None,
    tgt_keys=None,
    central=False,
    overwrite=False,  # if True, rewrite entire file!
    update=False,  # if True, go through make cat, and update existing values
    verbose=False,
    **kwargs
):

    simdir = sim.path

    remake_cat = overwrite or update
    if not remake_cat:
        try:
            cat = read_cat(
                simdir,
                snap,
                cat_type,
                rtype,
                rfact,
                id=id,
                tgt_keys=tgt_keys,
                central=central,
            )

            if tgt_keys is not None:

                found = 0
                # print(tgt_keys)
                for ktype in cat.keys():
                    # print(ktype)
                    # print(cat[ktype].keys())
                    for ktgt in tgt_keys:
                        if ktgt in cat[ktype].keys():
                            found += 1
                # print(found, len(tgt_keys))
                if found == 0:
                    raise KeyError

            if verbose:
                print("Read data catalogue")
        except (KeyError, FileNotFoundError):
            print('missing entries, remaking catalogue')
            # print(cat)
            remake_cat = True

    

    if remake_cat:

        cat = make_cat(
            sim,
            snap,
            cat_type,
            rtype,
            rfact,
            id=id,
            central=central,
            overwrite=overwrite,
            ignore_exist=False,
            **kwargs
        )
        if verbose:
            print("Made data catalogue")

        #check we can read it
        cat = read_cat(
            simdir,
            snap,
            cat_type,
            rtype,
            rfact,
            id=id,
            tgt_keys=tgt_keys,
            central=central,
        )

    # out_dict = {} #why did I try this?

    # if cat is not None:

    #     for ctype in cat.keys():
    #         out_dict[ctype] = {}
    #         for k in cat[ctype].keys():
    #             if tgt_keys is not None and k in tgt_keys:
    #                 out_dict[ctype][k] = cat[ctype][k]
    #             else:
    #                 out_dict[ctype][k] = cat[ctype][k]

    # return out_dict

    return cat


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("simdir", type=str, help="path to the simulation")
    parser.add_argument("snap", type=int, help="snap nb to read")
    parser.add_argument("id", type=int, help="object id to read")
    parser.add_argument(
        "--type",
        type=str,
        choices=["galaxy", "halo"],
        help="type of catalogue to build, either galaxy or halo based",
        default="galaxy",
    )
    parser.add_argument(
        "--rtype",
        type=float,
        help="rtype",
        choices=["rmax", "r50", "r90", "reff", "rvir", "ckpc", "as"],
    )
    parser.add_argument(
        "--rball",
        type=float,
        help="rballxr50 is the radius within which properties are to be measured",
        default=2.0,
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print more information", default=False
    )
    parser.add_argument(
        "--central", action="store_true", help="print more information", default=False
    )
    parser.add_argument(
        "--tgt_keys", type=str, nargs="+", help="keys to read", default=None
    )

    args = parser.parse_args()

    simdirs = args.simdir
    type = args.type
    snap = args.snap
    rtype = args.rtype
    rball = args.rball
    id = args.id
    verbose = args.verbose
    keys = args.tgt_keys
    central = args.central

    if verbose:
        print(
            f"Reading {type} {id} at snap {snap} in {simdirs}, in a ball of radius {rball}"
        )

    cat = read_cat(
        simdirs, snap, type, rtype, rball, id, tgt_keys=keys, central=central
    )
