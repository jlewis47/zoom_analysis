import argparse
import numpy as np
import os
import h5py


def read_lvl_h5(f, cat):
    for k, v in f.attrs.items():
        cat[k] = v
    for k, v in f.items():
        if isinstance(v, h5py.Group):
            cat[k] = {}
            read_lvl_h5(v, cat[k])
        else:
            cat[k] = v[()]
    return cat


def read_cat(simdir, snap, cat_type, rtype, rfact=1.0, id=None, tgt_keys=None):

    assert cat_type in ['galaxy',"halo"], "cat_type must be either galaxy or halo centred"
    assert rtype in ['r50','r90','reff','rvir'], 'rtype must be either r50, r90, reff, or rvir'        
        

    rfact_str = ("%.1f" % rfact).replace(".", "p")

    fdir = os.path.join(simdir, "catalogues")
    fname = os.path.join(fdir, f"{cat_type}_{rfact_str}X{rtype}_{snap}.hdf5")

    cat = {}

    with h5py.File(fname, "r") as f:

        # fill cat with attrs
        for k, v in f.attrs.items():
            cat[k] = v

        # fill cat with data, copying path of h5py file
        # if id is None, read all TODO
        # else read only id
        # if cat_type == "galaxy":
        #     arg = f["galaxy ids"] == id
        # elif cat_type == "halo":
        #     arg = f["halo ids"] == id

        if id is not None:
            key = f"{cat_type:s}_{int(id):07d}"
            if tgt_keys == None:
                read_lvl_h5(f[key], cat)
            else:
                for k in tgt_keys:
                    cat[k] = f[key][k][()]
        else:
            
            if tgt_keys == None:
                read_lvl_h5(f, cat)
            else:
                for obj in f.keys():
                    cat[obj]={}
                    for k in tgt_keys:
                        cat[k] = f[k][()]

    return cat

def obj_based_to_table(cat:dict,data_type,key):

    gids = cat['galaxy ids']

    vals = np.zeros_like(gids,dtype=np.float32)

    for igal,gid in enumerate(gids):

        vals[igal] = cat[f"galaxy_{gid:07d}"][data_type][key]

    return vals

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
        "--rball",
        type=float,
        help="rballxr50 is the radius within which properties are to be measured",
        default=2.0,
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print more information", default=False
    )
    parser.add_argument(
        "--tgt_keys", type=str, nargs="+", help="keys to read", default=None
    )

    args = parser.parse_args()

    simdirs = args.simdir
    type = args.type
    snap = args.snap
    rball = args.rball
    id = args.id
    verbose = args.verbose
    keys = args.tgt_keys

    if verbose:
        print(
            f"Reading {type} {id} at snap {snap} in {simdirs}, in a ball of radius {rball}"
        )

    cat = read_cat(simdirs, snap, type, rball, id, tgt_keys=keys)
