from datetime import date
from time import ctime, strftime
import numpy as np
import matplotlib.pyplot as plt

from h5tools.h5py_tools import h5_to_dict

from time import ctime

from scipy.ndimage import zoom

from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic_2d
import os

from skimage.transform import rescale


from scipy.ndimage import gaussian_filter, convolve, rotate
from scipy.signal import convolve2d, fftconvolve

from zoom_analysis.rascas.rascas_steps import (
    filt_file_to_dict,
    get_directions_cart,
    micron_to_fwhm_as,
)
from zoom_analysis.read.read_data import read_data_ball
from zoom_analysis.visu.visu_fct import plot_fields, plot_stars, basis_from_vect

from zoom_analysis.halo_maker.read_treebricks import (
    convert_star_units,
    read_zoom_brick,
    read_gal_stars,
    read_zoom_stars,
)
from zoom_analysis.stars import sfhs

from hagn.utils import get_hagn_sim

from zoom_analysis.zoom_helpers import starting_hid_from_hagn

import healpy as hp

# from zoom_analysis.stars.star_reader import read_part_ball,
from f90_tools.star_reader import read_part_ball_NCdust

from gremlin.read_sim_params import ramses_sim

from zoom_analysis.halo_maker.assoc_fcts import (
    get_halo_props_snap,
    get_gal_props_snap,
    find_zoom_tgt_halo,
    get_halo_assoc_file,
    get_assoc_pties_in_tree,
    smooth_props,
)

from zoom_analysis.rascas.field_fits import (
    load_psf,
    # rebin_factor
    # rebin
    dilate,
)

from zoom_analysis.rascas.filts.filts import (
    flamb_fnu,
    fnu_to_mAB,
)

import astropy.units as u

# from zoom_analysis.trees.tree_reader import read_tree_file_rev
from zoom_analysis.trees.tree_reader import read_tree_file_rev_correct_pos


# from matplotlib.gridspec import GridSpec

# from astropy.visualization import make_lupton_rgb

# from scipy.integrate import simps

from zoom_analysis.constants import *

from f90_tools.star_reader import read_part_ball_NCdust

# from zoom_analysis.rascas.errs import dumb_constant_mag, get_cl_err

from zoom_analysis.rascas.rascas_steps import read_params

from zoom_analysis.rascas.read_rascas import (
    compress_mock,
    cube_to_erg_s_A_cm2_as2,
    read_compress_mock,
    read_mock_dump,
    spec_to_erg_s_A_cm2,
    # spec_to_erg_s_A_cm2_rf,
    mock_spe_dump,
    read_PFS_dump,
)

from zoom_analysis.rascas.igm import T_IGM_Inoue2014

from zoom_analysis.rascas.filts.filts import (
    read_transmissions,
)

from zoom_analysis.rascas.rascas_plots import *


def find_line_in_file(fname, keyword):

    outs = []

    with open(fname, "r") as src:

        for line in src:

            if keyword in line:

                outs.append(line)

    return outs


if __name__ == "__main__":

    hm = "HaloMaker_stars2_dp_rec_dust/"

    model_name = "zoom_tgt_bc03_chabrier100_wlya_ZTEST_SMC"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_ZTEST_noDust"

    # sim_paths = locate_all_sims()

    sim_paths = [
        "/data101/jlewis/sims/NH_for_rascas",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288",
    ]
    # sim_paths = ["/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380"]
    # sim_paths = ["/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646"]
    # sim_paths = ["/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289"]
    # sim_paths = ["/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890"]
    # sim_paths = ["/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130"]

    for sim_path in sim_paths:

        zed_targets = None
        snap_targets = None

        zmax = np.inf
        ndir = 12
        # ndir = 12
        rfact = 1.0
        zstt = 2.0

        overwrite = True
        # overwrite = False

        nist_pc = 3.085678e18  # cm
        dist_10pc = 10.0 * nist_pc

        filter_select = [
            "MegaCam-u_new",
            "HSC-g",
            "HSC-i",
            "HSC-r",
            "HSC-y",
            "HSC-z",
            "NIRCAM-F115",
            "NIRCAM-F150",
            "NIRCAM-F277",
            "NIRCAM-F444",
            "HST-f814_res",
        ]

        filt_dict = filt_file_to_dict()

        # print(filt_dict)

        try:
            sim = ramses_sim(
                sim_path,
                nml="cosmo.nml",
                full_check=False
            )
        except FileNotFoundError:
            continue

        if len(sim.snap_numbers) < 1:
            continue

        name = sim.name
        # intID = int(name[2:].split("_")[0])

        if snap_targets is not None:
            snap_target = max(snap_targets)
        else:
            snap_target = sim.snap_numbers.max()

        if type(snap_target) in [np.ndarray, list]:

            if len(snap_target) == 0:
                continue

        # snap_target = sim.get_closest_snap(zed=zed_target)
        true_zed_snap = 1.0 / sim.get_snap_exps(snap_target)[0] - 1

        sim.init_cosmo()

        sim.get_snap_times()
        sim_aexps = sim.aexps

        rascas_path = os.path.join(sim_path, "rascas", model_name)

        if not os.path.exists(rascas_path):
            continue


        ztests_paths = np.array([
                    f
                    for f in os.listdir(path=rascas_path)
                    if f.startswith("Z")
                ])
        
        ztests = np.asarray(
                [
                    float(f[1:].replace('p','.'))
                    for f in ztests_paths
                ]
            )
        

        # order = np.argsort(ztests)[::-1]
        order = np.argsort(ztests)

        ztests = ztests[order]
        ztests_paths = ztests_paths[order]

        for iz,zmock in enumerate(ztests):


            rascas_snaps = np.sort(
                np.asarray(
                    [
                        int(f.split("_")[-1])
                        for f in os.listdir(path=os.path.join(rascas_path,ztests_paths[iz]))
                        if f.startswith("output_")
                    ]
                )
            )[::-1]

            aexp_mock = 1/(1+zmock)

            final_gid = None

            for i, snap in enumerate(rascas_snaps[::-1]):

                time = sim.get_snap_times(snap)[0]
                aexp = sim.get_snap_exps(snap)[0]

                # print(i, snap, aexp, time)
                if 1.0 / aexp - 1 > zmax:
                    print(f"skipping snap {snap:d} at z={1.0/aexp-1:.2f}")
                    continue

                # find_gals
                gal_folders = os.listdir(os.path.join(rascas_path,ztests_paths[iz], f"output_{snap:05d}"))
                gids = [int(gdir.split("_")[-1]) for gdir in gal_folders]

                for gid in gids:

                    rascas_dir = os.path.join(
                        rascas_path, ztests_paths[iz],f"output_{snap:05d}", f"gal_{gid:07d}"
                    )

                    try:
                        rascas_params = read_params(
                            os.path.join(rascas_dir, "params_rascas.cfg")
                        )
                    except FileNotFoundError:
                        print(f"no params_rascas.cfg in {rascas_dir}")
                        continue
                    print(f"found rascas dir or params_rascas.cfg in: {rascas_dir}")

                    fimg = os.path.join(rascas_dir, "mock_test.image")
                    fpfs = os.path.join(rascas_dir, "PFSDump", "pfsdump")
                    fcdd = os.path.join(rascas_dir, "CDDDump", "domain_1.mesh")
                    frascas = os.path.join(rascas_dir, "RASCASDump", "rascas_dump.dat")

                    fcompressd = os.path.join(rascas_dir, "mocks.h5")
                    fspec = os.path.join(rascas_dir, "mock_test.spectrum")
                    fcube = os.path.join(rascas_dir, "mock_test.cube")

                    if not os.path.exists(fcompressd) or overwrite:

                        ntodo = 0
                        if os.path.exists(fspec):
                            ntodo += 1
                            mocks_spe = read_mock_dump(fspec, ndir=ndir)
                        else:
                            mocks_spe = None
                        if os.path.exists(fcube):
                            ntodo += 1
                            mocks_cube = read_mock_dump(fcube, ndir=ndir)
                        else:
                            mocks_cube = None

                        if ntodo > 0:
                            if os.path.exists(fpfs):
                                PFS_header = read_PFS_dump(fpfs, hdr=True)
                            elif os.path.exists(os.path.join(rascas_dir, "log_PFS")):
                                print(f"no existing {fpfs}, making do...")

                                PFS_header = {}
                                tot_flux_line = find_line_in_file(
                                    os.path.join(rascas_dir, "log_PFS"), "Total luminosity"
                                )
                                if len(tot_flux_line) > 0:
                                    tot_flux_line = tot_flux_line[0]
                                else:
                                    print("no Total luminosity entry or log file found")
                                    continue
                                PFS_header["total_flux"] = float(
                                    tot_flux_line.split(":")[-1]
                                )

                                nphot_line = find_line_in_file(
                                    os.path.join(rascas_dir, "params_PFS.cfg"),
                                    "nPhotonPackets",
                                )
                                if len(nphot_line) > 0:
                                    nphot_line = nphot_line[0]
                                else:
                                    print("no nphotons or PFS param file found")
                                    continue
                                PFS_header["nphotons"] = float(nphot_line.split("=")[-1])
                            else:
                                print("no PFS log")
                                continue

                            print("compressing mock...")

                            compress_mock(
                                fcompressd, PFS_header, cubes=mocks_cube, specs=mocks_spe
                            )

                    if os.path.exists(fcompressd):
                        print(f"found {fcompressd}...")

                        readable = False

                        try:
                            # mocks_spe, mocks_cube, pfs_header = read_compress_mock(
                            #     fcompressd
                            # )
                            compressed_test = h5_to_dict(fcompressd)
                            readable = ("nphotons" in compressed_test.keys()) and (
                                "total_flux" in compressed_test.keys()
                            )
                            print("is readable")
                        except:
                            readable = False
                            print(f"{fcompressd} unreadable")
                            os.remove(fcompressd)
                            pass

                        if readable:
                            for f in [fimg, fspec, fcube, fcdd, fpfs, frascas]:
                                if os.path.exists(f):
                                    print(f"deleting {f}...")
                                    os.remove(f)

                            # for f in [fimg, fspec, fcube, fcdd, frascas]:  # keep fpfs
