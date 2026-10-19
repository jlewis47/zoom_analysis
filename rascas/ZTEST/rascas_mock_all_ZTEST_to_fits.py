import numpy as np
import matplotlib.pyplot as plt

from astropy import wcs as WCS
from astropy.coordinates import SkyCoord
from astropy import units as u

from scipy.stats import binned_statistic_2d
from scipy.signal import fftconvolve
from scipy.ndimage import zoom,rotate

from matplotlib.backends.backend_pdf import PdfPages

import os

from zoom_analysis.catalogue.read_cat import read_or_make_cat
from zoom_analysis.rascas.rascas_steps import filt_file_to_dict

from zoom_analysis.rascas.field_fits import (
    load_psf,
    dilate,
)

from zoom_analysis.sinks.sink_reader import read_sink_bin, read_sink_bin_NH

from f90_tools.star_reader import read_part_ball_NCdust, read_part_ball_NH


from gremlin.read_sim_params import ramses_sim

from zoom_analysis.halo_maker.assoc_fcts import (
    # get_halo_props_snap,
    get_gal_props_snap,
    # find_zoom_tgt_halo,
    get_halo_assoc_file,
    # get_assoc_pties_in_tree,
    # smooth_props,
)

from zoom_analysis.constants import *

from zoom_analysis.rascas.field_fits import (
    gals_to_fit_pos,
    read_fits,
    save_fits,
    tile_to_file,
    get_hdr_res,
)

from zoom_analysis.rascas.rascas_plots import *



def band_to_zpt(bands):

    if type(bands) is str:
        bands = [bands]

    test_bands = np.asarray([band.lower() for band in bands])

    zpts = np.zeros(len(bands))

    for i, band in enumerate(test_bands):
        if band in ["f814w", "f115w", "f150w", "f277w", "f444w"]:
            zpts[i] = 28.086519392
        elif band in ["f777w"]:
            zpts[i] = 26.58132
        else:
            # zpts[i] = 28.9
            zpts[i] = 28.086519392

    return zpts


def resc_flux(fls, pix_sizes_sr, bands, base_zpt=8.9):

    # fls in MJy

    new_zpts = band_to_zpt(bands)

    if len(fls.shape) == 2:
        fls = [fls]
    elif len(fls.shape) != 3:
        raise ValueError(
            "flux array must be 2D (n_bands,n_gals) or 3D (n_bands,n_gals,n_dirs)"
        )
    out_fls = np.zeros_like(fls)

    if type(pix_sizes_sr) not in [list, np.ndarray]:
        pix_sizes_sr = [pix_sizes_sr]

    for ifl, (fl, band, new_zpt, pix_size_sr) in enumerate(
        zip(fls, bands, new_zpts, pix_sizes_sr)
    ):

        # fls[ifl] = -2.5 * np.log10(10 ** (fl / -2.5) - base_zpt + new_zpt)
        # mags = -2.5 * np.log10(fl * 1e6) + base_zpt
        # fls[ifl] = 10 ** ((mags - base_zpt + new_zpt) / -2.5) * 1e-6  # back to MJy
        out_fls[ifl] = fl * 10 ** (
            (new_zpt - (base_zpt - 2.5 * np.log10(pix_size_sr * 1e6))) / 2.5
        )

    return out_fls


def walk_down_dirs(root, saved_paths):

    dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

    # print(root)

    for d in dirs:

        full_d = os.path.join(root, d)
        # print(full_d)

        if d.startswith("id"):
            if not os.path.islink(full_d):
                saved_paths.append(full_d)
        else:
            saved_paths = walk_down_dirs(full_d, saved_paths)

    # print(saved_paths)

    return saved_paths


def locate_all_sims():

    sim_paths = []

    root_dirs = [
        "/data101/jlewis/sims/dust_fid/lvlmax_22",
        "/data102/jlewis/sims/lvlmax_22",
        "/data103/jlewis/sims/lvlmax_22",
    ]

    for root_dir in root_dirs:
        # print(root_dir)
        sim_paths = walk_down_dirs(root_dir, sim_paths)

    sim_paths = np.unique(sim_paths)

    # print(sim_paths)

    sim_paths = [p for p in sim_paths if "common_plots" not in p]

    return sim_paths


if __name__ == "__main__":

    hm = "HaloMaker_stars2_dp_rec_dust/"

    sim_paths = [
        "/data101/jlewis/sims/NH_for_rascas",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130",
        "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756",
    ]

    zmax = 6.0
    # zmax = 4.0
    deltaT = 50  # Myr
    ndir = 12
    # ndir = 12
    rfact = 1.1
    zstt = 2.0

    # idir = 1
    idir = 3
    # idir = "random"

    if type(idir) in [int,np.int32,np.int64]:
        idir_str = f"{idir:d}"
    else:
        idir_str=str(idir)

    overwrite = True
    # overwrite = False
    debug = False

    nist_pc = 3.085678e18  # cm
    dist_10pc = 10.0 * nist_pc

    dust_model = "SMC"
    # dust_model = "noDust"
    # dust_model = "SB_calzetti_screen_1p5"

    # model_name = "LMC"
    # model_name = "SMC"
    # model_name = "ignoreDust"
    # model_name = "zoom_tgt_ignore_Dust"
    # model_name = "zoom_tgt"
    # model_name = "zoom_tgt_bpass_kroupa300_bpassV23"
    # model_name = "zoom_tgt_bpass_kroupa100_broken_metals"
    # model_name = "zoom_tgt_bpass_chabrier100_new_wlya"
    # model_name = "zoom_tgt_bpass_kroupa100_wlya_old_pos"
    # model_name = "zoom_tgt_bpass_chabrier100_new"
    # model_name = "zoom_tgt_bpass_kroupa100"
    # model_name = "zoom_tgt_bpass_kroupa100_wlya"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_ndust"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_ndust_SMC"
    model_name = f"zoom_tgt_bc03_chabrier100_wlya_ZTEST_{dust_model}"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_SMC_draine"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_YD"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_MW"
    # model_name = "zoom_tgt_bc03_chabrier100_wlya_noDust"
    # model_name = "zoom_tgt_bc03_chabrier100"
    # model_name = "zoom_tgt_bpass_kroupa100_wlya_noDust"
    # model_name = "zoom_tgt_bpass_chabrier100"
    # model_name = "zoom_tgt_bpass_chabrier100_ignoreDust"

    fnames = []
    all_rgals = []
    all_rapers = []
    all_rgals_px = []
    zeds = []
    zmocks = []
    snapshots = []
    galaxy_ids = []
    directions = []
    all_mgals = []
    all_r50s = []
    all_m_r50s = []
    all_m_2xr50s = []
    all_m_1p5xas = []
    all_sfrs_10 = []
    all_sfrs_10_r50 = []
    all_sfrs_10_2xr50 = []
    all_sfrs_10_1p5xas = []
    all_sfrs_100 = []
    all_sfrs_100_r50 = []
    all_sfrs_100_2xr50 = []
    all_sfrs_100_1p5xas = []
    angs_per_ckpc = []
    # img_size = []
    sim_origin = []

    for sim_path in sim_paths:

        zed_targets = None
        snap_targets = None

        psf_dict = filt_file_to_dict()

        if "NH_for_rascas" not in sim_path:
            try:
                sim = ramses_sim(sim_path, nml="cosmo.nml")
            except FileNotFoundError:
                print(f"Couldn't find ramses_sim here:{sim_path}")
                continue
            name = sim.name
            intID = str(int(name[2:].split("_")[0]))
            sim_id = intID
        else:
            sim = ramses_sim(sim_path, nml="cosmo.nml",tar_snaps=False,mini_snaps=True,full_snaps=False)
            name = "NewHorizon"
            intID = "NewHorizon"
            sim_id = "NewHorizon"


        print(name)

        tree_name = os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat")
        byte_file = os.path.join(sim.path, "TreeMakerDM_dust")  # , "tree_rev_nbytes")

        if snap_targets is not None:
            snap_target = max(snap_targets)
        else:
            snap_target = sim.snap_numbers.max()

        if type(snap_target) in [np.ndarray, list]:

            if len(snap_target) == 0:
                print("No snaps to work with")
                continue

        true_zed_snap = 1.0 / sim.get_snap_exps(snap_target)[0] - 1

        sim.init_cosmo()

        sim.get_snap_times()
        sim_aexps = sim.aexps


        _, data_snaps = sim.get_snaps(tar_snaps=True, full_snaps=True, mini_snaps=False)
        data_snap_aexps = sim.get_snap_exps(data_snaps)
        close_tol = 1e-3

        if len(data_snaps) == 0:
            print("No full snaps to work with")
            continue

        assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
        assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

        avail_aexps = np.intersect1d(
            sim.get_snap_exps(assoc_file_nbs, param_save=False), sim_aexps
        )

        if len(avail_aexps) == 0:
            print("No associated snaps to work with")
            continue

        avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

        if zed_targets is not None:
            snap_target_args = np.asarray(
                [
                    np.argmin(np.abs(data_snap_aexps - 1.0 / (z + 1.0)))
                    for z in zed_targets
                ]
            )
            snap_targets = data_snaps[snap_target_args]
            err = data_snap_aexps[snap_target_args] - 1.0 / (
                np.asarray(zed_targets) + 1.0
            )
            snap_targets = snap_targets[np.abs(err) < close_tol]
        elif snap_targets is None:
            snap_targets = sim.snap_numbers


        mock_path_int = os.path.join(
            "/data101/jlewis",
            "mock_spe",
            model_name,
            f"{intID}",
        )
        mock_path = os.path.join(
            "/data101/jlewis",
            "mock_spe",
            model_name,
            f"{sim.name:s}",
        )

        mock_ZTEST_paths = []

        for p in (mock_path, mock_path_int):

            if not os.path.exists(p):
                print(f"No existing mocks at {p}")
                continue
            mock_ZTEST_paths.extend(
                [
                    os.path.join(p, f)
                    for f in os.listdir(p)
                    if os.path.isdir(os.path.join(p, f)) and f.startswith("Z")
                ]
            )

        if len(mock_ZTEST_paths) == 0:
            print("No mocks found")

        # print(mock_ZTEST_paths)

        mock_ZTESTs = np.asarray(
                [
                    float(f.split('/')[-1][1:].replace('p','.'))
                    for f in mock_ZTEST_paths
                ]
            )

        print(f"Found ztests:{mock_ZTESTs} with mocks to place in flats")

        for iz, (zpath, zmock) in enumerate(zip(mock_ZTEST_paths, mock_ZTESTs)):


            

            mock_snap_paths=np.array(
                [
                    os.path.join(zpath,f)
                    for f in os.listdir(zpath)
                    if os.path.isdir(os.path.join(zpath,f))
                ]
            )
                
            if len(mock_snap_paths) == 0:
                print("No mocks found")

            # print(mock_snap_paths)

            mock_snaps = [
                int(p.split("/")[-1]) if p[-1] != "/" else int(p.split("/")[-2])
                for p in mock_snap_paths
            ]

            print(f"Found snapshots:{mock_snaps} with mocks to place in flats")


            for i, (snap_path, snap) in enumerate(zip(mock_snap_paths[:], mock_snaps[:])):
                if snap_targets is not None:
                    if snap not in snap_targets:
                        print(f"{snap} not in snap_targets")
                        continue

                time = sim.get_snap_times(snap)[0]
                aexp = sim.get_snap_exps(snap)[0]
                z = 1.0 / aexp - 1.0

                if 1.0 / aexp - 1 > zmax:
                    print(f"skipping snap {snap:d} at z={1.0/aexp-1:.2f}")
                    continue


                if not os.path.isfile(get_halo_assoc_file(sim_path, snap)):
                    print(f"skipping snap {snap:d} at time={time:.2f} no assoc file")
                    continue

                # find_gals
                gal_folders = os.listdir(snap_path)
                gids = [int(gdir.split("_")[-1]) for gdir in gal_folders]

                for gid in gids:

                    outp = os.path.join(
                        snap_path,
                        f"gal_{gid:07d}",
                    )

                    img_dir = os.path.join(outp, "MJy_images")

                    if not os.path.exists(img_dir):
                        os.makedirs(img_dir, exist_ok=True)

                    if idir=='random':
                        idir_nb = np.random.randint(low=0,high=11,size=1)[0]
                    else:
                        idir_nb = idir

                    fname_mock_img = os.path.join(
                        outp,
                        "MJy_images",
                        f"mock_images_{sim.name}_{snap}_{gid}_{idir_nb:d}_wIGM.h5",
                    )

                    if not os.path.exists(fname_mock_img):

                        fname_mock_img = os.path.join(
                            outp,
                            "MJy_images",
                            f"mock_images_{intID}_{snap}_{gid}_{idir_nb:d}_wIGM.h5",
                        )

                        if not os.path.exists(fname_mock_img):

                            print(f"Couldn't find {fname_mock_img}")
                            continue

                    print(
                        f"sim: {sim.name}, getting gal properties for galaxy {gid:d} in snapshot {snap:d}"
                    )

                    gal_cat = get_gal_props_snap(sim_path, snap)
                    cur_gal_props = get_gal_props_snap(sim_path, snap, gid)

                    last_time = time

                    # print(f"ok snap {snap:d}")

                    # continue
                    rgal = cur_gal_props["rmax"] * rfact
                    r50 = cur_gal_props["r50"]
                    mgal = cur_gal_props["mass"]
                    agegal = cur_gal_props["max age"]

                    [aexp] = sim.get_snap_exps(snap)
                    z = 1.0 / aexp - 1.0
                    if not hasattr(sim, "cosmo_model"):
                        sim.init_cosmo()
                    t = sim.cosmo_model.age(z).value  # Gyr
                    l = (
                        sim.cosmo.boxlen
                        * sim.unit_l(sim.aexp_stt)
                        / (ramses_pc * 1e6)
                        / sim.aexp_stt
                    )

                    rgal_ckpc = rgal * l * 1e3
                    rgal_as = sim.cosmo_model.arcsec_per_kpc_comoving(z).value * rgal_ckpc
                    r50_ckpc = r50 * l * 1e3
                    r50_as = sim.cosmo_model.arcsec_per_kpc_comoving(z).value * r50_ckpc


                    this_gal = gal_cat['gids']==gid

                    this_gals_subs = (gal_cat['host gid']==gid) * (gal_cat['gids']!=gid) * (gal_cat['mass']>=1e6)
                    if this_gals_subs.sum()>0:
                        subs_pos = gal_cat['pos'][:,this_gals_subs]
                        subs_rs = gal_cat['rmax'][this_gals_subs]
                        # subs_rs = all_gal_props['r90'][this_gals_subs] * 1

                        dists_to_subs = np.linalg.norm(gal_cat['pos'][:,this_gal] - subs_pos,axis=0)
                        max_extent_subs = dists_to_subs + subs_rs

                        kpc50_lim = max_extent_subs< (50/(sim.cosmo.lcMpc*1e3))

                        if np.any(kpc50_lim):
                            max_containinig_subs = max_extent_subs[kpc50_lim].max()

                            cur_rmax = max(max_containinig_subs, cur_gal_props['r90'])
                            # print("some close sub gal make aper larger")
                        else:

                            cur_rmax = cur_gal_props['r90']


                    else:

                        cur_rmax = cur_gal_props['r90']

                    raper_as = cur_rmax * 1e3 * sim.cosmo.lcMpc * sim.cosmo_model.arcsec_per_kpc_comoving(zmock).value

                    outp = os.path.join(
                        snap_path,
                        f"gal_{gid:07d}",
                    )

                    aper_cm = rgal * 2 * l * (1e6 * ramses_pc)
                    distl = sim.cosmo_model.luminosity_distance(z).value * (1e6 * ramses_pc)
                    aper_as = np.arctan(aper_cm / distl) * 180.0 / np.pi * 3600.0

                    if name != "NewHorizon":

                        ball_reader_fct = read_part_ball_NCdust
                        sink_reader_fct = read_sink_bin

                    else:

                        ball_reader_fct = read_part_ball_NH
                        sink_reader_fct = read_sink_bin_NH
        
                    try:
                        data_rgal = read_or_make_cat(
                            sim,
                            snap,
                            "galaxy",
                            "rmax",
                            1.0,
                            gid,
                            tgt_keys=[
                                "sfr10_Msun_per_yr",
                                "sfr100_Msun_per_yr",
                                "sfr500_Msun_per_yr",
                                "Mstar_msun",
                            ],
                            overwrite=False,
                            update=False,
                            ball_reader_fct = ball_reader_fct,
                            sink_reader_fct = sink_reader_fct

                        )
                        data_1p5xas = read_or_make_cat(
                            sim,
                            snap,
                            "galaxy",
                            "as",
                            1.5,
                            gid,
                            tgt_keys=[
                                "sfr10_Msun_per_yr",
                                "sfr100_Msun_per_yr",
                                "sfr500_Msun_per_yr",
                                "Mstar_msun",
                            ],
                            overwrite=False,
                            update=False,
                            ball_reader_fct = ball_reader_fct,
                            sink_reader_fct = sink_reader_fct
                        )
                        data_r50 = read_or_make_cat(
                            sim,
                            snap,
                            "galaxy",
                            "r50",
                            1.0,
                            gid,
                            tgt_keys=[
                                "sfr10_Msun_per_yr",
                                "sfr100_Msun_per_yr",
                                "sfr500_Msun_per_yr",
                                "Mstar_msun",
                            ],
                            overwrite=False,
                            ball_reader_fct = ball_reader_fct,
                            sink_reader_fct = sink_reader_fct
                        )
                        data_2xr50 = read_or_make_cat(
                            sim,
                            snap,
                            "galaxy",
                            "r50",
                            2.0,
                            gid,
                            tgt_keys=[
                                "sfr10_Msun_per_yr",
                                "sfr100_Msun_per_yr",
                                "sfr500_Msun_per_yr",
                                "Mstar_msun",
                            ],
                            overwrite=False,
                            ball_reader_fct = ball_reader_fct,
                            sink_reader_fct = sink_reader_fct
                        )
                    except (KeyError,OSError):  # means missing files usually
                        continue

                    if not hasattr(sim, "cosmo_model"):
                        sim.init_cosmo()

                    # print(data_rgal, data_r50, data_2xr50)

                    fnames.append(fname_mock_img)
                    all_rgals.append(rgal_as)
                    all_rapers.append(raper_as)
                    all_r50s.append(r50_as)
                    zeds.append(1.0 / aexp - 1)
                    zmocks.append(zmock)
                    all_mgals.append(mgal)
                    snapshots.append(snap)
                    galaxy_ids.append(gid)
                    directions.append(idir_nb)
                    all_sfrs_10.append(data_rgal["stars"]["sfr10_Msun_per_yr"])
                    all_sfrs_10_r50.append(data_r50["stars"]["sfr10_Msun_per_yr"])
                    all_sfrs_10_2xr50.append(data_2xr50["stars"]["sfr10_Msun_per_yr"])
                    all_sfrs_10_1p5xas.append(data_1p5xas["stars"]["sfr10_Msun_per_yr"])
                    all_sfrs_100.append(data_rgal["stars"]["sfr100_Msun_per_yr"])
                    all_sfrs_100_r50.append(data_r50["stars"]["sfr100_Msun_per_yr"])
                    all_sfrs_100_2xr50.append(data_2xr50["stars"]["sfr100_Msun_per_yr"])
                    all_sfrs_100_1p5xas.append(data_1p5xas["stars"]["sfr100_Msun_per_yr"])
                    all_m_r50s.append(data_r50["stars"]["Mstar_msun"])
                    all_m_2xr50s.append(data_2xr50["stars"]["Mstar_msun"])
                    all_m_1p5xas.append(data_1p5xas["stars"]["Mstar_msun"])
                    sim_origin.append(sim.name)
                    angs_per_ckpc.append(sim.cosmo_model.arcsec_per_kpc_comoving(z).value)


    if len(all_mgals) == 0:
        print(f"Something went wrong... Found NO galaxies")

    all_rgals = np.asarray(all_rgals)

    # find already inserted galaxies
    # first detect cats and combine them

    tgt_dir = os.path.join(
        "/data101/jlewis/zoom_flats", f"allsims", "ZTEST", f"{idir_str:s}", f"{dust_model:s}"
    )

    if not os.path.exists(tgt_dir):
        os.makedirs(os.path.join(tgt_dir, f"flat_{0}"))
        flat_nbs = [0]
        nflats = 0
    else:
        flat_nbs = [
            int(f.split("_")[1]) for f in os.listdir(tgt_dir) if f.startswith("flat")
        ]
        nflats = len(flat_nbs)

    cat_fname = f"out_cat_B3.txt"

    out_dir = os.path.join(
        "/data101/jlewis/zoom_flats",
        f"allsims",
        "ZTEST",
        f"{idir_str:s}",
        f"{dust_model:s}",
        f"flat_{flat_nbs[0]:d}",
    )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    found_all = False
    prev_total_found = 0
    total_found = 0

    if overwrite or nflats == 0:
        iflat = 0
    else:
        full_cat = (
            np.genfromtxt(
                os.path.join(out_dir, cat_fname),
                names=True,
                delimiter=" ",
            ),
        )

        if len(flat_nbs) > 1:

            for iflat in flat_nbs[1:]:
                out_dir = os.path.join(
                    "/data101/jlewis/zoom_flats",
                    f"allsims",
                    f"{idir_str:s}",
                    f"{dust_model:s}",
                    f"flat_{iflat:d}",
                )

                full_cat = np.squeeze(
                    np.concatenate(
                        [
                            np.squeeze(full_cat),
                            np.genfromtxt(
                                os.path.join(out_dir, cat_fname),
                                names=True,
                                delimiter=" ",
                                dtype=None,
                            ),
                        ]
                    )
                )

        max_iflat = np.max(flat_nbs)

        uniq_names_done = np.array(
            [
                "/".join([this_simname, str(this_snap), str(this_gid)])
                for this_simname, this_snap, this_gid in zip(
                    full_cat["simulation"].astype(str),
                    full_cat["snapshot"],
                    full_cat["galid"],
                )
            ]
        )
        uniq_names_current = np.array(
            [
                "/".join([this_simname, str(this_snap), str(this_gid)])
                for this_simname, this_snap, this_gid in zip(
                    sim_origin, snapshots, galaxy_ids
                )
            ]
        )

        _, match_all_args, _ = np.intersect1d(
            uniq_names_current, uniq_names_done, assume_unique=True, return_indices=True
        )

        total_found = len(match_all_args)

        all_args = np.arange(0, len(snapshots))

        not_matched_args = np.setdiff1d(all_args, match_all_args)
        iflat = max_iflat + 1

        l2trim = [
            fnames,
            all_rapers,
            all_rgals,
            all_r50s,
            zeds,
            zmocks,
            all_mgals,
            all_m_1p5xas,
            snapshots,
            galaxy_ids,
            directions,
            all_sfrs_10,
            all_sfrs_10_r50,
            all_sfrs_10_2xr50,
            all_sfrs_10_1p5xas,
            all_sfrs_100,
            all_sfrs_100_r50,
            all_sfrs_100_2xr50,
            all_sfrs_100_1p5xas,
            all_m_2xr50s,
            all_m_r50s,
            sim_origin,
            angs_per_ckpc,
        ]

        for il, l in enumerate(l2trim):
            l2trim[il] = np.asarray(l)[not_matched_args]

        (
            fnames,
            all_rapers,
            all_rgals,
            all_r50s,
            zmocks,
            all_mgals,
            all_m_1p5xas,
            snapshots,
            galaxy_ids,
            directions,
            all_sfrs_10,
            all_sfrs_10_r50,
            all_sfrs_10_2xr50,
            all_sfrs_10_1p5xas,
            all_sfrs_100,
            all_sfrs_100_r50,
            all_sfrs_100_2xr50,
            all_sfrs_100_1p5xas,
            all_m_2xr50s,
            all_m_r50s,
            sim_origin,
            angs_per_ckpc,
        ) = l2trim

    # iflat = 0

    filt_dict = {
        "f115w": "NIRCAM-F115",
        "f150w": "NIRCAM-F150",
        "f277w": "NIRCAM-F277",
        "f444w": "NIRCAM-F444",
        "HST-F814W": "HST-f814_res",
        "HSC-r": "HSC-r",
        "HSC-g": "HSC-g",
        "HSC-i": "HSC-i",
        "HSC-y": "HSC-y",
        "HSC-z": "HSC-z",
        "CFHT-u": "MegaCam-u_new",
    }

    filt_wav_dict = {
        "f115w": 1.150,
        "f150w": 1.501,
        "f277w": 2.760,
        "f444w": 4.400,
        "HST-F814W": 0.820,
        "CFHT-u": 0.386,
        "HSC-g": 0.475,
        "HSC-r": 0.623,
        "HSC-i": 0.770,
        "HSC-z": 0.890,
        "HSC-y": 1.000,
    }

    all_found_bools = np.full(len(all_rgals), False)

    while not found_all:

        print(
            f"flatnb {iflat}, so far {total_found} galaxies found out of {len(all_rgals)}"
        )

        search_args = np.where(all_found_bools == False)[0]

        print(f"searching for {len(search_args):d} extra galaxies to fill this flat")

        (
            tile_code,
            tile_shape_px,
            found_pos_px,
            all_rapers_px,
            found_all,
            done_bools,
            wcs,
        ) = gals_to_fit_pos(
            np.array(all_rapers)[search_args] * 1.1,
        )
        found_args = search_args[done_bools]

        # nfound = np.shape(found_pos_px)[0]
        nfound = done_bools.sum()
        print(f"found {nfound} spots for galaxies in flat #{iflat}")

        found_pos_deg = wcs.pixel_to_world(
            found_pos_px[done_bools, 0], found_pos_px[done_bools, 1]
        )

        out_dir = os.path.join(
            "/data101/jlewis/zoom_flats",
            f"allsims",
            'ZTEST',
            f"{idir_str:s}",
            f"{dust_model:s}",
            f"flat_{iflat:d}",
        )

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_flats_cat = os.path.join(out_dir, f"out_cat_{tile_code}.txt")
        out_flats_img_residual = os.path.join(
            out_dir, f"out_cat_{tile_code}_residual.png"
        )
        out_flats_img = os.path.join(out_dir, f"out_cat_{tile_code}.png")

        fname_res, _ = tile_to_file("f444w", tile_code)

        f444_hdr, _ = read_fits(fname_res)

        filt_list = [
            "CFHT-u",
            "HSC-g",
            "HSC-r",
            "HSC-i",
            "HST-F814W",
            "HSC-z",
            "HSC-y",
            "f115w",
            "f150w",
            "f277w",
            "f444w",
        ]
        # filt_list = [
        #     "CFHT-u",
        #     "HSC-g",
        #     "HSC-r",
        #     "HSC-i",
        #     "HST-F814W",
        #     "HSC-z",
        #     "HSC-y",
        #     "f115w",
        #     "f150w",
        #     "f277w",
        #     "f444w",
        # ]

        # fig, ax = plt.subplots(
        #     len(filt_list),
        #     3,
        #     layout="constrained",
        #     figsize=(3 * 4, 4 * len(filt_list)),
        # )

        band_names = []
        band_sums_mag = []
        band_sums_mag_noise = []
        band_sums_mag_r50 = []
        band_sums_mag_r50_noise = []
        band_sums_mag_2xr50 = []
        band_sums_mag_2xr50_noise = []
        band_sums_mag_1p5xas = []
        band_sums_mag_1p5xas_noise = []

        # correct zero points

        psf_filt_dict = filt_file_to_dict()

        psfs = [load_psf(psf_filt_dict[filt_dict[name]]["psf_fname"]) for name in filt_list[:]]
        print('loaded psfs')
        print([(filt_dict[filt_list[ipsf]], psf[1]) for ipsf, psf in enumerate(psfs)])

        psfs_rebinned=[]

        # load f444 header
        filt="f444w"
        

        fname_res, _ = tile_to_file(filt, tile_code)

        res_hdr, _ = read_fits(fname_res)

        wcs_f444 = WCS.WCS(res_hdr)

        f444_res = np.copy(get_hdr_res(res_hdr))

        # fig,axs = plt.subplots(len(filt_list),4,figsize=(4*4,4*11),layout='constrained',sharex=True,sharey=True)

        for ifilt, filt in enumerate(filt_list[:]):

            print(f'handling filter {filt}')

            band_names.append(filt)

            fname_res, _ = tile_to_file(filt, tile_code)

            res_hdr, res_data = read_fits(fname_res)

            wcs_cur_tile = WCS.WCS(res_hdr)

            instr_name = filt.split("-")[0]

            resolution_hdr_as = get_hdr_res(res_hdr)
            resolution_hdr_mas = resolution_hdr_as* 1e3


            res_data = np.copy(res_data.T)

            out_flats_fname = os.path.join(
                out_dir, f"out_flat_{filt:s}_{tile_code}.fits"
            )

            filt_sum_mag = []
            filt_sum_mag_noise = []
            filt_sum_mag_r50 = []
            filt_sum_mag_r50_noise = []
            filt_sum_mag_2xr50 = []
            filt_sum_mag_2xr50_noise = []
            filt_sum_mag_1p5xas = []
            filt_sum_mag_1p5xas_noise = []

            for igal, (fname, rgal_as, raper, raper_as, r50_as, found_pos) in enumerate(
                zip(
                    np.asarray(fnames)[found_args],
                    # all_rgals_px[done_bools],  # this has the size of search_args
                    all_rgals[found_args],
                    all_rapers_px[done_bools],  # this has the size of search_args
                    np.asarray(all_rapers)[found_args],
                    np.asarray(all_r50s)[found_args],
                    np.transpose([found_pos_deg.ra.value, found_pos_deg.dec.value])[:],
                )
            ):

                #load data
                with h5py.File(fname, "r") as src:
                    # print((src.keys()))
                    # print(src["header"].attrs["band_names"])

                    # print(ifilt,filt,src["header"].attrs["pixel_res_as"])
                    try:
                        img_gal_fine = src[filt_dict[filt]][()]
                        px_as = src["header"].attrs["pixel_res_as"]
                        # px_as = {
                        #     n: dx
                        #     for n, dx in zip(
                        #         src["header"].attrs["band_names"],
                        #         src["header"].attrs["pixel_res_as"],
                        #     )
                        # }[filt_dict[filt]]

                    except KeyError:
                        print(f"error reading {filt_dict[filt]} from {fname}")
                        continue

                size_gal_px = img_gal_fine.shape[0]
                size_gal_as = px_as * img_gal_fine.shape[0]

                # res_factor = px_as / resolution_hdr_as

                raper_fine_px = int(np.round(raper_as / px_as))

                #if we're bigger than the r_aper... trim
                if size_gal_px > 2*(raper_fine_px):

                    if size_gal_px%2!=0:
                        img_ctr = int(0.5*size_gal_px)+1

                        img_gal_fine = img_gal_fine[img_ctr-raper_fine_px:img_ctr+raper_fine_px+1,
                                                    img_ctr-raper_fine_px:img_ctr+raper_fine_px+1]
                    else:#even case
                        img_ctr = int(0.5*size_gal_px)

                        img_gal_fine = img_gal_fine[img_ctr-raper_fine_px-1:img_ctr+raper_fine_px,
                                                    img_ctr-raper_fine_px-1:img_ctr+raper_fine_px]






                #axs[ifilt,0].imshow(img_gal_fine.T,origin='lower',extent=[*(px_as*np.ravel(np.outer(np.transpose(img_gal_fine.shape),np.array([-0.5,0.5]))))])

                if len(psfs_rebinned)<ifilt+1:
                    psf_rebinned = dilate(psfs[ifilt][0],psfs[ifilt][1]/px_as) / (psfs[ifilt][1]/px_as)**2
                    print("rebinned psf")
                    # print(ifilt,filt,px_as,resolution_hdr_as,psfs[ifilt][1])
                    psfs_rebinned.append(psf_rebinned)
                else:
                    psf_rebinned = psfs_rebinned[ifilt]
                    print("loaded rebinned psf")

                #convolve with rescaled PSF
                img_MJy_wIGM_wPSF = fftconvolve(img_gal_fine, psf_rebinned)


                # reduce sizes - cut zero filled edges
                cur_img = img_MJy_wIGM_wPSF


                decal = 0

                while np.sum(cur_img[decal:-decal, decal:-decal]) == 0:
                    decal += 1

                img_MJy_wIGM_wPSF = cur_img[decal:-decal, decal:-decal]
                del cur_img

                size_after_psf_px = img_MJy_wIGM_wPSF.shape


                #axs[ifilt,1].imshow(img_MJy_wIGM_wPSF.T,origin='lower',extent=[*(px_as*np.ravel(np.outer(np.transpose(img_MJy_wIGM_wPSF.shape),np.array([-0.5,0.5]))))])

                #rebin to f444w resolution



                #get exact binning for insertion so that bands align

                xpos_deg, ypos_deg = found_pos
                sky = SkyCoord(xpos_deg * u.degree, ypos_deg * u.degree)
                cur_found_pos_px = np.round(wcs_f444.world_to_pixel(sky))
                cur_found_pos_deg = wcs_f444.pixel_to_world(*cur_found_pos_px)
                

                #we found centers in f444
                xpos_cell_f444,ypos_cell_f444 = np.squeeze(cur_found_pos_px)


                if abs(resolution_hdr_as - f444_res)<5e-3: #less than 1 mas difference 
                    #if resolution is equal, we rebin to center on the chosen pixel
                    bin_fact = px_as / resolution_hdr_as
                    tgt_size = np.int32(np.ceil(resolution_hdr_as / bin_fact))

                    img_gal = zoom(img_MJy_wIGM_wPSF, bin_fact,order=2)
                    fine_ctr = np.int32(0.5*np.array(img_gal.shape))

                    xmin = int(xpos_cell_f444-fine_ctr[0])
                    xmax = int(xpos_cell_f444+fine_ctr[0])
                    ymin = int(ypos_cell_f444-fine_ctr[1])
                    ymax = int(ypos_cell_f444+fine_ctr[1])

                    if xmax-xmin<img_gal.shape[0]:xmax+=1
                    if ymax-ymin<img_gal.shape[1]:ymax+=1

                else:
                    #if current resolution is coarser we need to do some resampling

                    #find nearest cell centre at this resolution
                    high_res_cell_ctr = (np.array(cur_found_pos_px)+0.5)*px_as

                    closest_coarse_cell= wcs_cur_tile.world_to_pixel(sky)
                    closest_coarse_cell_round = np.round(closest_coarse_cell)
                    # closest_coarse_as = np.array(closest_coarse_cell) * resolution_hdr_as
                    sky_coarse = wcs_cur_tile.pixel_to_world(*closest_coarse_cell_round)
                    closest_coarse_as = sky_coarse.ra.value*3600,sky_coarse.dec.value*3600

                    #this is the centering offset because of resolution differences, for rebinning
                    #pixel based difference doesn't work because of different discretizations of filters
                    # diff_to_tgt_as = #(closest_coarse_as+0.5*resolution_hdr_as) - high_res_cell_ctr
                    #let's just use the angles
                    diff_to_tgt_as = (closest_coarse_as + 0.5*resolution_hdr_as) - (np.array([cur_found_pos_deg.ra.value,cur_found_pos_deg.dec.value])*3600 + 0.5*f444_res)
                    print(ifilt,filt,diff_to_tgt_as)
                    print(fname)
                    print(closest_coarse_as,found_pos*3600)
                    # diff_to_tgt_as = diff_to_tgt_as
                    # diff_to_tgt_as = [1.0,0]
                    # diff_to_tgt_as = [0.,0.5]

                    #find shape containing coarse image plus one coarse resolution element margin on each axis
                    fine_size = img_MJy_wIGM_wPSF.shape[0]
                    ncoarse = int(np.ceil((fine_size * px_as)/resolution_hdr_as)) 
                    r_coarse_as = ncoarse * resolution_hdr_as
                    nxcoarse = int(((r_coarse_as+resolution_hdr_as) - (-resolution_hdr_as))//resolution_hdr_as)
                    nycoarse = int(((r_coarse_as+resolution_hdr_as) - (-resolution_hdr_as))//resolution_hdr_as)

                    if nxcoarse%2==0:
                        nxcoarse+=1
                    
                    if nycoarse%2==0:
                        nycoarse+=1
                    

                    xcoarse_bins = np.linspace(-resolution_hdr_as,r_coarse_as+resolution_hdr_as,nxcoarse)
                    ycoarse_bins = np.linspace(-resolution_hdr_as,r_coarse_as+resolution_hdr_as,nycoarse)

                    nxcoarse_ctr = int(0.5*nxcoarse)
                    nycoarse_ctr = int(0.5*nycoarse)    

                    #reference grid of high res data cells
                    xfine = np.linspace(0,fine_size*px_as,fine_size)
                    xg,yg = np.meshgrid(xfine,xfine)

    

                    #account for offset
                    offcenter_xg = xg + diff_to_tgt_as[1]
                    offcenter_yg = yg + diff_to_tgt_as[0]

                    rescale=np.ceil(resolution_hdr_as/px_as)

                    # smear image with a flat kernel of the right size
                    kern_size=int(rescale)
                    kernel = np.ones((kern_size,kern_size))
                    kernel/=kernel.sum()**2

                    #outside of image treated as 0
                    fine_degraded = convolve(img_MJy_wIGM_wPSF,kernel,mode='constant',cval=0.0)   

                    #now rebin to target resolution
                    img_gal = binned_statistic_2d(np.ravel(offcenter_xg),np.ravel(offcenter_yg),np.ravel(fine_degraded),'sum',bins=[xcoarse_bins,ycoarse_bins]).statistic.T     

                    # img_gal=rotate(img_gal, -20, reshape=False)
 
                    xmin=int(closest_coarse_cell[0]-nxcoarse_ctr)
                    xmax=int(closest_coarse_cell[0]+nxcoarse_ctr)
                    ymin=int(closest_coarse_cell[1]-nycoarse_ctr)
                    ymax=int(closest_coarse_cell[1]+nycoarse_ctr)



                
                px_as = max(resolution_hdr_as,px_as)

                #axs[ifilt,2].imshow(img_gal.T,origin='lower',extent=[*(px_as*np.ravel(np.outer(np.transpose(img_gal.shape),np.array([-0.5,0.5]))))])

                size = img_gal.shape

                px_sr = px_as**2 * (np.pi / 3600.0 / 180.0) ** 2

                if debug:
                    print(-2.5 * np.log10(img_gal.sum() * 1e6 * px_sr) + 8.9)
                zpt = band_to_zpt(filt)[0]
                img_gal_resc = resc_flux(img_gal, px_sr, filt_dict[filt])
                if debug:
                    print(-2.5 * np.log10(img_gal_resc.sum()) + zpt)
                img_gal_resc = img_gal_resc[0]

                # print(img_gal_resc.shape, size)


                # print(found_pos, rgal, xrgal, yrgal, res_data.shape)
                # print(filt, size)
                

                rgal_cm = rgal_as / np.asarray(angs_per_ckpc) * 1e3 * ramses_pc

                org_residuals = np.copy(
                    res_data[xmin : xmax, ymin : ymax]
                )

                # print(xmin, ymin, size, res_data.shape)

                # print(
                #     res_data[xmin : xmin + size[0], ymin : ymin + size[1]].shape,
                #     img_gal_resc.shape,
                # )  # j'ai des found_pos=0 mais que que pour

                print(cur_found_pos_px, cur_found_pos_deg, xpos_cell_f444, ypos_cell_f444, xmin, xmax, ymin, ymax, res_data.shape)


                #check for pixels out of imag
                xunder = xmin-0
                yunder = ymin-0
                xover = xmax-res_data.shape[0]
                yover = ymax-res_data.shape[1]
                
                tol=50

                if xunder < 0 and xunder > -tol:
                    xmin = 0
                if yunder < 0 and yunder > -tol:
                    ymin = 0

                if xover > 0 and xover < tol:
                    xmax -= xover
                if yover > 0 and yover < tol:
                    ymax -= yover

                res_data[xmin:xmax, ymin : ymax] = (
                    res_data[xmin : xmax, ymin : ymax]
                    + img_gal_resc
                )
                # print('warning: no noise!!!!!')
                # res_data[xmin : xmin + size[0], ymin : ymin + size[1]] = img_gal

                #axs[ifilt,3].imshow(res_data[xmin:xmax, ymin : ymax].T,origin='lower',extent=[*(px_as*np.ravel(np.outer(np.transpose(img_gal.shape),np.array([-0.5,0.5]))))])

                # aper_sr = (aper_as / 3600 / 180 * np.pi) ** 2
                rgal_sr = (2 * rgal_as / 3600 / 180 * np.pi) ** 2
                # rgal_sr = get_aper_sr(sim.cosmo_model.luminosity_distance, z, rgal_cm)

                rgal_px = abs(
                    int(
                        np.ceil(
                            wcs_cur_tile.wcs_world2pix(
                                xpos_deg + rgal_as / 3600.0,
                                ypos_deg,
                                0,
                            )[0]
                            - wcs_cur_tile.wcs_world2pix(xpos_deg, ypos_deg, 0)[0]
                        )
                    )
                )

                # r50_cm = r50_as / ang_per_ckpc * 1e3 * ramses_pc
                # r50_sr = get_aper_sr(sim.cosmo_model.luminosity_distance, z, r50_cm)
                r50_sr = (2 * r50_as / 3600 / 180 * np.pi) ** 2

                r50_px = rgal_px * r50_as / rgal_as
                # aper_ratio = r50_as / aper_as
                # aper_size_px = aper_ratio * img_gal.shape[0]
                # r_select_px = int(0.5 * aper_size_px)
                half_size_px = int(0.5 * img_gal.shape[0])

                Ximg, Yimg = (
                    np.mgrid[0 : img_gal.shape[0], 0 : img_gal.shape[1]]
                    - half_size_px
                    - 0.5
                )

                select_circ = np.linalg.norm([Ximg, Yimg], axis=0) <= rgal_px

                # px_sr=rgal_sr / select_circ.sum()

                aper_mag = (
                    -2.5 * np.log10(np.sum(img_gal[select_circ]) * 1e6 * px_sr) + 8.90
                )

                img_gal_w_noise = res_data[xmin : xmin + size[0], ymin : ymin + size[1]]

                aper_mag_w_noise = (
                    -2.5 * np.log10(np.sum(img_gal_w_noise[select_circ])) + zpt
                )
                # aper_mag_w_noise = (
                #     -2.5 * np.log10(np.sum(img_gal_w_noise[select_circ]) * 1e6 * px_sr)
                #     + 8.90
                # )

                filt_sum_mag.append(aper_mag)
                filt_sum_mag_noise.append(aper_mag_w_noise)

                if r50_px >= 0.5:

                    select_circ_r50 = np.linalg.norm([Ximg, Yimg], axis=0) <= r50_px

                    aper_mag_r50 = (
                        -2.5 * np.log10(np.sum(img_gal[select_circ_r50]) * 1e6 * px_sr)
                        + 8.90
                    )
                    aper_mag_r50_noise = (
                        -2.5 * np.log10(np.sum(img_gal_w_noise[select_circ_r50])) + zpt
                    )
                    # aper_mag_r50_noise = (
                    #     -2.5
                    #     * np.log10(np.sum(img_gal_w_noise[select_circ_r50]) * 1e6 * px_sr)
                    #     + 8.90
                    # )

                else:

                    flux_central_px = img_gal[half_size_px, half_size_px]
                    flux_central_px_noise = img_gal_w_noise[half_size_px, half_size_px]

                    aper_mag_r50 = (
                        -2.5
                        * np.log10(flux_central_px * r50_px**2 * np.pi * 1e6 * px_sr)
                        + 8.90
                    )
                    aper_mag_r50_noise = (
                        -2.5 * np.log10(flux_central_px_noise * r50_px**2 * np.pi) + zpt
                    )
                    # aper_mag_r50_noise = (
                    #     -2.5
                    #     * np.log10(flux_central_px_noise * r50_px**2 * np.pi * 1e6 * px_sr)
                    #     + 8.90
                    # )

                filt_sum_mag_r50.append(aper_mag_r50)
                filt_sum_mag_r50_noise.append(aper_mag_r50_noise)

                if (r50_px * 2) >= 0.5:

                    select_circ_2xr50 = np.linalg.norm([Ximg, Yimg], axis=0) <= (
                        r50_px * 2
                    )

                    aper_mag_2xr50 = (
                        -2.5
                        * np.log10(np.sum(img_gal[select_circ_2xr50]) * 1e6 * px_sr)
                        + 8.90
                    )
                    aper_mag_2xr50_noise = (
                        -2.5 * np.log10(np.sum(img_gal_w_noise[select_circ_2xr50]))
                        + zpt
                    )
                    # )
                    # aper_mag_2xr50_noise = (
                    #     -2.5
                    #     * np.log10(np.sum(img_gal_w_noise[select_circ_2xr50]) * 1e6 * px_sr)
                    #     + 8.90
                    # )

                else:

                    flux_central_px = img_gal[half_size_px, half_size_px]
                    aper_mag_2xr50 = (
                        -2.5
                        * np.log10(
                            flux_central_px * (r50_px * 2) ** 2 * np.pi * 1e6 * px_sr
                        )
                        + 8.90
                    )
                    flux_central_px_noise = img_gal_w_noise[half_size_px, half_size_px]
                    aper_mag_2xr50_noise = (
                        -2.5
                        * np.log10(flux_central_px_noise * (r50_px * 2) ** 2 * np.pi)
                        + zpt
                    )
                    # aper_mag_2xr50_noise = (
                    #     -2.5
                    #     * np.log10(
                    #         flux_central_px_oise * (r50_px * 2) ** 2 * np.pi * 1e6 * px_sr
                    #     )
                    #     + 8.90
                    # )

                filt_sum_mag_2xr50.append(aper_mag_2xr50)
                filt_sum_mag_2xr50_noise.append(aper_mag_2xr50_noise)

                as_ckpc = sim.cosmo_model.arcsec_per_kpc_comoving(z).value
                ckpc_code = 1.0 / sim.cosmo.lcMpc * 1e3
                as_code = as_ckpc / ckpc_code
                r_1p5xas_code = 1.5/as_code
                r_1p5xas_px = r_1p5xas_code * (2**sim.levelmax) / 1

                if (r_1p5xas_px) >= 0.5:

                    select_circ_1p5xas = np.linalg.norm([Ximg, Yimg], axis=0) <= (
                        r_1p5xas_px
                    )

                    aper_mag_1p5xas = (
                        -2.5
                        * np.log10(np.sum(img_gal[select_circ_1p5xas]) * 1e6 * px_sr)
                        + 8.90
                    )
                    aper_mag_1p5xas_noise = (
                        -2.5 * np.log10(np.sum(img_gal_w_noise[select_circ_1p5xas]))
                        + zpt
                    )

                else:

                    flux_central_px = img_gal[half_size_px, half_size_px]
                    aper_mag_1p5xas = (
                        -2.5
                        * np.log10(
                            flux_central_px * (r_1p5xas_px) ** 2 * np.pi * 1e6 * px_sr
                        )
                        + 8.90
                    )
                    flux_central_px_noise = img_gal_w_noise[half_size_px, half_size_px]
                    aper_mag_1p5xas_noise = (
                        -2.5
                        * np.log10(flux_central_px_noise * (r_1p5xas_px) ** 2 * np.pi)
                        + zpt
                    )
                    # aper_mag_2xr50_noise = (
                    #     -2.5
                    #     * np.log10(
                    #         flux_central_px_noise * (r50_px * 2) ** 2 * np.pi * 1e6 * px_sr
                    #     )
                    #     + 8.90
                    # )

                filt_sum_mag_1p5xas.append(aper_mag_1p5xas)
                filt_sum_mag_1p5xas_noise.append(aper_mag_1p5xas_noise)

                print(
                    filt,
                    aper_mag,
                    aper_mag_w_noise,
                    aper_mag_r50,
                    aper_mag_2xr50,
                    aper_mag_2xr50_noise,
                    aper_mag_1p5xas,
                    aper_mag_1p5xas_noise,
                    px_sr,
                    iflat,
                )

                # print(rgal, img_gal.shape)

                # input('')

            # newHDU = fits.PrimaryHDU(data=res_data,)

            # res_data = array, footprint = reproject_interp(newHDU, res_hdr)

            band_sums_mag.append(filt_sum_mag)
            band_sums_mag_r50.append(filt_sum_mag_r50)
            band_sums_mag_2xr50.append(filt_sum_mag_2xr50)
            band_sums_mag_1p5xas.append(filt_sum_mag_1p5xas)
            band_sums_mag_noise.append(filt_sum_mag_noise)
            band_sums_mag_r50_noise.append(filt_sum_mag_r50_noise)
            band_sums_mag_2xr50_noise.append(filt_sum_mag_2xr50_noise)
            band_sums_mag_1p5xas_noise.append(filt_sum_mag_1p5xas_noise)

            # res_hdr["BAND"] = filt

            save_fits(out_flats_fname, res_hdr, res_data, overwrite=True)

            # fig.savefig('diag_rebinning')

        datas_list = [
            np.int16(found_pos_px[done_bools, 0]),
            np.int16(found_pos_px[done_bools, 1]),
            # all_rgals_px[done_bools],
            all_rapers_px[done_bools],
            found_pos_deg.ra.value,
            found_pos_deg.dec.value,
            np.asarray(all_rgals)[found_args] / 3600.0,
            np.asarray(all_rapers)[found_args] / 3600.0,
            np.asarray(all_r50s)[found_args] / 3600.0,  # broken
            np.asarray(zeds)[found_args],
            np.asarray(zmocks)[found_args],
            np.asarray(sim_origin)[found_args],
            np.asarray(snapshots)[found_args],
            np.asarray(galaxy_ids)[found_args],
            np.asarray(directions)[found_args],
            np.asarray(all_mgals)[found_args],
            np.asarray(all_m_r50s)[found_args],
            np.asarray(all_m_2xr50s)[found_args],
            np.asarray(all_m_1p5xas)[found_args],
            np.asarray(all_sfrs_10)[found_args],
            np.asarray(all_sfrs_10_r50)[found_args],
            np.asarray(all_sfrs_10_2xr50)[found_args],
            np.asarray(all_sfrs_10_1p5xas)[found_args],
            np.asarray(all_sfrs_100)[found_args],
            np.asarray(all_sfrs_100_r50)[found_args],
            np.asarray(all_sfrs_100_2xr50)[found_args],
            np.asarray(all_sfrs_100_1p5xas)[found_args],
            *band_sums_mag,
            *band_sums_mag_r50,
            *band_sums_mag_2xr50,
            *band_sums_mag_1p5xas,
            # sfr100s,
            # sfr1000s,
        ]

        for ix, d in enumerate(datas_list):
            print(ix,d)
        #     print(ix, np.shape(d), d[0], d[-1])

        np.savetxt(
            out_flats_cat,
            np.transpose(
                [
                    np.int16(found_pos_px[done_bools, 0]),
                    np.int16(found_pos_px[done_bools, 1]),
                    # all_rgals_px[done_bools],
                    all_rapers_px[done_bools],
                    found_pos_deg.ra.value,
                    found_pos_deg.dec.value,
                    np.asarray(all_rgals)[found_args] / 3600.0,
                    np.asarray(all_rapers)[found_args] / 3600.0,
                    np.asarray(all_r50s)[found_args] / 3600.0,
                    np.asarray(zeds)[found_args],
                    np.asarray(zmocks)[found_args],
                    np.asarray(sim_origin)[found_args],
                    np.asarray(snapshots)[found_args],
                    np.asarray(galaxy_ids)[found_args],
                    np.asarray(directions)[found_args],
                    np.asarray(all_mgals)[found_args],
                    np.asarray(all_m_r50s)[found_args],
                    np.asarray(all_m_2xr50s)[found_args],
                    np.asarray(all_m_1p5xas)[found_args],
                    np.asarray(all_sfrs_10)[found_args],
                    np.asarray(all_sfrs_10_r50)[found_args],
                    np.asarray(all_sfrs_10_2xr50)[found_args],
                    np.asarray(all_sfrs_10_1p5xas)[found_args],
                    np.asarray(all_sfrs_100)[found_args],
                    np.asarray(all_sfrs_100_r50)[found_args],
                    np.asarray(all_sfrs_100_2xr50)[found_args],
                    np.asarray(all_sfrs_100_1p5xas)[found_args],
                    *band_sums_mag,
                    *band_sums_mag_r50,
                    *band_sums_mag_2xr50,
                    *band_sums_mag_1p5xas,
                    # sfr100s,
                    # sfr1000s,
                ]
            ),
            fmt="%s",
            # fmt="%.6e",
            header=(
                # "x(px) y(px) rgal(px) raper(px) ra(deg) dec(deg) rgal(deg) raper(deg) r50(deg) z zmock simulation snapshot galid direction mgal(msun) m_r50(msun) m_2xr50(msun) m_1.5as(msun) sfr_10Myr(msun/yr) sfr_10Myr_r50(msun/yr) sfr_10Myr_2xr50(msun/yr) sfr_10Myr_1.5as(msun/yr) sfr_100Myr(msun/yr) sfr_100Myr_r50(msun/yr) sfr_100Myr_2xr50(msun/yr) sfr_100Myr_1.5as(msun/yr) "
                "x(px) y(px) raper(px) ra(deg) dec(deg) rgal(deg) raper(deg) r50(deg) z zmock simulation snapshot galid direction mgal(msun) m_r50(msun) m_2xr50(msun) m_1.5as(msun) sfr_10Myr(msun/yr) sfr_10Myr_r50(msun/yr) sfr_10Myr_2xr50(msun/yr) sfr_10Myr_1.5as(msun/yr) sfr_100Myr(msun/yr) sfr_100Myr_r50(msun/yr) sfr_100Myr_2xr50(msun/yr) sfr_100Myr_1.5as(msun/yr) "
                + " ".join([b + "_rgal(magAB)" for b in band_names])
                + " "
                + " ".join([b + "_r50(magAB)" for b in band_names])
                + " "
                + " ".join([b + "_2xr50(magAB)" for b in band_names])
                + " "
                + " ".join([b + "_1.5as(magAB)" for b in band_names])
            ),
        )

        prev_total_found = total_found
        total_found += nfound
        all_found_bools[found_args] = True

        print(prev_total_found,total_found,nfound)
        print(len(found_args),all_found_bools.sum())

        if prev_total_found == total_found:
            print("this iteration found no extra spaces for galaxies")
            print(
                f"there are {len(search_args):d} missing galaxies that were not added to flats"
            )
            print(f"They were likely too big to fit, their sizes are {all_rgals} as")
            break
        else:
            iflat += 1
