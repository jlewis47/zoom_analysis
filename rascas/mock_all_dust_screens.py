import numpy as np
import matplotlib.pyplot as plt

from astropy import wcs as WCS
from astropy.coordinates import SkyCoord
from astropy import units as u

from scipy.stats import binned_statistic_2d
from scipy.signal import fftconvolve
from scipy.ndimage import zoom,rotate
from scipy.interpolate import BarycentricInterpolator,CubicSpline,make_interp_spline
from scipy.optimize import curve_fit as cvf

from h5tools.h5py_tools import h5_to_dict

from matplotlib.backends.backend_pdf import PdfPages

import os

from zoom_analysis.catalogue.read_cat import read_or_make_cat
from zoom_analysis.rascas.rascas_steps import filt_file_to_dict
from zoom_analysis.rascas.read_rascas import (
    mock_spe_dump,
)
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

def sum_mag_img(img):

    return -2.5*np.log10(np.nansum(10**(img/-2.5)))

def read_law(path):

    return np.genfromtxt(f"{path}", names=True, dtype=None)

def Aext_poly2(x,a,b,c):

    out = a*x**2+b*x+c
    out[out<0.]=0.
    return(out)
def Aext_poly3(x,a,b,c,d):
    out = a*x**3+b*x**2+c*x+d
    out[out<0.]=0.
    return(out)
def Aext_exp(x,a,b):
    out = a*np.exp(-x*b)
    out[out<0.]=0.
    return(out)


def write_mock_imgs(
    fname, aexp, time, z, aper_sr, px_res_as, px_res_stars_as, names, imgs, st_img
):

    with h5py.File(
        fname,
        "w",
    ) as f:
        hdr = f.create_group("header")

        hdr.attrs["redshift"] = z
        hdr.attrs["aexp"] = aexp
        hdr.attrs["time_Myr"] = time
        hdr.attrs["aperture_sr"] = aper_sr
        hdr.attrs["pixel_res_as"] = px_res_as
        hdr.attrs["pixel_res_stars_as"] = px_res_stars_as
        hdr.attrs["img_shapes"] = [i.shape for i in imgs]
        hdr.attrs["band_names"] = names
        hdr.attrs["flux_units"] = "MJy.sr^-1"
        hdr.attrs["stellar_surface_density_units"] = "M_sun.as^-2"
        hdr.attrs["stellar_surface_density_help"] = (
            "computed as a 2d histogram of the stellar particles in the aperture, taking stars that are in [-0.5,0.5] around the image in the projection direction"
        )

        for i, (band_name, img) in enumerate(zip(names, imgs)):
            f.create_dataset(band_name, data=img, compression="lzf")

        f.create_dataset("stellar_surface density", data=st_img, compression="lzf")


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

b_ctr = 4439 / 1e4
U_ctr = 0.365 
V_ctr = 0.551
r_ctr = 0.6219
J_ctr = 1.2525
NUV_ctr = 0.23065



if __name__ == "__main__":

    hm = "HaloMaker_stars2_dp_rec_dust/"

    sim_paths = [
        "/data101/jlewis/sims/NH_for_rascas",
        # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
        # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
        # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288",
        # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
        # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
        # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130",
        # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756",
    ]

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


    filt_names = [filt_dict[f] for f in filt_list]

    filt_wavs = [filt_wav_dict[f] for f in filt_list]

    rescale_A=1.5
    rescale_A_str = f"{rescale_A:.1f}".replace('.','p')

    dust_law = "SB_calzetti.dat"
    dust_dir = "/home/jlewis/codes/zoom_analysis/rascas/attenuation_laws"

    law = read_law(os.path.join(dust_dir,dust_law))

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

    overwrite = False 
    debug = False

    nist_pc = 3.085678e18  # cm
    dist_10pc = 10.0 * nist_pc

    model_dust_ref = "SMC"
    model_nodust = "noDust"

    model_types = [model_dust_ref,model_nodust]

    models = [f"zoom_tgt_bc03_chabrier100_wlya_{mt}" for mt in model_types]

    model_name_dust_ref = f"zoom_tgt_bc03_chabrier100_wlya_{model_dust_ref}"
    model_name_nodust = f"zoom_tgt_bc03_chabrier100_wlya_{model_nodust}"

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


        mock_path_ints = [os.path.join(
            "/data101/jlewis",
            "mock_spe",
            model_name,
            f"{intID}",
        )
        for model_name in models
        ]
        mock_paths = [os.path.join(
            "/data101/jlewis",
            "mock_spe",
            model_name,
            f"{sim.name:s}",
        )
        for model_name in models
        ]

        mock_snap_paths_dust = []
        mock_snap_paths_nodust = []

        for ps in mock_paths, mock_path_ints:

            p1,p2=ps

            if not os.path.exists(p1) or not os.path.exists(p2):
                print(f"No existing mocks at {p1} or {p2}")
                continue


            snaps1 = [f for f in os.listdir(p1) if os.path.isdir(os.path.join(p1, f))]
            snaps2 = [f for f in os.listdir(p2) if os.path.isdir(os.path.join(p2, f))]

            u_snaps = np.intersect1d(snaps1,snaps2)

            print(snaps1,snaps2)


            mock_snap_paths_dust.extend(
                [
                    os.path.join(p1, snap)
                    # for snap in snaps1
                    for snap in u_snaps
                ]
            )
            mock_snap_paths_nodust.extend(
                [
                    os.path.join(p2, snap)
                    # for snap in snaps2
                    for snap in u_snaps
                ]
            )

        if len(mock_snap_paths_dust) == 0:
            print("No mocks found")
            continue

        # print(mock_snap_paths)

        mock_snaps_dust = [
            int(p.split("/")[-1]) if p[-1] != "/" else int(p.split("/")[-2])
            for p in mock_snap_paths_dust
        ]
        mock_snaps_nodust = [
            int(p.split("/")[-1]) if p[-1] != "/" else int(p.split("/")[-2])
            for p in mock_snap_paths_nodust
        ]

        print(f"Found snapshots:{mock_snap_paths_dust} with mocks to place in flats")

        for i, ((snap_path_dust,snap_path_nodust), (snap)) in enumerate(zip(zip(mock_snap_paths_dust,mock_snap_paths_nodust), mock_snaps_dust[:])):
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
            gal_folders_dust = os.listdir(snap_path_dust)
            gal_folders_nodust = os.listdir(snap_path_nodust)
            gids_dust = [int(gdir.split("_")[-1]) for gdir in gal_folders_dust]
            gids_nodust = [int(gdir.split("_")[-1]) for gdir in gal_folders_nodust]

            gids = np.intersect1d(gids_dust,gids_nodust)

            for igid,gid in enumerate(gids):

                out_path = snap_path_dust.replace(model_dust_ref,f"{dust_law.replace('.dat',''):s}_screen_{rescale_A_str:s}")

                outp = os.path.join(
                    out_path,
                    f"gal_{gid:07d}",
                )

                if idir=='random':
                    idir_nb = np.random.randint(low=0,high=11,size=1)[0]
                else:
                    idir_nb = idir                

                fname_mock_img = os.path.join(
                    outp,
                    "MJy_images",
                    f"mock_images_{name}_{snap}_{gid}_{idir_nb:d}_wIGM.h5",
                )

                if os.path.exists(fname_mock_img) and not overwrite:
                    continue                

                inp_dust = os.path.join(
                    snap_path_dust,
                    f"gal_{gid:07d}",
                )
                inp_nodust = os.path.join(
                    snap_path_nodust,
                    f"gal_{gid:07d}",
                )

                img_dir = os.path.join(outp, "MJy_images")

                if not os.path.exists(img_dir):
                    os.makedirs(img_dir, exist_ok=True)

                fname_mock_img_dust = os.path.join(
                    inp_dust,
                    "MJy_images",
                    f"mock_images_{sim.name}_{snap}_{gid}_{idir_nb:d}_wIGM.h5",
                )
                if not os.path.exists(fname_mock_img_dust):

                    fname_mock_img_dust = os.path.join(
                        inp_dust,
                        "MJy_images",
                        f"mock_images_{intID}_{snap}_{gid}_{idir_nb:d}_wIGM.h5",
                    )

                    if not os.path.exists(fname_mock_img_dust):

                        print(f"Couldn't find {fname_mock_img_dust}")
                        continue
                
                fname_mock_img_nodust = os.path.join(
                    inp_nodust,
                    "MJy_images",
                    f"mock_images_{sim.name}_{snap}_{gid}_{idir_nb:d}_wIGM.h5",
                )
                if not os.path.exists(fname_mock_img_nodust):

                    fname_mock_img_nodust = os.path.join(
                        inp_nodust,
                        "MJy_images",
                        f"mock_images_{intID}_{snap}_{gid}_{idir_nb:d}_wIGM.h5",
                    )

                    if not os.path.exists(fname_mock_img_nodust):

                        print(f"Couldn't find {fname_mock_img_nodust}")
                        continue
                

                print(
                    f"sim: {sim.name}, getting gal properties for galaxy {gid:d} in snapshot {snap:d}"
                )



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

                aper_cm = rgal * 2 * l * (1e6 * ramses_pc)
                distl = sim.cosmo_model.luminosity_distance(z).value * (1e6 * ramses_pc)
                aper_as = np.arctan(aper_cm / distl) * 180.0 / np.pi * 3600.0

                imgs=[]
                imgs_flux=[]
                res=[]
                try:

                    for fpath in fname_mock_img_dust,fname_mock_img_nodust:

                        these_imgs=[]
                        these_imgs_flux=[]

                        with h5py.File(fpath) as src:

                            px_as = src["header"].attrs["pixel_res_as"]
                            px_as_star = src["header"].attrs["pixel_res_stars_as"]
                            aper_sr = src["header"].attrs["aperture_sr"]
                            time = src["header"].attrs["time_Myr"]
                            px_sr = (px_as / 3600 / 180 * np.pi) ** 2

                            stellar_density = src["stellar_surface density"][()]

                            for filt in filt_list:
                                flux_img=src[filt_dict[filt]][()]
                                mag_img= - 2.5*np.log10(flux_img * px_sr * 1e6)
                                these_imgs.append(mag_img)
                                these_imgs_flux.append(flux_img)



                        
                        imgs.append(these_imgs)
                        imgs_flux.append(these_imgs_flux)
                        res.append(px_as)

                except KeyError:
                    print(f"error reading from {fpath}")
                    continue


                if np.diff(res)>0.1:
                    print(res,np.diff(res))
                    print(f"dusty and dustless have different resolutions:{res}")
                    continue



                rest_frame_wavs = np.array(filt_wavs)/(1.+z)

                b_band = np.argmin(np.abs(rest_frame_wavs-b_ctr))
                V_band = np.argmin(np.abs(rest_frame_wavs-V_ctr))

                # ebv_dusty = sum_mag_img(imgs[0][b_band])-sum_mag_img(imgs[0][V_band])
                # ebv_dustless = sum_mag_img(imgs[1][b_band])-sum_mag_img(imgs[1][V_band])

                if imgs[0][b_band].shape != imgs[1][b_band].shape:
                    print('dusty and dustless have different shapes')

                    #do something about it

                    for iband in range(len(filt_list)):

                        shapes = [imgs[0][iband].shape[0],imgs[1][iband].shape[0]]

                        argmin = np.argmin(shapes)
                        argmax = np.argmax(shapes)

                        big_sh = shapes[argmax]
                        sml_sh = shapes[argmin]

                        pad_size = 0.5*(np.array(big_sh)-np.array(sml_sh))

                        if pad_size%1==0: #all ok

                            pad_size=np.int32(pad_size)
                            pad_size = np.repeat(pad_size,2) #split this
                            pad_size = np.array([pad_size,pad_size])

                        else: #uhoh might need to do something funky

                            
                            print('unhandled asymmetric padding')



                            pad_size = np.int32(np.repeat(pad_size,2)) #split this
                            pad_size[1]+=1
                            pad_size = np.array([pad_size,pad_size])

                            # break

    
                        imgs[argmin][iband] = np.pad(imgs[argmin][iband],pad_size,constant_values=0.0)
                        imgs_flux[argmin][iband] = np.pad(imgs_flux[argmin][iband],pad_size,constant_values=0.0)
                        

                        assert imgs[argmin][iband].shape == imgs[argmax][iband].shape, 'shapes still do not match...'

                    # print(ebv_dusty.shape,ebv_dustless.shape)
                    # raise ValueError
                    # continue

                ebv_dusty = imgs[0][b_band]-imgs[0][V_band]
                ebv_dustless = imgs[1][b_band]-imgs[1][V_band]

                ebv_img = ebv_dusty - ebv_dustless

                # ebv = ebv_dusty - ebv_dustless

                Ab = sum_mag_img(imgs[0][b_band]) - sum_mag_img(imgs[1][b_band])

                #use overall aperture B attenuation to scale/calibrate
                #anything else ends up very messy
              
                # Ab = imgs[0][b_band] - imgs[1][b_band]
                # Ab[np.isfinite(Ab)==False]=0.0
                # Ab[Ab<0]=0.0
                
                Aext_law = ebv_img[:,:,np.newaxis] * law['klambda']
                Aext_law_bands = np.asarray([Aext_law[:,:,np.argmin(np.abs(law['lambdaangstroms']/1e4-rfl))] for rfl in rest_frame_wavs])
                # Aext_law = ebv * law['klambda']
                # Aext_law_bands = np.asarray([Aext_law[np.argmin(np.abs(law['lambdaangstroms']/1e4-rfl))] for rfl in rest_frame_wavs])

                Ab_law= Aext_law_bands[np.argmin(np.abs(rest_frame_wavs-b_ctr))]

                # Aext = Aext_law_bands * (Ab/Ab_law * rescale_A)[:,:,np.newaxis]
                Aext = Aext_law_bands * (Ab/Ab_law * rescale_A)

                Aext[np.isfinite(Aext)==False] = 0.0
                Aext[Aext<0] = 0.0

                xsize = imgs[1][0].shape[0]
                xg,yg = np.mgrid[:xsize,:xsize]

                # img_screen = np.asarray(imgs[1]) + Aext[:,np.newaxis,np.newaxis]
                # imgs_screen_flux = imgs_flux[1] * 10**(Aext[:,np.newaxis,np.newaxis]/-2.5)
                # img_screen = np.asarray(imgs[1]) + Aext.T
                # imgs_screen_flux = imgs_flux[1] * 10**(Aext.T/-2.5)
                img_screen = np.asarray(imgs[1]) + Aext
                imgs_screen_flux = imgs_flux[1] * 10**(Aext/-2.5)

                xg = xg/xg.max()-0.5
                yg = yg/yg.max()-0.5
                rg = np.sqrt(xg**2+yg**2)
                cond = rg<=1.0

                Aext_aper = [(-2.5*np.log10(np.nansum(img_screen_flux) * 1e6 * px_sr) - 
                             (-2.5*np.log10(np.nansum(img_flux) * 1e6 * px_sr))) for img_screen_flux,img_flux in zip(imgs_screen_flux,imgs_flux[1])]

                in_spec = os.path.join(inp_nodust, f"mock_spectrum_{idir}.h5")
                in_spec_dusty = os.path.join(inp_dust, f"mock_spectrum_{idir}.h5")

                dustless_spec = h5_to_dict(in_spec)
                dusty_spec = h5_to_dict(in_spec_dusty)

                spec_dustless = dustless_spec['speectral_data']['spectrum']
                wav_dustless = dustless_spec['speectral_data']['lam spectrum']
                gal_data = dustless_spec['galaxy_data']

                # interp_Aext_aper = np.interp(wav_dustless/1e4, filt_wavs, Aext_aper)
                # interp = BarycentricInterpolator(filt_wavs, Aext_aper)
                interp = CubicSpline(filt_wavs, Aext_aper)
                interp_Aext_aper = interp(wav_dustless/1e4)
                # popts,pcov = cvf(Aext_exp,filt_wavs,Aext_aper)
                # interp_Aext_aper = Aext_exp(wav_dustless/1e4,*popts)
                # popts,pcov = cvf(Aext_poly3,filt_wavs,Aext_aper)
                # interp_Aext_aper = Aext_poly3(wav_dustless/1e4,*popts)
                # popts,pcov = cvf(Aext_poly2,filt_wavs,Aext_aper)
                # interp_Aext_aper = Aext_poly2(wav_dustless/1e4,*popts)
                # outside_red=wav_dustless/1e4>np.max(filt_wavs)
                # arg_red = np.min(np.where(outside_red))
                # arg_red = np.argmin(interp_Aext_aper)

                #extrapolate down to zero at red end
                arg_red = np.argmin(np.abs(wav_dustless/1e4-np.max(filt_wavs)))
                slope = np.diff(interp_Aext_aper[arg_red-5:arg_red]).sum()/(np.diff(wav_dustless[arg_red-5:arg_red]).sum()/1e4)
                interp_Aext_aper[arg_red:]=interp_Aext_aper[arg_red] + slope * (wav_dustless[arg_red:]-wav_dustless[arg_red])/1e4
                interp_Aext_aper[interp_Aext_aper<0]=0

                #extrapolate up at blue end
                arg_blue = np.argmin(np.abs(wav_dustless/1e4-np.min(filt_wavs)))
                slope = np.diff(interp_Aext_aper[arg_blue:arg_blue+5]).sum()/(np.diff(wav_dustless[arg_blue:arg_blue+5]).sum()/1e4)
                interp_Aext_aper[:arg_blue]=interp_Aext_aper[arg_blue] + slope * (wav_dustless[:arg_blue]-wav_dustless[arg_blue])/1e4

                spec_dusty_screen = spec_dustless + interp_Aext_aper

                ang_per_ckpc = sim.cosmo_model.arcsec_per_kpc_comoving(z).value

                #save the mock image



                write_mock_imgs(
                    fname_mock_img,
                    aexp,
                    time,
                    z,
                    aper_sr,
                    px_as,
                    px_as_star,
                    filt_names,
                    imgs_screen_flux,
                    stellar_density,
                )


                out_spec = os.path.join(outp, f"mock_spectrum_{idir}.h5")

                gal_data_adapt = {
                    "mass":gal_data["galaxy_mass(msun)"],
                    "sfr":gal_data["galaxy_sfr(msun.myr^-1)"],
                    "ssfr":gal_data["galaxy_ssfr(myr^-1)"],
                    "time":gal_data["t(myr)"],
                    "age":gal_data["max age(Gyr)"],
                    "age_wmstar":gal_data["mass weighted age(Gyr)"],
                }
                if 'aperture_mass(msun)' in gal_data.keys():
                    gal_data_adapt["mass_aper"]=gal_data["aperture_mass(msun)"]
                    gal_data_adapt["sfr_aper"]=gal_data["aperture_sfr(msun.myr^-1)"]
                    gal_data_adapt["ssfr_aper"]=gal_data["aperture_ssfr(myr^-1)"]

                #save to file
                mock_spe_dump(
                        out_spec,
                        z,
                        # 2 * rgal_ckpc / ang_per_ckpc,
                        rgal_ckpc / ang_per_ckpc,
                        filt_names,
                        wav_dustless,
                        spec_dusty_screen,
                        dustless_spec['speectral_data']['mag'] + Aext_aper,
                        dustless_spec['speectral_data']['mag_err'] + Aext_aper,
                        dustless_spec['speectral_data']['mag_rf'] + Aext_aper,
                        dustless_spec['speectral_data']['mag_err_rf'] + Aext_aper,
                        filt_wavs,
                        gal_data=gal_data_adapt,
                    )
                

