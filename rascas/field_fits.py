""" This module contains functions to read and write fits files, and to fit galaxies in a given field with a given mask. """


# import astropy
# import astropy.cosmology
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
import astropy.units as u
from h11 import Data
from matplotlib.patches import Circle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import zoom
from scipy.interpolate import NearestNDInterpolator

# from astropy.cosmology import FlatLambdaCDM
import os
from scipy.stats import binned_statistic_2d
from scipy.interpolate import RegularGridInterpolator


"""
convert star masks to pixel coords for every tile and move to infinity
with mask give catalogue of x,y,r_as,r_px,mgal,sfr
"""


def load_reg(fname, wcs):

    ra_deg, dec_deg, r_deg = np.loadtxt(fname, delimiter=" ").T

    degs = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)

    pxs = wcs.world_to_pixel(degs)

    return pxs


def read_fits_hdr(fname):

    with fits.open(fname, memmap=True) as hdul:

        # print(hdul.info())

        return hdul[0].header


def read_fits(fname):

    with fits.open(fname, memmap=True) as hdul:

        # print(hdul.info())

        return (hdul[0].header, hdul[0].data)


def save_fits(fname, hdr, data, **kwargs):

    hdu = fits.PrimaryHDU(np.transpose(data), hdr)
    hdul = fits.HDUList([hdu])

    hdul.writeto(fname, **kwargs)


def get_hdr_res(hdr):

    # keys = list(hdr.keys())

    # print(hdr)
    # print(keys)

    # if (
    #     "CDELT1" in keys and abs(hdr["CDELT1"]) != 1.0
    # ):  # don't know why but sometimes empty
    #     as_to_px = abs(hdr["CDELT1"]) * 3600.0
    # # elif "PC1_1" in keys:
    # #     as_to_px = abs(hdr["PC1_1"]) * 3600.0
    # # elif "CRDELT1" in keys:
    # #     as_to_px = abs(hdr["CRDELT1"]) * 3600.0
    # # elif "CD1_1" in keys:
    # #     as_to_px = abs(hdr["CD1_1"]) * 3600.0
    # # else:
    # #     raise KeyError("couldn't find pixel resolution in header")
    # else: #hardcoded can't find solution here...
    #     print("else")
    #     as_to_px={
    #     "HST":0.05, #as
    #     "HSC":0.168,
    #     "CFHT":0.187,
    #     # "":,
    #     # "":,
    #     # "":,
    #     # "":,
    #     # "":,
    #     # "":,
    #     # "":,
    #     # "":,
    #     }[name]

    # print(f"found {as_to_px} arcsec per pixel")

    # return as_to_px

    return proj_plane_pixel_scales(wcs.WCS(hdr))[0] * 3600.0


def get_res_from_img(fname_psf):

    lookup_dict = {
        "JWST-PSFEx_out_f115w_B3_psf_v5.0_s1-30mas.psf": "background_image_f115w.fits",
        "JWST-PSFEx_out_f150w_B3_psf_v5.0_s1-30mas.psf": "background_image_f150w.fits",
        "JWST-PSFEx_out_f277w_B3_psf_v5.0_s1-30mas.psf": "background_image_f277w.fits",
        "JWST-PSFEx_out_f444w_B3_psf_v5.0_s1-30mas.psf": "background_image_f444w.fits",
        "JWST-PSFEx_out_f814w_B3_psf_v5.0_s1.psf": "background_image_HST-F814W.fits",
        # "JWST-PSFEx_out_f770w_B3_psf_v5.0_s1.psf" : "background_image_",
        "OPT-PSFEx_out_CFHT-u_B3_psf_v4.0.psf": "background_image_CFHT-u.fits",
        "OPT-PSFEx_out_HSC-g_B3_psf_v4.0.psf": "background_image_HSC-g.fits",
        "OPT-PSFEx_out_HSC-r_B3_psf_v4.0.psf": "background_image_HSC-r.fits",
        "OPT-PSFEx_out_HSC-i_B3_psf_v4.0.psf": "background_image_HSC-i.fits",
        "OPT-PSFEx_out_HSC-y_B3_psf_v4.0.psf": "background_image_HSC-y.fits",
        "OPT-PSFEx_out_HSC-z_B3_psf_v4.0.psf": "background_image_HSC-z.fits",
        "OPT-PSFEx_out_SC-IB427_B3_psf_v4.0.psf": "background_image_SPC-L427_B3.fits",
        "OPT-PSFEx_out_SC-IA484_B3_psf_v4.0.psf": "background_image_SPC-L484_B3.fits",
        "OPT-PSFEx_out_SC-IB505_B3_psf_v4.0.psf": "background_image_SPC-L505_B3.fits",
        "OPT-PSFEx_out_SC-IA527_B3_psf_v4.0.psf": "background_image_SPC-L527_B3.fits",
        "OPT-PSFEx_out_SC-IB574_B3_psf_v4.0.psf": "background_image_SPC-L574_B3.fits",
        "OPT-PSFEx_out_SC-IA624_B3_psf_v4.0.psf": "background_image_SPC-L624_B3.fits",
        "OPT-PSFEx_out_SC-IA679_B3_psf_v4.0.psf": "background_image_SPC-L679_B3.fits",
        "OPT-PSFEx_out_SC-IB709_B3_psf_v4.0.psf": "background_image_SPC-L709_B3.fits",
        "OPT-PSFEx_out_SC-NB711_B3_psf_v4.0.psf": "background_image_SPC-L711_B3.fits",
        "OPT-PSFEx_out_SC-IA738_B3_psf_v4.0.psf": "background_image_SPC-L738_B3.fits",
        "OPT-PSFEx_out_SC-IA767_B3_psf_v4.0.psf": "background_image_SPC-L767_B3.fits",
        "OPT-PSFEx_out_SC-NB816_B3_psf_v4.0.psf": "background_image_SPC-L816_B3.fits",
        "OPT-PSFEx_out_SC-IB827_B3_psf_v4.0.psf": "background_image_SPC-L827_B3.fits",
        "OPT-PSFEx_out_UVISTA-H_B3_psf_v4.0.psf": "background_image_UVISTA-H_B3.fits",
        "OPT-PSFEx_out_UVISTA-J_B3_psf_v4.0.psf": "background_image_UVISTA-J_B3.fits",
        "OPT-PSFEx_out_UVISTA-Ks_B3_psf_v4.0.psf": "background_image_UVISTA-Ks_B3.fits",
        "OPT-PSFEx_out_UVISTA-NB118_B3_psf_v4.0.psf": "background_image_UVISTA-NB118_B3.fits",
        "OPT-PSFEx_out_UVISTA-Y_B3_psf_v4.0.psf": "background_image_UVISTA-Y_B3.fits",
    }

    res_root = "/automnt/data101/jlewis/marko_cosmosWeb_fields/residual_images/"

    res_fname = os.path.join(res_root, lookup_dict[fname_psf.strip(" ")])

    # print(fname_psf, res_fname)

    hdr = read_fits_hdr(res_fname)

    # print(hdr)
    # instr_name = lookup_dict[fname_psf.strip(" ")].split("_")[-1].split("-")[0].split(".")[0]

    return get_hdr_res(hdr)


def fit_galaxies_iter(
    rgals_px,
    flux_map,
    r_cat_px,
    cat_pos_px,
    grid_corner_px,
    stmask_fname,
    niter_max=10000,
    debug=False,
):

    found_pos = np.zeros((len(rgals_px), 2))

    # ncell_ra,ncell_dec = data_shape
    ncell_ra, ncell_dec = flux_map.shape

    ra_min, dec_min, ra_max, dec_max = grid_corner_px

    l_ra = ra_max - ra_min
    l_dec = dec_max - dec_min

    dx_px = l_ra / ncell_ra

    st_mask_hdr, st_mask = read_fits(stmask_fname)

    st_mask = st_mask.T

    # print("shapes")
    # print(data_shape)
    # print(st_mask.shape)

    X_mask_coords, Y_mask_coords = np.mgrid[0 : st_mask.shape[0], 0 : st_mask.shape[1]]

    ra_cells = np.arange(ra_min, l_ra + dx_px, dx_px)
    dec_cells = np.arange(dec_min, l_dec + dx_px, dx_px)

    ra_cells_big = np.linspace(0, st_mask.shape[0], len(ra_cells))
    dec_cells_big = np.linspace(0, st_mask.shape[1], len(dec_cells))

    # print(len(ra_cells),len(ra_cells_big))

    ra_compress = len(ra_cells_big) / float(len(ra_cells))
    dec_compress = len(dec_cells_big) / float(len(dec_cells))

    mask_rebinned = binned_statistic_2d(
        np.ravel(X_mask_coords),
        np.ravel(Y_mask_coords),
        np.ravel(st_mask),
        statistic="sum",
        bins=[ra_cells_big, dec_cells_big],
    )[0]

    # print(st_mask_hdr)

    mask = np.int8(np.zeros((ncell_ra, ncell_dec))) + mask_rebinned
    mask[mask > 1] = 1

    # handle empty noise regions for tile B3
    mask[:, -800:] = 50  # top
    mask[4500:5500, 7750:14000] = 50  # in field middle
    mask[8750:10000, 1000:5500] = 50  # in field middle
    mask[9000:9750, -2300:] = 50  # in field middle
    mask[13750:14000, 10500:13000] = 50  # in field middle

    edge_lim = 50
    mask[0:edge_lim, :] = 50
    mask[-edge_lim:, :] = 50
    mask[:, 0:edge_lim] = 50
    mask[:, -edge_lim:] = 50

    # mask = 1 -> contains unusable pixels
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].imshow(st_mask.T, origin="lower")
    ax[0, 1].imshow(mask_rebinned.T, origin="lower")
    ax[1, 0].imshow(mask.T, origin="lower")

    fig.savefig("st_mask")

    # ra_min,ra_max = ra_cells[[0,-1]]
    # dec_min,dec_max = dec_cells[[0,-1]]

    # RA,DEC = np.meshgrid(ra_cells,dec_cells)

    print("filling mask")

    for ra_cat, dec_cat in cat_pos_px:

        # wh = np.linalg.norm([(RA-ra_cat),(DEC-dec_cat)])<r_cat_px

        # mask[wh] = 1

        ira = int(ra_cat / dx_px)
        idec = int(dec_cat / dx_px)

        # print(ra_cat,dec_cat)

        # print(ira,idec)

        # print(ra_cells[ira],dec_cells[idec])

        mask[
            ira - r_cat_px : ira + r_cat_px + 1, idec - r_cat_px : idec + r_cat_px + 1
        ] = (
            mask[
                ira - r_cat_px : ira + r_cat_px + 1,
                idec - r_cat_px : idec + r_cat_px + 1,
            ]
            + 1
        )

    done_gals = 0
    niter = 0

    print("finding spaces")

    niter_pack = int(niter_max / len(rgals_px))
    cur_gal = -1
    cur_niter = 0

    # convolve_ok_map = fftconvolve(mask, np.ones((2*np.max(rgals_px)+1, 2*np.max(rgals_px)+1)), mode='same')

    while niter < niter_max and done_gals < len(rgals_px):

        if cur_niter == 0 or cur_niter == niter_pack:

            if debug:
                print("new pack")

            if cur_gal != done_gals:

                if debug:
                    print("new galaxy")

                cur_r = rgals_px[done_gals]
                cur_gal = done_gals

                kern=np.ones((int(2 * cur_r + 1), int(2 * cur_r + 1)))
                X,Y = np.mgrid[:kern.shape[0],:kern.shape[1]]
                X=X-int(0.5*kern.shape[0])
                Y=Y-int(0.5*kern.shape[1])
                R = np.linalg.norm([X,Y],axis=0)
                kern[R>cur_r]=0

                print(R)

                convolve_ok_map = fftconvolve(
                    mask, kern, mode="same"
                )
                ok_coords = np.where(convolve_ok_map == 0)  # no neighbours >0 in cur_r

                # ok_coords = np.where((convolve_ok_map == 0) * (mask == 0)) #no neighbours >0 in cur_r and central 0
                # ok_coords = np.where(convolve_ok_map == 0) #no neighbours >0 in cur_r

                if debug:
                    print((convolve_ok_map == 0).sum(), (mask == 0).sum())

                    fig, ax = plt.subplots(1, 3)
                    ax[0].imshow(mask.T, origin="lower", vmin=0, vmax=1)
                    ax[1].imshow(convolve_ok_map.T, origin="lower", vmin=0, vmax=1)
                    ax[1].imshow(
                        np.transpose((convolve_ok_map == 0) * (mask == 0)),
                        origin="lower",
                        vmin=0,
                        vmax=1,
                    )
                    # ax[1].scatter(ok_coords[0], ok_coords[1], s=1, color='r')
                    fig.savefig(f"debug_mask_r{cur_r}_gal{done_gals}")
                    plt.close(fig)

            # draw random pos uniform - vectorize later
            # ra_try = np.int32(np.random.uniform(low=cur_r*1.05,high=l_ra-cur_r*1.05,size=niter_pack)/dx_px)
            # dec_try = np.int32(np.random.uniform(low=cur_r*1.05,high=l_dec-cur_r*1.05,size=niter_pack)/dx_px)

            rand_idx = np.random.randint(low=0, high=len(ok_coords[0]), size=niter_pack)
            ra_try = ok_coords[0][rand_idx]
            dec_try = ok_coords[1][rand_idx]

            cur_niter = 0

        # print(ra_try,dec_try)

        # wh_try = np.linalg.norm([(RA-ra_try),(DEC-dec_try)])<cur_r

        # sum_flags = np.sum(mask[wh_try])

        ira = ra_try[cur_niter]
        idec = dec_try[cur_niter]

        # print(ira,idec,cur_r)

        # print(ra_try,dec_try)

        # print(ira,idec)

        if mask[ira, idec] == 0:

            # print(ra_cells[ira],dec_cells[idec])

            nb_zero_fl = np.sum(
                flux_map[ira - cur_r : ira + cur_r + 1, idec - cur_r : idec + cur_r + 1]
                == 0
            )
            sum_flags = np.sum(
                mask[ira - cur_r : ira + cur_r + 1, idec - cur_r : idec + cur_r + 1] > 0
            )

            # print(nb_zero_fl, cur_r, cur_r**2)

            if sum_flags == 0 and nb_zero_fl < (cur_r**2):

                print(f"{niter+1} draws, {done_gals+1}/{len(rgals_px)} galaxies placed")
                found_pos[done_gals, :] = ira, idec

                mask[ira - cur_r : ira + cur_r + 1, idec - cur_r : idec + cur_r + 1] = (
                    mask[ira - cur_r : ira + cur_r + 1, idec - cur_r : idec + cur_r + 1]
                    + 1
                )
                convolve_ok_map[
                    ira - cur_r : ira + cur_r + 1, idec - cur_r : idec + cur_r + 1
                ] = (
                    convolve_ok_map[
                        ira - cur_r : ira + cur_r + 1, idec - cur_r : idec + cur_r + 1
                    ]
                    + 1
                )

                # ok_coords = np.where(mask == 0)

                done_gals += 1
                cur_niter = (
                    -1
                )  # -1 because +1 below, so that we go to next block of randoms

            else:

                pass
                # print(sum_flags, nb_zero_fl, cur_r, cur_r**2)

        else:
            print("problem: I drew a central pixel that should be masked")
            if debug:
                print(mask[ira, idec], sum_flags)

        niter += 1
        cur_niter += 1

    return mask, found_pos, done_gals


def fit_galaxies(
    rgals_px,
    flux_map,
    r_cat_px,
    cat_pos_px,
    grid_corner_px,
    stmask_fname,
    debug=False,
    miri_mask=None,
    **kwargs
):

    n2fill = len(rgals_px)

    found_pos = np.zeros((len(rgals_px), 2))
    # found_pos = []
    # masks = []

    # ncell_ra,ncell_dec = data_shape
    ncell_ra, ncell_dec = flux_map.shape

    ra_min, dec_min, ra_max, dec_max = grid_corner_px

    l_ra = ra_max - ra_min
    l_dec = dec_max - dec_min

    dx_px = l_ra / ncell_ra

    mask_rebinned_stars = interp_mask(stmask_fname, ra_min, dec_min, l_ra, l_dec, dx_px).astype(np.int8)


    # print(st_mask_hdr)

    mask = np.int8(np.zeros((ncell_ra, ncell_dec))) + mask_rebinned_stars
    if miri_mask is not None:

        mask_rebinned_miri = interp_mask(miri_mask, ra_min, dec_min, l_ra, l_dec, dx_px).astype(np.int8)
        mask = mask + (1-mask_rebinned_miri) #in miri mask, 1s are good and 0s are bad... flip it!
    mask[mask > 1] = 1

    # print((mask>=1).sum(),np.prod(mask.shape))

    # handle empty noise regions for tile B3
    mask[:, -800:] = 50  # top
    mask[4500:5500, 7750:14000] = 50  # in field middle
    mask[8750:10000, 1000:5500] = 50  # in field middle
    mask[9000:9750, -2300:] = 50  # in field middle
    mask[13750:14000, 10500:13000] = 50  # in field middle

    edge_lim=25 #safety... should not really be necessary
    #but can overstep because of PSF

    # mask edges
    mask[0:edge_lim, :] = 50
    mask[-edge_lim:, :] = 50
    mask[:, 0:edge_lim] = 50
    mask[:, -edge_lim:] = 50

    assert (mask<1).sum()<np.prod(mask.shape),"Something went wrong, all pixels are masked"

    # mask = 1 -> contains unusable pixels
    # fig, ax = plt.subplots(2, 2)

    # ax[0, 0].imshow(st_mask.T, origin="lower")
    # ax[0, 1].imshow(mask_rebinned.T, origin="lower")
    # ax[1, 0].imshow(mask.T, origin="lower")

    # fig.savefig("st_mask")

    # ra_min,ra_max = ra_cells[[0,-1]]
    # dec_min,dec_max = dec_cells[[0,-1]]

    # RA,DEC = np.meshgrid(ra_cells,dec_cells)

    print("filling mask")

    for (ra_cat, dec_cat), r_px in zip(cat_pos_px, r_cat_px):

        # wh = np.linalg.norm([(RA-ra_cat),(DEC-dec_cat)])<r_cat_px

        # mask[wh] = 1
        
        if (ra_cat>=ra_min)*(ra_cat<ra_max) and (dec_cat>=dec_min)*(dec_cat<dec_max):

            ira = int(ra_cat / dx_px)
            idec = int(dec_cat / dx_px)

            # print(ra_cat, dec_cat)

            # print(ira, idec, r_cat_px)

            # print(ra_cells[ira],dec_cells[idec])

            mask[
                int(ira - r_px) : int(ira + r_px) + 1,
                int(idec - r_px) : int(idec + r_px) + 1,
            ] = (
                mask[
                    int(ira - r_px) : int(ira + r_px) + 1,
                    int(idec - r_px) : int(idec + r_px) + 1,
                ]
                + 1
            )

    print("finding spaces")

    # rmax = np.max(rgals_px)

    # fig, ax = plt.subplots(1, 2)

    # ax[0].imshow(mask.T, origin="lower", vmin=0, vmax=1)
    # ax[1].imshow(convolve_ok_map.T, origin="lower", vmin=0, vmax=1)

    # fig.savefig("test_maps")

    # print(rmax)

    # print(ok_coords)

    # compromise between optimum fitting and convolving with exact sizes, and fastest which is one
    # convolution with the largest gal size

    # tgt_nbin_min = min(5, len(rgals_px))
    nbins = max(int((rgals_px.max() - rgals_px.min()) / 5.0),2)

    # done_bins = False
    # while not done_bins:

    #     min_nper_bin = np.min(np.histogram(rgals_px, bins=nbins)[0])
    #     done_bins = min_nper_bin > tgt_nbin_min

    #     if not done_bins:
    #         nbins -= 1

    bins = np.linspace(rgals_px.min() - 1, rgals_px.max() + 1, nbins)
    bins=bins[::-1] #largest to smallest
    # bin_maxs = bins[1:] + np.diff(bins)

    # bins = bins
    # bin_maxs = bin_maxs

    print(
        f"I created {len(bins)} bins between {rgals_px.min():.2f} and {rgals_px.max():.2f}"
    )

    done_args = np.full(len(rgals_px), False)
    # print(len(done_args))

    for ibin in range(len(bins[:-1])):

        # rmin, rmax = bins[ibin], bins[ibin + 1]
        rmin, rmax = bins[ibin+1], bins[ibin]

        print(f"placing galaxies of sizes between {rmin:.1f} and {rmax:.1f} px")

        gals_in_bin = np.where((rgals_px >= rmin) * (rgals_px < rmax))[0]

        if len(gals_in_bin) == 0:
            print("no gals in bin, skipping")
            continue
        else:
            print(f"Found {len(gals_in_bin):d} galaxies in this size bin")



        print("performing convolution to find spaces")

        # convolve_r = bins[ibin + 1]
        convolve_r = bins[ibin]

        #mask edges (only for convolution) to make sure we get no <0 coordinates
        mask_edge_dist = int(np.ceil(convolve_r))
        conv_mask = np.copy(mask)
        conv_mask[:,:mask_edge_dist]=1
        conv_mask[:,-mask_edge_dist:]=1
        conv_mask[:mask_edge_dist,:]=1
        conv_mask[-mask_edge_dist:,:]=1

        kern=np.ones((int(2 * convolve_r + 1), int(2 * convolve_r + 1)))
        X,Y = np.mgrid[:kern.shape[0],:kern.shape[1]]
        X=X-int(0.5*kern.shape[0])
        Y=Y-int(0.5*kern.shape[1])
        R = np.linalg.norm([X,Y],axis=0)
        kern[R>convolve_r]=0

        convolve_ok_map = fftconvolve(
            conv_mask,
            kern,
            mode="same",
        )

        ok_coords_flat = np.flatnonzero(convolve_ok_map == 0)
        ok_coords = np.unravel_index(ok_coords_flat, convolve_ok_map.shape)

        # print(ok_coords)
        # print(ok_coords[0], len(ok_coords[0]))
        if len(ok_coords[0]) == 0:

            # print("No available spaces for this bin size... stopping")
            print("No available spaces for this bin size... trying smaller sizes")

            continue

        print("filling mask")

        for igal, gal in enumerate(gals_in_bin):

            true_r = rgals_px[gal]

            #ordered filling
            # idx = 0  # should be sorted, start with closest to lower left corner
            #random filling
            if len(ok_coords[0])>1:
                idx = np.random.randint(low=0,high=len(ok_coords[0]))
            else:
                idx=0
            try_ra = ok_coords[0][idx]
            try_dec = ok_coords[1][idx]

            ira = np.int32(try_ra / dx_px)
            idec = np.int32(try_dec / dx_px)

            update_mask = np.ones((2*true_r+1,2*true_r+1))
            X,Y = np.mgrid[:2*true_r+1,:2*true_r+1]
            X=X-true_r
            Y=Y-true_r
            R = np.linalg.norm([X,Y],axis=0)
            update_mask[R>convolve_r]=0

            mask[
                int(ira - true_r) : int(ira + true_r + 1),
                int(idec - true_r) : int(idec + true_r + 1),
            ] = (
                mask[
                    int(ira - true_r) : int(ira + true_r + 1),
                    int(idec - true_r) : int(idec + true_r + 1),
                ]
                + update_mask # here we use the object's true size
            )

            #could also update here to make not use squares...
            convolve_ok_map[  # update radius in 2d, need 2* margin to make sure I don't overlap in same bin
                int(ira - (convolve_r + true_r)) : int(ira + (convolve_r + true_r) + 1),
                int(idec - (convolve_r + true_r)) : int(
                    idec + (convolve_r + true_r) + 1
                ),
            ] = (
                convolve_ok_map[
                    int(ira - (convolve_r + true_r)) : int(
                        ira + (convolve_r + true_r) + 1
                    ),
                    int(idec - (convolve_r + true_r)) : int(
                        idec + (convolve_r + true_r) + 1
                    ),
                ]
                + 1
            )

            found_pos[gal, 0] = ira
            found_pos[gal, 1] = idec
            done_args[gal] = True

            # ok_coords = np.where(
            #     convolve_ok_map == 0
            # )  # this is long, can replace with indices that update ok_coords directly

            xlims = [
                int(ira - (true_r + convolve_r)),
                int(ira + (true_r + convolve_r) + 1),
            ]
            ylims = [
                int(idec - (true_r + convolve_r)),
                int(idec + (true_r + convolve_r) + 1),
            ]
            X, Y = np.meshgrid(np.arange(*xlims), np.arange(*ylims))

            # print(xlims,ylims)
            args_update = np.ravel(np.ravel_multi_index((X, Y), convolve_ok_map.shape))
            ok_coords_flat = ok_coords_flat[
                np.in1d(ok_coords_flat, args_update) == False
            ]
            # this is 20-40% faster than the np.where

            if debug:
                print("masks updated")

            if len(ok_coords_flat) == 0 and igal < len(gals_in_bin):
                gals_in_bin = gals_in_bin[: (igal + 1)]  # next one won't fit!
                break

            ok_coords = np.unravel_index(ok_coords_flat, convolve_ok_map.shape)

            # print(igal, gal, len(gals_in_bin))

        # print(found_pos[gals_in_bin, :].shape)

        n2fill -= len(gals_in_bin)

    if debug:
        print("found ", found_pos)

    return mask, found_pos, done_args

def interp_mask(mask_fname, ra_min, dec_min, l_ra, l_dec, dx_px):
    mask_hdr, mask = read_fits(mask_fname)

    mask = mask.T #this is true for the stellar mask... also for the miri one ?

    # print("shapes")
    # print(data_shape)
    # print(mask.shape)

    zoom_fact = l_ra/mask.shape[0],l_dec/mask.shape[1]

    print(zoom_fact)

    if not np.all(np.array(zoom_fact)==1.0):

        interpd_mask = zoom(mask, zoom_fact, order=0)

        # x,y = np.float64(np.mgrid[0:mask.shape[0],0:mask.shape[1]])
        # x/=mask.shape[0]
        # y/=mask.shape[1]

        # interp = NearestNDInterpolator((np.ravel(x),np.ravel(y)),np.ravel(mask))

        # xtgt,ytgt = np.float64(np.mgrid[0:l_ra,0:l_dec])
        # xtgt/=l_ra
        # ytgt/=l_dec

        # interpd_mask = interp(xtgt,ytgt)

        # print(mask.shape,interpd_mask.shape)

    else:

        return mask

    # print((mask>0).sum(),np.prod(mask.shape))

    return interpd_mask


def fake_gal_cat(ngal, box_corners_px):

    ra_min, dec_min, ra_max, dec_max = box_corners_px

    ras = np.random.uniform(low=ra_min, high=ra_max, size=ngal)
    decs = np.random.uniform(low=dec_min, high=dec_max, size=ngal)

    return (ras, decs, np.full_like(ras, 5))


def load_psf(fname, root_psf="/automnt/data101/jlewis/marko_psfs"):

    # res_as = fname.split('-')[2].split('.')[0]
    # res_as = float(res_as[:-3]) #rmv "mas"

    with fits.open(os.path.join(root_psf, fname.lstrip(" "))) as hdul:
        #
        # print(hdul[1].header)
        # res_as = hdul[1].header['CRDELT1']*3600.
        # res_as = get_hdr_res(hdul[1].header)
        # res_as = 30.0 / 1e3
        res_as = get_res_from_img(fname)
        # print(res_as)
        # return np.asarray(list(hdul[1].data["PSF_MASK"][0][0]))
        return (np.asarray(list(hdul[1].data["PSF_MASK"][0][0])), res_as)


# def rebin( a, newshape ):
#         '''Rebin an array to a new shape.
#         '''
#         assert len(a.shape) == len(newshape)


#         slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
#         coordinates = np.mgrid[slices]
#         indices = coordinates.astype('i')   #choose the biggest smaller integer index
#         return a[tuple(indices)]


def dilate(a, factor):
    """Rebin an array to a new shape using interpolation."""
    # print(a.shape,factor)
    # cur_x,cur_y = np.mgrid[0:a.shape[0],0:a.shape[1]]
    x, y = np.linspace(0, 1, a.shape[0]), np.linspace(0, 1, a.shape[1])
    # x = x - 0.5 * a.shape[0]
    # y = y - 0.5 * a.shape[1]

    interp2D = RegularGridInterpolator((x, y), a, method="linear")

    tgt_size = np.int32(np.round(factor * np.asarray(list(a.shape))))
    # print(a.shape,factor,tgt_size)
    # print(tgt_size/factor)
    # print(tgt_size/np.ceil(factor))
    # print(a)
    # print(tgt_size, factor)

    xtgt_bins = np.linspace(0, 1, tgt_size[0])  # - 0.5 * a.shape[0]
    ytgt_bins = np.linspace(0, 1, tgt_size[1])  # - 0.5 * a.shape[1]

    xtgt, ytgt = np.meshgrid(xtgt_bins, ytgt_bins)

    # xtgt, ytgt = np.mgrid[0 : tgt_size[0], 0 : tgt_size[1]]

    # xtgt = np.int64((xtgt - 0.5 * tgt_size[0]) / np.ceil(factor))
    # ytgt = np.int64((ytgt - 0.5 * tgt_size[1]) / np.ceil(factor))

    # print(xtgt.min(), xtgt.max())
    # print(ytgt.min(), ytgt.max())

    # if tgt_size[0] / np.ceil(factor) > a.shape[0]:
    #     xtgt = xtgt * a.shape[0] / (tgt_size[0] / np.ceil(factor))
    # if tgt_size[1] / np.ceil(factor) > a.shape[1]:
    #     ytgt = ytgt * a.shape[1] / (tgt_size[1] / np.ceil(factor))

    # print(a.shape)
    # print(xtgt.min(), xtgt.max())
    # print(ytgt.min(), ytgt.max())
    # print(x.min(), x.max())
    # print(y.min(), y.max())

    # tgt_size[0] = int(min(tgt_size[0] / np.ceil(factor), a.shape[0])*np.ceil(factor))
    # tgt_size[1] = int(min(tgt_size[1] / np.ceil(factor), a.shape[1])*np.ceil(factor))

    # print(tgt_size)

    # xtgt=np.int32((xtgt-0.5*tgt_size[0])/np.ceil(factor) + tgt_size[0])
    # ytgt=np.int32((ytgt-0.5*tgt_size[1])/np.ceil(factor) + tgt_size[1])
    # print(tgt_size)
    # print(xtgt,ytgt)

    # print(xtgt,ytgt)

    return interp2D(np.transpose([xtgt, ytgt]))


def pick_tile():

    return "B3"

    # itile = np.random.randint(low=0,high=19,size=1)[0]
    # tile=["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10",
    #         "B1","B2","B3","B4","B5","B6","B7","B8","B9","B10"][itile]
    # return(tile)


def tile_to_file(filt, tile):

    res = f"/data101/jlewis/marko_cosmosWeb_fields/residual_images/background_image_{filt:s}.fits"
    # res = f"/data101/jlewis/marko_cosmosWeb_fields/residual_images/CheckImg_COSMOSWeb__resid_mosaic_nircam_{filt:s}_COSMOS-Web_60mas_{tile:s}_v0_8_sci_1.fits"
    stmask = f"/automnt/data101/jlewis/marko_cosmosWeb_fields/starmask_{tile:s}_tweaked_painted_Jun27.fits"

    return (res, stmask)


def gals_to_fit_pos(rgals_as, cat_aper_norm=1.5, f277_lim=28, debug=False, **kwargs):

    tile = pick_tile()

    fname_data, fname_stmask = tile_to_file("f277w", tile)

    hdr, data = read_fits(fname_data)

    WCS = wcs.WCS(hdr)

    box_corners_px = [0, 0, hdr["NAXIS1"], hdr["NAXIS2"]]

    tile_shape_px = data.shape

    as_to_px = get_hdr_res(hdr)

    rgals_px = np.int32(abs(rgals_as / as_to_px))

    # reg_cat_fname = (
    #     "/automnt/data101/jlewis/marko_cosmosWeb_fields/olivier_reg_cat/joe_detect.reg"
    # )
    # cat_gals_xs, cat_gal_ys = load_reg(reg_cat_fname, WCS)

    # load_cat

    cweb_cat = "/data101/jlewis/COSMOSWeb_mastercatalog_v1.1.fits"
    with fits.open(cweb_cat) as hdul:
        # cweb_cat = hdul[2].data
        cweb_cat_se = hdul[1].data

    cweb_model_mag277 = cweb_cat_se[f"mag_model_f277w"][:]
    ra = cweb_cat_se["ra_model"]
    dec = cweb_cat_se["dec_model"]
    rsersic_deg = cweb_cat_se["radius_sersic"]
    bright = (cweb_model_mag277 < f277_lim)
    #make sure to include in tile and overlap with neighbours
    bright *= np.any([(cweb_cat_se["tile"] == "B3"), (cweb_cat_se["tile"] == "B4"),(cweb_cat_se["tile"] == "B8")])  
    # bright += (cweb_cat_se["tile"] == "B5") + (cweb_cat_se["tile"] == "B9")  

    cat_gals_r_pxs = (
        np.max([rsersic_deg[bright] * 3600 * cat_aper_norm, np.full(bright.sum(), 1.0)], axis=0)
        / as_to_px
    )  # either 1.5*fitted sersic or
    degs = SkyCoord(ra[bright] * u.deg, dec[bright] * u.deg)

    # print(cat_gals_xs)
    # cat_gals_r_pxs = np.full_like(cat_gals_xs[:], 2.0)

    # print(list(zip(ra[bright], dec[bright], rsersic_deg[bright])))

    cat_gal_xs, cat_gal_ys = WCS.world_to_pixel(degs)
    cat_gal_pxs = np.transpose([cat_gal_xs, cat_gal_ys])

    # print(list(zip(cat_gal_pxs[:50], rgals_px[:50])))

    mask, found_pos, done_bools = fit_galaxies(
        rgals_px,
        data.T,
        cat_gals_r_pxs,
        cat_gal_pxs,
        # np.transpose([cat_gals_xs, cat_gal_ys]),
        box_corners_px,
        fname_stmask,
        debug=debug,
        **kwargs
    )

    found_all = np.all(done_bools == True)

    return tile, tile_shape_px, found_pos, rgals_px, found_all, done_bools, WCS


if __name__ == "__main__":

    hdr, data = read_fits(
        "/data101/jlewis/marko_cosmosWeb_fields/residual_images/background_image_f277w.fits"
    )

    fig, ax = plt.subplots()

    img = ax.imshow(data, vmin=-0.16, vmax=0.16, cmap="bwr", origin="lower")
    # img=ax.imshow(np.log10(np.abs(data)),cmap='bwr',origin='lower')

    fig.savefig("residual")

    stmask_fname = "/automnt/data101/jlewis/marko_cosmosWeb_fields/starmask_A1_tweaked_painted_Jun27.fits"

    reg_cat_fname = (
        "/automnt/data101/jlewis/marko_cosmosWeb_fields/olivier_reg_cat/joe_detect.reg"
    )

    print(hdr)

    WCS = wcs.WCS(hdr)

    box_corners_px = [0, 0, hdr["NAXIS1"], hdr["NAXIS2"]]

    # mock_gals = fake_gal_cat(1000,box_corners_px)[:2]
    # mock_gals=(np.linspace(0,hdr['NAXIS1'],100),np.linspace(0,hdr['NAXIS2'],100),np.full(100,30))

    mock_gals = load_reg(reg_cat_fname, WCS)
    xcond = (mock_gals[0] > box_corners_px[0]) * (mock_gals[0] <= box_corners_px[2])
    ycond = (mock_gals[1] > box_corners_px[1]) * (mock_gals[1] <= box_corners_px[3])
    mock_gals = [mock_gals[0][xcond * ycond], mock_gals[1][xcond * ycond]]

    print(mock_gals)

    # print("mock_gals",mock_gals)
    # rgals_as = np.asarray([10,6,8,3,4,6,4,7])

    rgals_as = np.random.normal(loc=6, scale=2, size=50)

    as_to_px = hdr["CDELT1"] * 3600

    rgals_px = np.int32(abs(rgals_as / as_to_px))

    print(rgals_px, as_to_px, rgals_as)

    mask, found_pos, nfound = fit_galaxies(
        rgals_px,
        data.T,
        int(2.0 / as_to_px),
        np.transpose(mock_gals),
        box_corners_px,
        stmask_fname,
        niter_max=1000 * len(rgals_px),
        debug=True,
    )

    dmin, dmax = data.min(), data.max()

    print(dmin, dmax, data.mean())

    fig, ax = plt.subplots()

    # ax.imshow(data,vmin=dmin,vmax=dmax)
    # img=ax.imshow(data,vmin=-0.16,vmax=0.16,cmap='bwr')
    img = ax.imshow(
        mask.T,
        origin="lower",
        extent=[0, hdr["NAXIS1"], 1, hdr["NAXIS2"]],
        vmax=1,
        vmin=0,
    )

    plt.colorbar(img, ax=ax)

    print(mock_gals)

    # print('plotting mask')

    # for x,y in np.transpose(mock_gals):

    #     # ax.scatter(*pos,s=rgal)

    #     circ = Circle((x,y),3/as_to_px,edgecolor='b',facecolor='none')

    #     ax.add_artist(circ)

    print("plotting found pos")

    for rgal, pos in zip(rgals_px, found_pos):

        # ax.scatter(*pos,s=rgal)

        circ = Circle(pos, rgal, edgecolor="r", facecolor="none")

        ax.add_artist(circ)

    fig.savefig("test_fits")
