# import astropy
# import astropy.cosmology
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from h11 import Data
from matplotlib.patches import Circle
import numpy as np
import matplotlib.pyplot as plt

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

    keys = list(hdr.keys())

    # print(hdr)
    # print(keys)

    if (
        "CDELT1" in keys and abs(hdr["CDELT1"]) != 1.0
    ):  # don't know why but sometimes empty
        as_to_px = abs(hdr["CDELT1"]) * 3600.0
    elif "PC1_1" in keys:
        as_to_px = abs(hdr["PC1_1"]) * 3600.0
    elif "CRDELT1" in keys:
        as_to_px = abs(hdr["CRDELT1"]) * 3600.0
    elif "CD1_1" in keys:
        as_to_px = abs(hdr["CD1_1"]) * 3600.0
    else:
        raise KeyError("couldn't find pixel resolution in header")

    # print(f"found {as_to_px} arcsec per pixel")

    return as_to_px


def get_res_from_img(fname_psf):

    lookup_dict = {
        "JWST-PSFEx_out_f115w_B3_psf_v5.0_s1-30mas.psf": "background_image_f115w.fits",
        "JWST-PSFEx_out_f150w_B3_psf_v5.0_s1-30mas.psf": "background_image_f150w.fits",
        "JWST-PSFEx_out_f277w_B3_psf_v5.0_s1-30mas.psf": "background_image_f277w.fits",
        "JWST-PSFEx_out_f444w_B3_psf_v5.0_s1-30mas.psf": "background_image_f444w.fits",
        "JWST-PSFEx_out_f814w_B3_psf_v5.0_s1.psf": "background_image_HST-F814W.fits",
        "OPT-PSFEx_out_CFHT-u_B3_psf_v4.0.psf": "background_image_CFHT-u.fits",
        "OPT-PSFEx_out_HSC-g_B3_psf_v4.0.psf": "background_image_HSC-g.fits",
        "OPT-PSFEx_out_HSC-r_B3_psf_v4.0.psf": "background_image_HSC-r.fits",
        "OPT-PSFEx_out_HSC-i_B3_psf_v4.0.psf": "background_image_HSC-i.fits",
        "OPT-PSFEx_out_HSC-y_B3_psf_v4.0.psf": "background_image_HSC-y.fits",
        "OPT-PSFEx_out_HSC-z_B3_psf_v4.0.psf": "background_image_HSC-z.fits",
    }

    res_root = "/automnt/data101/jlewis/marko_cosmosWeb_fields/residual_images/"

    res_fname = os.path.join(res_root, lookup_dict[fname_psf.strip(" ")])

    # print(fname_psf, res_fname)

    hdr = read_fits_hdr(res_fname)

    return get_hdr_res(hdr)


def fit_galaxies(
    rgals_px,
    flux_map,
    r_cat_px,
    cat_pos_px,
    grid_corner_px,
    stmask_fname,
    niter_max=10000,
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

    mask[0:200, :] = 50
    mask[-200:, :] = 50
    mask[:, 0:200] = 50
    mask[:, -200:] = 50

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
        ] = 1

    done_gals = 0
    niter = 0

    print("finding spaces")

    niter_pack = int(niter_max / len(rgals_px))
    cur_gal = -1
    cur_niter = 0

    ok_coords = np.where(mask == 0)

    while niter < niter_max and done_gals < len(rgals_px):

        print(f"draw {niter}, gal {done_gals}")

        if cur_niter == 0 or cur_niter == niter_pack:

            if cur_gal != done_gals:
                cur_r = rgals_px[done_gals]
                cur_gal = done_gals

            # draw random pos uniform - vectorize later
            # ra_try = np.int32(np.random.uniform(low=cur_r*1.05,high=l_ra-cur_r*1.05,size=niter_pack)/dx_px)
            # dec_try = np.int32(np.random.uniform(low=cur_r*1.05,high=l_dec-cur_r*1.05,size=niter_pack)/dx_px)

            ra_try = ok_coords[0][
                np.random.randint(low=0, high=len(ok_coords[0]), size=niter_pack)
            ]
            dec_try = ok_coords[1][
                np.random.randint(low=0, high=len(ok_coords[1]), size=niter_pack)
            ]

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
                mask[ira - cur_r : ira + cur_r + 1, idec - cur_r : idec + cur_r + 1]
            )

            # print(nb_zero_fl, cur_r, cur_r**2)

            if sum_flags == 0 and nb_zero_fl < (cur_r**2):

                print(done_gals)
                found_pos[done_gals, :] = ira, idec

                mask[ira - cur_r : ira + cur_r + 1, idec - cur_r : idec + cur_r + 1] = 1

                ok_coords = np.where(mask == 0)

                done_gals += 1

        niter += 1
        cur_niter += 1

    return mask, found_pos, done_gals


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


def gals_to_fit_pos(rgals_as, niter=100000):

    tile = pick_tile()

    fname_data, fname_stmask = tile_to_file("f277w", tile)

    hdr, data = read_fits(fname_data)

    # print(hdr)

    WCS = wcs.WCS(hdr)

    box_corners_px = [0, 0, hdr["NAXIS1"], hdr["NAXIS2"]]

    tile_shape_px = data.shape

    as_to_px = get_hdr_res(hdr)

    rgals_px = np.int16(abs(rgals_as / as_to_px))

    reg_cat_fname = (
        "/automnt/data101/jlewis/marko_cosmosWeb_fields/olivier_reg_cat/joe_detect.reg"
    )
    mock_gals = load_reg(reg_cat_fname, WCS)

    mask, found_pos, ndone_gals = fit_galaxies(
        rgals_px,
        data.T,
        int(3 / as_to_px),
        np.transpose(mock_gals),
        box_corners_px,
        fname_stmask,
        niter_max=niter,
    )

    found_all = ndone_gals == len(rgals_px)

    return tile, tile_shape_px, found_pos, rgals_px, found_all, WCS


if __name__ == "__main__":

    hdr, data = read_fits(
        "/data101/jlewis/marko_cosmosWeb_fields/residual_images/CheckImg_COSMOSWeb__resid_mosaic_nircam_f277w_COSMOS-Web_60mas_A1_v0_8_sci_1.fits"
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
        int(2 / as_to_px),
        np.transpose(mock_gals),
        box_corners_px,
        stmask_fname,
        niter_max=1000 * len(rgals_px),
    )

    dmin, dmax = data.min(), data.max()

    print(dmin, dmax, data.mean())

    fig, ax = plt.subplots()

    # ax.imshow(data,vmin=dmin,vmax=dmax)
    # img=ax.imshow(data,vmin=-0.16,vmax=0.16,cmap='bwr')
    img = ax.imshow(mask.T, origin="lower", extent=[0, hdr["NAXIS1"], 1, hdr["NAXIS2"]])

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
