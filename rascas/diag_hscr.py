import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec
from h5tools.h5py_tools import h5_to_dict
import os

from astropy import units as u
from astropy import wcs
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord

# from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales

from zoom_analysis.rascas.field_fits import tile_to_file


filt_wav_dict = {
    "NIRCAM-F115": 1.150,
    "NIRCAM-F150": 1.501,
    "NIRCAM-F277": 2.760,
    "NIRCAM-F444": 4.400,
    "HST-f814_res": 0.820,
    "MegaCam-u_new": 0.386,
    "HSC-g": 0.475,
    "HSC-r": 0.623,
    "HSC-i": 0.770,
    "HSC-z": 0.890,
    "HSC-y": 1.000,
}

vmin, vmax = -0.33, 0.33
# flatnb = 5
# flatnb = 0
zpt = 28.086519392
# ndir = 3
ndir = "3"

found = False

flatnb=0

cat = np.genfromtxt(
    # f"/data101/jlewis/zoom_flats/NH/{ndir}/SMC/flat_{flatnb}/out_cat_B3.txt",
    f"/data101/jlewis/zoom_flats/allsims/{ndir}/SMC/flat_{flatnb}/out_cat_B3.txt",
    names=True,
    dtype=None
)

# while not found and flatnb<=7:

    # cat = np.genfromtxt(
    #     # f"/data101/jlewis/zoom_flats/NH/{ndir}/SMC/flat_{flatnb}/out_cat_B3.txt",
    #     f"/data101/jlewis/zoom_flats/allsims/{ndir}/SMC/flat_{flatnb}/out_cat_B3.txt",
    #     names=True,
    #     dtype=None
    # )

    # NH
    # arg = 2894
    # arg=np.where(((np.abs(cat["z"]-2))<0.1) * (cat["mgalmsun"]>2e10))
    # arg = np.where((cat['snapshot']==403)*(cat['galid']==130))
    # arg = np.where((cat['snapshot']==419)*(cat['galid']==241))
    # arg = np.where((cat['snapshot']==318)*(cat['galid']==183))
    # arg = np.where((cat['snapshot']==309)*(cat['galid']==135))
    # arg = np.where((cat['snapshot']==199)*(cat['galid']==44))
    # arg = np.where((cat['snapshot']==286)*(cat['galid']==6))
    # arg = np.where((cat['snapshot']==391)*(cat['galid']==288))
    # arg = np.where((cat['snapshot']==162)*(cat['galid']==4))
    # arg = np.where((cat['snapshot']==163)*(cat['galid']==15))
    # arg = np.where((cat['snapshot']==307)*(cat['galid']==235))
    # arg = np.where((cat['snapshot']==416)*(cat['galid']==2))
    # arg = np.where((cat['snapshot']==388)*(cat['galid']==171))
    # arg = np.where((cat['snapshot']==434)*(cat['galid']==214))
    # arg = np.where((cat['snapshot']==391)*(cat['galid']==288))
    # arg = np.where((cat['snapshot']==276)*(cat['galid']==10))
    # allsims
    # print(cat['simulation'])
    # arg = np.where((cat['snapshot']==207)*(cat['galid']==4)*(cat['simulation'].astype(str)=='id52380'))
    # arg = np.where((cat['snapshot']==206)*(cat['galid']==2)*(cat['simulation'].astype(str)=='id52380'))
    # arg = np.where((cat['snapshot']==206)*(cat['galid']==2)*(cat['simulation'].astype(str)=='id52380'))
    # arg = np.where((cat['snapshot']== 193)*(cat['galid']==2)*(cat['simulation'].astype(str)=='id26646'))
    # arg = np.where((cat['snapshot']== 191)*(cat['galid']==44)*(cat['simulation'].astype(str)=='id74890'))
    # arg = np.where((cat['snapshot']== 143)*(cat['galid']==2)*(cat['simulation'].astype(str)=='id180130'))
    # arg = np.where((cat['snapshot']== 112)*(cat['galid']==7)*(cat['simulation'].astype(str)=='id242756'))
    # arg = np.where((cat['snapshot']==195)*(cat['galid']==4)*(cat['simulation'].astype(str)=='id52380'))
    # arg = np.where((cat['snapshot']==118)*(cat['galid']==11)*(cat['simulation'].astype(str)=='id180130'))
    # arg = np.where((cat['snapshot']==111)*(cat['galid']==2)*(cat['simulation'].astype(str)=='id242756'))
    # arg = np.where((cat['snapshot']==174)*(cat['galid']==1)*(cat['simulation'].astype(str)=='id52380'))



    # found = len(arg[0])==1

    # if not found:
    #     flatnb+=1
    # else:
    #     arg = arg[0][0]



# assert found, "couldn't find the desired sim/snap/galid combination"


for arg in range(len(cat['r50deg'])):

    ldata = type(cat['r50deg']) == np.ndarray
    if ldata:ldata=cat['r50deg'].shape!=()

    if not ldata:

        snap = cat["snapshot"]
        sim = cat["simulation"].astype(str)
        zed = cat["z"]
        gid = cat["galid"]
        # rgalas = cat["rgaldeg"] * 3600
        rgalas=2.1
        r50as = cat['r50deg'] * 3600
        rgalpx = cat["rgalpx"]
        print(f"{cat['mgalmsun']:.1e}")
        xpx, ypx = (
            cat["xpx"],
            cat["ypx"],
        )
        radeg, decdeg = (
            cat["radeg"],
            cat["decdeg"],
        )
    else:

        snap = cat["snapshot"][arg]
        sim = cat["simulation"].astype(str)[arg]
        zed = cat["z"][arg]
        gid = cat["galid"][arg]
        # rgalas = cat["rgaldeg"][arg] * 3600
        rgalas=2.1
        r50as = cat['r50deg'][arg] * 3600
        rgalpx = cat["rgalpx"][arg]
        print(f"{cat['mgalmsun'][arg]:.1e}")
        xpx, ypx = (
            cat["xpx"][arg],
            cat["ypx"][arg],
        )
        radeg, decdeg = (
            cat["radeg"][arg],
            cat["decdeg"][arg],
        )

    print(zed,snap,gid,r50as)

    tr = transforms.Affine2D().rotate_deg(-20)


    fig = plt.figure(figsize=(11 * 4, 4 * 6), layout="constrained")
    gs = gridspec.GridSpec(6, 11, figure=fig)

    ax = []

    # First 3 rows: normal 11 columns
    for r in range(5):
        row_axes = []
        for c in range(11):
            row_axes.append(fig.add_subplot(gs[r, c]))
        ax.append(row_axes)

    # Last row: single axis spanning all columns
    ax_last = fig.add_subplot(gs[5, :])
    ax.append([ax_last])  # keep structure similar to original ax

    noise_bins = np.logspace(-2, 2, 100)

    # fig,ax=plt.subplots(4,11,figsize=(11*4,4*4),sharex='row',sharey='row')

    # fig=plt.figure(figsize=(11*4,4*4))
    # spec=fig.add_gridspec(4,11)
    # ax=[
    #     fig.
    # ]



    spec_file = f"/data101/jlewis/mock_spe/zoom_tgt_bc03_chabrier100_wlya_SMC/{sim}/{int(snap):d}/gal_{int(gid):07d}/mock_spectrum_{3}.h5"
    spec = h5_to_dict(spec_file)

    ax[-1][-1].plot(
        spec["speectral_data"]["lam spectrum"] / 1e4, spec["speectral_data"]["spectrum"]
    )
    ax[-1][-1].plot(spec["speectral_data"]["lam"] / 1e4, spec["speectral_data"]["mag"])


    hscr_wav = 0.623

    img_file = f"/data101/jlewis/mock_spe/zoom_tgt_bc03_chabrier100_wlya_SMC/{sim}/{int(snap):d}/gal_{int(gid):07d}/MJy_images/mock_images_{sim}_{int(snap):d}_{int(gid):d}_{3}_wIGM.h5"
    img = h5_to_dict(img_file)
    wavs = []
    mags = []
    for iband, band in enumerate(
        [
            "MegaCam-u_new",
            "HSC-g",
            "HSC-i",
            "HSC-r",
            "HST-f814_res",
            "HSC-y",
            "HSC-z",
            "NIRCAM-F115",
            "NIRCAM-F150",
            "NIRCAM-F277",
            "NIRCAM-F444",
        ]
    ):

        band_names = img["header"]["band_names"]

        band_arg = np.where(band_names == "NIRCAM-F444")[0][0]
        # wav = filt_wav_dict[band]
        img_HSC_r_nopsf_f444 = img[band]

        band_names = img["header"]["band_names"]
        band_arg = np.where(band_names == band)[0][0]
        wav = filt_wav_dict[band]
        img_HSC_r_nopsf = img[band]
        img_HSC_resas_nopsf = img["header"]["pixel_res_as"]
        img_size = img_HSC_r_nopsf.shape
        rgalpx = rgalas / img_HSC_resas_nopsf



        print(img_size)

        x, y = np.mgrid[: img_size[0], : img_size[1]]
        ctr = np.int32(0.5 * np.array(img_size))
        rs = np.sqrt((x - ctr[0]) ** 2 + (y - ctr[1]) ** 2)
        select_px = rs <= rgalpx
        px_sr = img_HSC_resas_nopsf**2 * (np.pi / 3600.0 / 180.0) ** 2

        mag_img = -2.5 * np.log10(img_HSC_r_nopsf[select_px].sum() * 1e6 * px_sr) + 8.9

        ctr = np.int32(0.5 * np.array(img_HSC_r_nopsf.shape))
        extent = [
            -ctr[0] * img_HSC_resas_nopsf,
            ctr[0] * img_HSC_resas_nopsf,
            -ctr[1] * img_HSC_resas_nopsf,
            ctr[1] * img_HSC_resas_nopsf,
        ]

        transp = np.ones_like(img_HSC_r_nopsf_f444)*0.5
        transp[img_HSC_r_nopsf_f444<1E-2]=0


        if "NIRCAM" not in band and "HST" not in band:
            ax[0][iband].imshow(
                img_HSC_r_nopsf.T,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap="bwr",
                extent=extent,
                transform=tr + ax[0][iband].transData,
            )
            # ax[0][iband].imshow(img_HSC_r_nopsf_f444.T, origin='lower', vmin=vmin, vmax=vmax, cmap='Greens', extent=extent, alpha=transp.T, transform=tr + ax[0][iband].transData)
        else:
            ax[0][iband].imshow(
                img_HSC_r_nopsf.T,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap="bwr",
                extent=extent,
            )
            # ax[0][iband].imshow(img_HSC_r_nopsf_f444.T, origin='lower', vmin=vmin, vmax=vmax, cmap='Greens', extent=extent, alpha=transp.T)

        wavs.append(wav)
        mags.append(mag_img)

    ax[-1][-1].scatter(wavs, mags, c="tab:blue")


    img_file = f"/data101/jlewis/mock_spe/zoom_tgt_bc03_chabrier100_wlya_SMC/{sim}/{int(snap):d}/gal_{int(gid):07d}/MJy_images/mock_images_{sim}_{int(snap):d}_{int(gid):d}_{3}_wIGM_wPSF_rebinned.h5"
    img = h5_to_dict(img_file)


    wavs = []
    mags = []
    for iband, band in enumerate(
        [
            "MegaCam-u_new",
            "HSC-g",
            "HSC-i",
            "HSC-r",
            "HST-f814_res",
            "HSC-y",
            "HSC-z",
            "NIRCAM-F115",
            "NIRCAM-F150",
            "NIRCAM-F277",
            "NIRCAM-F444",
        ]
    ):

        band_names = img["header"]["band_names"]
        band_arg = np.where(band_names == band)[0][0]
        wav = filt_wav_dict[band]
        img_HSC_r_nopsf = img[band]
        img_HSC_resas_nopsf = img["header"]["pixel_res_as"][band_arg]
        img_size = img_HSC_r_nopsf.shape
        rgalpx = rgalas / img_HSC_resas_nopsf
        r50px = r50as / img_HSC_resas_nopsf

        print(img_size)

        # print(band,img_HSC_resas_nopsf)

        x, y = np.mgrid[: img_size[0], : img_size[1]]
        ctr = np.int32(0.5 * np.array(img_size))
        rs = np.sqrt((x - ctr[0]) ** 2 + (y - ctr[1]) ** 2)
        select_px = rs <= rgalpx
        # select_px = rs <= 2*r50px
        px_sr = img_HSC_resas_nopsf**2 * (np.pi / 3600.0 / 180.0) ** 2

        ctr = np.int32(0.5 * np.array(img_HSC_r_nopsf.shape))
        extent = [
            -ctr[0] * img_HSC_resas_nopsf,
            ctr[0] * img_HSC_resas_nopsf,
            -ctr[1] * img_HSC_resas_nopsf,
            ctr[1] * img_HSC_resas_nopsf,
        ]

        if "NIRCAM" not in band and "HST" not in band:
            ax[1][iband].imshow(
                img_HSC_r_nopsf.T,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap="bwr",
                extent=extent,
                transform=tr + ax[1][iband].transData,
            )
        else:
            ax[1][iband].imshow(
                img_HSC_r_nopsf.T,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap="bwr",
                extent=extent,
            )

        mag_img = -2.5 * np.log10(img_HSC_r_nopsf[select_px].sum() * 1e6 * px_sr) + 8.9

        wavs.append(wav)
        mags.append(mag_img)

    ax[-1][-1].scatter(wavs, mags, c="tab:orange")


    filt_dict = {
        "NIRCAM-F115": "f115w",
        "NIRCAM-F150": "f150w",
        "NIRCAM-F277": "f277w",
        "NIRCAM-F444": "f444w",
        "HST-f814_res": "HST-F814W",
        "HSC-r": "HSC-r",
        "HSC-g": "HSC-g",
        "HSC-i": "HSC-i",
        "HSC-y": "HSC-y",
        "HSC-z": "HSC-z",
        "MegaCam-u_new": "CFHT-u",
    }

    if sim=="NewHorizon":
        flat_sim="NH"
    else:
        flat_sim="allsims"

    flat_path = f"/data101/jlewis/zoom_flats/{flat_sim}/{ndir}/SMC/flat_{flatnb}/"

    wavs = []
    mags = []


    for iband, band in enumerate(
        [
            "MegaCam-u_new",
            "HSC-g",
            "HSC-i",
            "HSC-r",
            "HST-f814_res",
            "HSC-y",
            "HSC-z",
            "NIRCAM-F115",
            "NIRCAM-F150",
            "NIRCAM-F277",
            "NIRCAM-F444",
        ]
    ):

        band_flat = filt_dict[band]

        # fflat = os.path.join(flat_path,f"out_flat_{band_flat}_B3.fits")

        fflat, _ = tile_to_file(filt_dict[band], "B3")

        with fits.open(fflat) as src:
            hdul = src[0]
            flat_data, flat_hdr = hdul.data, hdul.header

        this_wcs = wcs.wcs.WCS(flat_hdr)

        px_res_as = proj_plane_pixel_scales(this_wcs) * 3600.0
        px_sr = px_res_as[0] ** 2 * (np.pi / 3600.0 / 180.0) ** 2

        # print(px_res_as)

        if "HSC" in band or "HST" in band:
            pos_deg = SkyCoord((radeg + 5*px_res_as[0]/3600.)*u.deg, decdeg * u.deg)
        else:
            pos_deg = SkyCoord((radeg)*u.deg, decdeg * u.deg)

        print(radeg, 1*px_res_as[0]/3600)

        rgalpx = rgalas / px_res_as[0]

        res_cutout = Cutout2D(
            flat_data,
            position=pos_deg,
            size=2 * rgalas * u.arcsec,
            wcs=this_wcs,
        )

        cutout_shape = res_cutout.shape

        print(band,rgalpx,cutout_shape)

        cutout_data = res_cutout.data

        ctr = np.int32(0.5 * np.int32(cutout_data.shape))
        extent = [
            -ctr[0] * px_res_as[0],
            ctr[0] * px_res_as[0],
            -ctr[1] * px_res_as[1],
            ctr[1] * px_res_as[1],
        ]

        if "NIRCAM" not in band and "HST" not in band:
            ax[2][iband].imshow(
                cutout_data,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap="bwr",
                extent=extent,
                transform=tr + ax[2][iband].transData,
            )
        else:
            ax[2][iband].imshow(
                cutout_data, origin="lower", vmin=vmin, vmax=vmax, cmap="bwr", extent=extent
            )

        xs, ys = np.mgrid[: cutout_shape[0], : cutout_shape[1]]
        xs = xs / xs.max() - 0.5
        ys = ys / ys.max() - 0.5
        rs = np.sqrt(xs**2 + ys**2)

        select = rs <= 0.5

        print(band, cutout_data[select].min(), cutout_data[select].max())

        ax[3][iband].hist(np.abs(cutout_data[select]), bins=noise_bins, density=True)
        # ax[3][iband].axvline(cutout_data[select].sum(),c='k',ls='--')
        ax[3][iband].text(
            0.8,
            0.9,
            f"{cutout_data[select].sum():.1e} MJy/sr",
            transform=ax[3][iband].transAxes,
            va="center",
            ha="center",
        )
        # ax[3][iband].text(0.8,0.8,f"{-2.5*np.log10(cutout_data[select].sum()*1e6*px_sr)+8.9:.1f}",transform=ax[3][iband].transAxes,va="center",ha="center")
        ax[3][iband].text(
            0.8,
            0.8,
            f"{-2.5*np.log10(cutout_data[select].sum())+zpt:.1f}",
            transform=ax[3][iband].transAxes,
            va="center",
            ha="center",
        )
        ax[3][iband].set_yscale("log")
        ax[3][iband].set_xscale("log")

        # mag=-2.5 * np.log10(cutout_data[select].sum() * 1e6 * px_sr) + 8.9
        mag = -2.5 * np.log10(np.abs(cutout_data[select].sum())) + zpt
        wav = filt_wav_dict[band]

        wavs.append(wav)
        mags.append(mag)
    ax[-1][-1].scatter(wavs, mags, c="tab:green")

    img_file = f"/data101/jlewis/mock_spe/zoom_tgt_bc03_chabrier100_wlya_SMC/{sim}/{int(snap):d}/gal_{int(gid):07d}/MJy_images/mock_images_{sim}_{int(snap):d}_{int(gid):d}_{3}_wIGM_wPSF_rebinned.h5"
    img = h5_to_dict(img_file)


    wavs = []
    mags = []
    for iband, band in enumerate(
        [
            "MegaCam-u_new",
            "HSC-g",
            "HSC-i",
            "HSC-r",
            "HST-f814_res",
            "HSC-y",
            "HSC-z",
            "NIRCAM-F115",
            "NIRCAM-F150",
            "NIRCAM-F277",
            "NIRCAM-F444",
        ]
    ):

        band_names = img["header"]["band_names"]
        # iband = np.where(band_names == "NIRCAM-F444")[0][0]
        # iband = np.where(band_names == band)[0][0]
        # tgt_band = band_names[band_arg]
        wav = filt_wav_dict[band]
        img_HSC_r_nopsf_nopsf = img[band]
        img_HSC_resas_nopsf = img["header"]["pixel_res_as"][iband]
        img_size = img_HSC_r_nopsf.shape
        rgalpx = rgalas / img_HSC_resas_nopsf
        band_flat = filt_dict[band]

        fflat = os.path.join(flat_path, f"out_flat_{band_flat}_B3.fits")

        with fits.open(fflat) as src:
            hdul = src[0]
            flat_data, flat_hdr = hdul.data, hdul.header

        this_wcs = wcs.wcs.WCS(flat_hdr)

        px_res_as = proj_plane_pixel_scales(this_wcs) * 3600.0
        px_sr = px_res_as[0] ** 2 * (np.pi / 3600.0 / 180.0) ** 2

        # print(px_res_as)

        # pos_deg = SkyCoord((radeg+0*px_res_as[0]/3600) * u.deg, (decdeg+0*px_res_as[1]/3600) * u.deg)

        if "HSC" in band or "CFHT" in band:
            pos_deg = SkyCoord((radeg + 1*px_res_as[0]/3600.)*u.deg, decdeg * u.deg)
        else:
            pos_deg = SkyCoord((radeg)*u.deg, decdeg * u.deg)


        rgalpx = rgalas / px_res_as[0]

        res_cutout = Cutout2D(
            flat_data,
            position=pos_deg,
            size=2 * rgalas * u.arcsec,
            wcs=this_wcs,
        )

        cutout_shape = res_cutout.shape

        print(band,rgalpx,cutout_shape)

        cutout_data = res_cutout.data

        ctr = np.int32(0.5 * np.int32(cutout_data.shape))
        extent = [
            -ctr[0] * px_res_as[0],
            ctr[0] * px_res_as[0],
            -ctr[1] * px_res_as[1],
            ctr[1] * px_res_as[1],
        ]


        ctr2 = np.int32(0.5 * np.int32(img_HSC_r_nopsf_nopsf.shape))
        extent_nopsf = [
            -ctr2[0] * img_HSC_resas_nopsf,
            ctr2[0] * img_HSC_resas_nopsf,
            -ctr2[1] * img_HSC_resas_nopsf,
            ctr2[1] * img_HSC_resas_nopsf,
        ]    

        if "NIRCAM" not in band and "HST" not in band:
            ax[4][iband].imshow(
                cutout_data,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap="bwr",
                extent=extent,
                transform=tr + ax[4][iband].transData,
            )

            # transp = np.ones_like(img_HSC_r_nopsf_nopsf)*0.5
            # transp[img_HSC_r_nopsf_nopsf<1E-2]=0
            # ax[4][iband].imshow(img_HSC_r_nopsf_nopsf.T, origin='lower', vmin=vmin, vmax=vmax, cmap='Greens', extent=extent_nopsf, alpha=transp.T, transform=tr + ax[4][iband].transData)
        else:

            ax[4][iband].imshow(
                cutout_data, origin="lower", vmin=vmin, vmax=vmax, cmap="bwr", extent=extent
            )
            # transp = np.ones_like(img_HSC_r_nopsf_nopsf)*0.5
            # transp[img_HSC_r_nopsf_nopsf<1E-1]=0
            # ax[4][iband].imshow(img_HSC_r_nopsf_nopsf.T, origin='lower', vmin=vmin, vmax=vmax, cmap='Greens', extent=extent_nopsf, alpha=transp.T)



        print(rgalas,r50as*2)
        circ = Circle((0,0),2*r50as,facecolor="none",edgecolor="k",ls='--')
        ax[4][iband].add_patch(circ)
        
        # print(rgalas,r50as*2)
        circ = Circle((0,0),rgalas,facecolor="none",edgecolor="k",ls='-')
        ax[4][iband].add_patch(circ)

        xs, ys = np.mgrid[: cutout_shape[0], : cutout_shape[1]]
        xs = xs / xs.max() - 0.5
        ys = ys / ys.max() - 0.5
        rs = np.sqrt(xs**2 + ys**2)

        select = rs <= 0.5

        print(band, cutout_data[select].min(), cutout_data[select].max())

        # mag=-2.5 * np.log10(cutout_data[select].sum() * 1e6 * px_sr) + 8.9
        mag = -2.5 * np.log10(cutout_data[select].sum()) + zpt
        wav = filt_wav_dict[band]

        wavs.append(wav)
        mags.append(mag)
    ax[-1][-1].scatter(wavs, mags, c="tab:red")

    # size_as = 1.5

    for iax in [0, 1, 2, 4]:
        for iay in range(11):
            ax[iax][iay].set_xlim(-rgalas, rgalas)
            ax[iax][iay].set_ylim(-rgalas, rgalas)
            # ax[iax][iay].set_xlim(-size_as, size_as)
            # ax[iax][iay].set_ylim(-size_as, size_as)


    ax[-1][-1].invert_yaxis()

    for band_name in filt_wav_dict.keys():

        ax[-1][-1].text(filt_wav_dict[band_name],spec["speectral_data"]["mag"].max()*1.095,band_name,va='center',ha='center')


    ymin,ymax=spec["speectral_data"]["mag"].max()*1.1,spec["speectral_data"]["mag"].min()*0.9

    if not np.isfinite(ymin):
        ymin=31
    if not np.isfinite(ymax):
        ymax=22

    # ax.set_ylim(26,21)
    ax[-1][-1].set_ylim(ymin,ymax)
    ax[-1][-1].set_xlim(3e-1, 6)
    ax[-1][-1].set_xscale("log")

    fig.savefig(f"figs_diag/diag_hscr_{arg}")
