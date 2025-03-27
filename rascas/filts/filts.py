from unittest import skip
import numpy as np
import os

# import scipy convoluion (1d and 2d)
# from scipy.signal import convolve
# from scipy.ndimage import convolve1d
from scipy.stats import binned_statistic, binned_statistic_2d
from scipy.integrate import simpson, trapz

from scipy.interpolate import interp1d

from astropy import units as u
from astropy.cosmology import Planck18 as cosmo


def read_transmission(fin):

    # print(fin)
    wav, trans = np.genfromtxt(fin, skip_header=1, comments="#").T
    return wav, trans


def read_transmissions(tgt_bands=None, IB=False, NB=False, MIRI=False, YJHK=True):
    root = "/home/jlewis/zoom_analysis/rascas/filts"

    filters_files = [
        f
        for f in os.listdir(root)
        if not ".py" in f and os.path.isfile(os.path.join(root, f))
    ]

    if IB == False:
        filters_files = [f for f in filters_files if "IB" not in f]
    if NB == False:
        filters_files = [f for f in filters_files if "NB" not in f]
    if MIRI == False:
        filters_files = [f for f in filters_files if "MIRI" not in f]
    if YJHK == False:
        filters_files = [f for f in filters_files if not f.startswith("Y.")]
        filters_files = [f for f in filters_files if not f.startswith("J.")]
        filters_files = [f for f in filters_files if not f.startswith("H.")]
        filters_files = [f for f in filters_files if not f.startswith("K.")]

    filters_names = [f.split(".")[0] for f in filters_files]
    # print(filters_names)

    if tgt_bands is not None:
        filters_files = [f for f in filters_files if tgt_bands in f]

    filters_wavs = []
    filters_trans = []

    for f in filters_files:
        wav, trans = read_transmission(os.path.join(root, f))

        filters_wavs.append(wav)
        filters_trans.append(trans)

    return filters_names, filters_wavs, filters_trans


def convolve_spe(spe_wav, spe_flux, filts_wav, filts_trans):

    out = np.zeros(len(filts_wav))
    ctr = np.zeros(len(filts_wav))

    ifilt = 0

    for ifilt, (filt_wav, filt_trans) in enumerate(zip(filts_wav[:], filts_trans[:])):

        # interpolate filter to match spectrum
        inter_fct = interp1d(
            filt_wav, filt_trans, "linear", bounds_error=False, fill_value=0.0
        )

        bin_filt_trans = inter_fct(spe_wav)

        bin_filt_trans[np.isnan(bin_filt_trans)] = 0.0
        if np.sum(bin_filt_trans) == 0:
            continue

        # convolve spectrum with filter
        out[ifilt] = simpson(spe_flux[:] * bin_filt_trans, spe_wav[:]) / simpson(
            bin_filt_trans, spe_wav[:]
        )

        # print(ifilt, out[ifilt])

        ctr[ifilt] = np.median(filt_wav)

        ifilt += 1

    # print(px_weights)

    return ctr, out  # / px_weights


def convolve_cube(spe_wav, spe_cube, filts_wav, filts_trans):

    # print(filts_wav, filts_trans)

    # incoming cubes are dims x,y,wav
    out_cube = np.zeros((len(filts_wav), spe_cube.shape[0], spe_cube.shape[1]))
    ctr = np.zeros(len(filts_wav))
    # px_weights = np.zeros_like(spe_flux)

    ifilt = 0

    for ifilt, (filt_wav, filt_trans) in enumerate(zip(filts_wav[:], filts_trans[:])):
        # reample filter to match spectrum
        inter_fct = interp1d(
            filt_wav, filt_trans, "linear", bounds_error=False, fill_value=0.0
        )

        bin_filt_trans = inter_fct(spe_wav)
        if np.sum(bin_filt_trans) == 0:
            continue

        # convolve spectrum with filter
        # px_weights[:-1] += np.int32(bin_filt_trans > 0)
        out_cube[ifilt, :, :] = simpson(
            spe_cube[:, :, :] * bin_filt_trans[np.newaxis, np.newaxis, :],
            spe_wav[:],
            axis=2,
        ) / simpson(bin_filt_trans, spe_wav[:])

        ctr[ifilt] = np.median(filt_wav)

        ifilt += 1

    return ctr, out_cube  # / px_weights


def norm_band(input_dat, filt_name):

    filts_names, filts_wav, filts_trans = read_transmissions()

    # band_tots = np.asarray([np.sum(filt_trans) for filt_trans in filts_trans])
    band_tots = np.asarray([trapz(filt_trans) for filt_trans in filts_trans])
    band_norm = np.max(band_tots) / band_tots[np.isin(filts_names, filt_name)]
    # print(band_norm, filt_name, filts_names)
    return input_dat * band_norm


def fnu_flamb(fnu, wav):
    # fnu * nu = flamb * lam
    # flamb = fnu * nu / lam
    nu = 2.99792458e18 / wav
    return fnu * nu / wav


def flamb_fnu(flamb, wav):
    # fnu * nu = flamb * lam
    # fnu = flamb * lam / nu
    nu = 2.99792458e18 / wav  # wav in angstroms

    return flamb * wav / nu


def fnu_to_mAB(fnu):
    # fnu in erg/s/cm^2/Hz
    # apparent
    out = -2.5 * np.log10(fnu) - 48.60
    out[np.isinf(out)] = -999
    return out


def flamb_to_mAB(flamb, wav):
    # flamb in erg/s/cm^2/Ang
    # print(flamb, fnu, wav)
    return fnu_to_mAB(flamb_fnu(flamb, wav))


def m_to_M(m, z):
    dl = cosmo.luminosity_distance(z).to(u.parsec).value
    mu = 5 * np.log10(dl / 10)
    return m - mu


def lintrinsic_to_fapparent(l, z):
    ang_dist = cosmo.angular_diameter_distance(z).to(u.cm).value
    return l / (ang_dist**2)


def degrade_res(img, px_cm, px_mas, z):
    # smooth to resolution of JWST at z

    ang_dist = cosmo.angular_diameter_distance(z).to(u.cm).value

    tgt_px_cm = (px_mas * 1e-3) * (ang_dist / 3600 * np.pi / 180)

    degrade_fact = tgt_px_cm / px_cm

    X, Y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    tgt_bins = np.arange(0, img.shape[0], degrade_fact)
    degrade_size = len(tgt_bins)

    degrade_img, _, _ = binned_statistic_2d(
        np.ravel(X),
        np.ravel(Y),
        np.ravel(img),
        statistic="mean",
        bins=[tgt_bins, tgt_bins],
    )

    return (degrade_img, degrade_size)
