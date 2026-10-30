import numpy as np
import os

# import scipy convoluion (1d and 2d)
# from scipy.signal import convolve
# from scipy.ndimage import convolve1d
from scipy.stats import binned_statistic, binned_statistic_2d
from scipy.integrate import simpson, trapezoid

from scipy.interpolate import interp1d

from astropy import units as u
from astropy.cosmology import Planck18 as cosmo

import matplotlib.pyplot as plt


def read_transmission(fin):

    print(fin)
    wav, trans = np.genfromtxt(fin, skip_header=1, comments="#").T
    return wav, trans


def read_transmissions(
    tgt_bands=None,
    NIRCAM=True,
    HST=True,
    HSC=True,
    IRAC=True,
    SuprimeCam=True,
    ULTRAVISTA=True,
    MIRI=True,
    VISTA=True,
    NISP=True,
    VIS=True,
    LSST=True,
    GALEX=True,
    MegaCam=True,
    Euclid_EXT=True,
    CLUTCH=True,
):
        """Read the filter transmission curves stored in the filter directory.

    All files of the hard-coded directory
    ``/home/jlewis/codes/zoom_analysis/rascas/filts`` are read, except ``.py``
    and ``.png`` files and files whose name contains ``UVIS1``. Instrument
    families can be excluded with the boolean flags below. Each file is read
    with :func:`read_transmission` (two columns: wavelength and transmission;
    the first line is skipped and lines starting with ``#`` are ignored).

    Parameters
    ----------
    tgt_bands : str, optional
        Substring that the file names must contain. Currently broken, see
        Notes.
    NIRCAM, HST, SuprimeCam, ULTRAVISTA, MIRI, IRAC : bool, optional
        If False, exclude the files whose name *contains* the flag name.
        Default is True (keep).
    VISTA, NISP, VIS, LSST, GALEX, MegaCam, HSC : bool, optional
        If False, exclude the files whose name *starts with* the flag name.
        Default is True (keep).
    Euclid_EXT : bool, optional
        If False, exclude the files starting with ``PS`` and the file
        ``MegaCam-r.pb``. Default is True.
    CLUTCH : bool, optional
        If False, exclude ``NIRCAM-F200W.txt``, ``NIRCAM-F356W.txt``,
        ``HST-F098W.pb``, ``HST-F435W.pb``, ``HST-F606W.pb`` and the files
        starting with ``UVIS``. Default is True.

    Returns
    -------
    filters_names : list of str
        File names up to the first dot (e.g. ``NIRCAM-F277``), sorted
        alphabetically.
    filters_wavs : list of np.ndarray
        Wavelength grid of each filter, in Angstrom.
    filters_trans : list of np.ndarray
        Transmission of each filter, as stored in the file (not normalised).

    Notes
    -----
    * The lists are sorted by name, not by wavelength.
    * ``VIS=False`` also removes the files starting with ``VISTA``, and
      ``CLUTCH=False`` also removes any file starting with ``UVIS`` (this
      includes names such as ``UVISTA-H``). ``ULTRAVISTA=False`` only matches
      names containing ``ULTRAVISTA``, not ``UVISTA``.
    * ``tgt_bands`` filters the files after the names have been listed, so the
      lists get out of sync: an ``IndexError`` is raised unless the substring
      matches every file.
    * The files are read again at each call, and the path of each file and the
      number of filters loaded are printed.
    """
    
    root = "/home/jlewis/codes/zoom_analysis/rascas/filts"

    filters_files = [
        f
        for f in os.listdir(root)
        if not f.endswith(".py") and os.path.isfile(os.path.join(root, f))
        if not f.endswith(".png") and os.path.isfile(os.path.join(root, f))
        if not "UVIS1" in f
    ]

    # print(filters_files)

    if NIRCAM == False:
        filters_files = [f for f in filters_files if "NIRCAM" not in f]
    if HST == False:
        filters_files = [f for f in filters_files if "HST" not in f]
    if SuprimeCam == False:
        filters_files = [f for f in filters_files if "SuprimeCam" not in f]
    if ULTRAVISTA == False:
        filters_files = [f for f in filters_files if "ULTRAVISTA" not in f]
    if MIRI == False:
        filters_files = [f for f in filters_files if "MIRI" not in f]
    if VISTA == False:
        filters_files = [f for f in filters_files if not f.startswith("VISTA")]
    if NISP == False:
        filters_files = [f for f in filters_files if not f.startswith("NISP")]
    if VIS == False:
        filters_files = [f for f in filters_files if not f.startswith("VIS")]
    if LSST == False:
        filters_files = [f for f in filters_files if not f.startswith("LSST")]
    if GALEX == False:
        filters_files = [f for f in filters_files if not f.startswith("GALEX")]
    if MegaCam == False:
        filters_files = [f for f in filters_files if not f.startswith("MegaCam")]
    if HSC == False:
        filters_files = [f for f in filters_files if not f.startswith("HSC")]
    if IRAC == False:
        filters_files = [f for f in filters_files if "IRAC" not in f]
    if CLUTCH == False:
        filters_files = [
            f
            for f in filters_files
            if f not in ["NIRCAM-F200W.txt", "NIRCAM-F356W.txt"]
        ]
        filters_files = [
            f
            for f in filters_files
            if f not in ["HST-F098W.pb", "HST-F435W.pb", "HST-F606W.pb"]
        ]
        filters_files = [f for f in filters_files if not f.startswith("UVIS")]
        # filters_files = [f for f in filters_files if f != "HST-F098W.pb"]
    if Euclid_EXT == False:
        filters_files = [f for f in filters_files if not f.startswith("PS")]
        filters_files = [f for f in filters_files if f != "MegaCam-r.pb"]

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

    # print(filters_names)

    order = np.argsort(filters_names)

    filters_names = [filters_names[o] for o in order]
    filters_wavs = [filters_wavs[o] for o in order]
    filters_trans = [filters_trans[o] for o in order]

    print(f"Loaded {len(filters_names)} filters")

    return filters_names, filters_wavs, filters_trans


def check_filters(path="."):

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), layout="constrained")

    names, wavs, trans = read_transmissions()

    last_instr = ""
    for n, w, t in zip(names, wavs, trans):
        if last_instr == "" or last_instr != n.split("-")[0]:
            color = None
            last_instr = n.split("-")[0]
        else:
            color = l.get_color()
        (l,) = ax.plot(w, t / t.sum(), label=n, color=color)
        print(t.min(), t.max(), n)

    ax.set_ylabel("transmission")
    ax.set_xlabel("Angstrom")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_ylim(
        1e-4,
    )

    fig.legend(loc="outside lower center", ncols=4, framealpha=0.0)

    fig.savefig("filt_test_all")


def convolve_spe(spe_wav, spe_flux, filts_wav, filts_trans):

    out = np.zeros(len(filts_wav))
    ctr = np.zeros(len(filts_wav))

    ifilt = 0

    for ifilt, (filt_wav, filt_trans) in enumerate(zip(filts_wav[:], filts_trans[:])):

        # interpolate filter to match spectrum
        inter_fct = interp1d(
            filt_wav, filt_trans, "linear", bounds_error=False, fill_value=0.0
        )

        wav_min = filt_wav[0]
        wav_max = filt_wav[-1]
        i0 = np.searchsorted(spe_wav, wav_min)-1
        i1 = np.searchsorted(spe_wav, wav_max)+1        

        bin_filt_trans = inter_fct(spe_wav[i0:i1])

        bin_filt_trans[np.isnan(bin_filt_trans)] = 0.0
        if np.sum(bin_filt_trans) == 0:
            continue

        # convolve spectrum with filter
        out[ifilt] = simpson(spe_flux[i0:i1] * bin_filt_trans, spe_wav[i0:i1]) / simpson(
            bin_filt_trans, spe_wav[i0:i1]
        )
        # out[ifilt] = trapezoid(spe_flux[:] * bin_filt_trans, spe_wav[:]) / trapezoid(
        #     bin_filt_trans, spe_wav[:]
        # )

        # print(ifilt, out[ifilt])

        ctr[ifilt] = np.median(filt_wav)

    # print(px_weights)

    return ctr, out  # / px_weights


def convolve_cube(spe_wav, spe_cube, filts_wav, filts_trans, rpix=None):

    left_half = int(np.floor(rpix * 0.5))
    right_half = int(np.ceil(rpix * 0.5))
    half = int(np.floor(spe_cube.shape[0] / 2))

    work_cube = spe_cube[
                half - left_half : half + right_half,
                half - left_half : half + right_half,
                :,
            ]

    # incoming cubes are dims x,y,wav
    if rpix == None:
        rpix = spe_cube.shape[0]
    out_cube = np.zeros((len(filts_wav), rpix, rpix))
    ctr = np.zeros(len(filts_wav))
  

    ifilt = 0

    for ifilt, (filt_wav, filt_trans) in enumerate(zip(filts_wav[:], filts_trans[:])):
        # reample filter to match spectrum
        inter_fct = interp1d(
            filt_wav, filt_trans, "linear", bounds_error=False, fill_value=0.0
        )


        wav_min = filt_wav[0]
        wav_max = filt_wav[-1]
        i0 = np.searchsorted(spe_wav, wav_min)-1
        i1 = np.searchsorted(spe_wav, wav_max)+1

        bin_filt_trans = inter_fct(spe_wav[i0:i1])


        if np.sum(bin_filt_trans) == 0:
            continue
        # print(half, rpix, half_rpix, half+half_rpix, half-half_rpix)

        # convolve spectrum with filter
        # px_weights[:-1] += np.int32(bin_filt_trans > 0)
        out_cube[ifilt, :, :] = simpson(
        # out_cube[ifilt, :, :] = trapezoid(
            work_cube[:,:,i0:i1]
            * bin_filt_trans[np.newaxis, np.newaxis, :],
            # np.einsum('ijk,k->ijk', work_cube[:,:,i0:i1], bin_filt_trans),
            spe_wav[i0:i1],
            axis=2,
        # ) / trapezoid(bin_filt_trans, spe_wav[:])
        ) / simpson(bin_filt_trans, spe_wav[i0:i1])

        # print(filt_wav)
        

        ctr[ifilt] = np.median(filt_wav)

 

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
