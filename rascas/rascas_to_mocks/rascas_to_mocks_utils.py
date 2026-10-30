import os
import numpy as np
import h5py
from scipy.ndimage import zoom, gaussian_filter

def rescale_image_flux_conserving(image, scale_factor, order=2):
    """Resize an image by a factor ``1 / scale_factor`` with spline interpolation.

    Notes
    -----
    Despite the function name, the total flux is currently NOT conserved:
    the correction ``rescaled_image *= scale_factor**2`` is commented out, so
    pixel values are preserved (surface brightness) and the total sum changes
    by about ``1 / scale_factor**2``.

    Parameters
    ----------
    image : np.ndarray
        2D image to rescale.
    scale_factor : float
        Values larger than 1 shrink the image (output size is about
        input size / ``scale_factor``, rounded to an integer number of pixels).
    order : int, optional
        Spline interpolation order passed to ``scipy.ndimage.zoom``
        (0 to 5). Default is 2 (quadratic). 

    Returns
    -------
    np.ndarray
        The rescaled image.
    """
    
    
    # Rescale image using zoom
    rescaled_image = zoom(
        image, 1 / scale_factor, order=order
    )  # binomial interpolation

    # Conserve flux: scale intensity by (scale_factor)^2
    # rescaled_image *= scale_factor**2

    return rescaled_image


def pad_to_multiple(img, factor):
    """Pad a 2D array so both dimensions are multiples of ``factor``.

    This is useful for image operations that require block-aligned shapes, e.g.
    integer rebinning.

    Padding is added only at the end of each axis (bottom and right) and is
    filled with zeros. This preserves the sum of the image but biases block
    averages near the edges. A new array is always returned, even if no
    padding is needed.

    Parameters
    ----------
    img : np.ndarray
        2D array to pad.
    factor : int
        Positive integer. Each output dimension is the smallest multiple of
        ``factor`` that is greater than or equal to the input dimension.

    Returns
    -------
    np.ndarray
        Zero-padded array whose two dimensions are multiples of ``factor``.
    """
    ny, nx = img.shape
    ny_pad = (factor - ny % factor) % factor
    nx_pad = (factor - nx % factor) % factor
    return np.pad(img, ((0, ny_pad), (0, nx_pad)), mode="constant", constant_values=0)


def rebin_flux_conserving(img, cur_res_as, tgt_res_as):
    """
    Rebin a 2D image from current resolution to target resolution,
    approximately conserving total flux and avoiding aliasing artifacts.
    
    Notes    -----
    Total flux is only approximately conserved because:

    * rows and columns that do not fill a complete block at the bottom and
      right edges are trimmed, and their flux is lost;
    * the Gaussian smoothing uses scipy's default ``reflect`` boundary mode,
      which slightly alters the flux near the edges.

    The ratio ``tgt_res_as / cur_res_as`` is rounded to the nearest integer
    (a warning is printed if it is not close to an integer). The output
    resolution is therefore ``factor * cur_res_as``, which may differ from
    ``tgt_res_as``. If the ratio is below 1.5 the factor is 1 and the image
    is only smoothed, not rebinned. If ``tgt_res_as <= cur_res_as``, a copy
    of the input is returned unchanged.

    Parameters
    ----------
    img : 2D ndarray
        Input image with square pixels.
    cur_res_as : float
        Current resolution per pixel (in arcseconds).
    tgt_res_as : float
        Target resolution per pixel (in arcseconds).

    Returns
    -------
    img_rebinned : 2D ndarray
        Rebinned image with total flux preserved approximately.
    """

    if tgt_res_as <= cur_res_as:
        # No rebinning needed
        return img.copy()

    # Compute integer rebinning factor
    rebin_factor = tgt_res_as / cur_res_as
    int_factor = int(np.round(rebin_factor))

    if not np.isclose(rebin_factor, int_factor, rtol=1e-2):
        print(
            f"Warning: rebin factor {rebin_factor:.3f} is not integer — rounding to {int_factor}"
        )

    rebin_factor = int_factor

    # Apply Gaussian smoothing to suppress high-frequency noise (anti-aliasing)
    sigma = rebin_factor / 2.355  # ~match FWHM of target resolution
    img_smoothed = gaussian_filter(img, sigma=sigma)

    # Trim image to be divisible by rebin_factor
    ny, nx = img_smoothed.shape
    ny_trim = ny - (ny % rebin_factor)
    nx_trim = nx - (nx % rebin_factor)
    img_trimmed = img_smoothed[:ny_trim, :nx_trim]

    # Reshape and sum blocks to conserve flux
    img_rebinned = img_trimmed.reshape(
        ny_trim // rebin_factor, rebin_factor, nx_trim // rebin_factor, rebin_factor
    ).sum(axis=(1, 3))

    return img_rebinned


def write_mock_imgs(
    fname, aexp, time, z, aper_sr, px_res_as, px_res_stars_as, names, imgs, st_img
):
    """Write mock-image data and metadata to an HDF5 file.

    An existing file is overwritten. Images are written as provided: no unit
    conversion is done, ``imgs`` are assumed to be in MJy.sr^-1 and ``st_img`` in
    M_sun.as^-2 (units are only recorded in the header). Each band is stored in
    a dataset named after ``names``; the stellar map is stored in the dataset
    ``"stellar_surface density"`` (spelling kept for compatibility).

    Parameters
    ----------
    fname : str or os.PathLike
        Output filename.
    aexp : float
        Cosmic scale factor.
    time : float
        Simulation time in Myr.
    z : float
        Redshift.
    aper_sr : float
        Aperture size in steradians.
    px_res_as : float
        Pixel resolution of the main mock images in arcseconds.
    px_res_stars_as : float
        Pixel resolution used for the stellar surface-density image.
    names : sequence of str
        Names of the mock bands.
    imgs : sequence of np.ndarray
        2D intensity maps for each band.
    st_img : np.ndarray
        Stellar surface-density map.
        
    Notes
    -----
    ``names`` and ``imgs`` must have the same length: extra entries in the
    longer one are silently ignored for the datasets, while the header still
    lists all ``names``.
    """

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


def walk_down_dirs(root, saved_paths):
    """Recursively collect simulation directories under ``root``.

    A directory whose name starts with ``id`` is treated as a simulation
    directory: it is added to ``saved_paths`` (unless it is a symbolic link) and
    the search does not descend into it. Other directories are explored
    recursively (symbolic links to directories are followed).

    Parameters
    ----------
    root : str
        Directory to scan. Must exist, otherwise ``os.listdir`` raises.
    saved_paths : list of str
        List the paths are appended to. It is modified in place and also returned.

    Returns
    -------
    list of str
        The ``saved_paths`` list, in directory listing order (not sorted).
    """

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
    """Locate simulation directories in a fixed set of data directories.

    Scans the hard-coded roots ``/data101/jlewis/sims/dust_fid/lvlmax_22``,
    ``/data102/jlewis/sims/lvlmax_22`` and ``/data103/jlewis/sims/lvlmax_22``
    with :func:`walk_down_dirs`. All three must exist, otherwise a
    ``FileNotFoundError`` is raised.

    Returns
    -------
    list of str
        Sorted, unique simulation paths (as ``numpy.str_``), excluding any path
        containing the substring ``common_plots``.
    """
    
    
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