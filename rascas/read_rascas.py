import numpy as np
import matplotlib.pyplot as plt
import h5py
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from f90_tools.IO import read_record

c = 299792458  # m/s
h = 6.62607015e-34  # J s

# dump photons from module_photon.f90
# np = size(pgrid)
# open(unit=14, file=trim(file), status='unknown', form='unformatted', action='write')
# write(14) np
# write(14) (pgrid(i)%ID,      i=1,np)
# write(14) (pgrid(i)%status,  i=1,np)
# write(14) (pgrid(i)%xlast(:),i=1,np)
# write(14) (pgrid(i)%nu_ext,  i=1,np)
# write(14) (pgrid(i)%k(:),    i=1,np)
# write(14) (pgrid(i)%nb_abs,  i=1,np)
# write(14) (pgrid(i)%time,    i=1,np)
# close(14)

# type def of pgrid
#   type photon_current
#      integer(kind=4)           :: ID
#      integer(kind=4)           :: status       ! =0 if flying, =1 if escape, =2 if absorption (by dust)
#      real(kind=8),dimension(3) :: xlast        ! coordinates of last interaction in box units
#      real(kind=8),dimension(3) :: xcurr        ! current position of the photon in box units
#      real(kind=8)              :: nu_ext       ! external frame frequency (Hz)
#      real(kind=8),dimension(3) :: k            ! normalised propagation vector
#      integer(kind=4)           :: nb_abs       ! number of interactions before escape
#      real(kind=8)              :: time         ! time in [s] from emission to escape/absorption
#      real(kind=8)              :: tau_abs_curr ! current optical depth (useful when photon change mesh domain)
#      integer(kind=4)           :: iran         ! state of the random generator
#   end type photon_current


def read_photon_dump(fname):
    with open(fname, "rb") as src:
        ngrid = np.fromfile(src, dtype=np.int32, count=3)[1]

        pgrid = {}

        np.fromfile(src, dtype=np.int32, count=1)
        pgrid["ID"] = np.fromfile(src, dtype=np.int32, count=ngrid)
        np.fromfile(src, dtype=np.int32, count=1)

        np.fromfile(src, dtype=np.int32, count=1)
        pgrid["status"] = np.fromfile(src, dtype=np.int32, count=ngrid)
        np.fromfile(src, dtype=np.int32, count=1)

        np.fromfile(src, dtype=np.int32, count=1)
        pgrid["xlast"] = np.fromfile(src, dtype=np.float64, count=ngrid * 3).reshape(
            ngrid, 3
        )
        np.fromfile(src, dtype=np.int32, count=1)

        np.fromfile(src, dtype=np.int32, count=1)
        pgrid["nu_ext"] = np.fromfile(src, dtype=np.float64, count=ngrid)
        np.fromfile(src, dtype=np.int32, count=1)

        np.fromfile(src, dtype=np.int32, count=1)
        pgrid["k"] = np.fromfile(src, dtype=np.float64, count=ngrid * 3).reshape(
            ngrid, 3
        )
        np.fromfile(src, dtype=np.int32, count=1)

        np.fromfile(src, dtype=np.int32, count=1)
        pgrid["nb_abs"] = np.fromfile(src, dtype=np.int32, count=ngrid)
        np.fromfile(src, dtype=np.int32, count=1)

        np.fromfile(src, dtype=np.int32, count=1)
        pgrid["time"] = np.fromfile(src, dtype=np.float64, count=ngrid)
        np.fromfile(src, dtype=np.int32, count=1)

    return pgrid


def read_PFS_dump(fname, hdr=False):
    out = {}

    with open(fname, "rb") as src:
        nphotons = read_record(src, 1, np.int32)
        total_flux = read_record(src, 1, np.float64)
        ranseed = read_record(src, 1, np.int32)

        out["nphotons"] = nphotons
        out["total_flux"] = total_flux
        out["ranseed"] = ranseed
        # print(nphotons, total_flux, ranseed)

        if not hdr:
            photID = read_record(src, nphotons, np.int32)
            nu_em = read_record(src, nphotons, np.float64)
            x_em = read_record(src, nphotons * 3, np.float64).reshape(nphotons, 3)
            k_em = read_record(src, nphotons * 3, np.float64).reshape(nphotons, 3)
            phot_seeds = read_record(src, nphotons, np.int32)
            v_em = read_record(src, nphotons * 3, np.float64).reshape(nphotons, 3)

            out["photID"] = photID
            out["nu_em"] = nu_em
            out["x_em"] = x_em
            out["k_em"] = k_em
            out["phot_seeds"] = phot_seeds
            out["v_em"] = v_em

    return out


def spec_to_erg_s_A_cm2_rf(mock, pfs_path):  # in rest frame
    phot_hdr = read_PFS_dump(pfs_path, hdr=True)

    # print(mock["lambda_min"], mock["lambda_max"], mock["lambda_npix"])

    lambs = np.linspace(
        mock["lambda_min"], mock["lambda_max"], mock["lambda_npix"]
    )  # angstrom
    dlamb = np.diff(lambs)[0]
    nrg = c * h / (lambs * 1e-10) * 1e7  # erg
    nrg *= phot_hdr["total_flux"] / phot_hdr["nphotons"] / dlamb
    # nrg /= aperture_cm**2 * np.pi
    # nrg *= mock["aperture"] ** 2  # to luminisity at aperture
    # nrg /= 1 + z  # dimming

    # mock["spectrum"] = nrg * mock["spectrum"]
    return nrg * mock["spectrum"]

    # as in rascas/py/mocks.py by Leo Michel Dansac
    # nPhotPerPacket = phot_hdr["total_flux"] / phot_hdr["nphotons"]
    # dl = (mock["lambda_max"] - mock["lambda_min"]) / mock["lambda_npix"]
    # l = np.arange(mock["lambda_min"], mock["lambda_max"], dl)
    # l = l * 1e-10  # m
    # nrg = c * h / l * 1e7  # erg
    # nrg *= nPhotPerPacket
    # nrg /= dl
    # mock["spectrum"] *= nrg


def spec_to_erg_s_A_cm2(mock, pfs_path, z):  # , aperture_cm):  # in obs frame

    PC_cm = 3.08e18 #cm

    phot_hdr = read_PFS_dump(pfs_path, hdr=True)

    # print(mock["lambda_min"], mock["lambda_max"], mock["lambda_npix"])

    if z>0:
        dL = cosmo.luminosity_distance(z).to(u.cm).value
    else:
        dL = 10 * PC_cm

    lambda_bins = np.linspace(
        mock["lambda_min"]*(1+z), mock["lambda_max"]*(1+z), mock["lambda_npix"]
    ) #* (1+z)  # angstrom
    dlamb = np.diff(lambda_bins)[0]  # * 1e-10
    nrg = c * h / (lambda_bins * 1e-10) * 1e7  # erg
    nrg *= phot_hdr["total_flux"] / phot_hdr["nphotons"] / dlamb


    nrg /= (
        4 * np.pi * dL ** 2
    )  # to flux at observer
    if z > 0:
        nrg *= 1 + z  # dimming
    # nrg /= aperture_cm**2 * np.pi
    # nrg *= mock["aperture"] ** 2  # to luminisity at aperture
    # nrg /= 1 + z  # dimming

    return nrg * mock["spectrum"]
    # mock["spectrum"] = nrg * mock["spectrum"]

    # as in rascas/py/mocks.py by Leo Michel Dansac
    # nPhotPerPacket = phot_hdr["total_flux"] / phot_hdr["nphotons"]
    # dl = (mock["lambda_max"] - mock["lambda_min"]) / mock["lambda_npix"]
    # l = np.arange(mock["lambda_min"], mock["lambda_max"], dl)
    # l = l * 1e-10  # m
    # nrg = c * h / l * 1e7  # erg
    # nrg *= nPhotPerPacket
    # nrg /= dl
    # mock["spectrum"] *= nrg


def cube_to_erg_s_A_cm2_as2(mock, pfs_path, z, aper_as):  # in obs frame

    PC_cm = 3.08e18

    phot_hdr = read_PFS_dump(pfs_path, hdr=True)  # why pfs here ?

    # print(mock["lambda_min"], mock["lambda_max"], mock["lambda_npix"])
    if z>0:
        dL = cosmo.luminosity_distance(z).to(u.cm).value  # cm
    else:
        dL = 10 * PC_cm

    # dx_as = np.arctan(dx_cm / dL) * 180 / np.pi * 3600  # as
    dx_as = aper_as / mock["cube"].shape[0]

    lambs = np.linspace(
        mock["lambda_min"]*(1+z), mock["lambda_max"]*(1+z), mock["lambda_npix"]
    )  # angstrom
    dlamb = np.diff(lambs)[0]
    nrg = c * h / (lambs * 1e-10) * 1e7  # erg
    nrg *= phot_hdr["total_flux"] / phot_hdr["nphotons"] / dlamb
    nrg /= dx_as**2  # cell surface
    nrg /= 4 * np.pi * dL**2  # to flux at observer

    nrg *= 1 + z  # dimming

    mock["cube"] = nrg * mock["cube"]


def read_mock_dump(fname, ndir=1):
    file_type = fname.split(".")[-1]

    mocks = {}

    with open(fname, "rb") as src:

        for idir in range(ndir):

            mock = {}
            mock["type"] = file_type

            if file_type == "flux":
                pass

            elif file_type == "spectrum":
                npix = read_record(src, 1, np.int32)
                aperture, lambda_min, lambda_max = read_record(src, 3, np.float64)
                spec = np.zeros(npix, dtype=np.float64)
                spec = read_record(src, npix, np.float64)

                mock["aperture"] = aperture
                mock["lambda_min"] = lambda_min
                mock["lambda_max"] = lambda_max
                mock["lambda_npix"] = npix
                mock["spectrum"] = spec

            elif file_type == "image":
                img_npix = read_record(src, 1, np.int32)
                # print(img_npix)
                img_side = read_record(src, 1, np.float64)
                img_ctr = read_record(src, 3, np.float64)
                img = np.zeros((img_npix, img_npix), dtype=np.float64)
                img = read_record(src, img_npix * img_npix, np.float64).reshape(
                    img_npix, img_npix
                )

                mock["img_npix"] = img_npix
                mock["img_side"] = img_side
                mock["img_ctr"] = img_ctr

                mock["image"] = img

            elif file_type == "cube":
                lambda_npx, img_npx = read_record(src, 2, np.int32)
                lambda_min, lambda_max, cube_side = read_record(src, 3, np.float64)
                cube_ctr = read_record(src, 3, np.float64)
                cube = np.zeros((lambda_npx, img_npx, img_npx), dtype=np.float64)
                cube = read_record(
                    src, lambda_npx * img_npx * img_npx, np.float64
                ).reshape(img_npx, img_npx, lambda_npx)

                mock["lambda_npix"] = lambda_npx
                mock["lambda_min"] = lambda_min
                mock["lambda_max"] = lambda_max

                mock["img_npix"] = img_npx

                mock["cube_side"] = cube_side
                mock["cube_ctr"] = cube_ctr
                mock["cube"] = cube

            mocks[f"direction_{idir:d}"] = mock

    return mocks


def mock_spe_dump(
    fout,
    z,
    aperture_as,
    band_names,
    wav_micron_full_spec,
    mag_full_spec,
    mag,
    mag_err,
    mag_rf,
    mag_err_rf,
    wav_microm,
    gal_data,
    mag_units="MAB",
    lam_units="micrometers",
):
    with h5py.File(fout, "w") as f:
        hdr = f.create_group("header")
        hdr.create_dataset("z", data=z)
        hdr.create_dataset("aperture_as", data=aperture_as)

        gal_dat = f.create_group("galaxy_data")
        gal_dat.create_dataset("galaxy_mass(msun)", data=gal_data["mass"])
        gal_dat.create_dataset("galaxy_sfr(msun.myr^-1)", data=gal_data["sfr"])
        gal_dat.create_dataset("galaxy_ssfr(myr^-1)", data=gal_data["ssfr"])
        if "mass_aper" in gal_data.keys():
            gal_dat.create_dataset("aperture_mass(msun)", data=gal_data["mass_aper"])
            gal_dat.create_dataset(
                "aperture_sfr(msun.myr^-1)", data=gal_data["sfr_aper"]
            )
            gal_dat.create_dataset("aperture_ssfr(myr^-1)", data=gal_data["ssfr_aper"])
        gal_dat.create_dataset("t(myr)", data=gal_data["time"])
        gal_dat.create_dataset("max age(Gyr)", data=gal_data["age"])
        gal_dat.create_dataset("mass weighted age(Gyr)", data=gal_data["age_wmstar"])

        dat = f.create_group("speectral_data")
        dat.create_dataset("units", data=(lam_units, mag_units))
        dat.create_dataset("lam spectrum", data=wav_micron_full_spec, compression="lzf")
        dat.create_dataset("spectrum", data=mag_full_spec, compression="lzf")
        dat.create_dataset("bands", data=band_names, compression="lzf")
        dat.create_dataset("lam", data=wav_microm, compression="lzf")
        dat.create_dataset("mag", data=mag, compression="lzf")
        dat.create_dataset("mag_err", data=mag_err, compression="lzf")
        dat.create_dataset("mag_rf", data=mag_rf, compression="lzf")
        dat.create_dataset("mag_err_rf", data=mag_err_rf, compression="lzf")


def read_mock_spe(fin, debug=False):

    with h5py.File(fin, "r") as f:
        dhdr = {}
        ddata = {}
        dgaldata = {}

        if debug:
            print(f.keys())

        hdr = f["header"]
        if debug:
            print(hdr.keys())
        for k in hdr.keys():
            if debug:
                print(k)
            dhdr[k] = hdr[k][()]

        dat = f["speectral_data"]
        if debug:
            print(dat.keys())
        for k in dat.keys():
            if debug:
                print(k)

        gal_data = f["galaxy_data"]
        if debug:
            print(gal_data.keys())
        for k in gal_data.keys():
            # if "_sfr(msun" in k:
            # continue
            if debug:
                print(k)
            dgaldata[k] = gal_data[k][()]
            # ddata[k] = dat[k][()]

    return dhdr, ddata, dgaldata
