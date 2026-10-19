# import contextlib
import mmap
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from f90_tools.IO import read_record,skip_record,skip_ndata
from h5tools.h5py_tools import h5_to_dict

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


def find_line_in_file(fname, keyword):

    outs = []

    with open(fname, "r") as src:

        for line in src:

            if keyword in line:

                outs.append(line)

    return outs


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
    return nrg * np.copy(mock["spectrum"])

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

    PC_cm = 3.08e18  # cm

    if type(pfs_path) == str:
        phot_hdr = read_PFS_dump(pfs_path, hdr=True)
    else:
        phot_hdr = pfs_path

    # print(mock["lambda_min"], mock["lambda_max"], mock["lambda_npix"])

    if z > 0:
        dL = cosmo.luminosity_distance(z).to(u.cm).value
    else:
        dL = 10 * PC_cm

    lambda_bins = np.linspace(
        mock["lambda_min"] * (1 + z), mock["lambda_max"] * (1 + z), int(mock["lambda_npix"])
    )  # * (1+z)  # angstrom
    dlamb = np.diff(lambda_bins)[0]  # * 1e-10
    nrg = c * h / (lambda_bins * 1e-10) * 1e7  # erg
    nrg *= phot_hdr["total_flux"] / int(phot_hdr["nphotons"]) / dlamb

    nrg /= 4 * np.pi * dL**2  # to flux at observer
    if z > 0:
        nrg *= 1 + z  # dimming
    # nrg /= aperture_cm**2 * np.pi
    # nrg *= mock["aperture"] ** 2  # to luminisity at aperture
    # nrg /= 1 + z  # dimming

    return nrg * np.copy(mock["spectrum"])
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


def cube_to_erg_s_A_cm2_as2(mock, pfs, z, aper_as2):  # in obs frame if z!=0

    PC_cm = 3.08e18

    if type(pfs) == str:
        phot_hdr = read_PFS_dump(pfs, hdr=True)
    else:
        phot_hdr = pfs

    # print(mock["lambda_min"], mock["lambda_max"], mock["lambda_npix"])
    if z > 0:
        dL = cosmo.luminosity_distance(z).to(u.cm).value  # cm
    else:
        dL = 10 * PC_cm

    # dx_as = np.arctan(dx_cm / dL) * 180 / np.pi * 3600  # as
    dx_as2 = aper_as2 / np.prod(mock["cube"].shape[:2])

    lambs = np.linspace(
        mock["lambda_min"] * (1 + z), mock["lambda_max"] * (1 + z), mock["lambda_npix"]
    )  # angstrom
    dlamb = np.diff(lambs)[0]
    nrg = c * h / (lambs * 1e-10) * 1e7  # erg
    nrg *= phot_hdr["total_flux"] / phot_hdr["nphotons"] / dlamb
    nrg /= dx_as2  # cell solid angle
    nrg /= 4 * np.pi * dL**2  # to flux at observer

    nrg *= (1 + z)  # dimming

    mock["cube"] = nrg * mock["cube"]


def read_mock_dump(fname, ndir=1,debug=False):
    file_type = fname.split(".")[-1]

    mocks = {}

    if type(ndir)==int:
        dir_args = np.arange(ndir+1)
        print(f"Reading all to direction {ndir}")
    else:
        dir_args = [nd for nd in ndir]
        print(f"Reading only directions {ndir}")

    print(dir_args)

    with open(fname, "rb") as src:
    # with open(fname,'rb') as f:
        # with contextlib.closing(mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)) as src:

            for idir in dir_args:

                if debug:print(f"direction:{idir}")

                mock = {}
                mock["type"] = file_type

                if file_type == "flux":
                    pass

                elif file_type == "spectrum":

                    npix = read_record(src, 1, np.int32,debug=debug)
                    if idir not in dir_args:
                        skip_record(src,1,debug=debug)
                        skip_ndata(src,npix,np.float64,debug=debug)
                        continue
                    else:
                        aperture, lambda_min, lambda_max = read_record(src, 3, np.float64,debug=debug)
                        spec = read_record(src, npix, np.float64,debug=debug)

                        mock["aperture"] = aperture
                        mock["lambda_min"] = lambda_min
                        mock["lambda_max"] = lambda_max
                        mock["lambda_npix"] = npix
                        mock["spectrum"] = spec

                elif file_type == "image":
                    img_npix = read_record(src, 1, np.int32,debug=debug)
                    if idir not in dir_args:
                        skip_record(src,1,debug=debug)
                        skip_record(src,1,debug=debug)                        
                        skip_ndata(src,img_npix**2,np.float64,debug=debug)  
                        continue 
                    else:
                        img_side = read_record(src, 1, np.float64,debug=debug)
                        img_ctr = read_record(src, 3, np.float64,debug=debug)
                        img = read_record(src, img_npix * img_npix, np.float64,debug=debug).reshape(
                            img_npix, img_npix
                        )

                        mock["img_npix"] = img_npix
                        mock["img_side"] = img_side
                        mock["img_ctr"] = img_ctr

                        mock["image"] = img

                elif file_type == "cube":
                    lambda_npx, img_npx = read_record(src, 2, np.int32,debug=debug)
                    if idir not in dir_args:
                        skip_record(src,1,debug=debug)
                        skip_record(src,1,debug=debug)                        
                        skip_ndata(src,img_npx**2 * lambda_npx,np.float64,debug=debug)  
                        continue
                    else:
                        lambda_min, lambda_max, cube_side = read_record(src, 3, np.float64,debug=debug)
                        cube_ctr = read_record(src, 3, np.float64,debug=debug)
                        if debug:print(lambda_npx, img_npx, lambda_min, lambda_max, cube_side, cube_ctr)
                        cube = read_record(src, lambda_npx * img_npx * img_npx, np.float64,
                        debug=debug).reshape(img_npx, img_npx, lambda_npx)
                        # cube = np.memmap(fname,dtype=np.float64,offset=64 * (3 + 3 + 2),shape=(img_npx, img_npx, lambda_npx),order='F')
                        # cube = np.memmap(fname,dtype=np.float64,offset=64 * (3 + 3 + 2),shape=(img_npx**2*lambda_npx)).reshape(img_npx, img_npx, lambda_npx)

                        mock["lambda_npix"] = lambda_npx
                        mock["lambda_min"] = lambda_min
                        mock["lambda_max"] = lambda_max

                        mock["img_npix"] = img_npx

                        mock["cube_side"] = cube_side
                        mock["cube_ctr"] = cube_ctr
                        mock["cube"] = cube

                mocks[f"direction_{idir:d}"] = mock

    return mocks


def compress_mock(fout, PFSheader, cubes=None, specs=None):

    assert (
        cubes is not None or specs is not None
    ), "must either give a cubes or spec dict (or both)"

    with h5py.File(fout, "w") as dest:

        dest.create_dataset(
            "nphotons", data=PFSheader["nphotons"]
        )  # essential! used for inferring physical units for mocks
        dest.create_dataset("total_flux", data=PFSheader["total_flux"])

        if specs is not None:

            spec_grp = dest.create_group("spectrum")

            ndir = sum([1 for k in specs.keys() if k.startswith("direction")])

            for idir in range(ndir):

                dir_key = f"direction_{idir}"

                if idir == 0:
                    spec_grp.create_dataset("aperture", data=specs[dir_key]["aperture"])
                    spec_grp.create_dataset(
                        "lambda_min", data=specs[dir_key]["lambda_min"]
                    )
                    spec_grp.create_dataset(
                        "lambda_max", data=specs[dir_key]["lambda_max"]
                    )
                    spec_grp.create_dataset(
                        "lambda_npix", data=specs[dir_key]["lambda_npix"]
                    )

                spec_grp.create_dataset(
                    f"spectrum_{idir}",
                    data=specs[dir_key]["spectrum"],
                    compression="lzf",
                )

        if cubes is not None:

            cube_grp = dest.create_group("cube")

            ndir = sum([1 for k in cubes.keys() if k.startswith("direction")])

            for idir in range(ndir):

                dir_key = f"direction_{idir}"

                if idir == 0:
                    cube_grp.create_dataset(
                        "cube_side", data=cubes[dir_key]["cube_side"]
                    )
                    cube_grp.create_dataset("cube_ctr", data=cubes[dir_key]["cube_ctr"])
                    cube_grp.create_dataset("img_npix", data=cubes[dir_key]["img_npix"])
                    cube_grp.create_dataset(
                        "lambda_min", data=cubes[dir_key]["lambda_min"]
                    )
                    cube_grp.create_dataset(
                        "lambda_max", data=cubes[dir_key]["lambda_max"]
                    )
                    cube_grp.create_dataset(
                        "lambda_npix", data=cubes[dir_key]["lambda_npix"]
                    )

                cube_grp.create_dataset(
                    f"cube_{idir}",
                    data=cubes[dir_key]["cube"],
                    compression="lzf",
                )

    return 1


# def read_compress_mock(fname,dtype=np.float32):
def read_compress_mock(fname,ndirs=None,dtype=None):

    read_mocks = h5_to_dict(fname,dtype=dtype)
    # print(fname, fname.split("/")[:-1])
    rascas_dir = "/".join(fname.split("/")[:-1])
    # print(rascas_dir)

    # dicts going out
    mocks_spe = {}
    mocks_cube = {}
    pfs_header = {}

    ndir = len(
        [
            k
            for k in read_mocks["cube"].keys()
            if k.startswith("cube") and "side" not in k and "ctr" not in k
        ]
    )

    if type(ndirs)==int:
        ndirs=[ndirs]

    # to old format
    for idir in range(ndir):

        if ndirs is not None:
            if idir not in ndirs:
                print(f'skipping direction {idir}')
                continue

        dirkey = f"direction_{idir}"

        if "spectrum" in read_mocks.keys():
            mocks_spe[dirkey] = {}
            for k in ["aperture", "lambda_min", "lambda_max", "lambda_npix"]:
                mocks_spe[dirkey][k] = read_mocks["spectrum"][k]
            mocks_spe[dirkey]["spectrum"] = read_mocks["spectrum"][f"spectrum_{idir}"]
        else:
            print("no spectrum entries found in compressed mock file")

        if "cube" in read_mocks.keys():
            mocks_cube[dirkey] = {}
            for k in [
                "cube_side",
                "cube_ctr",
                "img_npix",
                "lambda_min",
                "lambda_max",
                "lambda_npix",
            ]:
                mocks_cube[dirkey][k] = read_mocks["cube"][k]
            mocks_cube[dirkey]["cube"] = read_mocks["cube"][f"cube_{idir}"]
        else:
            print("no cube entries found in compressed mock file")

    for kpfs in ["nphotons", "total_flux"]:
        if kpfs in read_mocks.keys():
            pfs_header[kpfs] = read_mocks[kpfs]

    #handle old cases with missing nphotons and total luminosity
    if pfs_header == {}:

        tot_flux_line = find_line_in_file(
            os.path.join(rascas_dir, "log_PFS"), "Total luminosity"
        )[0]
        # print(tot_flux_line)
        pfs_header["total_flux"] = float(tot_flux_line.split(":")[-1])

        nphot_line = find_line_in_file(
            os.path.join(rascas_dir, "params_PFS.cfg"), "nPhotonPackets"
        )[0]
        # print(nphot_line)
        pfs_header["nphotons"] = float(nphot_line.split("=")[-1])

    return mocks_spe, mocks_cube, pfs_header


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
