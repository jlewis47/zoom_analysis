import numpy as np
from astropy.io import fits


def dumb_constant_err(flux, val=0.01):

    return val * flux


def dumb_constant_mag(mag, val=0.01):

    flux_err = 10 ** (mag / -2.5) * val

    return -2.5 * np.log10(flux_err)


###### from CL ######
def AddErrorOnFlux(mag, depth, fact_err, nsig):
    """
    f is the filter name
    depth, fact_err, nsig are dictionnaries which give the depth at nsig sigma, and fact_err is the square root of the inverse of the gain

    it returns the perturbed flux, flux error, perturbed magnitudes, magnitudes errors
    """
    # seed = np.random.seed(0)

    onesig = depth + 2.5 * np.log10(nsig)
    onesig = 10 ** ((-onesig + 23.9) / 2.5)

    flux = 10 ** ((-mag + 23.9) / 2.5)  # jl: ab mag to micro jansky

    fluxerr = np.sqrt(
        np.random.normal(loc=onesig, scale=onesig / 4.0, size=len(mag)) ** 2
    )  # jl: sqrt of ^2 ?????

    # joe clamp fluxerr to flux so no negative fluxes
    fluxerr_max = np.clip(fluxerr, -flux, flux)
    # fluxerr_max = fluxerr

    newflux = np.random.normal(loc=flux, scale=fluxerr_max, size=len(mag))

    # print(list(zip(flux, fluxerr, newflux)))
    fluxerr = np.sqrt((fluxerr) ** 2 + flux * fact_err**2)

    newmag = -2.5 * np.log10(newflux) + 23.9
    magerr = 1.086 * np.abs(fluxerr / newflux)
    return newflux, fluxerr, newmag, magerr


###### from CL ######
def estimate_pseudogain(flux, flux_err):
    w = np.where((flux_err > 0) * (flux > 20) * (flux < 600))[0]

    gain = np.mean(flux_err[w] ** 2 / flux[w])
    return 1.0 / gain


###### from CL ######
def cosmos_stuff(lephare_bands):

    lephare_to_err_bands = {
        "f814_res": "HSTF814W",
        "F444": "F444W",
        "F115": "F115W",
        "F150": "F150W",
        "F277": "F277W",
        "F770": "F770W",
        "u_new": "u",
        "gHSC": "gHSC",
        "rHSC": "rHSC",
        "iHSC": "iHSC",
        "zHSC": "zHSC",
        "yHSC": "yHSC",
        "Y": "Y",
        "K": "Ks",
        "H": "H",
        "J": "J",
        "IB427": "IB427",
        "IB467": "IB467",
        "IB484": "IB484",
        "IB505": "IB505",
        "IB527": "IB527",
        "IB574": "IB574",
        "IB624": "IB624",
        "IB679": "IB679",
        "IB709": "IB709",
        "IB738": "IB738",
        "IB767": "IB767",
        "IB827": "IB827",
        "NB711": "NB711",
        "NB816": "NB816",
        "B": "B",
        "V": "V",
        "r": "r",
        "ip": "ip",
        "zpp": "zpp",
        "ch1": "ch1",
        "ch2": "ch2",
        "NUV": "NUV",
        "FUV": "FUV",
    }

    bands = [lephare_to_err_bands[lephare_band] for lephare_band in lephare_bands]

    fact_err = {
        "HSTF814W": 0.01,
        "F444W": 0.01,
        "F115W": 0.01,
        "F150W": 0.01,
        "F277W": 0.01,
        "F770W": 0.01,
        "u": 0.0021564078,
        "gHSC": 0.012513637,
        "rHSC": 0.012198602,
        "iHSC": 0.010987033,
        "zHSC": 0.012856143,
        "yHSC": 0.0153,
        "Y": 0.01,
        "Ks": 0.01,
        "H": 0.01,
        "J": 0.01,
        "IB427": 0.010,
        "IB467": 0.010,
        "IB484": 0.010,
        "IB505": 0.010,
        "IB527": 0.010,
        "IB574": 0.010,
        "IB624": 0.010,
        "IB679": 0.010,
        "IB709": 0.010,
        "IB738": 0.010,
        "IB767": 0.010,
        "IB827": 0.010,
        "NB711": 0.010,
        "NB816": 0.010,
        "B": 0.0046,
        "V": 0.006,
        "r": 0.006,
        "ip": 0.006,
        "zpp": 0.006,
        "ch1": 0.006,
        "ch2": 0.006,
        "NUV": 0.006,
        "FUV": 0.006,
    }
    depth = {
        "HSTF814W": 27.8,
        "F444W": 28.17,
        "F115W": 27.45,
        "F150W": 27.66,
        "F277W": 28.28,
        "F770W": 24,
        "u": 27,
        "gHSC": 28.1,
        "rHSC": 27.8,
        "iHSC": 27.6,
        "zHSC": 27.2,
        "yHSC": 26.5,
        "Y": 25.3,
        "Ks": 25.3,
        "H": 24.9,
        "J": 25.2,
        "IB427": 26.1,
        "IB467": 25.6,
        "IB484": 26.5,
        "IB505": 26.1,
        "IB527": 26.4,
        "IB574": 25.8,
        "IB624": 26.4,
        "IB679": 25.6,
        "IB709": 25.9,
        "IB738": 26.1,
        "IB767": 25.6,
        "IB827": 25.6,
        "NB711": 25.5,
        "NB816": 25.6,
        "B": 27.8,
        "V": 26.8,
        "r": 27.1,
        "ip": 26.7,
        "zpp": 26.3,
        "ch1": 26.4,
        "ch2": 26.3,
        "NUV": 26,
        "FUV": 26,
    }
    nsig = {
        "HSTF814W": 3,
        "F444W": 5,
        "F115W": 5,
        "F150W": 5,
        "F277W": 5,
        "F770W": 5,
        "u": 3,
        "gHSC": 3,
        "rHSC": 3,
        "iHSC": 3,
        "zHSC": 3,
        "yHSC": 3,
        "Y": 3,
        "Ks": 3,
        "H": 3,
        "J": 3,
        "IB427": 3,
        "IB467": 3,
        "IB484": 3,
        "IB505": 3,
        "IB527": 3,
        "IB574": 3,
        "IB624": 3,
        "IB679": 3,
        "IB709": 3,
        "IB738": 3,
        "IB767": 3,
        "IB827": 3,
        "NB711": 3,
        "NB816": 3,
        "B": 3,
        "V": 3,
        "r": 3,
        "ip": 3,
        "zpp": 3,
        "ch1": 3,
        "ch2": 3,
        "NUV": 3,
        "FUV": 3,
    }

    return (
        np.asarray([fact_err[band] for band in bands]),
        np.asarray([depth[band] for band in bands]),
        np.asarray([nsig[band] for band in bands]),
    )


# def cosmos_fluxes(band):

# f = fits.open("/data101/jlewis/COSMOS2020/CATALOGS/COSMOS2020_v1.8_formatted.fits")
# cat20 = f[1].data

# ### estimate gain in a given COSMOS band
# # go from band name to fits column name
# str_u = "CFHT_" + "u"
# str_u = "GALEX_NUV"

# str_mag = "_MAG"
# str_flux = "_FLUX"
# str_fluxerr = "_FLUXERR"
# str_magerr = "_MAGERR"
# str_aper = ""  #'_APER2'

# flux_u = cat20.field(str_u + str_flux + str_aper)
# fluxerr_u = cat20.field(str_u + str_fluxerr + str_aper)
# mag_u = cat20.field(str_u + str_mag + str_aper)
# magerr_u = cat20.field(str_u + str_magerr + str_aper)

# return flux_u, fluxerr_u  # , mag_u, magerr_u


def get_cl_err(mags_sim, bands):

    # flux, flux_err = cosmos_fluxes(f)

    # fact_err = np.sqrt(1.0 / estimate_pseudogain(flux, flux_err))

    fact_err, depth, nsig = cosmos_stuff(bands)

    newflux, newflux_err, newmag, newmag_err = AddErrorOnFlux(
        mags_sim, depth, fact_err, nsig
    )

    return newmag, newmag_err, newflux, newflux_err
