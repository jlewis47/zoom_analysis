from f90_tools.IO import read_record
import os
import numpy as np
from scipy.interpolate import (
    interp2d,
    LinearNDInterpolator,
    RegularGridInterpolator,
    griddata,
)


def read_st_wind_f(fname):

    with open(fname, "rb") as f:

        nt_SW, nz_SW = read_record(f, 2, np.int32)

        log_tSW = np.zeros(nt_SW, dtype=np.float32)  # log yr
        log_zSW = np.zeros(nz_SW, dtype=np.float32)  # log abs Z
        log_cmSW = np.zeros(
            (nt_SW, nz_SW), dtype=np.float32
        )  # log cumulative mass fraction, per Msun
        log_ceSW = np.zeros(
            (nt_SW, nz_SW), dtype=np.float32
        )  # log cumulative energy fraction, ergs per erg
        log_cmzSW = np.zeros(
            (nt_SW, nz_SW), dtype=np.float32
        )  # log cumulative metal mass fraction, per Msun

        log_tSW[:] = read_record(f, nt_SW, np.float64)

        log_zSW[:] = read_record(f, nz_SW, np.float64)

        for iz in range(nz_SW):
            log_cmSW[:, iz] = read_record(f, nt_SW, np.float64)

        for iz in range(nz_SW):
            log_ceSW[:, iz] = read_record(f, nt_SW, np.float64)

        for iz in range(nz_SW):
            log_cmzSW[:, iz] = read_record(f, nt_SW, np.float64)

    stellar_wind = {}
    stellar_wind["log_tSW"] = log_tSW
    stellar_wind["log_zSW"] = log_zSW
    stellar_wind["log_cmSW"] = log_cmSW
    stellar_wind["log_ceSW"] = log_ceSW
    stellar_wind["log_cmzSW"] = log_cmzSW

    return stellar_wind


# just use interp_2d to get the mass loss rate...
def interp_table(stellar_wind, table="log_cmSW"):

    # return interp2d(
    # stellar_wind["log_tSW"], stellar_wind["log_zSW"], stellar_wind[table].T
    # )

    # print(stellar_wind[table].T.shape)
    # print(stellar_wind["log_tSW"].shape)
    # print(stellar_wind["log_zSW"].shape)

    points = np.meshgrid(stellar_wind["log_zSW"], stellar_wind["log_tSW"])
    point_inds = np.meshgrid(
        np.arange(len(stellar_wind["log_zSW"])), np.arange(len(stellar_wind["log_tSW"]))
    )

    values = np.asarray(stellar_wind[table]).T[
        np.ravel(point_inds[0]), np.ravel(point_inds[1])
    ]

    points = list(zip(np.ravel(points[0]), np.ravel(points[1])))

    # print(points[0].shape, points[1].shape)
    # print(point_inds[0].shape, point_inds[1].shape)
    # print(values.shape)
    # print(points)

    # print(list(zip([points[0].flatten(), points[1].flatten()])))

    # print(len(list(points)))
    # print(values.shape)

    return LinearNDInterpolator(points, values)

    # return RegularGridInterpolator(
    #     # (stellar_wind["log_tSW"], stellar_wind["log_zSW"]),
    #     points,
    #     np.asarray(stellar_wind[table]).T,
    #     bounds_error=False,
    #     fill_value=None,
    # )


def get_mass_loss(ages, Zs, stellar_wind):

    f = interp_table(stellar_wind)

    # check if the input is a scalar or an array
    if np.isscalar(ages):
        ages = np.array([ages])
        Zs = np.array([Zs])

    agemin = 10 ** stellar_wind["log_tSW"].min()
    Zmin = 10 ** stellar_wind["log_zSW"].min()

    ages[ages < agemin] = agemin
    Zs[Zs < Zmin] = Zmin

    agemax = 10 ** stellar_wind["log_tSW"].max()
    Zmax = 10 ** stellar_wind["log_zSW"].max()

    ages[ages > agemax] = agemax
    Zs[Zs > Zmax] = Zmax

    # print(np.log10([ages.min(), ages.max(), ages.mean(), np.median(ages)]))
    # print(np.log10([Zs.min(), Zs.max(), Zs.mean(), np.median(Zs)]))

    log_loss_rates = np.zeros_like(ages)

    # for i, (age, Z) in enumerate(zip(ages, Zs)):

    # print(f(-2, 2))

    log_loss_rates = f(np.log10(Zs), np.log10(ages))

    # print(
    #     log_loss_rates.shape,
    # )

    # print(log_loss_rates[np.isfinite(log_loss_rates)])

    if np.any(~np.isfinite(log_loss_rates)):
        print("Warning: some mass loss rates are not finite")
        print(log_loss_rates[np.isfinite(log_loss_rates)])

    return 10**log_loss_rates
