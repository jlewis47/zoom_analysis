import numpy as np
import os
from matplotlib import gridspec, pyplot as plt
from gremlin.read_sim_params import ramses_sim
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.cosmology import z_at_value
from scipy.stats import binned_statistic

# from f90_tools.IO import read_record

from zoom_analysis.stars.stellar_winds_reader import (
    get_mass_loss,
    # interp_table,
    read_st_wind_f,
)
from zoom_analysis.halo_maker.read_treebricks import read_gal_stars, convert_star_units

# from scipy.spatial import cKDTree


def plot_sf_stuff(
    axs, sfh, sfr, ssfr, t, t_err, cosmo, color=None, label="", ticks=False, **kwargs
):
    """Plot a star formation history

     Parameters
     ----------
     ax : matplotlib.axes.Axes
         The axes to plot on.
     sfh : numpy.ndarray
         The star formation history.
    sfr : numpy.ndarray
         The star formation rate.
     ssfr : numpy.ndarray
         The specific star formation rate.
     t : numpy.ndarray
         The time steps.
     t_err : numpy.ndarray
         The error on the time steps.
     color : str, optional
         The color to plot with.
     label : str, optional
         The label for the plot.

     Returns
     -------
     None.

    """

    datas = [ssfr, sfr, sfh]
    # print(sfh / sfh.max())
    # xind = np.where(sfh / sfh.max() > 0.33)[0][-1]
    # print(xind)

    non_nul = sfh > 0

    for ax, data in zip(axs, datas):
        l = ax.errorbar(
            t[non_nul] * 1e-3,
            data[non_nul],
            xerr=t_err,
            color=color,
            label=label,
            **kwargs,
        )
        # print(data.min(), data.max())
        # if ticks:
        # ax.set_ylim(data[xind] * 0.95, data.max() * 1.05)
    # ax.set_ylim(np.min(datas) * 0.5, np.max(datas) * 1.5)

    if ticks:
        # xlim to get at least 75% of the mass
        # print(sfh / sfh.max())
        # print(sfh[xind], sfh.max())
        # axs[0].set_xlim(t[xind] * 1e-3 * 0.95, t.max() * 1e-3 * 1.01)
        # print(xind)
        # print(xind, len(sfh)), print(t[xind] * 1e-3 * 0.95, t.max() * 1e-3 * 1.05)

        # twin x axis for redshift
        ax2 = axs[0].twiny()
        ax2.set_xlabel("Redshift")
        ax2.set_xlim(axs[0].get_xlim())
        # ax2.set_xticks(ax.get_xticks())
        xticks = axs[0].get_xticks()
        # print(xticks)
        zticks = [z_at_value(cosmo.age, (x + 1e-3) * u.Gyr).value for x in xticks]
        ax2.set_xticklabels(["{:.2f}".format(z) for z in zticks])

        axs[0].tick_params(
            axis="x",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
            labeltop=False,
        )
        axs[1].tick_params(
            axis="x",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
            labeltop=False,
        )
        axs[2].tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=False,
            labelbottom=True,
            labeltop=False,
        )

    # axs[0].set_xtick_labels([])
    # axs[1].set_xtick_labels([])

    return l


def setup_sfh_plot():
    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    fig = plt.figure(figsize=(8, 8))

    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 3, 3], hspace=0.0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    axs3 = fig.add_subplot(gs[2], sharex=ax1)
    axs = [ax1, ax2, axs3]

    axs[-1].set_xlabel("Time (Gyr)")
    axs[0].set_ylabel("sSFR (Myr$^{-1}$)")
    axs[1].set_ylabel("SFR (M$_\odot$/Myr)")
    axs[2].set_ylabel("Stellar Mass (M$_\odot$)")

    # ax.set_xscale("log")

    for ax in axs[:]:
        ax.set_yscale("log")
        # ax.grid()

    return (fig, axs)


def get_sf_stuff(starlist, z, sim: ramses_sim, deltaT=100, debug=False):

    if not hasattr(sim, "cosmo_model"):
        sim.init_cosmo()

    cosmo = sim.cosmo_model

    try:
        ages = starlist["agepart"]  # Myr
    except KeyError:
        ages = starlist["age"]

    try:
        masses = starlist["mpart"]  # Msun
    except KeyError:
        masses = starlist["mass"]

    try:
        Zs = starlist["Zpart"]
    except KeyError:
        Zs = starlist["metallicity"]

    # print(ages, masses)

    masses = correct_mass(sim, ages, masses, Zs, debug=debug)

    time_steps = deltaT  # Myr

    time_snap = cosmo.age(z).value * 1e3  # Myr

    # print(time_snap, ages.max())

    t = np.arange(0, ages.max() + time_steps, time_steps)

    # print(t)

    # print(time_snap)

    bin_masses = binned_statistic(ages, masses, statistic="sum", bins=t)[0]

    sfh = np.cumsum(bin_masses)[::-1]  # Msun

    # print(bin_masses, time_steps)

    sfr = bin_masses / time_steps  # Msun/Myr

    ssfr = sfr / sfh  # Myr^-1

    if debug:
        print(ages.min(), ages.max())  # , time_steps)
        print(sfh, sfr, ssfr)

    return sfh, sfr, ssfr, time_snap - t[:-1], cosmo


def correct_mass(sim, ages, masses, Zs, debug=False):
    if "physics_params" in sim.namelist:
        if "t_delay" not in sim.namelist["physics_params"]:
            t_delay = 10  # Myr
        else:
            t_delay = float(sim.namelist["physics_params"]["t_delay"])
        cond = ages > t_delay

        if "eta_sn" not in sim.namelist["physics_params"]:
            eta_sn = 0.05
        else:
            eta_sn = float(sim.namelist["physics_params"]["eta_sn"])

        masses[cond] = masses[cond] / (1.0 - eta_sn)

    else:
        fname_winds = os.path.join(
            sim.path, sim.namelist["feedback_params"]["stellar_winds_file"]
        )
        if not os.path.exists(fname_winds):
            fname_winds = "./ramses_swind_Sikey.dat"
            if debug:
                print(
                    f"didn't find winds file at location given by .nml, assuming local file:{fname_winds}"
                )

        st_wind_tab = read_st_wind_f(fname_winds)

        mass_loss = get_mass_loss(ages * 1e6, Zs, st_wind_tab)
        # print(list(zip(ages, Zs, mass_loss)))
        # print(mass_loss.min(), mass_loss.max(), mass_loss.mean())
        masses = masses / (1.0 - mass_loss)
    return masses


def get_gal_sfh(hm, sim, snap, gid):

    z = 1.0 / sim.get_snap_exps(snap, param_save=False) - 1.0

    fstar = os.path.join(sim.path, hm, f"GAL_{snap:05d}", f"gal_stars_{gid:07d}")
    # print(fstar)
    stars = read_gal_stars(fstar)
    convert_star_units(stars, snap, sim)

    sfh, sfr, ssfr, t_sfr, _ = get_sf_stuff(stars, z, sim, deltaT=10.0)
    return sfh, sfr, ssfr, t_sfr
