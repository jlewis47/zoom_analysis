import numpy as np
import os
from matplotlib import gridspec, pyplot as plt
from gremlin.read_sim_params import ramses_sim
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.cosmology import z_at_value
from zoom_analysis.sinks.sink_reader import find_massive_sink, get_sink_mhistory
from zoom_analysis.constants import ramses_pc

from hagn.utils import get_hagn_sim
from hagn.catalogues import make_super_cat, get_cat_hids
from hagn.tree_reader import read_tree_rev

from scipy.stats import binned_statistic


def plot_bh_stuff(
    axs,
    bhm,
    mdot,
    tsink,
    fmerger,
    tmerge,
    t_err,
    cosmo,
    color=None,
    label="",
    ticks=False,
    **kwargs
):
    """Plot a BH history

     Parameters
     ----------
     ax : matplotlib.axes.Axes
         The axes to plot on.
     bhm : numpy.ndarray
         The black hole mass history.
    mdot : numpy.ndarray
         BH bondi accretion rate.
    tsink; numpy.ndarray
            The time steps.
     fmerger : numpy.ndarray
         mass fraction of main branch.
     tmerge : numpy.ndarray
         The time steps for merger mass.
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

    datas = [fmerger, np.abs(mdot), bhm]
    xdatas = [tmerge, tsink, tsink]
    dss = ["steps-mid", "default", "default"]
    # print(bhm / bhm.max())
    # xind = np.where(bhm / bhm.max() > 0.33)[0][-1]
    # print(xind)

    for ax, xdata, data, ds in zip(axs, xdatas, datas, dss):
        l = ax.errorbar(
            xdata * 1e-3, data, xerr=t_err, color=color, label=label, ds=ds, **kwargs
        )
        # print(data.min(), data.max())
        # if ticks:
        # ax.set_ylim(data[xind] * 0.95, data.max() * 1.05)
    # ax.set_ylim(np.min(datas) * 0.5, np.max(datas) * 1.5)

    if ticks:
        # xlim to get at least 75% of the mass
        # print(bhm / bhm.max())
        # print(bhm[xind], bhm.max())
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


def setup_bh_plot():
    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    fig = plt.figure(figsize=(8, 8))

    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 3, 3], hspace=0.0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    axs3 = fig.add_subplot(gs[2], sharex=ax1)
    axs = [ax1, ax2, axs3]

    axs[-1].set_xlabel("Time (Gyr)")
    axs[0].set_ylabel("$\mathrm{f_{merger}}$")
    axs[1].set_ylabel(r"$\mathrm{\dot{m_{BH}}}$ (M$_\odot$/yr)")
    axs[2].set_ylabel("Black Hole Mass (M$_\odot$)")

    # ax.set_xscale("log")

    # for ax in axs[:]:
    # ax.set_yscale("log")
    # ax.grid()
    axs[0].set_ylim(50, 100)
    axs[1].set_yscale("log")
    axs[2].set_yscale("log")

    return (fig, axs)


def get_bh_stuff(sim, hid, tgt_zed):

    hagn_sim = get_hagn_sim()

    tgt_snap = hagn_sim.get_closest_snap(zed=tgt_zed)

    super_cat = make_super_cat(tgt_snap, "hagn")  # , overwrite=True)

    # print(tgt_zed, tgt_snap)

    l_hagn = hagn_sim.unit_l(hagn_sim.aexp_stt) / (ramses_pc * 1e6) / hagn_sim.aexp_stt

    # sim_path = sim.path
    # sim_name = sim.name

    gal_pties = get_cat_hids(super_cat, [hid])

    snap = sim.get_closest_snap(zed=tgt_zed)
    cur_aexp = sim.get_snap_exps(snap)

    # pos = np.asarray(
    #     [
    #         sim.namelist["refine_params"]["xzoom"],
    #         sim.namelist["refine_params"]["yzoom"],
    #         sim.namelist["refine_params"]["zzoom"],
    #     ]
    # )

    hagn_tree_hids, hagn_tree_datas, hagn_tree_aexps = read_tree_rev(
        tgt_zed,
        gal_pties["gid"],
        tree_type="gal",
        target_fields=["m", "x", "y", "z", "r", "m_father", "nb_father"],
    )

    # aexp = 1.0 / (tgt_zed + 1.0)

    # cur_snap = sim.snaps[-1]
    # cur_aexp = sim.aexps[-1]

    # print(hagn_tree_datas["x"])
    if len(hagn_tree_datas["x"]) == 0:
        return -1, -1, -1, -1

    # get rid of bad steps
    # print(hagn_tree_hids)
    # print(hagn_tree_hids, hagn_tree_datas, hagn_tree_aexps)

    # good_steps = np.where(hagn_tree_hids != -1)[0]
    # hagn_tree_hids = [hagn_tree_hids[0][good_steps]]
    # hagn_tree_datas = {k: [v[0][good_steps]] for k, v in hagn_tree_datas.items()}
    # hagn_tree_aexps = [hagn_tree_aexps[good_steps]]

    # print(hagn_tree_hids)

    # # get hagn main branch galaxy id at this time
    tree_arg = np.argmin(np.abs(hagn_tree_aexps - cur_aexp))

    # hagn_main_branch_id = hagn_tree_hids[np.argmin(np.abs(hagn_tree_aexps - cur_aexp))]
    hagn_ctr = np.asarray(
        [
            hagn_tree_datas["x"][0][tree_arg],
            hagn_tree_datas["y"][0][tree_arg],
            hagn_tree_datas["z"][0][tree_arg],
        ]
    )

    hagn_ctr += 0.5 * (l_hagn * cur_aexp)
    hagn_ctr /= l_hagn * cur_aexp

    hagn_rvir = hagn_tree_datas["r"][0][tree_arg] / (l_hagn * cur_aexp)  # * 10

    print(hagn_rvir, l_hagn)
    print(hagn_ctr)
    print("%.1e" % hagn_tree_datas["m"][0][tree_arg])

    # try:
    massive_sink = find_massive_sink(
        # pos, snap, sim, rmax=sim.namelist["refine_params"]["rzoom"]
        # pos,
        hagn_ctr,
        snap,
        sim,
        # rmax=0.3 * sim.namelist["refine_params"]["rzoom"],
        rmax=hagn_rvir,
    )
    # except ValueError:

    #     print("didn't find a massive sink withing hagn halo")
    #     return -1, -1, -1, -1

    sid = massive_sink["identity"]

    # print(sid)

    central_sink = get_sink_mhistory(
        sid,
        snap,
        sim,
    )

    fmerger = hagn_tree_datas["m_father"][0]  # / hagn_tree_datas["nb_father"][0]

    zeds_sinks = central_sink["zeds"]

    zeds_tree = 1.0 / hagn_tree_aexps - 1

    if not hasattr(hagn_sim, "cosmo_model"):
        sim.init_cosmo()

    time_sinks = sim.cosmo_model.age(zeds_sinks).value * 1e3  # Myr
    time_tree = sim.cosmo_model.age(zeds_tree).value * 1e3  # Myr

    tgt_DT_sinks = 50  # Myr
    bin_size_sinks = np.median(np.abs(np.diff(time_sinks)))

    kern_size_sinks = int(tgt_DT_sinks / bin_size_sinks)

    # print(kern_size)

    # smooth with flat convolution
    binned_sinks_mass = np.convolve(
        central_sink["mass"], np.ones(kern_size_sinks) / kern_size_sinks, mode="same"
    )
    binned_sinks_dmbh = np.convolve(
        np.abs(central_sink["dMBH_coarse"]),
        np.ones(kern_size_sinks) / kern_size_sinks,
        mode="same",
    )

    bin_size_tree = np.median(np.abs(np.diff(time_tree)))
    kern_size_tree = int(tgt_DT_sinks / bin_size_tree)

    # print(bin_size_tree, kern_size_tree)

    binned_fmerger = np.convolve(
        fmerger, np.ones(kern_size_tree) / kern_size_tree, mode="same"
    )

    # tgt_bins = np.arange(0, time_tree.max() + tgt_DT_sinks, tgt_DT_sinks)

    # # rebin sinks to lower resolution a bit... take mean
    # binned_sinks_mass = binned_statistic(
    #     time_sinks, central_sink["mass"], bins=tgt_bins, statistic=np.nanmean
    # )[0]
    # binned_sinks_dmbh = binned_statistic(
    #     time_sinks,
    #     np.abs(central_sink["dMBH_coarse"]),
    #     bins=tgt_bins,
    #     statistic=np.nanmean,
    # )[0]

    # print(time_sinks, time_tree)

    # print(np.mean(time_sinks), np.mean(time_tree))

    # print(central_sink["mass"], central_sink["dMBH_coarse"])

    # linearly interpolate to get mass and mdot at same times as time_tree

    # args = np.digitize(time_tree, time_sinks)
    # args[args > len(time_sinks) - 1] = len(time_sinks) - 1

    # print(args)

    # binned_mbh = central_sink["mass"][args]
    # binned_dmbh_dt = central_sink["dMBH_coarse"][args]

    # binned_dmbh_dt = np.interp(time_tree, time_sinks, central_sink["dMBH_coarse"])

    # print(time_tree, time_sinks)

    # print(list(zip(time_tree, [time_sinks[arg] for arg in args], args)))

    # print(central_sink["mass"], binned_mbh)

    # print(binned_fmerger[::kern_size_tree], time_tree[::kern_size_tree])

    return (
        # central_sink["mass"],
        # central_sink["dMBH_coarse"],
        # time_sinks,
        binned_sinks_mass[::kern_size_sinks],
        binned_sinks_dmbh[::kern_size_sinks],
        time_sinks[::kern_size_sinks],
        # fmerger,
        # time_tree * 1e3,
        binned_fmerger[::kern_size_tree],
        time_tree[::kern_size_tree],
    )
