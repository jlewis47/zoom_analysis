import os
from turtle import pos
from astropy.units.cgs import C
import numpy as np
import matplotlib.pyplot as plt

from zoom_analysis.zoom_helpers import project_direction


def mass_dot_product(v1, v2):
    """
    V1,V2 are arrays of N vectors of dimension M
    return N dot products
    """

    return np.sum(v1 * v2, axis=1)


def compute_ang_mom(m_stars, pos_stars, vel_stars, pos_gal):
    # Compute the angular momentum of the galaxy
    # Input: m_stars, pos_stars, vel_stars, pos_gal
    # Output: ang_mom vector
    # pos_gal is the position of the galaxy
    # pos_stars is the position of the stars
    # vel_stars is the velocity of the stars
    # m_stars is the mass of the stars
    # ang_mom is the angular momentum of the galaxy
    # ang_mom = sum_i m_i (r_i - r_gal) x v_i
    # where r_i is the position of the star, r_gal is the position of the galaxy, and v_i is the velocity of the star
    # x is the cross product
    # sum_i is the sum over all stars

    ctr_star_pos = pos_stars - pos_gal
    # ctr_vel_stars = vel_stars - np.average(vel_stars, axis=0, weights=m_stars)

    ang_mom = np.sum(
        np.cross(ctr_star_pos, vel_stars) * m_stars[:, np.newaxis],
        axis=0,
        # np.cross(ctr_star_pos, m_stars[:, np.newaxis] * ctr_vel_stars),
        # axis=0,
    )

    return ang_mom


def project_vels(m_stars, pos_stars, vel_stars, ang_mom, debug=False):
    """
    pos_stars is the position of the stars in the galaxy frame
    """

    vbulk = np.average(vel_stars, axis=0, weights=m_stars)
    bulkless_vel = vel_stars - vbulk

    # project vel_stars into ang_mom basis

    l_ang_mom = np.linalg.norm(ang_mom)
    ang_mom_norm = ang_mom / l_ang_mom

    R = project_direction(ang_mom_norm, [0, 0, 1])

    new_z = ang_mom_norm
    new_x = np.dot(R, [1, 0, 0])
    new_x = new_x / np.linalg.norm(new_x)
    new_y = np.dot(R, [0, 1, 0])
    new_y = new_y / np.linalg.norm(new_y)

    new_xs = np.dot(pos_stars, new_x)
    if debug:
        print("new_xs", new_xs)
    new_ys = np.dot(pos_stars, new_y)
    if debug:
        print("new_ys", new_ys)

    new_vels_xs = np.dot(bulkless_vel, b=new_x)
    new_vels_ys = np.dot(bulkless_vel, new_y)

    new_rhos = np.asarray([new_xs, new_ys, np.zeros_like(new_xs)]).T
    if debug:
        print("new_rhos", new_rhos)
    new_rhos[new_rhos < 1e-10] = 1e-10
    new_rhos /= np.linalg.norm(new_rhos, axis=1)[:, np.newaxis]
    new_phis = np.cross(new_rhos, new_z[np.newaxis, :], axisa=1, axisb=1)
    if debug:
        print("new_phis", new_phis)
    new_phis /= np.linalg.norm(new_phis, axis=1)[:, np.newaxis]

    if debug:
        print(R)
        print(new_rhos, new_phis, new_z)

    # norm_yz = np.linalg.norm(ang_mom[1:])
    # norm_xz = np.linalg.norm(ang_mom[[0, 2]])

    # get rotation matrix to go from cartesian basis to ang_mom basis
    # ang1 = np.arccos(norm_yz / ang_mom_norm[1])  # between L and y
    # ang2 = np.arccos(norm_xz / ang_mom_norm[0])  # between L and x

    # c1 = norm_yz / ang_mom[1]  # np.cos(ang1)
    # s1 = norm_yz / ang_mom[2]
    # s1 = np.sin(ang1)
    # c2 = norm_xz / ang_mom[0]  # np.cos(ang2)
    # s2 = norm_xz / ang_mom[2]
    # s2 = np.sin(ang2)

    # # c1 = l_ang_mom / ang_mom[1]
    # # c2 = l_ang_mom / ang_mom[0]
    # # s1

    # R1 = [[1, 0, 0], [0, c1, -s1], [0, s1, c1]]
    # R2 = [[c2, 0, +s2], [0, 1, 0], [-s2, 0, c2]]

    # R = [[c2, 0, -s2], [-s1 * s2, c1, -s1 * c2], [c1 * s2, s1, c1 * c2]]

    # # # rho vector is position vector from gal ctr for each star
    # new_rho = pos_stars
    # # # normalise vectors
    # new_rho /= np.linalg.norm(new_rho)

    # new_phi = np.cross(new_z[np.newaxis, :], new_rho, axis=1)
    # # normalise vectors
    # new_phi /= np.linalg.norm(new_phi)

    # compute bulk motion

    # print(new_rho, new_phi, new_z)

    # project velocities
    # vel_rad = np.dot(bulkless_vel, new_rho)
    # vel_tan = np.dot(bulkless_vel, new_phi)
    # vel_vert = np.dot(bulkless_vel, new_z[np.newaxis, :])
    vel_rad = mass_dot_product(bulkless_vel, new_rhos)
    vel_tan = mass_dot_product(bulkless_vel, new_phis)
    vel_vert = mass_dot_product(bulkless_vel, new_z[np.newaxis, :])

    # new_vels = np.array([vel_rad, vel_tan, vel_vert]).T

    pos_rad = mass_dot_product(pos_stars, new_rhos)
    pos_tan = mass_dot_product(pos_stars, new_phis)
    pos_vert = mass_dot_product(pos_stars, new_z[np.newaxis, :])

    # new_pos = np.array([pos_rad, pos_tan, pos_vert]).T

    # if m_stars.sum() > 1e11:
    #     # if (
    #     #     np.abs(np.average(vel_tan, weights=m_stars))
    #     #     > 5
    #     #     * np.sqrt(
    #     #         (
    #     #             weighted_variance(vel_tan, m_stars)
    #     #             + weighted_variance(vel_rad, m_stars)
    #     #             + weighted_variance(vel_vert, m_stars)
    #     #         )
    #     #         / 3.0
    #     #     )
    #     #     and len(m_stars) > 2500
    #     # ):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")

    # # test the vectors
    # ax.quiver(0, 0, 0, new_z[0], new_z[1], new_z[2], color="r", label="z")
    # ax.quiver(0, 0, 0, new_rho[0], new_rho[1], new_rho[2], color="g", label="rho")
    # ax.quiver(0, 0, 0, new_phi[0], new_phi[1], new_phi[2], color="b", label="phi")

    # # plot angular momentum
    # ax.quiver(
    #     0, 0, 0, ang_mom[0], ang_mom[1], ang_mom[2], color="k", label="ang_mom"
    # )

    # ax.scatter(
    #     pos_stars[::1, 0] - pos_stars[:, 0].mean(),
    #     pos_stars[::1, 1] - pos_stars[:, 1].mean(),
    #     pos_stars[::1, 2] - pos_stars[:, 2].mean(),
    #     color="c",
    #     label="stars",
    # )

    # plot in new rho, new phi plane
    # # if m_stars.sum() > 1e10:
    # fig, ax = plt.subplots(3, 1)

    # xy_vel = np.sqrt(bulkless_vel[:, 0] ** 2 + bulkless_vel[:, 1] ** 2)
    # ax[0].scatter(
    #     pos_stars[:, 0],
    #     pos_stars[:, 1],
    #     s=1,
    #     alpha=0.5,
    #     label="stars",
    #     c=np.log10(xy_vel),
    # )
    # xz_vel = np.sqrt(bulkless_vel[:, 0] ** 2 + bulkless_vel[:, 2] ** 2)
    # ax[1].scatter(
    #     pos_stars[:, 0],
    #     pos_stars[:, 2],
    #     s=1,
    #     alpha=0.5,
    #     label="stars",
    #     c=np.log10(xz_vel),
    # )
    # yz_vel = np.sqrt(bulkless_vel[:, 1] ** 2 + bulkless_vel[:, 2] ** 2)
    # ax[2].scatter(
    #     pos_stars[:, 1],
    #     pos_stars[:, 2],
    #     s=1,
    #     alpha=0.5,
    #     label="stars",
    #     c=np.log10(yz_vel),
    # )

    # ax[-1].colorbar = plt.colorbar(ax[-1].collections[0], ax=ax, orientation="vertical")

    # ax[0].quiver(0, 0, ang_mom[0], ang_mom[1], color="k", label="ang_mom")
    # ax[1].quiver(0, 0, ang_mom[0], ang_mom[2], color="k", label="ang_mom")
    # ax[2].quiver(0, 0, ang_mom[1], ang_mom[2], color="k", label="ang_mom")

    # ax[0].text(0, 0, f"vrot: {np.average(vel_tan,weights=m_stars):.1f} km/s", color="k")

    # ax[0].legend()
    # fig.savefig("tan_velocity")

    # input("press enter to continue")

    # plt.close()

    if debug:

        from scipy.stats import binned_statistic_2d

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            pos_stars[:, 0],
            pos_stars[:, 1],
            pos_stars[:, 2],
            s=1,
            alpha=0.01,
            c=np.log10(vel_vert),
        )

        ax.quiver(
            0, 0, 0, ang_mom[0], ang_mom[1], ang_mom[2], color="k", label="ang_mom"
        )

        ax.quiver(
            0,
            0,
            0,
            new_phis[0, 0],
            new_phis[0, 1],
            new_phis[0, 2],
            color="r",
            label="phi",
        )
        ax.quiver(
            0,
            0,
            0,
            new_rhos[0, 0],
            new_rhos[0, 1],
            new_rhos[0, 2],
            color="g",
            label="rho",
        )
        ax.quiver(0, 0, 0, new_z[0], new_z[1], new_z[2], color="b", label="z")

        fig.savefig("ang_mom")

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        xbins = np.linspace(new_xs.min(), new_xs.max(), 30)
        ybins = np.linspace(new_ys.min(), new_ys.max(), 30)

        X, Y = np.meshgrid(xbins, ybins)

        max_zdist = 1e-5

        plane_stars = np.abs(pos_vert) < max_zdist

        mean_tanx_vel, _, _, _ = binned_statistic_2d(
            new_xs[plane_stars],
            new_ys[plane_stars],
            new_vels_xs[plane_stars],
            bins=(xbins, ybins),
            statistic="mean",
        )
        mean_tany_vel, _, _, _ = binned_statistic_2d(
            new_xs[plane_stars],
            new_ys[plane_stars],
            new_vels_ys[plane_stars],
            bins=(xbins, ybins),
            statistic="mean",
        )

        print(X.shape, Y.shape, mean_tanx_vel.shape)

        ax.quiver(X[:-1, :-1], Y[:-1, :-1], mean_tanx_vel.T, mean_tany_vel.T)

        # plot some of the particles bases
        for i in range(10):

            ind = np.random.randint(0, len(new_xs))

            plt_r = new_rhos[ind]
            plt_t = new_phis[ind]
            plt_z = new_z

            plt_x = new_xs[ind]
            plt_y = new_ys[ind]
            # plane_dist = np.linalg.norm([plt_x, plt_y])

            plt_vel_tan = vel_tan[ind]
            plt_vel_rad = vel_rad[ind]
            plt_vel_vert = vel_vert[ind]

            plt_vel_x, plt_vel_y, plt_vel_z = bulkless_vel[ind]

            print(plt_vel_rad, plt_vel_tan, plt_vel_vert)
            print(plt_vel_x, plt_vel_y, plt_vel_z)

            print(plt_r)
            print(plt_t)
            print(plt_z)

            print(
                np.dot(bulkless_vel[ind], plt_r),
                np.dot(bulkless_vel[ind], plt_t),
                np.dot(bulkless_vel[ind], plt_z),
            )

            ax.quiver(0, 0, plt_r[0], plt_r[1], color="b", label="r")
            ax.quiver(plt_x, plt_y, plt_t[0], plt_t[1], color="r", label="t")

        # ax.scatter(new_xs, new_ys, s=1, alpha=0.01, c=np.log10(vel_vert))
        # ax.quiver(new_xs[::400], new_ys[::400], vel_xs[::400], vel_ys[::400])

        fig.savefig("disk_plane_velocities")

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        ax.hist(vel_tan, bins=100, histtype="step", color="r", label="tan")
        ax.hist(vel_rad, bins=100, histtype="step", color="b", label="rad")
        ax.hist(vel_vert, bins=100, histtype="step", color="g", label="vert")

        ax.legend()

        ax.set_yscale("log")

        fig.savefig("velocities")

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        ax.hist(bulkless_vel[:, 0], bins=100, histtype="step", color="r", label="x")
        ax.hist(bulkless_vel[:, 1], bins=100, histtype="step", color="b", label="y")
        ax.hist(bulkless_vel[:, 2], bins=100, histtype="step", color="g", label="z")

        ax.legend()

        fig.savefig("velocities_cartesian")

    # projected map of velocities in xy xz yz planes
    # fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    # pos_mins = np.min(new_pos, axis=0)
    # pos_maxs = np.max(new_pos, axis=0)

    # print(pos_mins)
    # print(pos_maxs)

    # npos_bins = 100
    # pos_bins = [
    #     np.linspace(pos_mins[0], pos_maxs[0], npos_bins),
    #     np.linspace(pos_mins[1], pos_maxs[1], npos_bins),
    #     np.linspace(pos_mins[2], pos_maxs[2], npos_bins),
    # ]

    # vmin = 0.1
    # vmax = 500

    # for i in range(3):

    #     ind0 = i
    #     ind1 = i + 1
    #     ind2 = i + 2

    #     ind1 = ind1 % 3
    #     ind2 = ind2 % 3

    #     print(f"Plotting {ind0} vs {ind1}, view from {ind2}")

    #     # plane_stars = np.abs(stars["pos"][:, ind2] - tgt_pos[ind2]) < plane_dist_tol
    #     #
    #     plane_vels = np.linalg.norm(
    #         np.transpose([new_vels[:, ind0], new_vels[:, ind1]]), axis=1
    #     )

    #     means, _, _, _ = binned_statistic_2d(
    #         new_pos[:, ind0],
    #         new_pos[:, ind1],
    #         plane_vels,
    #         bins=(pos_bins[ind0], pos_bins[ind1]),
    #         statistic="mean",
    #     )
    #     img = ax[i].imshow(
    #         means.T,
    #         origin="lower",
    #         extent=(pos_mins[ind0], pos_maxs[ind0], pos_mins[ind1], pos_maxs[ind1]),
    #         # norm=LogNorm(vmin=vmin, vmax=vmax),
    #         vmin=vmin,
    #         vmax=vmax,
    #     )

    #     ax[i].set_xlabel(f"pos {ind0}")
    #     ax[i].set_ylabel(f"pos {ind1}")

    # plt.colorbar(img, ax=ax)

    # fig.savefig(f"velocities_angmom_basis.png")

    return vel_rad, vel_tan, vel_vert


def weighted_variance(values, weights):

    average = np.average(values, weights=weights)
    average2 = np.average(values**2, weights=weights)
    variance = average2 - average**2
    return variance


def extract_nh_kinematics(m_stars, pos_stars, vel_stars, pos_gal, debug=False):
    """
    From dubois et al 2021
    The stellar rotation V is the average of the tangential component of velocities,
    while the 1D stellar dispersion sigma is the dispersion obtained from the dispersion
    around each mean component, that is sigma**2 = (sigmar**2 +sigmat**2 +sigmaz**2 )/3.

    """

    ang_mom = compute_ang_mom(m_stars, pos_stars, vel_stars, pos_gal)

    if debug:
        print(ang_mom)

    vel_rad, vel_tan, vel_vert = project_vels(
        m_stars, pos_stars - pos_gal, vel_stars, ang_mom, debug=debug
    )

    if debug:
        print(
            np.average(vel_rad, weights=m_stars),
            np.average(vel_tan, weights=m_stars),
            np.average(vel_vert, weights=m_stars),
        )

    disp = np.sqrt(
        (
            weighted_variance(vel_rad, m_stars)
            + weighted_variance(vel_tan, m_stars)
            + weighted_variance(vel_vert, m_stars)
        )
        / 3.0
    )
    Vrot = np.average(vel_tan, weights=m_stars)

    rots = {"Vrot": Vrot, "disp": disp}

    if debug:
        print(f"Vrot: {Vrot:.2e} km/s")
        print(f"disp: {disp:.2e} km/s")
        print(f"Vrot/disp: {Vrot/disp:.2e}")

    # print(np.std(vel_rad), np.std(vel_tan), np.std(vel_vert))

    # print(Vrot, disp, Vrot / disp)

    m_above = m_stars[vel_tan > 0].sum()
    m_below = m_stars[vel_tan < 0].sum()
    m_tot = m_stars.sum()

    if m_below > m_above:
        fbulge = m_above * 2 / m_tot
    else:
        fbulge = m_below * 2 / m_tot

    fdisk = 1 - fbulge

    m_disk = fdisk * m_tot

    kin_sep = {"fdisk": fdisk, "fbulge": fbulge, "Mdisk": m_disk}

    if debug:
        print(f"fdisk: {fdisk:.2e}")
        print(f"fbulge: {fbulge:.2e}")

    return rots, kin_sep

    # construct radial axis as the normalized vector perpendicular in the plane of ang_mom and z

    # we know alpha the angle between new_z and z
    # alpha = np.arcsin((np.sqrt(new_z[0] ** 2 + new_z[1] ** 2)) / np.linalg.norm(new_z))
    # print(alpha*180/np.pi)
    # we want the angle beta between new_z and the radial axis to be pi/2
    # beta = np.pi / 2 - alpha
    # print(beta*180/np.pi)
    # R = np.sqrt(rhox**2 + rhoy**2)
    # R = -np.sin(alpha) + np.sin(beta) - np.sqrt(new_z[0]**2 + new_z[1]**2)
    # print(R)

    # flip in the x,y plane
    # rho_y = -new_z[1]
    # rho_x = -new_z[0]
    # R = np.sqrt(rho_x**2 + rho_y**2)

    # we can infer the z component imposing the angle between new_z and the radial axis to be pi/2
    # rho_z = np.sqrt((R**2 * (1.0 - np.sin(beta) ** 2)) / np.sin(beta) ** 2)

    # #using null scalar product between new_z and the radial axis
    # #2nd order polynomial equation to get rho_y
    # a = -(1  + new_z[1]**2/new_z[0]**2)
    # b = - 2 * new_z[1] * new_z[2] * rho_z/new_z[0]**2
    # c = 1 - rho_z**2 * new_z[2]**2*rho_z**2/new_z[0]**2
    # print(a,b,c)
    # rho_y = np.roots([a,b,c])
    # print(rho_y)
    # rho_x = np.sqrt(1 - rho_y**2 - rho_z**2)

    # new_rho1 = np.array([rho_x[0], rho_y[0], rho_z])
    # if np.dot(new_rho1, new_z) > 1e-5 or np.any(np.isnan(new_rho1)):
    #     new_rho = np.array([rho_x[1], rho_y[1], rho_z]) #other solution
    # else:
    #     new_rho = new_rho1
    # print(rho_x)
