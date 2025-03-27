import numpy as np

# https://arxiv.org/pdf/2404.17945 #for for def quenched
# https://iopscience.iop.org/article/10.3847/1538-4357/ac887d#apjac887deqn9 #fit and definitions


def quad_z(z, x0, x1, x2):
    return x2 * z**2 + x1 * z + x0


def get_z_a(z):
    return quad_z(z, 0.03746, 0.3448, -0.1156)


def get_z_b(z):
    return quad_z(z, 0.9605, 0.04990, -0.05984)


def get_z_c(z):
    return quad_z(z, 0.2516, 1.118, -0.2006)


def get_z_logMt(z):
    return quad_z(z, 10.22, 0.3826, -0.04491)


def sfr_ridge_leja22(z, Mt_gals):
    """
    z redshift
    Mt total stellar mass
    """

    assert np.any(
        (0.3 <= z) * (z <= 2.7)
    ), "Redshift out of range, Leja+22 fits are only valid for 0.3<z<=2.7"

    a = get_z_a(z)
    b = get_z_b(z)
    c = get_z_c(z)
    logMt = get_z_logMt(z)

    # print(a, b, c, logMt)

    log_Mt_gals = np.log10(Mt_gals)

    logSFRs = a * (log_Mt_gals - logMt)
    logSFRs[log_Mt_gals <= logMt] *= b / a

    return 10 ** (logSFRs + c)
