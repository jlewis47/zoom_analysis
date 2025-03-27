"""Module regrouping functions relevant for IGM physics

WRITTEN BY MAXIME TREBITSCH, 2024

Content:
    - For now, only the Inoue+2014 model

"""

import numpy as np

"""
Analytic model

At a given observed wavelength lobs, for a source at redshift zs, we sum "line series" + "Lyman Continuum"
    tau_IGM(lobs, zs) = tau_LS^LAF(lobs, zs) + tau_LS^DLA(lobs, zs) + tau_LC^LAF(lobs, zs) + tau_LC^DLA(lobs, zs)

The LAF component corresponds to the "Lya Forest" and DLA to "Damped Lya systems"

LS component (for each LAF and DLA)
    tau_LS = sum_j tau_j(lobs, zs)
where the sum is over the lines

Inoue+2014 find that for the bins such that lj < lobs <= lj(1+zs)
            Aj1^LAF (lobs/lj)**1.2 if lobs < 2.2 lj
tau_j^LAF = Aj2^LAF (lobs/lj)**3.7 if 2.2 lj <= lobs < 5.7 lj
            Aj3^LAF (lobs/lj)**5.5 if 5.7 lj <= obs
and tau_j^LAF otherwise

With the same constraint on lj, the DLA component becomes
tau_j^DLA = Aj1^DLA (lobs/lj)**2 for lobs < 3 lj
            Aj2^DLA (lobs/lj)**3 for lobs >= 3 lj

Inoue+2014 also provides analytical expressions for the LC components.
"""


def tau_igm_LC_LAF(lobs, zs):
    """Equations 25-27 of Inoue+2014, lobs in angstrom"""
    lL = 911.8  # angstrom
    if zs < 1.2:
        tau = np.where(
            lobs < lL * (1 + zs),
            0.325 * ((lobs / lL) ** 1.2 - (1 + zs) ** (-0.9) * (lobs / lL) ** 2.1),
            0,
        )
    elif np.logical_and(1.2 <= zs, zs < 4.7):
        tau_a = (
            2.55e-2 * (1 + zs) ** 1.6 * (lobs / lL) ** 2.1
            + 0.325 * (lobs / lL) ** 1.2
            - 0.250 * (lobs / lL) ** 2.1
        )
        tau_b = 2.55e-2 * ((1 + zs) ** 1.6 * (lobs / lL) ** 2.1 - (lobs / lL) ** 3.7)
        conditions = [
            (lobs < 2.2 * lL),
            np.logical_and(2.2 * lL <= lobs, lobs < lL * (1 + zs)),
        ]
        tau = np.select(conditions, [tau_a, tau_b], 0)
    else:
        tau_a = (
            5.22e-4 * (1 + zs) ** 3.4 * (lobs / lL) ** 2.1
            + 0.325 * (lobs / lL) ** 1.2
            - 3.14e-2 * (lobs / lL) ** 2.1
        )
        tau_b = (
            5.22e-4 * (1 + zs) ** 3.4 * (lobs / lL) ** 2.1
            + 0.218 * (lobs / lL) ** 2.1
            - 2.55e-2 * (lobs / lL) ** 3.7
        )
        tau_c = 5.22e-4 * ((1 + zs) ** 3.4 * (lobs / lL) ** 2.1 - (lobs / lL) ** 5.5)
        conditions = [
            (lobs < 2.2 * lL),
            np.logical_and(2.2 * lL <= lobs, lobs < 5.7 * lL),
            np.logical_and(5.7 * lL <= lobs, lobs < lL * (1 + zs)),
        ]
        tau = np.select(conditions, [tau_a, tau_b, tau_c], 0)
    return tau


def tau_igm_LC_DLA(lobs, zs):
    """Equations 28-29 of Inoue+2014, lobs in angstrom"""
    lL = 911.8  # angstrom
    if zs < 2:
        tau = np.where(
            lobs < lL * (1 + zs),
            0.211 * (1 + zs) ** 2
            - 7.66e-2 * (1 + zs) ** 2.3 * (lobs / lL) ** (-0.3)
            - 0.135 * (lobs / lL) ** 2.0,
            0,
        )
        return tau
    else:
        tau_a = (
            0.634
            + 4.70e-2 * (1 + zs) ** 3
            - 1.78e-2 * (1 + zs) ** 3.3 * (lobs / lL) ** (-0.3)
            - 0.135 * (lobs / lL) ** 2.0
            - 0.291 * (lobs / lL) ** (-0.3)
        )
        tau_b = (
            4.70e-2 * (1 + zs) ** 3
            - 1.78e-2 * (1 + zs) ** 3.3 * (lobs / lL) ** (-0.3)
            - 2.92e-2 * (lobs / lL) ** 3
        )
        conditions = [
            (lobs < 3.0 * lL),
            np.logical_and(3.0 * lL <= lobs, lobs < lL * (1 + zs)),
        ]
        tau = np.select(conditions, [tau_a, tau_b], 0)
    return tau


j_LAF, lj_LAF, Aj1_LAF, Aj2_LAF, Aj3_LAF = np.loadtxt(
    "/home/trebitsc/Models/InoueIGM/LAFcoeff.txt", unpack=True
)
j_DLA, lj_DLA, Aj1_DLA, Aj2_DLA = np.loadtxt(
    "/home/trebitsc/Models/InoueIGM/DLAcoeff.txt", unpack=True
)


def tau_igm_LS_LAF(lobs, zs):
    """Equation 21 from Inoue+2014, lobs in angstrom"""

    def tau_j_LAF(lobs, j=0):
        conditions = [
            lobs < 2.2 * lj_LAF[j],
            np.logical_and(2.2 * lj_LAF[j] <= lobs, lobs < 5.7 * lj_LAF[j]),
            5.7 * lj_LAF[j] <= lobs,
        ]
        taus = [
            Aj1_LAF[j] * (lobs / lj_LAF[j]) ** 1.2,
            Aj2_LAF[j] * (lobs / lj_LAF[j]) ** 3.7,
            Aj3_LAF[j] * (lobs / lj_LAF[j]) ** 5.5,
        ]
        return np.select(conditions, taus)

    tau_LS_LAF = np.zeros_like(lobs)
    for j, jline in enumerate(j_LAF):
        tau_j = np.where(
            np.logical_and(lj_LAF[j] < lobs, lobs <= lj_LAF[j] * (1 + zs)),
            tau_j_LAF(lobs, j),
            0.0,
        )
        tau_LS_LAF += tau_j
    return tau_LS_LAF


def tau_igm_LS_DLA(lobs, zs):
    """Equation 22 from Inoue+2014, lobs in angstrom"""

    def tau_j_DLA(lobs, j=0):
        return np.where(
            lobs < 3 * lj_DLA[j],
            Aj1_DLA[j] * (lobs / lj_DLA[j]) ** 2,
            Aj2_DLA[j] * (lobs / lj_DLA[j]) ** 3,
        )

    tau_LS_DLA = np.zeros_like(lobs)
    for j, jline in enumerate(j_DLA):
        tau_j = np.where(
            np.logical_and(lj_DLA[j] < lobs, lobs <= lj_DLA[j] * (1 + zs)),
            tau_j_DLA(lobs, j),
            0.0,
        )
        tau_LS_DLA += tau_j
    return tau_LS_DLA


def tau_igm_Inoue2014(lobs, zs):
    return (
        tau_igm_LC_LAF(lobs, zs)
        + tau_igm_LC_DLA(lobs, zs)
        + tau_igm_LS_LAF(lobs, zs)
        + tau_igm_LS_DLA(lobs, zs)
    )


def T_IGM_Inoue2014(lobs, zs):
    return np.exp(-tau_igm_Inoue2014(lobs, zs))
