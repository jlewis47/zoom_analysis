import numpy as np
from gremlin.read_sim_params import ramses_sim


def hagn_injection(sink_dict, hagn_sim: ramses_sim):

    pass


def zoom_injection(sink_dict, sim: ramses_sim):

    X_floor = sim.namelist["smbh_params"]["x_floor"]
    mad_jet = sim.namelist["smbh_params"]["mad_jet"]
    eff_kin = sim.namelist["smbh_params"]["eagn_k"]
    eff_rad = sim.namelist["smbh_params"]["eagn_t"]

    dmbh_kg = sink_dict["dMBH_coarse"] * 2e30
    dsmbh_ks = sink_dict["dMsmbh"] * 2e30 / 1e5
    dmed_kg = sink_dict["dMEd_coarse"] * 2e30
    spin_mag = np.linalg.norm(sink_dict["spins"], axis=1)

    ZZ1 = 1 + np.cbrt(1 - spin_mag**2) * np.cbrt(1 + spin_mag) + np.cbrt(1 - spin_mag)
    ZZ2 = np.sqrt(3 * spin_mag**2 + ZZ1**2)

    pos_spin = spin_mag > 0
    r_lso = np.where(
        pos_spin,
        3 + ZZ2 - np.sqrt((3 - ZZ1) * (3 + ZZ1 + 2 * ZZ2)),
        3 + ZZ2 + np.sqrt((3 - ZZ1) * (3 + ZZ1 + 2 * ZZ2)),
    )

    eps_r = 1 - np.sqrt(1 - 2 / (3 * r_lso))

    X_radio = dmbh_kg / dmed_kg

    EAGN = np.zeros(len(dsmbh_ks))

    radio_mode = X_radio < X_floor

    if mad_jet:

        eff_mad = (
            4.10507
            + 0.328712 * spin_mag
            + 76.0849 * spin_mag**2
            + 47.9235 * spin_mag**3
            + 3.86634 * spin_mag**4
        ) / 100.0

        EAGN[radio_mode] = eff_mad[radio_mode] * dsmbh_ks[radio_mode] * 3e8**2

    else:

        EAGN[radio_mode] = eff_kin * dsmbh_ks[radio_mode] * eps_r[radio_mode] * 3e8**2

    EAGN[radio_mode == False] = (
        eff_rad * eps_r[radio_mode == False] * dsmbh_ks[radio_mode == False] * 3e8**2
    )

    sink_dict["EAGN"] = EAGN  # J/s
