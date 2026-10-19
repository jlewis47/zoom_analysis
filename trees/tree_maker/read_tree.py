import numpy as np
import os
from f90_tools.IO import read_record
from gremlin.read_sim_params import ramses_sim

# from astropy.comsology import z_at_value


def read_tree(tfile, sim: ramses_sim):

    with open(tfile, "rb") as src:

        nsteps = read_record(src, 1, np.int32)

        nb_halos = np.zeros(nsteps, np.float32)
        nb_subhalos = np.zeros(nsteps, np.float32)
        aexps = np.zeros(nsteps, np.float32)
        omega_ts = np.zeros(nsteps, np.float32)
        age_univs = np.zeros(nsteps, np.float32)

        nb_halos[:] = read_record(src, nsteps, np.float64)
        nb_subhalos[:] = read_record(src, nsteps, np.float64)
        aexps[:] = read_record(src, nsteps, np.float64)
        omega_ts[:] = read_record(src, nsteps, np.float64)
        age_univs[:] = read_record(src, nsteps, np.float64)

        snap_aexps = sim.get_aexps()
        # snap_times = sim.times
        snap_zeds = 1.0 / snap_aexps - 1.0
        age_univ_snaps = np.asarray(
            [sim.cosmo_model.lookback_time(aexp).value for aexp in snap_aexps]
        )
        snaps = sim.snaps
        snap_nbs = sim.snap_numbers

        damax = abs(snap_aexps[0] - aexps[0])

        matches = np.where(np.any(np.abs(aexps[:, None] - snap_aexps) < damax, axis=1))

        nfiles = snap_nbs[matches]

        stepping = True

        while stepping:

            istep += 1

            ntot = nb_halos[istep] + nb_subhalos[istep]

            if istep == 1:
                idselec_tmp = np.zeros(ntot, np.int32)
                idmainfather_tmp = np.zeros(ntot, np.int32)
                mselec_tmp = np.zeros(ntot, np.int32)
                ok = np.zeros(ntot, np.int32)

            fname = "tree_%05d" % nfiles[istep]
