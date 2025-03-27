#    subroutine ct_friedman(O_mat_0,O_vac_0,O_k_0,alpha,axp_min, &
#    & axp_out,hexp_out,tau_out,t_out,ntable,age_tot)

#       implicit none
#       integer::ntable
#       real(kind=8)::O_mat_0, O_vac_0, O_k_0
#       real(kind=8)::alpha,axp_min,age_tot
#       real(kind=8),dimension(0:ntable)::axp_out,hexp_out,tau_out,t_out
#       ! ######################################################!
#       ! This subroutine assumes that axp = 1 at z = 0 (today) !
#       ! and that t and tau = 0 at z = 0 (today).              !
#       ! axp is the expansion factor, hexp the Hubble constant !
#       ! defined as hexp=1/axp*daxp/dtau, tau the conformal    !
#       ! time, and t the look-back time, both in unit of 1/H0. !
#       ! alpha is the required accuracy and axp_min is the     !
#       ! starting expansion factor of the look-up table.       !
#       ! ntable is the required size of the look-up table.     !
#       ! ######################################################!
#       real(kind=8)::axp_tau, axp_t
#       real(kind=8)::axp_tau_pre, axp_t_pre
#       real(kind=8)::dtau,dt
#       real(kind=8)::tau,t
#       integer::nstep,nout,nskip

#       !  if( (O_mat_0+O_vac_0+O_k_0) .ne. 1.0D0 )then
#       !     write(*,*)'Error: non-physical cosmological constants'
#       !     write(*,*)'O_mat_0,O_vac_0,O_k_0=',O_mat_0,O_vac_0,O_k_0
#       !     write(*,*)'The sum must be equal to 1.0, but '
#       !     write(*,*)'O_mat_0+O_vac_0+O_k_0=',O_mat_0+O_vac_0+O_k_0
#       !     stop
#       !  end if

#       axp_tau = 1.0D0
#       axp_t = 1.0D0
#       tau = 0.0D0
#       t = 0.0D0
#       nstep = 0

#       do while ( (axp_tau .ge. axp_min) .or. (axp_t .ge. axp_min) )

#          nstep = nstep + 1
#          dtau = alpha * axp_tau / dadtau(axp_tau,O_mat_0,O_vac_0,O_k_0)
#          axp_tau_pre = axp_tau - dadtau(axp_tau,O_mat_0,O_vac_0,O_k_0)*dtau/2.d0
#          axp_tau = axp_tau - dadtau(axp_tau_pre,O_mat_0,O_vac_0,O_k_0)*dtau
#          tau = tau - dtau

#          dt = alpha * axp_t / dadt(axp_t,O_mat_0,O_vac_0,O_k_0)
#          axp_t_pre = axp_t - dadt(axp_t,O_mat_0,O_vac_0,O_k_0)*dt/2.d0
#          axp_t = axp_t - dadt(axp_t_pre,O_mat_0,O_vac_0,O_k_0)*dt
#          t = t - dt

#       end do

#       age_tot=-t
# !!$    write(*,666)-t
# !!$666 format(' Age of the Universe (in unit of 1/H0)=',1pe10.3)

#       nskip=nstep/ntable

#       axp_t = 1.d0
#       t = 0.d0
#       axp_tau = 1.d0
#       tau = 0.d0
#       nstep = 0
#       nout=0
#       t_out(nout)=t
#       tau_out(nout)=tau
#       axp_out(nout)=axp_tau
#       hexp_out(nout)=dadtau(axp_tau,O_mat_0,O_vac_0,O_k_0)/axp_tau

#       do while ( (axp_tau .ge. axp_min) .or. (axp_t .ge. axp_min) )

#          nstep = nstep + 1
#          dtau = alpha * axp_tau / dadtau(axp_tau,O_mat_0,O_vac_0,O_k_0)
#          axp_tau_pre = axp_tau - dadtau(axp_tau,O_mat_0,O_vac_0,O_k_0)*dtau/2.d0
#          axp_tau = axp_tau - dadtau(axp_tau_pre,O_mat_0,O_vac_0,O_k_0)*dtau
#          tau = tau - dtau

#          dt = alpha * axp_t / dadt(axp_t,O_mat_0,O_vac_0,O_k_0)
#          axp_t_pre = axp_t - dadt(axp_t,O_mat_0,O_vac_0,O_k_0)*dt/2.d0
#          axp_t = axp_t - dadt(axp_t_pre,O_mat_0,O_vac_0,O_k_0)*dt
#          t = t - dt

#          if(mod(nstep,nskip)==0)then
#             nout=nout+1
#             t_out(nout)=t
#             tau_out(nout)=tau
#             axp_out(nout)=axp_tau
#             hexp_out(nout)=dadtau(axp_tau,O_mat_0,O_vac_0,O_k_0)/axp_tau
#          end if

#       end do
#       t_out(ntable)=t
#       tau_out(ntable)=tau
#       axp_out(ntable)=axp_tau
#       hexp_out(ntable)=dadtau(axp_tau,O_mat_0,O_vac_0,O_k_0)/axp_tau

#    contains
#       function dadtau(axp_tau,O_mat_0,O_vac_0,O_k_0)
#          implicit none
#          real(kind=8)::dadtau,axp_tau,O_mat_0,O_vac_0,O_k_0
#          dadtau = axp_tau*axp_tau*axp_tau *  &
#          &   ( O_mat_0 + &
#          &     O_vac_0 * axp_tau*axp_tau*axp_tau + &
#          &     O_k_0   * axp_tau )
#          dadtau = sqrt(dadtau)
#          return
#       end function dadtau

#       function dadt(axp_t,O_mat_0,O_vac_0,O_k_0)
#          implicit none
#          real(kind=8)::dadt,axp_t,O_mat_0,O_vac_0,O_k_0
#          dadt   = (1.0D0/axp_t)* &
#          &   ( O_mat_0 + &
#          &     O_vac_0 * axp_t*axp_t*axp_t + &
#          &     O_k_0   * axp_t )
#          dadt = sqrt(dadt)
#          return
#       end function dadt

#    end subroutine ct_friedman

import numpy as np
import os

from zoom_analysis.constants import *


def ct_friedman(
    O_mat_0,
    O_vac_0,
    O_k_0,
    alpha,
    axp_min,
    axp_out,
    hexp_out,
    tau_out,
    t_out,
    ntable,
    age_tot,
):
    def dadtau(axp_tau, O_mat_0, O_vac_0, O_k_0):
        return (axp_tau**3 * (O_mat_0 + O_vac_0 * axp_tau**3 + O_k_0 * axp_tau)) ** 0.5

    def dadt(axp_t, O_mat_0, O_vac_0, O_k_0):
        return ((1.0 / axp_t) * (O_mat_0 + O_vac_0 * axp_t**3 + O_k_0 * axp_t)) ** 0.5

    axp_tau = 1.0
    axp_t = 1.0
    tau = 0.0
    t = 0.0
    nstep = 0

    while axp_tau >= axp_min or axp_t >= axp_min:
        # print(nstep, axp_tau, axp_t, axp_min)
        dadtau_axp_tau = dadtau(axp_tau, O_mat_0, O_vac_0, O_k_0)
        dtau = alpha * axp_tau / dadtau_axp_tau
        axp_tau_pre = axp_tau - dadtau_axp_tau * dtau / 2.0
        axp_tau = axp_tau - dadtau(axp_tau_pre, O_mat_0, O_vac_0, O_k_0) * dtau
        tau = tau - dtau

        dadt_axp_t = dadtau(axp_tau, O_mat_0, O_vac_0, O_k_0)
        dt = alpha * axp_t / dadt_axp_t
        axp_t_pre = axp_t - dadt_axp_t * dt / 2.0
        axp_t = axp_t - dadt(axp_t_pre, O_mat_0, O_vac_0, O_k_0) * dt
        t = t - dt
        nstep += 1

    age_tot = -t

    nskip = int(nstep / ntable)
    # print(nskip, nstep, ntable)

    axp_t = 1.0
    t = 0.0
    axp_tau = 1.0
    tau = 0.0
    nstep = 0
    nout = 0
    t_out[nout] = t
    tau_out[nout] = tau
    axp_out[nout] = axp_tau
    hexp_out[nout] = dadtau(axp_tau, O_mat_0, O_vac_0, O_k_0) / axp_tau

    while axp_tau >= axp_min or axp_t >= axp_min:
        dadtau_axp_tau = dadtau(axp_tau, O_mat_0, O_vac_0, O_k_0)
        dtau = alpha * axp_tau / dadtau_axp_tau
        axp_tau_pre = axp_tau - dadtau_axp_tau * dtau / 2.0
        axp_tau = axp_tau - dadtau(axp_tau_pre, O_mat_0, O_vac_0, O_k_0) * dtau
        tau = tau - dtau

        dadt_axp_t = dadt(axp_t, O_mat_0, O_vac_0, O_k_0)
        dt = alpha * axp_t / dadt_axp_t
        axp_t_pre = axp_t - dadt_axp_t * dt / 2.0
        axp_t = axp_t - dadt(axp_t_pre, O_mat_0, O_vac_0, O_k_0) * dt
        t = t - dt
        nstep += 1

        if nstep % nskip == 0:
            # print("fill", nout)
            t_out[nout] = t
            tau_out[nout] = tau
            axp_out[nout] = axp_tau
            hexp_out[nout] = dadtau(axp_tau, O_mat_0, O_vac_0, O_k_0) / axp_tau
            nout += 1

    t_out[ntable - 1] = t
    tau_out[ntable - 1] = tau
    axp_out[ntable - 1] = axp_tau
    hexp_out[ntable - 1] = dadtau(axp_tau, O_mat_0, O_vac_0, O_k_0) / axp_tau


#    subroutine ct_init_cosmo(omega_m,omega_l,omega_k,h0)

#       ! h0 is in km/s/Mpc

#       implicit none
#       real(kind=8),intent(in) :: omega_m,omega_l,omega_k,h0
#       real(kind=8)            :: time_tot

#       allocate(aexp_frw(0:n_frw),hexp_frw(0:n_frw))
#       allocate(tau_frw(0:n_frw),t_frw(0:n_frw))
#       call ct_friedman(omega_m,omega_l,omega_k,1.d-6,1.d-3,aexp_frw,hexp_frw,tau_frw,t_frw,n_frw,time_tot)
#       ! convert time to yr
#       t_frw = t_frw / (h0 / 3.08d19) / (365.25*24.*3600.)

#       return


def ct_init_cosmo(fname, omega_m, omega_l, omega_k, H0, n_frw=1000):
    if not os.path.isfile(fname):
        aexp_frw = np.zeros(n_frw)
        hexp_frw = np.zeros(n_frw)
        tau_frw = np.zeros(n_frw)
        t_frw = np.zeros(n_frw)

        time_tot = 0.0

        ct_friedman(
            omega_m,
            omega_l,
            omega_k,
            1e-6,
            1e-3,
            aexp_frw,
            hexp_frw,
            tau_frw,
            t_frw,
            n_frw,
            time_tot,
        )

        # convert time to yr
        t_frw = t_frw / (H0 / ramses_pc * 1e-2 * 1e6) / (365.25 * 24.0 * 3600.0)

        np.savetxt(
            fname,
            np.transpose([aexp_frw, hexp_frw, tau_frw, t_frw]),
            header="aexp_frw, hexp_frw, tau_frw, t_frw",
        )

    else:
        aexp_frw, hexp_frw, tau_frw, t_frw = np.loadtxt(fname, unpack=True, skiprows=1)

    return aexp_frw, hexp_frw, tau_frw, t_frw
