from zoom_analysis.sinks.sink_reader import read_sink_snap,sid_to_hid
from zoom_analysis.halo_maker.assoc_fcts import get_halo_assoc_file
from gremlin.read_sim_params import ramses_sim

from zoom_analysis.read.read_data import read_data_ball

import os
import numpy as np
import matplotlib.pyplot as plt



# sid = 1
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380"
sid = 158
sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id36755"
# sid = 2
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed/restart"
# sid = 11
# sim_dir = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130"
sim = ramses_sim(sim_dir)

snaps = sim.snap_numbers


res = np.zeros(len(snaps))
dens = np.zeros(len(snaps))
cs = np.zeros(len(snaps))
vrel = np.zeros(len(snaps))
growth = np.zeros(len(snaps))

for i,snap in enumerate(snaps[::-1]):
    print(i,len(snaps),snap)
    if os.path.exists(get_halo_assoc_file(sim.path, snap)):
        sink = read_sink_snap(sim, snap, sid=sid)
        if sink !={}:
            print('...found sink')
            try:
                hid = sid_to_hid(sim, sid, snap)
            except:
                hid = None

            if hid is not None:
                print('...found halo')
                print(i,snap,1./sim.get_snap_exps(snap)[0]-1)
                # print(f"Snap {snap}: sink {sid} in halo {hid}")
                try:
                    res[i] = np.max(read_data_ball(sim,snap, sink['position'], 1e-3/sim.cosmo.lcMpc, host_halo=hid,data_types=["gas"],tgt_fields=['ilevel'])['ilevel'])
                except:
                    continue
                print('...found data')
                dens[i] = sink['dens']
                growth[i] = sink['dMBH_coarse']
                cs[i] = sink['csound']
                vrel[i] = sink['vrel']


zeds = 1./sim.aexps[::-1] - 1

non_nul = dens>0

zeds = zeds[non_nul]
res = np.int32(res[non_nul])
dens = dens[non_nul]
growth = growth[non_nul]
cs = cs[non_nul]
vrel = vrel[non_nul]

fig,ax = plt.subplots(2,3,layout="constrained")

ax_res = ax[0,0]
ax_dens = ax[0,1]
ax_growth = ax[1,0]
ax_cs = ax[1,1]
ax_vrel = ax[0,2]
ax_evol = ax[1,2]





ax_res.plot(zeds,res)
ax_dens.plot(zeds,dens)
ax_cs.plot(zeds,cs)
ax_vrel.plot(zeds,vrel)
ax_growth.plot(zeds,growth)
    
for ia,a in enumerate(ax.flatten()[:-1]):
    a.set_xlabel("z")
    if ia>0:a.set_yscale("log")
    a.grid()

ax_res.set_ylabel("max level")
ax_dens.set_ylabel("sink density")
ax_growth.set_ylabel("sink growth rate")
ax_cs.set_ylabel("sink sound speed")
ax_vrel.set_ylabel("sink relative velocity")

# ax_evol.plot(res,dens)
dens_split_res=[dens[res==r] for r in np.unique(res)]
ax_evol.hist(dens_split_res, label=[f"Level {r}" for r in np.unique(res)],bins=np.logspace(-2,3,10))

ax_evol.set_xlabel("density")
ax_evol.set_ylabel("counts")
ax_evol.set_xscale("log")
ax_evol.legend()


fig.savefig('evol_sink_dens_resolution')