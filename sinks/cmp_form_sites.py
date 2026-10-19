from zoom_analysis.sinks import sink_histories, sink_reader
from gremlin.read_sim_params import ramses_sim

import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import binned_statistic


sim_dirs = [
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_form_stellar_peak",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_no_sink_desc",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_sink_desc_goodfix",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag",

]

# tgt_z = 7.37
# tgt_z = 7.33
# tgt_z = 6.78
tgt_z = 6.53



fig,ax = plt.subplots(1,3,figsize=(18,6),sharey=True)


vrel_bins = 10**np.arange(0.5,3.5,0.25)
csound_bins = 10**np.arange(0.,3.5,0.25)
rho_bins = 10**np.arange(-2,3,0.25)

#for every sim
for sdir in sim_dirs:
    sim = ramses_sim(sdir)


    #find all sinks at tgt_z
    tgt_snap = sim.get_closest_snap(zed=tgt_z)
    sinks = sink_reader.read_sink_snap(sim,tgt_snap)

    nsinks = len(sinks['mass'])

    vrels = np.zeros(nsinks)
    rhos = np.zeros(nsinks)
    csounds = np.zeros(nsinks)
    
    #get their histories and their first measured local density, sound speed, and vrel
    for isink in range(nsinks):

        sid = sinks['identity'][isink]

        hist = sink_reader.get_sink_mhistory(sid,tgt_snap,sim,out_keys=[("zeds",1),("vrel",1),("csound",1),("dens",1)],debug=False)

        arg = np.argmax(hist['zeds'])

        vrels[isink] = hist['vrel'][arg]
        csounds[isink] = hist['csound'][arg]
        rhos[isink] = hist['dens'][arg]


    for iprop,prop in enumerate([vrels,csounds,rhos]):
        if iprop == 0:
            bins = vrel_bins
        elif iprop == 1:
            bins = csound_bins
        elif iprop == 2:
            bins = rho_bins

        counts = binned_statistic(prop,prop,statistic='count',bins=bins).statistic

        ax[iprop].plot(bins[:-1],counts/counts.sum(),label=sim.name,ds='steps-pre')

ax[0].set_xlabel("vrel [km/s]")
ax[0].set_ylabel("Fraction of total counts")
ax[1].set_xlabel("csound [km/s]")
ax[2].set_xlabel("rho [H/cc]")
ax[0].legend(framealpha=0.0)

for a in ax:
    # a.set_yscale('log')
    a.set_xscale('log')
    a.grid()

fig.savefig("figs/sink_local_props_hist.png",bbox_inches='tight')