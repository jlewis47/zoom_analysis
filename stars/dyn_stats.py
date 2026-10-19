import matplotlib.pyplot as plt
import numpy as np

# from zoom_analysis.halo_maker.assoc_fcts import get_gal_props_snap
from zoom_analysis.catalogue.read_cat import read_cat,read_or_make_cat,obj_based_to_table
from zoom_analysis.sinks.sink_reader import gid_to_sid, read_sink_snap

from gremlin.read_sim_params import ramses_sim


sim_dirs = ["/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380/",
            "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130/",
            "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id36755/",
            "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
            "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288",
            "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
            "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
]

mbulge=[]
fbulge=[]
mstar=[]
mBH=[]
vrot=[]
vdisp=[]

for sim_dir in sim_dirs:

    sim = ramses_sim(sim_dir)

    # snap = sim.get_closest_snap(zed=2.)
    snap = sim.get_closest_snap(zed=3.8)

    cat = read_or_make_cat(sim.path,snap,'galaxy','r50',rfact=1)
    

    this_vrot = obj_based_to_table(cat,'stars/vrot_kms')
    this_vdisp = obj_based_to_table(cat,'stars/vdisp_kms')
    this_fdisk = obj_based_to_table(cat,'stars/fdisk')
    this_fbulge = obj_based_to_table(cat,'stars/fbulge')
    this_mdisk = obj_based_to_table(cat,'stars/mdisk_msun')
    this_mstar = obj_based_to_table(cat,'stars/Mstar_msun')
    this_SFR100 = obj_based_to_table(cat,'stars/sfr100_Msun_per_yr')
    this_Zstar = obj_based_to_table(cat,'stars/Zstar')
    this_age_max_Myr = obj_based_to_table(cat,'stars/age_max_Myr')

    gids = cat['galaxy_ids']

    print(sim.name,this_mstar.sum())

    if len(gids)==0:continue


    this_mBH = np.zeros_like(this_mstar)

    sinks = read_sink_snap(sim, snap)

    step=None

    for igal,gid in enumerate(gids):

        sid,found,step = gid_to_sid(sim,gid,snap,step=step)
        if found:
            sid_arg = sinks['identity']==sid
            # if sid_arg.sum()>0:
            this_mBH[igal] = sinks['mass'][sid_arg]




    this_mbulge = this_mdisk / this_fdisk * this_fbulge

    mstar.append(this_mstar)
    mBH.append(this_mBH)
    mbulge.append(this_mbulge)
    fbulge.append(this_fbulge)
    vrot.append(this_vrot)
    vdisp.append(this_vdisp)

fig,ax = plt.subplots(1,2,figsize=(12,5))

sc=ax[0].scatter(np.concatenate(mstar),np.concatenate(vrot)/np.concatenate(vdisp), c=np.concatenate(fbulge), label='fbulge')
plt.colorbar(sc,ax=ax[0])

sc=ax[1].scatter(np.concatenate(mstar), np.concatenate(mBH), c=np.concatenate(fbulge), label='fbulge')
plt.colorbar(sc,ax=ax[1])

ax[0].set_xlabel('Mstar/Msun')
ax[1].set_xlabel('Mstar/Msun')

ax[0].set_ylabel('Vrot/Vsig/Msun')
ax[1].set_ylabel('MBH/Msun')

for a in ax:
    a.set_xscale('log')
    a.set_yscale('log')


fig.savefig('figs/test_morpho.png')