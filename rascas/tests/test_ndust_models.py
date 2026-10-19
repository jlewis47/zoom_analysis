from gremlin.read_sim_params import ramses_sim

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

from scipy.spatial import KDTree

from zoom_analysis.read.read_data import read_data_ball

from zoom_analysis.halo_maker.assoc_fcts import get_gal_props_snap

from scipy.stats import binned_statistic_2d

sim = ramses_sim('/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756/')

snap=150
gid = 10

aexp = sim.get_snap_exps(snap)[0]

gprops = get_gal_props_snap(sim.path,snap,gid=gid)

pos = gprops['pos']
rmax = gprops['rmax']

print(gprops['host purity'], np.log10(gprops['mass']), pos, rmax)

rload = 0.00045

gas = read_data_ball(sim,snap,pos,rload,gid=gid,data_types=['gas'],tgt_fields=['metallicity','density','dust_bin01','dust_bin02','dust_bin03','dust_bin04'])

Zref = 5e-3
SioverSil = 0.163

ndust_rascas = gas['metallicity']/Zref * gas['density']  / 1.67e-24

ndust_yd = (gas['dust_bin01']+gas['dust_bin02']+gas['dust_bin03']/SioverSil+gas['dust_bin04']/SioverSil) * gas['density'] / 1.67e-24

fig,ax=plt.subplots(2,2,figsize=(12,12),layout='constrained')

density_bins = np.logspace(-6,3,15)
metallicity_bins = np.logspace(-4,0,15)

for iz,z in enumerate([ndust_rascas,ndust_yd]):


    img = binned_statistic_2d(gas['density']/ 1.67e-24,gas['metallicity'],z,'mean',bins=[density_bins,metallicity_bins]).statistic
    img_DTM = binned_statistic_2d(gas['density']/ 1.67e-24,gas['metallicity'],z/(gas['metallicity'] * gas['density']  / 1.67e-24),'mean',bins=[density_bins,metallicity_bins]).statistic


    pl_img=ax[0,iz].imshow(img.T,origin="lower",norm=LogNorm(1e-8,1e3),extent=np.log10([*density_bins[[0,-1]],*metallicity_bins[[0,-1]]]))
    pl_img_dtm=ax[1,iz].imshow(img_DTM.T,origin="lower",extent=np.log10([*density_bins[[0,-1]],*metallicity_bins[[0,-1]]]),cmap='hot')

    print(np.nanmax(img_DTM),np.nanmin(img_DTM))

    ax[1,iz].set_xlabel('density, Hpcc')

    ax[iz,0].set_ylabel('metallicity')

ax[0,0].set_title('RASCAS, Laursen+09 recipe')
ax[0,1].set_title('sum of YD dust density bins')

plt.colorbar(pl_img,ax=ax[0,:],label='mean assumed dust density in Hpcc units')
plt.colorbar(pl_img_dtm,ax=ax[1,:],label='mean DTM from assumed dust density')

fig.savefig('test_dust_density_models')


npath=100

gas_coords=np.transpose([gas['x'],gas['y'],gas['z']])
tree = KDTree(gas_coords)

xpath = np.linspace(0,rload,npath)
ypath = np.full_like(xpath,rload)
zpath = np.full_like(xpath,rload)

path_points = np.transpose([xpath,ypath,zpath])

#to box coords 
path_points += pos

path_dists,path_args = tree.query(path_points,k=1,distance_upper_bound=1/2**gas['ilevel'].min())


fig,ax=plt.subplots(1,2,layout='constrained',figsize=(8,4))

xpath_kpc = xpath * sim.unit_l(aexp)/3.06e21

# ax[2].plot(xpath_kpc,np.cumsum(ndust_rascas[path_args])/np.cumsum(ndust_yd[path_args]))
ax[1].plot(xpath_kpc,ndust_rascas[path_args],label='nhi * Z/5e-3 (rascas default), Hpcc')
ax[1].plot(xpath_kpc,ndust_yd[path_args],label='sum(dust density bins), Hpcc')
ax[1].plot(xpath_kpc,gas['density'][path_args]/1.67e-24,label='gas density, Hpcc')
# ax.plot(xpath_kpc,(ndust_yd/ndust_rascas)[path_args])
ax[0].plot(xpath_kpc,gas['metallicity'][path_args],label="metallicity")
ax[0].plot(xpath_kpc,gas['dust_bin01'][path_args],label="small carbon")
ax[0].plot(xpath_kpc,gas['dust_bin02'][path_args],label="large carbon")
ax[0].plot(xpath_kpc,gas['dust_bin03'][path_args]/SioverSil,label="small sillicates")
ax[0].plot(xpath_kpc,gas['dust_bin04'][path_args]/SioverSil,label="large sillicates")

ax[0].set_ylabel('metals')
ax[1].set_ylabel('dust density')

ax[1].set_xlabel('distance to gal centre, ckpc')

ax[0].set_yscale('log')
ax[1].set_yscale('log')
# ax[2].set_yscale('log')
ax[0].legend(framealpha=0)
ax[1].legend(framealpha=0)

fig.savefig('test_dusty_models_path')



