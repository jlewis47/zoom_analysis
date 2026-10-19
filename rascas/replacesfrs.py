import numpy as np
import os

from gremlin.read_sim_params import ramses_sim
from zoom_analysis.halo_maker.assoc_fcts import get_gal_props_snap


cat_dir = "/data101/jlewis/cats2fix"

cats=[]

for root,dirs,files in os.walk(cat_dir):
    if "fixed" in root:continue
    for file in files:
        if file.endswith('.txt'):
            cats.append(os.path.join(root,file))

print(cats)

for fcat in cats:

    print(fcat)

    cat = np.genfromtxt(fcat,names=True,delimiter=" ",dtype=None)
    if not 'simulation' in cat.dtype.names:
        continue
    sims,snaps,gals = cat['simulation'].astype(str),cat['snapshot'],cat['galid']

    sfrs_100myr_rgal=np.zeros(len(gals))
    sfrs_10myr_rgal=np.zeros(len(gals))

    for igal,(sim,snap,gal) in enumerate(zip(sims,snaps,gals)):


        if sim.startswith('id'):
            sim_dir = os.path.join("/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/",sim)
        else:
            sim_dir = "/data101/jlewis/sims/NH_for_rascas"

        print(f"{igal:d}/{len(gals):d}")

        gprops = get_gal_props_snap(sim_dir,snap,gal)

        sfrs_100myr_rgal[igal] = gprops['sfr100']/1e6
        sfrs_10myr_rgal[igal] = gprops['sfr10']/1e6

    cat['sfr_100Myrmsunyr']=sfrs_100myr_rgal
    cat['sfr_10Myrmsunyr']=sfrs_10myr_rgal


    cat_path,name = os.path.split(fcat)

    replace_cat_path = cat_path.replace('cats2fix/','cats2fix/fixedcats/')

    os.makedirs(replace_cat_path,exist_ok=True)

    cat['simulation'] = cat['simulation'].astype(str)

    print(cat['simulation'])

    np.savetxt(os.path.join(replace_cat_path,name),cat,fmt='%s',header=' '.join(cat.dtype.names))

    