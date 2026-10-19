
from zoom_analysis.catalogue.read_cat import read_or_make_cat
# from zoom_analysis.stars.star_reader import yt_read_star_ball, read_part_ball
from zoom_analysis.stars.leja_quench_fit import sfr_ridge_leja22
from zoom_analysis.halo_maker.read_treebricks import (
    # read_brickfile,
    # convert_brick_units,
    convert_star_units,
    read_zoom_stars,
    # convert_star_time,
)
from astropy.cosmology import Planck18 as cosmo
from zoom_analysis.read.read_data import read_data_ball
from zoom_analysis.sinks.sink_reader import gid_to_sid, get_sink

# from zoom_analysis.trees.tree_reader import read_tree_file_rev as read_tree_rev_sim
from zoom_analysis.trees.tree_reader import (
    convert_adaptahop_pos,
    read_tree_file_rev_correct_pos as read_tree_rev_sim,
)
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import (
    find_zoom_tgt_gal,
    get_central_gal_for_hid,
    get_gal_props_snap,
    get_halo_props_snap,
    get_halo_assoc_file,
    get_assoc_pties_in_tree,
    smooth_props,
    find_snaps_with_gals,
    get_gal_assoc_file,
    get_halo_assoc_file,
)
from zoom_analysis.halo_maker.read_treebricks import read_zoom_stars

from zoom_analysis.zoom_helpers import find_starting_position, starting_hid_from_hagn, find_hagn_match_halo, decentre_coordinates, recentre_coordinates
from hagn.utils import get_hagn_sim
from hagn.tree_reader import read_tree_rev as read_tree_rev_hagn
from gremlin.read_sim_params import ramses_sim

from matplotlib.lines import Line2D

import os
import numpy as np
import h5py

# from hagn.association import gid_to_stars
from hagn.utils import get_hagn_sim, adaptahop_to_code_units
from hagn.tree_reader import interpolate_tree_position

from hagn.catalogues import make_super_cat, get_cat_hids

# from f90_tools.star_reader import read_part_ball_hagn, read_part_ball_NCdust

from matplotlib import gridspec, pyplot as plt
# from matplotlib.ticker import FormatStrFormatter

import argparse


# fig = plt.figure(figsize=(15,10),layout="constrained")
fig = plt.figure(figsize=(15,10),layout="tight")
# fig = plt.figure(figsize=(15,10))

#add gas mass?
#rotation support?
#outflows?
#galaxy merger tree instead?

gs = gridspec.GridSpec(3, 3, hspace=0.0)
ax=[]
ax.append(fig.add_subplot(gs[0,0]))
ax.append(fig.add_subplot(gs[0,1], sharex=ax[0]))
ax.append(fig.add_subplot(gs[1,0], sharex=ax[0]))
ax.append(fig.add_subplot(gs[1,1], sharex=ax[0]))
ax.append(fig.add_subplot(gs[0,2], sharex=ax[0]))
ax.append(fig.add_subplot(gs[1,2], sharex=ax[0]))
ax.append(fig.add_subplot(gs[2,0], sharex=ax[0]))
ax.append(fig.add_subplot(gs[2,1], sharex=ax[0]))
ax.append(fig.add_subplot(gs[2,2], sharex=ax[0]))

ax_sSFR, ax_SFR, ax_mass, ax_mBH, ax_size, ax_dummy,ax_vVSsigma,_, ax_legend   = ax

ax_sSFR.set_ylabel('sSFR, 1/yr')



ax_SFR.set_ylabel('SFR, M$_\odot$/yr')
ax_mass.set_ylabel('M$_\star$, M$_\odot$')
ax_mBH.set_ylabel('M$_{BH}$, M$_\odot$')
ax_size.set_ylabel('r50, ckpc')
ax_dummy.set_ylabel('Non main progenitor halo mass ratio')
ax_vVSsigma.set_ylabel('v/$\sigma$')
# ax_dummy.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

for a in ax[-4:-1]:
    a.set_xlabel('Time, Gyr')


# ax_mass.set_xlabel('Time, Gyr')
# ax_mBH.set_xlabel('Time, Gyr')
# ax_dummy.set_xlabel('Time, Gyr')



args = argparse.ArgumentParser(description="Get gals SFH and MBH growth and compare to HAGN")

args.add_argument("sim_dirs", type=str, nargs="+", help="List of zoom sim dirs to compare to HAGN")
args.add_argument("--ztgt", type=float,default=None, help="Target redshift")
args.add_argument("--gids", type=int, nargs="+", help="List of gids to compare to HAGN", default=None)
args.add_argument("--mlim", type=float, default=1e10, help="min stellar mass to search, not used if gids are provided")
args.add_argument("--ssfrlim", type=float, nargs=2, default=[-np.inf,+np.inf], help="ssfr100 boundaries to search, not used if gids are provided")
args.add_argument("--main_stars", help="Use main stars for SFH calculation", default=False, action="store_true")
args.add_argument("--fpure", type=float, help="Minimum host purity for selecting galaxies", default=0.9999)
args.add_argument("--overwrite", action='store_true', help="overwrite used catalogue data", default=False)

args=args.parse_args()


if args.main_stars:
    main_star_str = "mainStars"
else:
    main_star_str = ""

overwrite=args.overwrite

hagn_sim = get_hagn_sim()
hagn_sim.init_cosmo()
l_hagn = hagn_sim.unit_l(hagn_sim.aexp_stt) / (ramses_pc * 1e6) / hagn_sim.aexp_stt


sim_names=[]

for sdir in args.sim_dirs:

    if sdir.endswith("/"):
        sdir = sdir[:-1]
    name = sdir.split("/")[-1].strip()

    if name.startswith('id'):
        intID = int(name[2:].split("_")[0])
    else:
        intID=name

    sim_names.append(intID)

if len(np.unique(sim_names))==1:
    unique=True
else:
    unique=False

lines=[]
labels=[]

ztgt_args = args.ztgt

for sim_dir in args.sim_dirs:

    sim = ramses_sim(sim_dir)

    if ztgt_args is None:
        hsnaps = find_snaps_with_gals(sim.snap_numbers,sim.path)
        ztgt = 1./(sim.get_snap_exps(hsnaps).max())-1.
    else:
        ztgt = ztgt_args

    hagn_snap = hagn_sim.get_closest_snap(zed=ztgt)

    print(f"Running {sim_dir} at z={ztgt}")

    if sim.name == "NH_for_rascas" or sim.name=='NewHorizon':
        sim.zoom_ctr=np.array([0.5,0.5,0.5])

    sim.init_cosmo()

    snap = sim.get_closest_snap(zed=ztgt)
    

    if args.gids is None:
        try:
            gals = get_gal_props_snap(sim_dir, snap)#, main_stars=args.main_stars)
        except FileNotFoundError:
            print(f'no galaxies found at snap {snap}')
            continue
        mlim = gals['mass']>args.mlim
        ssfrlim = (gals['sfr100']/gals['mass']/1e6>args.ssfrlim[0])*(gals['sfr100']/gals['mass']/1e6<=args.ssfrlim[-1])

    else:
        try:
            gals = get_gal_props_snap(sim_dir, snap, gid=args.gids, main_stars=args.main_stars)
        except FileNotFoundError:
            print(f'no galaxies found at snap {snap}')
            continue
        mlim = True
        ssfrlim = True

    #only pure galaxies
    pure = gals['host purity']> args.fpure 

    for k in gals.keys():
        if type(gals[k]) in [list, np.ndarray]:
            if len(np.shape(gals[k]))>1:
                gals[k] = gals[k][:,mlim*pure]
            else:
                gals[k] = gals[k][mlim*pure]

    

    # nsteps = len(sim.snaps[sim.snap_numbers<=snap])
    # nstep_tree_hagn = len(hagn_sim.snap_numbers[hagn_sim.snap_numbers<=hagn_snap])    

    for igal,gid in enumerate(gals['gids']):

        color = None #new colour for every galaxy

        #look for hagn pair
        # if gals['central'][igal]:
        hid=gals['host hid'][igal]
        try:
            host_props = get_halo_props_snap(sim_dir, snap, hids=hid)[f'halo_{hid:07d}']
        except FileNotFoundError:
            print(f'no halos found at snap {snap}')
            continue
        # else:
        #     central_gal = get_gal_props_snap(sim_dir, snap, gid=gals['host gid'][igal], main_stars=args.main_stars)
        #     hid=central_gal['host hid']
        #     host_props = get_halo_props_snap(sim_dir, snap, hids=hid)[f'halo_{hid:07d}']

        host_pos = host_props['pos']
        # host_props['pos'] = decentre_coordinates(host_pos, sim.path)
        if hasattr(sim,"zoom_ctr"):
            if np.all((sim.zoom_ctr == [0.5,0.5,0.5])):
                host_props['pos'] = recentre_coordinates(host_pos,sim.path,sim.zoom_ctr)

        hagn_hid = find_hagn_match_halo(hagn_sim,ztgt,host_props,rtol=4.0,mtol=0.3)

        mstel_hagn = 0.0

        if hagn_hid is None:
            print(f"Didn't find matching HAGN halo, skipping {gid:d}, m={gals['mass'][igal]:.1e}")
        else:

            hagn_tree_hids, hagn_tree_datas, hagn_tree_aexps = read_tree_rev_hagn(
            ztgt,
            [hagn_hid],
            tree_type="halo",
            # [gid],
            # tree_type="gal",
            target_fields=["m", "x", "y", "z", "r"],
            sim="hagn",
            
            )

            nstep_tree_hagn = len(hagn_tree_aexps)  

            mstel_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
            mBH_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
            mvir_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
            sfr_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
            time_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
            edd_ratio_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)
            # size_hagn = np.zeros(nstep_tree_hagn, dtype=np.float32)

            for istep in range(nstep_tree_hagn):
                cur_snap = hagn_sim.get_closest_snap(aexp=hagn_tree_aexps[istep])

                hagn_cat = make_super_cat(cur_snap)
                matching=hagn_cat['hid']==hagn_tree_hids[0][istep]
                if np.any(matching):
                    cat_arg = np.where(matching)[0][0]
                    mstel_hagn[istep] = hagn_cat['mgal'][cat_arg]
                    mvir_hagn[istep] = hagn_cat['mhalo'][cat_arg]
                    sfr_hagn[istep] = hagn_cat['sfr100'][cat_arg]
                    # size_hagn[istep] = hagn_cat['r50'][cat_arg]
                    if 'mbh' in hagn_cat:
                        mBH_hagn[istep] = hagn_cat['mbh'][cat_arg]
                        sid_hagn = hagn_cat['sid'][cat_arg]

                        sink_hagn = get_sink(sid_hagn, cur_snap, hagn_sim, hagn=True)
                        edd_ratio_hagn[istep] = min(sink_hagn['dMBH_coarse']/sink_hagn['dMEd_coarse'],1)

                    time_hagn[istep] = hagn_sim.cosmo_model.age(1.0/hagn_tree_aexps[istep]-1.0).value * 1e3

            non_null_hagn = mstel_hagn>0



            mstel_hagn = mstel_hagn[non_null_hagn]
            mvir_hagn = mvir_hagn[non_null_hagn]
            sfr_hagn = sfr_hagn[non_null_hagn]
            time_hagn = time_hagn[non_null_hagn]
            mBH_hagn = mBH_hagn[non_null_hagn]
            edd_ratio_hagn = edd_ratio_hagn[non_null_hagn]
            ssfr_hagn = sfr_hagn/mstel_hagn            
            quasar_hagn = edd_ratio_hagn >=0.01
            radio_hagn = quasar_hagn==False

        sim_halo_tree_rev_fname = os.path.join(
            sim.path, "TreeMakerDM_dust", "tree_rev.dat"
        )
        sim_halo_tree_rev_fname2 = os.path.join(
            sim.path, "TreeDM_dust", "tree_rev.dat"
        )

        names = [sim_halo_tree_rev_fname,sim_halo_tree_rev_fname2]
        tree_hids = None
        for name in names:
            print(name,'/'.join(name.split('/')[:-1]))
            if os.path.exists(name):

                try:
                    tree_hids, tree_datas, tree_aexps = read_tree_rev_sim(
                    name,
                    sim,
                    snap,
                    # fbytes=os.path.join(sim.path, "TreeMakerDM_dust"),
                    fbytes='/'.join(name.split('/')[:-1]),
                    zstart=ztgt,
                    tgt_ids=[hid],
                    star=False,
                    verbose=False,
                    overwrite=True,
                    # overwrite=False,
                    speed_lim=1e5,
                    # speed_lim=np.inf,
                    fpure_min=0.999,
                    tgt_fields=["m", "x", "y", "z", "r","m_father"]
                    )  
                except:
                    print("tree failed for some reason")
                    pass
        if np.any(tree_hids==None):
            print("no tree")
            continue

        nsteps = len(tree_aexps)

        mstel_zoom = np.zeros(nsteps, dtype=np.float32)
        mBH_zoom = np.zeros(nsteps, dtype=np.float32)
        mvir_zoom = np.zeros(nsteps, dtype=np.float32)
        sfr_zoom = np.zeros(nsteps, dtype=np.float32)
        size_zoom = np.zeros(nsteps, dtype=np.float32)
        time_zoom = np.zeros(nsteps, dtype=np.float32) 
        edd_ratio_zoom = np.zeros(nsteps, dtype=np.float32) 
        vOverSigma_zoom = np.zeros(nsteps, dtype=np.float32)
        
        for istep in range(nsteps):
            cur_snap = sim.get_closest_snap(aexp=tree_aexps[istep])

            if os.path.exists(os.path.join(sim.path,"association",f"assoc_{cur_snap:03d}_halo_lookup.h5")):
                try:
                    hprops,gprops = get_halo_props_snap(sim_dir,cur_snap,hids=tree_hids[0][istep],hosted_gals=True)
                    print(istep,tree_aexps[istep],cur_snap,tree_hids[0][istep])
                except KeyError:
                    continue
            else:
                print(f'no halo found at step {istep}')
                continue

            # this_gprops = get_gal_props_snap(sim.path,cur_snap)
            # this_hid = tree_hids[0][istep]


            if "gids" in gprops:

                #most massive central
                # central = gprops['central']
                this_gids = gprops['gids']

                if type(gprops['mass']) not in [float,np.float64,np.float32]:
                    most_massive = np.argmax(gprops['mass'])
                    tgt_gid = gprops['gids'][most_massive]
                    gprops_tgt = get_gal_props_snap(sim.path,cur_snap,tgt_gid)
                else:
                    gprops_tgt = gprops

                
                mstel_zoom[istep] = gprops_tgt['mass']
                size_zoom[istep] = gprops_tgt['r50']
                mvir_zoom[istep] = gprops_tgt['host mass']
                sfr_zoom[istep] = gprops_tgt['sfr100']

                # try:
                #     cat = read_or_make_cat(sim,cur_snap,'galaxy','r50',rfact=1.0,id=tgt_gid,tgt_keys=['vrot_kms','vdisp_kms','Mgas_msun','tot_outflow_kms'],overwrite=overwrite)
                #     # cat = read_or_make_cat(sim,cur_snap,'halo','rvir',rfact=0.1,id=gprops_tgt['host hid'],tgt_keys=['vrot_kms','vdisp_kms'])
                #     vOverSigma_zoom[istep] = abs(cat['stars']['vrot_kms'])/cat['stars']['vdisp_kms']
                # except (AssertionError,KeyError,TypeError):
                #     pass

                

                sid,found,step = gid_to_sid(sim, gprops_tgt['gids'], cur_snap, gprops_tgt)
                if found:
                    sprops = get_sink(sid, cur_snap, sim)

                    mBH_zoom[istep] = sprops['mass']
                    edd_ratio_zoom[istep] = min(sprops['dMBH_coarse']/sprops['dMEd_coarse'],1)

                time_zoom[istep] = sim.cosmo_model.age(1.0/tree_aexps[istep]-1.0).value * 1e3

        non_null = mstel_zoom>0

        mstel_zoom = mstel_zoom[non_null]
        size_zoom = size_zoom[non_null]
        mvir_zoom = mvir_zoom[non_null]
        sfr_zoom = sfr_zoom[non_null]
        mBH_zoom = mBH_zoom[non_null]
        vOverSigma_zoom = vOverSigma_zoom[non_null]
        delta_m_zoom = 1.0 - tree_datas["m_father"][0][non_null] / 100.0
        time_zoom = time_zoom[non_null]
        edd_ratio_zoom = edd_ratio_zoom[non_null]

        quasar_zoom = edd_ratio_zoom >=0.01
        radio_zoom = quasar_zoom==False

        ssfr_zoom = sfr_zoom/mstel_zoom

        size_zoom*= sim.cosmo.lcMpc*1e3

        print(sim.name,len(mstel_zoom),non_null.sum())

        # print(len(time_zoom),len(ssfr_zoom),color)

        l = ax_sSFR.errorbar(
            time_zoom * 1e-3,
            ssfr_zoom/1e6,
            # xerr=t_err,
            color=color,
            # ls=zoom_ls[zoom_style],
            # label=label,
            # **kwargs,
        )

        if color is None:
            color = l[0].get_color()

        ax_SFR.errorbar(
            time_zoom * 1e-3,
            sfr_zoom/1e6,
            # xerr=t_err,
            color=color,
            # ls=zoom_ls[zoom_style],
            # label=label,
            # **kwargs,
        )    

        ax_mass.errorbar(
            time_zoom * 1e-3,
            mstel_zoom,
            # xerr=t_err,
            color=color,
            # ls=zoom_ls[zoom_style],
            # label=label,
            # **kwargs,
        )    

        ax_mBH.errorbar(
            time_zoom * 1e-3,
            mBH_zoom,
            # xerr=t_err,
            color=color,
            ls='-',
            # ls=zoom_ls[zoom_style],
            # label=label,
            # **kwargs,
        )    
        # ax_mBH.errorbar(
        #     time_zoom[radio_zoom] * 1e-3,
        #     mBH_zoom[radio_zoom],
        #     # xerr=t_err,
        #     color=color,
        #     ls='-',
        #     # ls=zoom_ls[zoom_style],
        #     # label=label,
        #     # **kwargs,
        # )    
        # ax_mBH.errorbar(
        #     time_zoom[quasar_zoom] * 1e-3,
        #     mBH_zoom[quasar_zoom],
        #     # xerr=t_err,
        #     color=color,
        #     ls='-',
        #     # lw=3,
        #     marker='|',
        #     # markerevery=5,
        #     # ls=zoom_ls[zoom_style],
        #     # label=label,
        #     # **kwargs,
        # )    

        ax_size.errorbar(
            time_zoom * 1e-3,
            size_zoom,
            # xerr=t_err,
            color=color,
            # ls=zoom_ls[zoom_style],
            # label=label,
            # **kwargs,
        )  

        ax_vVSsigma.errorbar(
            time_zoom*1e-3,
            vOverSigma_zoom,
            color=color,
        )  

        ax_dummy.plot(time_zoom * 1e-3,
                    delta_m_zoom,
                    color=color,
                    lw=1,
                    label=f"{sim.name:s}:gid={gid:d}"
                      )

        labels.append(f"{sim.name:s}:gid={gid:d}")
        lines.append(Line2D([0], [0], color=color, lw=1))


        if hagn_hid is not None:
            l = ax_sSFR.errorbar(
                time_hagn * 1e-3,
                ssfr_hagn,
                # xerr=t_err,
                color=color,
                lw=1
                # label=label,
                # **kwargs,
            )

            ax_SFR.errorbar(
                time_hagn * 1e-3,
                sfr_hagn,
                # xerr=t_err,
                color=color,
                lw=1
                # # label=label,
                # **kwargs,
            )    

            ax_mass.errorbar(
                time_hagn * 1e-3,
                mstel_hagn,
                # xerr=t_err,
                color=color,
                lw=1
                # # label=label,
                # **kwargs,
            )    

            ax_mBH.errorbar(
                time_hagn * 1e-3,
                mBH_hagn,
                # xerr=t_err,
                color=color,
                # lw=1.5,
                ls='-',
                marker='|',
                # # label=label,
                # **kwargs,
            )
            # ax_mBH.errorbar(
            #     time_hagn[quasar_hagn] * 1e-3,
            #     mBH_hagn[quasar_hagn],
            #     # xerr=t_err,
            #     color=color,
            #     # lw=1.5,
            #     ls='-',
            #     marker='|',
            #     # # label=label,
            #     # **kwargs,
            # )
            # ax_mBH.errorbar(
            #     time_hagn[radio_hagn] * 1e-3,
            #     mBH_hagn[radio_hagn],
            #     # xerr=t_err,
            #     color=color,
            #     lw=1,
            #     ls='-',
            #     # # label=label,
            #     # **kwargs,
            # )

ax_dummy.axhline(0.5,ls='--',c='k',label='')

xtime = ax[0].get_xticks()
ax_sSFR.plot(xtime, 0.2/(xtime*1e9),ls='--',c='k',label='sSFR=0.2/t yr^${-1}$')
lines.append(Line2D([0], [0], color='k', lw=1, ls='--'))
labels.append('sSFR=0.2/t yr$^{-1}$')
ax_legend.legend(lines,labels,framealpha=0,loc='center right')


xlim = ax[0].get_xlim()

y0 = ax[0].twiny()
y0.set_xlim(xlim)
zlabels = [f"{cosmo.age(zed).value:.1f}" for zed in ax[0].get_xticks()]
# y0.set_xticks(ax[0].get_xticks())
y0.set_xticklabels(zlabels)
y0.set_xlabel("redshift")

y1 = ax[1].twiny()
y1.set_xlim(xlim)
zlabels = [f"{cosmo.age(zed).value:.1f}" for zed in ax[1].get_xticks()]
# y1.set_xticks(ax[1].get_xticks())
y1.set_xticklabels(zlabels)
y1.set_xlabel("redshift")

y1 = ax[4].twiny()
y1.set_xlim(xlim)
zlabels = [f"{cosmo.age(zed).value:.1f}" for zed in ax[1].get_xticks()]
# y1.set_xticks(ax[1].get_xticks())
y1.set_xticklabels(zlabels)
y1.set_xlabel("redshift")

for a in ax[:-4]:
    a.set_yscale('log')

for a in ax[-1:]:
    a.axis('off')

xlim=ax_dummy.get_xlim()
ax_dummy.set_xlim(xlim)

ax_dummy.set_ylim(0,1.0)

# ax_dummy.fill(ax_dummy.get_xlim(),0.0,y2=0.5,color='blue',alpha=0.2)
# ax_dummy.text(0.5,0.01,"No or minor merger",ha='centre',va='centre',color='blue')
ax_dummy.fill_between(xlim,0.5,y2=1.0,color='red',alpha=0.2)
ax_dummy.text(0.5,0.95,"Major merger",ha='center',va='center',color='red',transform=ax_dummy.transAxes)
   
if not unique:
    fname=f"gals_to_history_vs_HAGN.png"
elif len(args.sim_dirs)==1:
    fname=f"gals_to_history_vs_HAGN_{name}.png"
else:
    fname=f"gals_to_history_vs_HAGN_{intID}.png"

fig_dir = os.path.join("figs", main_star_str)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
f_fig = os.path.join(fig_dir, fname)
print("saving to ", f_fig)
fig.savefig(f_fig)

        



