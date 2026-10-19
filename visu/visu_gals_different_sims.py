import numpy as np
import os
import matplotlib.pyplot as plt

from visu_fct import plot_fields, plot_zoom_BHs, plot_zoom_gals, plot_zoom_halos

from zoom_analysis.halo_maker.assoc_fcts import get_halo_props_snap, get_gal_props_snap, find_shared_gals

from gremlin.read_sim_params import ramses_sim

vmin = vmax = None
cmap = None
cb = False

directions = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

mode = "sum"
marker_color = "white"

# field = "density"
# cmap = "magma"
# vmin = 1e-26
# vmax = 1e-20

# field = "stellar mass"
# cmap = "viridis"
# vmin = 6e4
# vmax = 1e9

# field = "dm mass"
# cmap = "viridis"
# vmin=1e4
# vmax=1e10


mlim= 1e10

sim_dirs=[
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_lowerSFE_NHboost",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel",
    ]

tgt_zed = 2.4

hids,gids,snaps,hmasses,zeds = find_shared_gals(sim_dirs,tgt_zed,mlim,rfact=1.0)

nsims,ngals = hids.shape

nplots = nsims

nrows = int(np.floor(np.sqrt(nplots)))
ncols = int(np.ceil(nplots/nrows))

overlap = nrows*ncols - nplots

for igal in range(ngals):

    fig,ax = plt.subplots(nrows,ncols,figsize=(8*nrows,8*ncols))
    if overlap>0:
        for iax in range(nplots,nplots+overlap):
            ax[iax].off()

    ax = np.ravel(ax)

    for isim in range(nsims):

        ax[isim].grid('off')


        sim_dir = sim_dirs[isim]

        sim = ramses_sim(sim_dir, nml="cosmo.nml")
        ax[isim].set_title(f"{sim.name}")

        cur_zed = zeds[isim]

        snap = sim.get_closest_snap(zed=cur_zed)

        print(sim.name,snap,hids[isim,igal],gids[isim,igal])

        if hids[isim,igal] == -1 or gids[isim,igal] == -1:
            continue


        hprops,_ = get_halo_props_snap(sim.path,snap,hid=hids[isim,igal])

        _,gprops = get_gal_props_snap(sim.path,snap,gid=gids[isim,igal])


        hpos = hprops["pos"]
        gpos = gprops["pos"]
        hmass = hprops["mvir"]
        gmass = gprops["mass"]
        rvir = hprops["rvir"]
        r50 = gprops["r50"]

        tgt_pos = None
        rad_tgt=None

        if tgt_pos==None:
            tgt_pos = hpos
        if rad_tgt==None:
            rad_tgt= rvir*0.1

        zdist=rad_tgt / 1 * sim.cosmo.lcMpc * 1e3


        args = {
            "cb": cb,
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
            "mode": mode,
            "transpose": True,
            "color": marker_color,
            "hid": int(hids[isim,igal]),
        }

        img = plot_fields(field, fig, ax[isim], 1./(cur_zed+1), directions, tgt_pos, rad_tgt, sim, **args)        

        circ = plt.Circle((gpos[0],gpos[1]),r50,color=marker_color,fill=False)
        ax[isim].add_artist(circ)

        plot_zoom_halos(
            ax[isim],
            snap,
            sim,
            tgt_pos,
            rad_tgt,
            zdist,
            hm="HaloMaker_DM_dust",
            gal_markers=True,
            annotate=True,
            direction=directions[0],
            # transpose=True,
            **args,
        )

        plot_zoom_gals(
            ax[isim],
            snap,
            sim,
            tgt_pos,
            rad_tgt,
            zdist,
            hm="HaloMaker_stars2_dp_rec_dust",
            gal_markers=True,
            annotate=True,
            direction=directions[0],
            # transpose=True,
            **args,
        )      

        plot_zoom_BHs(
                ax[isim], snap, sim, tgt_pos, rad_tgt, zdist, direction=directions[0], **args
            )

    fout=f"./figs/visu_gals_different_sims_{igal}.png"
    print(f"Saving file {fout}")
    fig.savefig(fout)