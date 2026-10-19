import numpy as np
from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from zoom_analysis.visu.visu_fct import basis_from_vect
# from matplotlib.colors import LogNorm
# from matplotlib.patches import Circle
import os

from hagn.utils import get_hagn_sim
from hagn.IO import read_hagn_snap_brickfile, read_hagn_sink_bin
from scipy.spatial import KDTree


from zoom_analysis.constants import *
from zoom_analysis.halo_maker.read_treebricks import read_zoom_brick
from zoom_analysis.sinks.sink_reader import (
    check_if_superEdd,
    # find_zoom_massive_central_sink,
    hid_to_sid,
    read_sink_bin,
)

from zoom_analysis.stars.sfhs import correct_mass
from zoom_analysis.visu.visu_fct import (
    # make_amr_img,
    make_amr_img_smooth,
    plot_stars,
    # make_yt_img,
    plot_zoom_BHs,
    plot_zoom_gals,
    plot_zoom_halos,
    plot_trail,
    plot_fields,
)

# from f90_tools.star_reader import read_part_ball_NCdust, read_part_ball_hagn

# from zoom_analysis.halo_maker.read_treebricks import read_zoom_stars
from zoom_analysis.halo_maker.assoc_fcts import (
    # find_zoom_tgt_gal,
    # get_gal_assoc_file,
    # find_snaps_with_gals,
    find_snaps_with_halos,
    get_gal_props_snap,
    get_halo_props_snap,
)

from zoom_analysis.sinks.sink_reader import (
    get_sink_mhistory,
    # find_zoom_central_massive_sink,
)
from zoom_analysis.zoom_helpers import (
    decentre_coordinates,
    recentre_coordinates,
    starting_hid_from_hagn,
)

lines=[]
labels=[]

ztgt=2.51

sims = [
"/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
"/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_NClike",
"/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_model5_100eps",
# "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_no_sink_desc",
# "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_sink_desc",
# "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_sink_desc_coarse",
# "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag",
# "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130/",
# "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id180130_novrel_lowerSFE_stgNHboost_strictSF/",
]

fig = plt.figure(figsize=(12,8),layout='tight')

gs = GridSpec(5,2)

ax_mdot = fig.add_subplot(gs[0,:])

ax_mass = fig.add_subplot(gs[1,0])
ax_dens = fig.add_subplot(gs[1,1])
ax_cs = fig.add_subplot(gs[2,0])
ax_vrel = fig.add_subplot(gs[2,1])
ax_mstar = fig.add_subplot(gs[3,0])
ax_dist = fig.add_subplot(gs[3,1])
ax_sfr = fig.add_subplot(gs[4,0])
ax_leg = fig.add_subplot(gs[4,1])


zmin=2

for sim_dir in sims:

    hagn = False
    delta_aexp = 0.05

    intID = int(sim_dir.split("/")[-1].split("_")[0][2:])

    if not hagn:
        sim = ramses_sim(sim_dir, nml="cosmo.nml")
        bh_fct = read_sink_bin
        gal_fct = read_zoom_brick
        hm = "HaloMaker_stars2_dp_rec_dust/"
        hm_dm = "HaloMaker_DM_dust/"

    else:
        # for getting hagn plots... need to lookup massive hagn sink id. Can use one of the codes in ../sinks
        sim = get_hagn_sim()
        bh_fct = read_hagn_sink_bin
        gal_fct = read_hagn_snap_brickfile
        hm = None
        # intID = 242756
        sim_dir = os.path.join("/data101/jlewis/hagn", "maps", "halos", f"{intID}")

    name = sim.name


    snaps = sim.snap_numbers
    sim_aexps = sim.get_snap_exps(param_save=False)
    sim_times = sim.get_snap_times(param_save=False)
    sim_snaps = sim.snap_numbers


    zstt = 2.0
    
    # tgt_zed = 2.0
    max_zed = 8.0
    zmax=max_zed

    # delta_t = 5  # Myr
    # every_snap = True  # try and interpolate between tree nodes if not found




    # sim = ramses_sim(sim_dir, nml="cosmo.nml")
    sim.init_cosmo()
    l = sim.unit_l(sim.aexp_stt) / (ramses_pc * 1e6) / sim.aexp_stt

    h_snaps = find_snaps_with_halos(snaps, sim.path)[:-1]
    h_aexps = sim.get_snap_exps(h_snaps)
    h_times = sim.get_snap_times(h_snaps)

    decal = len(h_aexps) - 1
    
    ztgt = 1.0 / h_aexps[decal] - 1

    sink_files = os.listdir(sim.sink_path)
    sink_fnbs = np.asarray([int(f.split("_")[1].split(".")[0]) for f in sink_files])
    last_sink_file = sink_files[np.argmax(sink_fnbs) - 1]

    last_sinks = read_sink_bin(
        os.path.join(sim.sink_path, last_sink_file), tgt_fields=["identity", "mass"]
    )
    last_sink_aexp = last_sinks["aexp"]
    last_sink_zed = 1.0 / last_sink_aexp - 1.0

    # closest_arg = np.argmin(np.abs(1.0 / h_aexps - 1.0 - last_sink_aexp))
    # mass = last_sinks["mass"]
    # mass_order = np.sort(mass)
    # mass_arg = np.where(mass == mass_order[-4])[0][0]
    # sid = last_sinks["identity"][mass_arg]

    find_sink_zed = max(ztgt, last_sink_zed)


    last_snap_aexp = h_aexps[-1]
    last_snap_zed = 1.0 / last_snap_aexp - 1.0


    print(last_snap_zed, find_sink_zed)

    found = False

    # while not found and decal > 0:

    print(1.0 / h_aexps[decal] - 1)

    sim_snap = sim.get_closest_snap(zed=ztgt)

    gprops = get_gal_props_snap(sim.path,sim_snap)

    massive = gprops['mass']>5e9

    hids = gprops['host hid'][massive]

    for hid_start in hids:


        sid, found = hid_to_sid(sim, hid_start, sim_snap, debug=False)


        # sid = 3

        # assert found, "Didn't find a sink at "

        print("Found starting massive sink:", sid, "at z=", 1.0 / h_aexps[decal] - 1)

        # sid = find_zoom_massive_central_sink(sim, 1.0 / (find_sink_zed + 1.0))
        # sid = 120

        sink_hist = get_sink_mhistory(
            sid,
            sim_snap,
            sim,
            hagn=hagn,
            out_keys=[
                ("position", 3),
                ("mass", 1),
                ("dMBH_coarse", 1),
                ("dMEd_coarse", 1),
                ("dens", 1),
                ("csound", 1),
                ("vrel", 1),
            ],
        )

        found_zed = 1/sim.get_snap_exps(sim_snap)[-1]-1

        zmin = min(zmin,found_zed)
        # zmax = max(zmax,sink_hist["zeds"].max())

        print(f"found sink mass {sink_hist["mass"].max():.1e}")

        sink_hist_time = sim.cosmo_model.age(sink_hist["zeds"]).value * 1e3  # Myr

        print("Got sink history")

        dists2sink = np.full_like(sink_hist_time,-1)


        if check_if_superEdd(sim):
            sink_growth = np.min([sink_hist["dMEd_coarse"], sink_hist["dMBH_coarse"]], axis=0)
        else:
            sink_growth = sink_hist["dMBH_coarse"]

        # print(sink_hist["zeds"].min(), sink_hist["zeds"].max())
        ngals = len(h_snaps)
        gal_masses = np.full_like(sink_hist_time,-1)
        gal_sfrs = np.full_like(sink_hist_time,-1)

        print('computing distance evolution')
        
        for igal,(snap, aexp, time) in enumerate(zip(h_snaps[::-1], h_aexps[::-1], h_times[::-1])):

            print(snap, 1.0 / aexp - 1)

            zed = 1.0 / aexp - 1.0
            if zed > max_zed or zed < find_sink_zed:
                print("outside of redshift range")
                continue

            l_pMpc = sim.cosmo.lcMpc * aexp

            # zed_diff = np.abs(zed - sink_hist["zeds"])
            # # print(zed_diff.min(), zed, sink_hist["zeds"][np.argmin(zed_diff)])
            # if np.all(zed_diff > 0.1):
            #     continue
            # sink_hist_arg = np.argmin(zed_diff)

            aexp_diff = np.abs(aexp - 1.0 / (sink_hist["zeds"] + 1.0))
            # print(zed_diff.min(), zed, sink_hist["zeds"][np.argmin(zed_diff)])
            if np.all(aexp_diff > delta_aexp):
                print("no sink data for this snap")
                continue

            sink_hist_arg = np.argmin(aexp_diff)

            if (
                sink_hist["zeds"][sink_hist_arg] > zed + delta_aexp
                or zed > sink_hist["zeds"].max()
            ):
                print("outside of sink history")
                continue


            # print(zed, sink_hist["zeds"][sink_hist_arg])
            tgt_pos = sink_hist["position"][sink_hist_arg]  # * sim.cosmo.lcMpc * 1e3
            cur_mass = sink_hist["mass"][sink_hist_arg]
            cur_mdot = sink_growth[sink_hist_arg]
            cur_dens = sink_hist["dens"][sink_hist_arg]
            cur_cs = sink_hist["csound"][sink_hist_arg]
            cur_vrel = sink_hist["vrel"][sink_hist_arg]


            halos_snap = get_halo_props_snap(sim.path, snap)

            rvirs = np.asarray(
                [halos_snap[halo_k]["rvir"] for halo_k in halos_snap if "halo_" in halo_k]
            )
            hpos = np.asarray(
                [halos_snap[halo_k]["pos"] for halo_k in halos_snap if "halo_" in halo_k]
            )

            fpure = halos_snap["fpure"] > 1 - 1e-4

            # h_tree = KDTree([tgt_pos], boxsize=1 + 1e-6)
            # sink_args = h_tree.query_ball_point(hpos[fpure], rvirs[fpure])

            hdists = np.linalg.norm(hpos[fpure]-tgt_pos,axis=1)
            sink_args = np.where(hdists<rvirs[fpure])[0]

            # for isink, sink_arg in enumerate(sink_args):
            #     if len(sink_arg) > 0:
            #         contains_sink = isink
            #         break

            hmasses = halos_snap["mvir"][fpure][sink_args]

            if len(hmasses)>0:
                possible_hids = halos_snap["hid"][fpure][sink_args]

                cur_hid = possible_hids[np.argmax(hmasses)]

                hprops,cur_gals = get_halo_props_snap(sim.path,snap,hids=cur_hid,hosted_gals=True)
                if 'gids' in cur_gals: #halos have galaxies
                    cur_gids = cur_gals['gids']
                    cur_gals = get_gal_props_snap(sim.path,snap,cur_gids)

                    gals_pos = cur_gals['pos']
                    gals_r= cur_gals['rmax']
                    gals_mass= cur_gals['mass']
                    gals_sfr= cur_gals['sfr100']


                    if type(gals_r) == np.ndarray:
                        try:
                            gal2sink_dist = np.linalg.norm(gals_pos.T-tgt_pos,axis=1)
                            sink_in_gal_args = np.where(gal2sink_dist<gals_r*3)[0]

                            host_arg = sink_in_gal_args[np.argmax(gals_mass[sink_in_gal_args])]

                            dists2sink[sink_hist_arg] = gal2sink_dist[host_arg] * sim.cosmo.lcMpc*1e3
                            gal_masses[sink_hist_arg] = gals_mass[host_arg]
                            gal_sfrs[sink_hist_arg] = gals_sfr[host_arg]
                        except (IndexError,ValueError):
                            print('no nearby galaxy! setting distance to -1')
                            # break

                    else:
                        dists2sink[sink_hist_arg] = np.linalg.norm(gals_pos.T-tgt_pos) * sim.cosmo.lcMpc*1e3
                        gal_masses[sink_hist_arg] = gals_mass
                        gal_sfrs[sink_hist_arg] = gals_sfr

                cur_dist2sink = dists2sink[sink_hist_arg]        


        print(f'{sim.name} done')

        ax_mdot.plot(sink_hist["zeds"], sink_growth, lw=1)
        # ax_mdot.scatter(
        #     found_zed,
        #     cur_mdot,
        #     marker='+',
        #     s=100,
        #     zorder=999,
        #     ec='k'
        # )
        ax_mdot.set_yscale("log")
        # ax_mdot.grid()

        ax_mdot.set_xlabel("redshift")
        ax_mdot.set_ylabel("BH growth rate, $M_\odot$/yr")
        ax_mdot.set_xlim(xmax=max_zed, xmin=zmin)
        ax_mdot.invert_xaxis()

        ax_mass.plot(
            sink_hist["zeds"],
            sink_hist["mass"],
            lw=1,
        )
        # ax_mass.scatter(
        #     found_zed,
        #     cur_mass,
        #     s=100,
        #     marker='+',
        #     zorder=999,
        #     ec='k'
        # )
        ax_mass.set_yscale("log")
        # ax_mass.grid()

        ax_mass.set_xlabel("redshift")
        ax_mass.set_ylabel("BH mass, $M_\odot$")

        ax_mass.set_xlim(xmax=max_zed, xmin=zmin)
        ax_mass.invert_xaxis()

        ax_dens.plot(
            sink_hist["zeds"],
            sink_hist["dens"],
            lw=1,
        )

        # ax_dens.scatter(zed, cur_dens, s=100, marker='+', zorder=999,ec='k')

        ax_dens.set_yscale("log")
        # ax_dens.grid()

        ax_dens.set_xlabel("redshift")
        ax_dens.set_ylabel("cloud density, H/cm$^3$")
        ax_dens.set_xlim(xmax=max_zed, xmin=zmin)
        ax_dens.invert_xaxis()

        ax_vrel.plot(
            sink_hist["zeds"],
            sink_hist["vrel"],
            lw=1,
        )

        # ax_vrel.scatter(
        #     found_zed,
        #     cur_vrel,
        #     s=100,
        #     marker='+',
        #     zorder=999,
        #     ec='k'
        # )

        ax_vrel.set_yscale("log")
        # ax_vrel.grid()

        ax_vrel.set_xlabel("redshift")
        ax_vrel.set_ylabel("vrel, km/s")
        ax_vrel.set_xlim(xmax=max_zed, xmin=zmin)
        ax_vrel.invert_xaxis()

        ax_cs.plot(
            sink_hist["zeds"],
            sink_hist["csound"],
            lw=1,
        )

        # ax_cs.scatter(
        #     found_zed,
        #     cur_cs,
        #     s=100,
        #     marker='+',
        #     zorder=999,
        #     ec='k'
        # )

        ax_cs.set_yscale("log")
        # ax_cs.grid()

        ax_cs.set_xlabel("redshift")
        ax_cs.set_ylabel("sound speed, km/s")
        ax_cs.set_xlim(xmax=max_zed, xmin=zmin)
        ax_cs.invert_xaxis()


        l,=ax_dist.plot(
            sink_hist["zeds"][dists2sink>=0],
            dists2sink[dists2sink>=0] / (1+sink_hist["zeds"][dists2sink>=0]) ,
            lw=1
        )



        # sc=ax_dist.scatter(
        #     found_zed,
        #     cur_dist2sink/(zed+1),
        #     s=100,
        #     marker='+',
        #     zorder=999,
        #     ec='k'
        # )

        ax_dist.set_yscale("log")
        # ax_dist.grid()

        ax_dist.set_xlabel("redshift")
        ax_dist.set_ylabel("distance from host galaxy, kpc")
        ax_dist.set_xlim(xmax=max_zed, xmin=zmin)
        ax_dist.invert_xaxis()

        l,=ax_mstar.plot(
            sink_hist["zeds"][dists2sink>=0],
            gal_masses[dists2sink>=0] / (1+sink_hist["zeds"][dists2sink>=0]) ,
            lw=1
        )

        # sc=ax_mstar.scatter(
        #     found_zed,
        #     cur_dist2sink/(zed+1),
        #     s=100,
        #     marker='+',
        #     zorder=999,
        #     ec='k'
        # )

        ax_mstar.set_yscale("log")
        # ax_mstar.grid()

        ax_mstar.set_xlabel("redshift")
        ax_mstar.set_ylabel("Galaxy mass, Msun")
        ax_mstar.set_xlim(xmax=max_zed, xmin=zmin)
        ax_mstar.invert_xaxis()


        l,=ax_sfr.plot(
            sink_hist["zeds"][dists2sink>=0],
            (gal_sfrs[dists2sink>=0] / 1e6) / (1+sink_hist["zeds"][dists2sink>=0]) ,
            lw=1
        )

        # sc=ax_sfr.scatter(
        #     found_zed,
        #     cur_dist2sink/(zed+1),
        #     s=100,
        #     marker='+',
        #     zorder=999,
        #     ec='k'
        # )

        ax_sfr.set_yscale("log")
        # ax_sfr.grid()

        ax_sfr.set_xlabel("redshift")
        ax_sfr.set_ylabel("SFR100, Msun/yr")
        ax_sfr.set_xlim(xmax=max_zed, xmin=zmin)
        ax_sfr.invert_xaxis()


        labels.append(f"{sim.name}, sid={sid}")
        # lines.append((l,sc))
        lines.append((l))


    # plt.tight_layout()

for a in ax_mass,ax_dens,ax_dist,ax_mdot,ax_vrel,ax_cs,ax_mstar,ax_sfr:
    a.set_xlim(zmax,zmin)
#     a.grid()

ax_leg.legend(lines,labels,framealpha=0.0)
ax_leg.grid()
ax_leg.axis('off')

fout = "./sink_mdot_ingredients_multi_sinks.png"

print(f"writing {fout}")

fig.savefig(
    fout,
    dpi=300,
    format="png",
)

plt.close()

    # break
