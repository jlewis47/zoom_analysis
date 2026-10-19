from zoom_analysis.sinks.sink_reader import read_sink_snap, find_massive_sink, get_sink_mhistory, sid_to_hid, gid_to_sid
from zoom_analysis.halo_maker.assoc_fcts import get_gal_assoc_file, get_gal_props_snap, get_halo_props_snap
from gremlin.read_sim_params import ramses_sim

import matplotlib.pyplot as plt
import os
import numpy as np

from zoom_analysis.sinks.sink_constraints import (
    calculate_reines15_mbh_bulges,
    calculate_reines15_mbh_local,
    NH_smbh,
)

sim_dirs=[
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_merge_main_branch",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel_lowerSFE_stgNHboost_stricterSF",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_superEdd_drag",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_DynBondiGravSinkMass",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_form_stellar_peak",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_form_stellar_peak_boostAcc",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_form_stellar_peak_boostAcc_1d7_dynMass",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_no_sink_desc",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_lowSNe",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_boostAcc",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_boostAcc_lowSNe",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_boostAcc_novrel",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_boostAcc_lowerSFE",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_boostAcc_superEdd",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_boostAcc_noSpin",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_boostAcc_lowTherm",
    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_nsc_biernacki_1d7_dynSeed_boostAcc_lowKin",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_sink_desc_goodfix",
    # "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_drag",

    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_NClike",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_model5_100eps",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_model0_0p02eps",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_model0_0p05eps",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_model0_0p10eps",
    # "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380_manual_sink1D7",

]


zmin=2.7

ymax_plot=-np.inf
ymin_plot=+np.inf


hist_keys=[
            ("position", 3),
            ("zeds", 1),
            ("coarse_step", 1),
            ("mass", 1),
            # ("position",3),
            # ("velocity",3),
            ("birth_time", 1),
            ("dMBH_coarse", 1),
            ("dMEd_coarse", 1),
            ("dMsmbh", 1),
            ("vrel", 1),
            ("csound", 1),
        ]


fig, ax = plt.subplots(1, 1, figsize=(8, 8))

zmax=12
tstep = 50

zscale = np.arange(zmin,zmax,0.3)


all_same_sim = True


if ramses_sim(sim_dirs[0], nml="cosmo.nml", full_check=False).name.startswith('id'):
    prev_ID = int(
        ramses_sim(sim_dirs[0], nml="cosmo.nml", full_check=False).name.split("_")[0][2:]
    )
else:
    prev_ID = ramses_sim(sim_dirs[0], nml="cosmo.nml", full_check=False).name

for sim_dir in sim_dirs[1:]:

    sim = ramses_sim(sim_dir, nml="cosmo.nml", full_check=False)
    if sim.name.startswith('id'):
        intID = int(sim.name.split("_")[0][2:])
    else:
        intID = sim.name
        if intID == "NH_for_rascas":
            intID = "NewHorizon"

    all_same_sim = all_same_sim * (intID == prev_ID)

    prev_ID = intID

for sdir in sim_dirs:


    sim = ramses_sim(sdir)

    times = sim.get_snap_times()

    # last_snap = sim.snap_numbers.max()
    last_snap = sim.get_closest_snap(zed=zmin)
    found=False

    while not found:
        last_time = sim.get_snap_times(last_snap)[0]

        # last_sinks = read_sink_snap(sim,last_snap)

        # most_massive = np.argmax(last_sinks['mass'])

        try:
            last_gals = get_gal_props_snap(sim.path,last_snap)
            found=True
        except FileNotFoundError:
            last_snap-=1
            continue

    if np.abs((1/sim.get_snap_exps(last_snap)[0]-1)-zmin) > 0.1:
        continue

    fpure = last_gals['host purity']>0.9999

    most_massive_pure = np.argmax(last_gals['mass'][fpure])
    gid = last_gals['gids'][fpure]

    tgt_sid, found, step = gid_to_sid(sim,gid,last_snap)
    # tgt_sid = last_sinks['identity'][most_massive]

    # sink_hist = get_sink_mhistory(tgt_sid,last_snap,sim,max_z=10,skip=5,out_keys=hist_keys)

    first_time = sim.cosmo_model.age(1./sim.aexps.min()-1).value*1e3

    tbins_myr = np.arange(first_time,last_time,tstep)

    # tbins_myr = sim.get_snap_times()
    # zbins = np.arange(1./sim.aexps.max()-1, min(1./sim.aexps.min()-1,10), 0.5)
    
    # tbins_myr = sim.get_snap_times([sim.get_closest_snap(zed=z) for z in zbins])

    gal_masses = np.zeros(len(tbins_myr))
    sink_masses = np.zeros(len(tbins_myr))
    zeds = np.zeros(len(tbins_myr))


    for it,tbin in enumerate(tbins_myr[::-1]):

        print(it,tbin)

        snap_arg = np.argmin(np.abs(times-tbin))

        snap = sim.snap_numbers[snap_arg]
        zed = 1./sim.get_snap_exps(snap)[0] - 1.
        zeds[it]=zed

        # hist_arg = np.argmin(np.abs(zed-sink_hist['zeds']))

        this_sink = read_sink_snap(sim,snap,sid=tgt_sid)
        if this_sink =={}:continue

        try:
            all_gals = get_gal_props_snap(sim.path,snap)
        except FileNotFoundError:
            continue
        pure = all_gals['host purity']>0.9999
        pure_pos = all_gals['pos'][:,pure]
        pure_mass = all_gals['mass'][pure]

        sink_pos = this_sink['position']

        dists = np.linalg.norm(pure_pos.T-sink_pos,axis=1)

        if len(dists)>0:

            arg_pos = np.argmin(dists)
            #pick closest hosted galaxy
            gal_masses[it] = pure_mass[arg_pos]
        else:#implicit
            continue        

        sink_masses[it] = this_sink['mass']


        sink_pos = this_sink['position']


    # normd_zeds = (zeds - zscale.min())/(zscale.max() - zscale.min())

    order = np.argsort(zeds)
    ax.plot(gal_masses[order],sink_masses[order],label=sim.name)
    sc=ax.scatter(gal_masses[order],sink_masses[order],c=zeds[order],vmin=zscale.min(),vmax=zscale.max())

    if len(sink_masses)>0:
        ymax_plot = 1.1*max(ymax_plot,np.max(sink_masses))
        ymin_plot = 0.9*min(ymin_plot,np.min(sink_masses))

plt.colorbar(sc,ax=ax,label='redshift')
ax.legend(framealpha=0.0)

ax.grid()

ax.set_xlabel("Stellar mass [M$_\odot$]")
ax.set_ylabel("BH Mass [M$_\odot$]")

ax.set_xscale('log')
ax.set_yscale('log')

gal_mass_bins = np.logspace(6, 12, 12)

reines_masses_bulges = calculate_reines15_mbh_bulges(gal_mass_bins)
reines_masses_local = calculate_reines15_mbh_local(gal_mass_bins)

ax.plot(
    gal_mass_bins[gal_mass_bins > 1e8],
    reines_masses_bulges[gal_mass_bins > 1e8],
    label="Reines,Volonteri 2015 $M_{BH}/M_{\star}$",
    c="k",
    lw=0.6,
)
ax.plot(
    gal_mass_bins[gal_mass_bins > 1e8],
    reines_masses_local[gal_mass_bins > 1e8],
    # label="Reines+ 2015 $M_{BH}/M_{\star}$",
    c="k",
    lw=0.6,
)

ax.set_ylim(ymax=ymax_plot, ymin=ymin_plot)

zstr = f"{zeds.min():.1f}".replace(".","p")

NH_smbh(ax)

if all_same_sim:
    name_prefix = f"{intID}_"
else:
    name_prefix = ""

fig.savefig(name_prefix + "zoom_BHSMR_history_"+zstr+".png")

