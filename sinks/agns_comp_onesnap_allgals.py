from zoom_analysis.sinks.sink_reader import get_sink_mhistory, find_massive_sink
from gremlin.read_sim_params import ramses_sim
import matplotlib.pyplot as plt
import os
import numpy as np
from zoom_analysis.constants import ramses_pc
from zoom_analysis.halo_maker.assoc_fcts import get_gal_props_snap


# sim_path=    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id21892_leastcoarse"
# sim_path=    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id180130_leastcoarse"
# sim_path=    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id242704"
# sim_path=    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id242756"
# sim_path=    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/small_rgal/id74099"
# sim_path=    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892"
# sim_path=    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892"
# sim_path=    "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_meanBondi"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id21892_novrel"
# sim_path="/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id147479"
# sim_path="/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130"
# sim_path="/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_highMseed"
# sim_path="/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi"
# sim_path="/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_novrel"
# sim_path="/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_meanBondi_lowerSFE"
# sim_path="/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_novrel"
# sim_path="/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_lowerSFE"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_maxiBH"
# sim_path="/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id180130_noAGN"
# sim_path="/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242704_lower_nstar"
# sim_path="/data101/jlewis/sims/dust_fid/lvlmax_19/mh1e12/id242756_lower_nstar"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242704_novrel"

# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel"
sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE"
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_lowerSFE_NHboost"
# sim_path = (
#     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_lowerSFE_stgNHboost"
# )
# sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id242756_novrel_drag"
# sim_path="/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id242756_nh"
# sim_path=     "/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/id74099"


fpure = 0.9999
# mlim = 1e11
mlim = 1e10

# tgt_zed = 1.9
tgt_zed = 2.5
nsmooth = 60
# fig = plt.figure()
# ax = fig.add_subplot(111)

fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# hagn_sim = get_hagn_sim()
# hagn_sim.init_cosmo()
# super_cat = make_super_cat(197, "hagn")  # , overwrite=True)

# l_hagn = hagn_sim.unit_l(hagn_sim.aexp_stt) / (ramses_pc * 1e6) / hagn_sim.aexp_stt
# l_hagn = hagn_sim.unit_l(1.0 / tgt_zed - 1.0) / (ramses_pc * 1e6)  # / hagn_sim.aexp_stt
# print(l_hagn)

done_hagn_ids = []
done_hagn_colors = []

sim_name = sim_path.split("/")[-1]
sdir = sim_path.split(sim_name)[0]

# print(sim_name)

sim_id = int(sim_name[2:].split("_")[0])

sim = ramses_sim(sim_path, nml="cosmo.nml")

if not hasattr(sim, "cosmo_model"):
    sim.init_cosmo()

# snap = sim.get_closest_snap(zed=tgt_zed)
# snap_aexp = sim.get_snap_exps(snap)

# find last output with assoc files
assoc_files = np.asarray(os.listdir(os.path.join(sim.path, "association")))
assoc_file_nbs = np.asarray([int(f.split("_")[1]) for f in assoc_files])

avail_aexps = np.intersect1d(
    sim.get_snap_exps(assoc_file_nbs, param_save=False), sim.aexps
)
avail_times = sim.cosmo_model.age(1.0 / avail_aexps - 1.0).value * 1e3

avail_snaps = np.intersect1d(sim.snap_numbers, assoc_file_nbs)

avail_zeds = 1.0 / avail_aexps - 1.0

real_start_zed = avail_zeds[np.argmin(np.abs(avail_zeds - tgt_zed))]
real_start_snap = avail_snaps[np.argmin(np.abs(avail_zeds - tgt_zed))]

snap = real_start_snap

print(snap)


gal_props = get_gal_props_snap(sim.path, snap)

valid = (gal_props["mass"] > mlim) * (gal_props["host purity"] > fpure)
valid_gids = gal_props["gids"][valid]
valid_pos = gal_props["pos"][:, valid]
valid_rmax = gal_props["rmax"][valid]

print(valid_gids)


for igal, gid in enumerate(valid_gids):

    # print(pos, rvir)
    # hagn_massive_sid = find_massive_sink(pos, hagn_snap, hagn_sim, rmax=rvir * 2)[

    try:
        massive_sid = find_massive_sink(
            valid_pos[:, igal], snap, sim, rmax=valid_rmax[igal], hagn=True
        )["identity"]
        print(f"massive sink has id: {massive_sid}")
    except ValueError:
        print(f"no massive sink in gal {gid:d}")
        continue
        # print()

    sink_hist = get_sink_mhistory(massive_sid, snap, sim, hagn=False)

    # print(list(zip(sink_hist["zeds"], sink_hist["mass"])))

    zeds = sink_hist["zeds"]

    if not hasattr(sim, "cosmo_model"):
        sim.init_cosmo()

    times = sim.cosmo_model.age(zeds).value * 1e3  # Myr

    nnZero = np.where(sink_hist["mass"] > 0.0)[0]

    (l,) = axs[0].plot(
        zeds[nnZero],
        sink_hist["mass"][nnZero],
        lw=2,
        label="gid=" + str(gid) + ", sid=" + str(massive_sid),
    )  # , c=l.get_color())

    mdot_data = sink_hist["dMBH_coarse"]
    mdot_edd_data = sink_hist["dMEd_coarse"]
    # print(mdot_data, mdot_edd_data)

    # print(len(mdot_data), len(mdot_edd_data))
    min_mdot_data = np.min(
        [mdot_data, mdot_edd_data], axis=0
    )  # take minimum to only have relavent rate
    # print(len(zeds), len(min_mdot_data))

    # smooth using convolution
    min_mdot_data = np.convolve(min_mdot_data, np.ones(nsmooth) / nsmooth, mode="same")
    # min_mdot_data = np.convolve(mdot_data, np.ones(nsmooth) / nsmooth, mode="same")

    axs[1].plot(zeds[nnZero], min_mdot_data[nnZero], lw=2, c=l.get_color())

axs[0].legend(framealpha=0.0)

axs[1].legend(title=f"smoothed X{nsmooth}", framealpha=0.0)


axs[1].set_xlabel("z")

axs[0].set_ylabel("BH mass [M$_\odot$]")
axs[1].set_ylabel("Mdot [M$_\odot$/yr]")

# axs[1].set_ylim(1e-23, 1e-17)
# ax.grid()

for ax in axs:
    ax.set_yscale("log")
    ax.tick_params(direction="in", top=True, right=True)

# second xaxis times...
ticks = axs[0].get_xticks()
axs[0].set_xticks(ticks)
axs[0].invert_xaxis()
ax2 = axs[0].twiny()
ax2.set_xlim(axs[0].get_xlim())
ax2.set_xticks(ticks)
ax2.set_xticklabels([f"{sim.cosmo_model.age(xtick).value:.2f}" for xtick in ticks])
ax2.set_xlabel("time [Gyr]")
ax2.tick_params(direction="in", top=True, right=True)

zstr = f"{tgt_zed:.2f}".replace(".", "p")
fig.savefig(f"sinks_all_gals_{sim.name:s}_z{zstr:s}")
