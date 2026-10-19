import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy.stats import binned_statistic_2d

from zoom_analysis.halo_maker.assoc_fcts import get_gal_props_snap, get_gal_assoc_file

from gremlin.read_sim_params import ramses_sim

tgt_zed = 2.0

mcut = 1e6

sim_dirs = [
    "/data103/jlewis/sims/lvlmax_22/mh1e12/id180130",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id26646",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id52380",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id18289",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id74890",
    "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/id112288",
]
tot_nsubs = []
tot_msubs = []
tot_mfrac_subs = []

tot_masses = []
tot_sfrs100 = []

for sim_dir in sim_dirs:

    sim = ramses_sim(sim_dir)

    # last_snap = np.max(sim.snap_numbers)

    tgt_snap = sim.get_closest_snap(zed=tgt_zed)

    if not os.path.exists(get_gal_assoc_file(sim.path, tgt_snap)):
        continue

    gal_dict = get_gal_props_snap(sim.path, tgt_snap)

    nsubs = np.zeros(len(gal_dict["gids"]))
    msubs = np.zeros(len(gal_dict["gids"]))
    mfrac_subs = np.zeros(len(gal_dict["gids"]))

    masses = gal_dict["mass"]
    sfrs100 = gal_dict["sfr100"]

    for igal, gid in enumerate(gal_dict["gids"]):

        subs = gal_dict["host gid"] == gid
        subs[gal_dict["gids"] == gid] = False

        if subs.sum() == 0:
            continue

        if mcut is not None:
            subs = subs * (gal_dict["mass"] > mcut)

        nsubs[igal] = subs.sum()
        msubs[igal] = gal_dict["mass"][subs].sum()
        mfrac_subs[igal] = msubs[igal] / gal_dict["mass"][igal]

    tot_nsubs.append(nsubs)
    tot_msubs.append(msubs)
    tot_mfrac_subs.append(mfrac_subs)

    tot_masses.append(masses)
    tot_sfrs100.append(sfrs100)

tot_masses = np.concatenate(tot_masses)
tot_nsubs = np.concatenate(tot_nsubs)
tot_sfrs100 = np.concatenate(tot_sfrs100) / 1e6
tot_msubs = np.concatenate(tot_msubs)
tot_mfrac_subs = np.concatenate(tot_mfrac_subs)

fig, ax = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")

# plt.subplots_adjust(hspace=0.1,wspace=0.1)

ax = np.ravel(ax)

mbins = np.logspace(5, 11, 18)
sfrbins = np.logspace(-5, 3, 21)

img_nbins = binned_statistic_2d(
    tot_masses, tot_sfrs100, tot_nsubs, "mean", bins=[mbins, sfrbins]
)[0]
img_nwsubs = binned_statistic_2d(
    tot_masses, tot_sfrs100, tot_nsubs > 1, "sum", bins=[mbins, sfrbins]
)[0]
img_mbins = binned_statistic_2d(
    tot_masses, tot_sfrs100, tot_msubs, "mean", bins=[mbins, sfrbins]
)[0]
img_mfrac = binned_statistic_2d(
    tot_masses, tot_sfrs100, tot_mfrac_subs, "mean", bins=[mbins, sfrbins]
)[0]
img_ngals = binned_statistic_2d(
    tot_masses, tot_sfrs100, tot_nsubs, "count", bins=[mbins, sfrbins]
)[0]


img_frac_wsubs = img_nwsubs / img_ngals


for a in ax:

    a.set_xlabel("log$_{10}$(Mgal, M$_\odot$)")
    a.set_ylabel("log$_{10}$(SFR100, M$_\odot$/yr)")

img = ax[0].imshow(
    img_ngals.T,
    extent=np.log10([mbins[0], mbins[-1], sfrbins[0], sfrbins[-1]]),
    origin="lower",
    norm=LogNorm(),
    aspect="auto",
)
cb = plt.colorbar(img, ax=ax[0])
ax[0].set_title("Number of galaxies")

img = ax[1].imshow(
    img_nbins.T,
    extent=np.log10([mbins[0], mbins[-1], sfrbins[0], sfrbins[-1]]),
    origin="lower",
    norm=LogNorm(),
    aspect="auto",
)
cb = plt.colorbar(img, ax=ax[1])
ax[1].set_title("Number of substructures")

img = ax[2].imshow(
    img_mbins.T,
    extent=np.log10([mbins[0], mbins[-1], sfrbins[0], sfrbins[-1]]),
    origin="lower",
    norm=LogNorm(),
    aspect="auto",
)
cb = plt.colorbar(img, ax=ax[2])
ax[2].set_title("M substructures, M$_\odot$")

vmin, vmax = np.nanpercentile(img_mfrac, [10.0, 90.0])

img = ax[3].imshow(
    img_mfrac.T,
    extent=np.log10([mbins[0], mbins[-1], sfrbins[0], sfrbins[-1]]),
    origin="lower",
    vmin=vmin,
    vmax=vmax,
    aspect="auto",
)
cb = plt.colorbar(img, ax=ax[3])
ax[3].set_title("M$\mathrm{_{frac}}$ substructures")

img = ax[4].imshow(
    img_frac_wsubs.T,
    extent=np.log10([mbins[0], mbins[-1], sfrbins[0], sfrbins[-1]]),
    origin="lower",
    # vmin=vmin,
    # vmax=vmax,
    aspect="auto",
)
cb = plt.colorbar(img, ax=ax[4])
ax[4].set_title("Fraction with substructures")

ax[5].axis("off")

zstr = f"{tgt_zed:.1f}".replace(".", "p")

title = f"z={tgt_zed:.1f}"

fig_name = f"clump_stats_{zstr}"

if mcut is not None:
    fig_name += f"_Msub>{mcut:.1e}_Msun".replace(".", "p")
    title += f", Msub>{mcut:.1e} M$_\odot$"

fig.suptitle(title)
fig.savefig(fig_name)
