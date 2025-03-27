import numpy as np
import matplotlib.pyplot as plt
import os
import h5py


def get_spec_stuff(p):

    with h5py.File(p, "r") as src:

        spec_src = src["speectral_data"]

        return (
            spec_src["lam"][()] / 1e4,
            spec_src["mag"][()],
            spec_src["lam spectrum"][()] / 1e4,
            spec_src["spectrum"][()],
        )


good_paths = [
    "/data101/jlewis/mock_spe/zoom_tgt_bpass_kroupa100_wlya/id242756_novrel/189/gal_0000031/mock_spectrum_1.h5",
    "/data101/jlewis/mock_spe/zoom_tgt_bpass_kroupa100_wlya/id242756_novrel/200/gal_0000031/mock_spectrum_3.h5",
    "/data101/jlewis/mock_spe/zoom_tgt_bpass_kroupa100_wlya/id242756_novrel/205/gal_0000031/mock_spectrum_3.h5",
    "/data101/jlewis/mock_spe/zoom_tgt_bpass_kroupa100_wlya/id242756_novrel/214/gal_0000031/mock_spectrum_7.h5",
]


bad_paths = [
    "/data101/jlewis/mock_spe/zoom_tgt_bpass_kroupa100_wlya/id242756_novrel/189/gal_0000031/mock_spectrum_9.h5",
    "/data101/jlewis/mock_spe/zoom_tgt_bpass_kroupa100_wlya/id242756_novrel/200/gal_0000031/mock_spectrum_8.h5",
    "/data101/jlewis/mock_spe/zoom_tgt_bpass_kroupa100_wlya/id242756_novrel/205/gal_0000031/mock_spectrum_7.h5",
    "/data101/jlewis/mock_spe/zoom_tgt_bpass_kroupa100_wlya/id242756_novrel/214/gal_0000031/mock_spectrum_3.h5",
]

lss = ["-", "--", "-.", ":"]

gd_bands_lam, gd_mags, gd_lam, gd_spec = get_spec_stuff(good_paths[0])
nbands = len(gd_bands_lam)

fig, axs = plt.subplots(nbands, 1, figsize=(3, 15), sharex=True)

for ax, gd_path, bd_path, ls in zip(axs, good_paths, bad_paths, lss):

    snap = gd_path.split("/")[-3]

    # ax.plot([], [], color="k", linestyle=ls, label=f"snap {snap}")

    gd_bands_lam, gd_mags, gd_lam, gd_spec = get_spec_stuff(gd_path)
    # ax.plot(lam, spec, color=good_color, linestyle=ls)
    bd_bands_lam, bd_mags, bd_lam, bd_spec = get_spec_stuff(bd_path)
    # ax.plot(lam, spec, color=bad_color, linestyle=ls)

    print(bd_mags[2] - bd_mags[3], gd_mags[2] - gd_mags[3])

#     ax.scatter(
#         gd_bands_lam,
#         gd_mags - bd_mags,
#         color="k",
#         marker="x",
#         s=5,
#         label=f"snap {snap}",
#     )
#     # ax.plot(gd_lam, gd_spec - bd_spec, color="k", linestyle=ls, label=f"snap {snap}")

#     ax.set_xscale("log")
#     ax.grid()

#     # ax.invert_yaxis()

#     ax.axhline(0, color="k", linestyle="--", lw=0.5)

#     ax.set_xlim(ax.get_xlim())
#     ax.set_ylim(-1, 1.5)

#     ax.fill_between(
#         ax.get_xlim(),
#         0,
#         max(ax.get_ylim()),
#         color="gray",
#         alpha=0.25,
#         label="Good>Bad",
#     )

#     ax.set_xlabel(f"Wavelength, $\mu$m")
#     # ax.set_ylabel("diff magnitude")

#     ax.legend(framealpha=0)
# # ax.plot([], [], color=good_color, label="Good")
# # ax.plot([], [], color=bad_color, label="Bad")


# fig.savefig("compare_goodVbad_diff")
