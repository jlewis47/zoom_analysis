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
good_color = "tab:green"
bad_color = "tab:red"

lss = ["-", "--", "-.", ":"]
fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

for ax, gd_path, bd_path, ls in zip(axs, good_paths, bad_paths, lss):

    snap = gd_path.split("/")[-3]

    # ax.plot([], [], color="k", linestyle=ls, label=f"snap {snap}")

    bands_lam, mags, lam, spec = get_spec_stuff(gd_path)
    # ax.plot(lam, spec, color=good_color, linestyle=ls)
    ax.plot(lam, spec, color=good_color, linestyle=ls, label=f"Good snap {snap}")
    ax.scatter(bands_lam, mags, color=good_color)

    bands, mags, lam, spec = get_spec_stuff(bd_path)
    # ax.plot(lam, spec, color=bad_color, linestyle=ls)
    ax.plot(lam, spec, color=bad_color, linestyle=ls, label=f"Bad snap {snap}")
    ax.scatter(bands_lam, mags, color=bad_color)

    ax.set_xscale("log")
    ax.grid()

    ax.invert_yaxis()

    ax.set_xlabel(f"Wavelength, $\mu$m")
    ax.set_ylabel("Magnitude, AB")

    ax.legend(framealpha=0)
# ax.plot([], [], color=good_color, label="Good")
# ax.plot([], [], color=bad_color, label="Bad")


fig.savefig("compare_goodVbad")
