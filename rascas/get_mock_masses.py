from hagn.catalogues import make_super_cat
import numpy as np
import os
import matplotlib.pyplot as plt

mock_dir = "/data101/jlewis/color_mocks/zoom_tgt_bc03_chabrier100_wlya_SMC/NewHorizon/"
direction = 1

snap_dirs = [os.path.join(mock_dir, d) for d in os.listdir(mock_dir)]

gal_dirs = [
    os.path.join(d, gd)
    for d in snap_dirs
    for gd in os.listdir(d)
    if os.path.exists(
        os.path.join(os.path.join(d, gd), f"mock_rgb_{direction}_wIGM.png")
    )
]

print(len(gal_dirs))

print(f"found {len(gal_dirs)} galaxies with mocks")

masses = np.zeros(len(gal_dirs))

last_snap = -1
for i, gd in enumerate(gal_dirs):
    snap = int(gd.split("/")[-2])
    if snap != last_snap:
        print(f"Processing snap {snap}")
        gal_cat = make_super_cat(snap, sim="nh")
        last_snap = snap
    gid = int(gd.split("_")[-1])

    mass = gal_cat["mgal"][gal_cat["gid"] == gid][0]

    if mass > 1e10:
        print(f"{mass:.2e}, {os.path.join(gd, f'mock_rgb_{direction}_wIGM.png')}")

    masses[i] = mass

print(gal_dirs[np.argmax(masses)])

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

ax.hist(np.log10(masses), bins=20)

ax.set_xlabel("log10 Stellar Mass [Msun]")
ax.set_ylabel("Number of galaxies")

fig.savefig("NH_mock_masses_hist.png")
