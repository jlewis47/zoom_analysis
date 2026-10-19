import numpy as np
import os
import matplotlib.pyplot as plt


mock_dir = "/data101/jlewis/color_mocks/zoom_tgt_bc03_chabrier100_wlya_SMC/NewHorizon/"
direction = 1


snap_dirs = [os.path.join(mock_dir, d) for d in os.listdir(mock_dir)]

snap_nbs = [d.split("/")[-1] for d in snap_dirs]

order = np.argsort(snap_nbs)
snap_dirs = np.asarray(snap_dirs)[order]

gal_dirs = [
    os.path.join(d, gd)
    for d in snap_dirs
    for gd in os.listdir(d)
    if os.path.exists(
        os.path.join(os.path.join(d, gd), f"mock_rgb_{direction}_wIGM.png")
    )
]

print(len(gal_dirs))

npick = 576
shape = "slide"
# shape="squaez"

picked = 0

pick = []

while picked < npick:
    pick.extend(
        np.unique(np.random.randint(low=0, high=len(gal_dirs), size=npick - picked))
    )
    pick = np.unique(pick).tolist()
    picked = len(pick)

npick = len(pick)

if shape == "square":

    nrows = int(npick**0.5)
    ncols = npick // nrows

elif shape == "slide":
    i = int(npick / (16 * 9))  # make a slide shaped one

    ncols = int(i**0.5) * 16
    nrows = int(i**0.5) * 9

    if nrows * ncols < npick:
        ncols += 1
else:
    print(f"unsupported shape: {shape}")

overstep = nrows * ncols - npick

fig, axs = plt.subplots(
    nrows, ncols, figsize=(8 * ncols, 8 * nrows), facecolor="k", layout="tight"
)  # , layout="constrained")


axs = np.ravel(axs)

plt.subplots_adjust(hspace=0.0, wspace=0.0)

for ia, a in enumerate(axs):

    if ia < npick:

        img = plt.imread(
            os.path.join(gal_dirs[pick[ia]], f"mock_rgb_{direction}_wIGM.png")
        )

        # if ia != 0:
        img[-33:, :] = 0, 0, 0, 0
        img[:20, -20:] = 0, 0, 0, 0

        a.imshow(img, origin="upper")

    a.set_facecolor("k")
    a.axis("off")

fig.savefig("test_pretty_mosaic")
