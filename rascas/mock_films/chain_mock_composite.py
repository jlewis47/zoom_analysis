from mpl_toolkits.axes_grid1.parasite_axes import parasite_axes_class_factory
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import os

# from astropy.visualization import make_lupton_rgb

from scipy.signal import convolve2d

from zoom_analysis.rascas.rascas_plots import plot_mock_composite

from scipy.ndimage import gaussian_filter
from gremlin.read_sim_params import ramses_sim
from scipy.stats import binned_statistic_2d
from zoom_analysis.rascas.plot_mock_composite import make_snap_gal_mock
from zoom_analysis.halo_maker.assoc_fcts import get_gal_props_snap

spath = "/data101/jlewis/mock_spe"

# sim_path = f"/data101/jlewis/sims/dust_fid/lvlmax_20/mh1e12/"
# sim_path = f"/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12/"
# model_name = "zoom_tgt_bc03_chabrier100_wlya"
# model_name = "zoom_tgt_bc03_chabrier100_wlya_noDust"
model_name = "zoom_tgt_bc03_chabrier100_wlya_SMC"
# model_name = "zoom_tgt_bc03_chabrier100_wlya_ZTEST_SMC"
# model_name = "zoom_tgt_bc03_chabrier100_wlya_ndust_SMC"

sim_path = "/data101/jlewis/sims/dust_fid/lvlmax_22/mh1e12"
# sim_name = "id180130"
sim_name = "id52380"
# sim_name = "id26646"
# sim_name = "id18289"
# sim_name = "id74890"

# sim_name = "NH_for_rascas"
# sim_path = "/data101/jlewis/sims/"


overwrite=True
# clean=False
clean=True

# depth = 15
# depths = [15, 15, 30] # only 2
depths = [15, 30]  # only 2
# depths = [40, 30, 30]

# los_nbs = np.arange(12)
los_nbs = [1,5, 6]


sim = ramses_sim(os.path.join(sim_path, sim_name), param_save=False, full_check=False)

ztest = 'ZTEST' in model_name

zstr='Z3p0'

if not ztest:

    path_to_data = os.path.join(sim.path, "rascas", model_name)
    path_to_mocks = os.path.join("/data101/jlewis/mock_spe/", model_name,sim.name)
    
else:

    path_to_data = os.path.join(sim.path, "rascas", model_name,zstr)
    path_to_mocks = os.path.join("/data101/jlewis/mock_spe/", model_name,sim.name,zstr)

snaps = [
    d
    for d in os.listdir(path_to_data)
    if d.startswith("output")
]

snap_nbs = np.sort([int(s.split("_")[1]) for s in snaps])[::-1]

# snap_nbs = np.sort(snap_nbs)


for snap in snap_nbs:

    # print(snap)

    mock_path = os.path.join(path_to_mocks, f"{snap:d}")

    if not os.path.exists(mock_path):continue

    gals = os.listdir(
        mock_path
    )
    gal_nbs = [int(g.split("_")[1]) for g in gals if "vignette" not in g]

    for gal in gal_nbs:

        # print("...", gal)

        out_dir = os.path.join(
            "/data101/jlewis/color_mocks",
            model_name,
            sim_name,
            f"{snap:d}",
            f"gal_{gal:07d}",
        )

        gprops = get_gal_props_snap(sim.path,snap,gal)

        # if gprops['mass']<1e11:continue

        print(snap,gal)
        print(f"mass:{gprops['mass']:.2e} Msun, sfr100:{gprops['sfr100']/1e6:.2e} Msun/yr")

        if sim_name == "NH_for_rascas" and not ztest:
            mock_sim_name = "NewHorizon"
        else:
            mock_sim_name = sim_name

        if not os.path.exists(
            f"{path_to_mocks}/{snap:d}/gal_{gal:07d}/MJy_images/"
        ):
            print(f"didn't find a target for snapshot {snap} and galaxy {gal},")
            continue

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # cur_aexp = sim.get_snap_exps(snap, param_save=False)[0]
        # cur_z = 1.0 / cur_aexp - 1.0

        # sim.init_cosmo()

        # ang_per_ckpc = sim.cosmo_model.arcsec_per_kpc_comoving(cur_z).value

        try:
            fig,ax=make_snap_gal_mock(
                sim,
                model_name,
                snap,
                gal,
                depths=depths,
                tgt_img_size_as=2.0,
                los_nbs=los_nbs,
                overwrite=overwrite,
                zstr=zstr,
                clean=clean,
                # vmax=2,
                # vmin=50
                # zeros=-15,
                norms=4
            )
        except (SystemError, FileNotFoundError, KeyError):
            continue
