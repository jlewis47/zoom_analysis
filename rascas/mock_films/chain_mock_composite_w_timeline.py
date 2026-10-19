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

from zoom_analysis.trees.tree_reader import read_tree_file_rev_correct_pos

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
depths = [15]  
suffixes=["wIGM.h5"]
# depths = [30]  
# suffixes=["wIGM_wPSF_rebinned.h5"]
# depths = [15, 15, 30] # only 2
# depths = [15, 30]  # only 2
# depths = [40, 30, 30]

# los_nbs = np.arange(12)
# los_nbs = [1,5, 6]
los_nbs = [6]


sim = ramses_sim(os.path.join(sim_path, sim_name), param_save=False, full_check=False)

tree_name = os.path.join(sim.path, "TreeMakerDM_dust", "tree_rev.dat")
byte_file = os.path.join(sim.path, "TreeMakerDM_dust")
# tree_name = os.path.join(sim.path, "TreeMakerstars2_dp_rec_dust/", "tree_rev.dat")
# byte_file = os.path.join(sim.path, "TreeMakerstars2_dp_rec_dust/")

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

isnap=0
tree=False

# snap_start = sim.snap_numbers.max()
snap_start=209



gprops = get_gal_props_snap(sim.path,snap_start)

pure = gprops['host purity']>0.9999
most_massive = np.argmax(gprops['mass'][pure])

hid_most_massive = gprops['host hid'][pure][most_massive]

found_ids, found_fields, tree_aexps= read_tree_file_rev_correct_pos(
                                                        tree_name,
                                                        sim,
                                                        snap_start,
                                                        byte_file,
                                                        1/sim.get_snap_exps(snap_start)[0]-1,
                                                        [hid_most_massive],
                                                        # tree_type="halo",
                                                        tgt_fields=["m", "x", "y", "z", "r"],
                                                        verbose=False,
                                                        star=False,
                                                    )


for istep,aexp in enumerate(tree_aexps):

    snap = sim.get_closest_snap(aexp=aexp)

    # print(snap)

    mock_path = os.path.join(path_to_mocks, f"{snap:d}")

    if not os.path.exists(mock_path):continue

    gals = os.listdir(
        mock_path
    )

    halo = found_ids[0][istep] 

    gals_step = get_gal_props_snap(sim.path,snap)

    possible_gals = gals_step['gids'][gals_step['host hid']==halo]
    most_massive_gal = np.argmax(gals_step['mass'][np.in1d(gals_step['gids'],possible_gals)])
    gal = gals_step['gids'][gals_step['host hid']==halo][most_massive_gal]

    if f"gal_{gal:07d}" in gals:

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
                tgt_img_size_as=2.0,
                los_nbs=los_nbs,
                overwrite=overwrite,
                zstr=zstr,
                clean=clean,
                # vmax=2,
                # vmin=50
                # zeros=-15,
                # norms=3, #more suited to rebinned psf images
                norms=4,
                depths=depths, #needs to match len of suffixes
                suffixes=suffixes
            )
        

            time_BB = -13.72
            cur_time_gyr = sim.get_snap_times(snap)[0]/1e3 + time_BB
            present_time = 0.0

            xleft=0.1
            dx = 0.8
            xright = xleft+dx

            y_text = 0.8

            # ax.plot([0.1,0.9],[0.95,0.95],c='w',ls='-',lw=2,transform=ax.transAxes)
            ax.arrow(xleft,0.9,dx,0.0,color='w',ls='-',lw=10,transform=ax.transAxes)

            ax.text(xleft,y_text,f'Big bang\n{time_BB:.2f} Ma', c='w',transform=ax.transAxes, ha='center',size='xx-large')
            ax.text(xright,y_text,f'Présent\n{present_time:.2f} Ma', c='w',transform=ax.transAxes, ha='center',size='xx-large')

            x_cur = (cur_time_gyr - time_BB)/(present_time-time_BB) * (xright-xleft) + xleft
            ax.text(x_cur,y_text+0.15,f'\n{cur_time_gyr:.2f} Ma', c='w',transform=ax.transAxes, ha='center',size='xx-large')
            ax.plot([x_cur,x_cur],[0.9,0.93],c='w',lw=3,transform=ax.transAxes)

            # fig.savefig('test')

            if not os.path.exists(str(sim.name)):
                os.makedirs(str(sim.name))
            fig.savefig(f'{sim.name}/{isnap:05d}.png')
        except (SystemError, FileNotFoundError, KeyError):
            continue

    isnap+=1


