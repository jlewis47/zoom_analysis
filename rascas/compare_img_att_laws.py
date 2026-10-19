import numpy as np
import matplotlib.pyplot as plt
import os

from gremlin.read_sim_params import ramses_sim
from zoom_analysis.halo_maker.assoc_fcts import get_gal_props_snap
from zoom_analysis.rascas.igm import T_IGM_Inoue2014
from zoom_analysis.visu.visu_fct import plot_fields
from zoom_analysis.rascas.rascas_steps import filt_file_to_dict, get_directions_cart
from zoom_analysis.rascas.read_rascas import (
    cube_to_erg_s_A_cm2_as2,
    read_compress_mock,
    read_mock_dump,
    spec_to_erg_s_A_cm2,
    mock_spe_dump,
)
from zoom_analysis.rascas.rascas_plots import gen_imgs
from f90_tools.star_reader import read_part_ball_NH
from zoom_analysis.rascas.rascas_plots import gen_spec

from scipy.ndimage import zoom
import healpy as hp



ramses_pc = 3.08e16

filt_wav_dict = {
    "F115W": 1.150,
    "F150W": 1.501,
    "F277W": 2.760,
    "F444W": 4.400,
    "HST-F814W": 0.820,
    "CFHT-u": 0.386,
    "HSC-g": 0.475,
    "HSC-r": 0.623,
    "HSC-i": 0.770,
    "HSC-z": 0.890,
    "HSC-y": 1.000,
}


filter_select = [
    "GALEX-NUV",
    "MegaCam-u_new",
    "HSC-g",
    "HSC-i",
    "HSC-r",
    "HSC-y",
    "HSC-z",
    "NIRCAM-F115",
    "NIRCAM-F150",
    "NIRCAM-F277",
    "NIRCAM-F444",
    "HST-f814_res",
]


from zoom_analysis.rascas.filts.filts import (
    read_transmissions,
    flamb_fnu
)

def read_law(name):

    return np.genfromtxt(f"attenuation_laws/{name}", names=True, dtype=None)



def interp_spec(tgt_wav, wav, flux):

    arg = np.digitize(tgt_wav, wav)

    argp1 = arg + 1
    argm1 = arg - 1

    if argm1 < 0 or argp1 > len(wav) - 1:
        return flux[arg]

    wavp1 = wav[argp1]
    wavm1 = wav[argm1]
    fluxp1 = flux[argp1]
    fluxm1 = flux[argm1]

    delta = (fluxp1 - fluxm1) / (wavp1 - wavm1)

    return (tgt_wav - wavm1) * delta + fluxm1




# nuv_wav = 0.228
nuv_wav = 0.23065
# nuv_wav = 0.3858 #cfht unew #MS paper
# r_wav = 0.6240 #HSC-r
r_wav = 0.6219  # HSC-r #MS paper
# J_wav = 1.250 #
J_wav = 1.2525  # #MS paper

norm_tgt = 0.55  # microns
b_ctr = 4439 / 1e4
v_ctr = 5448 / 1e4

ndir = 12

sim_dir = "/data101/jlewis/sims/NH_for_rascas"
snap = 416
galid = 1



# dirs = np.arange(0,12)
idir=3

sim = ramses_sim(sim_dir, param_save=False, full_check=False, mini_snaps=True, full_snaps=False)

gprops = get_gal_props_snap(sim.path,snap,gid=galid)

sim.init_cosmo()

l = sim.cosmo.lcMpc

aexp = sim.get_snap_exps(416)[0]
cur_z = 1/sim.get_snap_exps(snap)[0]-1
z = 1 / aexp - 1

model_names = np.array(["SMC", "MW5p5", "MW3p1", "LMCavg", "noDust", "MW3p1_Zref5EXPm2","MW3p1_Zref1EXPm3"])
# model_names = ["SMC", "MW5p5"]  # , "MW3p1", "LMCavg"]

maps = []
tots = []

names_all, wavs_all, trans_all = read_transmissions()


tgt_res_as = 33e-3

tgt_aper = 0.75 #as


for im,model_name in enumerate(model_names):



    print(f"loading model {model_name}")

    path_model = f"zoom_tgt_bc03_chabrier100_wlya_{model_name}"

    rascas_path = os.path.join(sim_dir, "rascas", path_model)

    rascas_dir = os.path.join(rascas_path, f"output_{snap:05d}", f"gal_{galid:07d}")

    h5mock = False
    if os.path.exists(os.path.join(rascas_dir, "mocks.h5")):
        h5mock = True
    rawmock = False
    if os.path.exists(os.path.join(rascas_dir, "mock_test.spectrum")):
        rawmock = True

    if h5mock:
        mocks_spe, mocks_cube, pfs_header = read_compress_mock(
            os.path.join(rascas_dir, "mocks.h5"),ndirs=idir
        )
        print('read cube')
        pfs = pfs_header

    else:
        pfs_path = os.path.join(rascas_dir, "PFSDump", f"pfsdump")
        pfs = pfs_path
        mocks_spe = read_mock_dump(
            os.path.join(rascas_dir, "mock_test.spectrum"), ndir=[idir]
        )
        mocks_cube = read_mock_dump(
            os.path.join(rascas_dir, "mock_test.cube"), ndir=[idir]
        )
        print('read cube')

    spec = mocks_spe[f"direction_{idir:d}"]

    lmin = spec["lambda_min"]
    lmax = spec["lambda_max"]
    nlamb = int(spec["lambda_npix"])
    lambda_bins_rf = np.linspace(lmin, lmax, nlamb)    

    # ditch filters outside of spectral coverage
    names = [
        names_all[i]
        for i in range(len(names_all))
        if np.all(wavs_all[i] <= lmax * (1.0 + z))
        and np.all(wavs_all[i] >= lmin * (1 + z))
        and names_all[i] in filter_select
    ]

    trans = [
        trans_all[i]
        for i in range(len(names_all))
        if np.all(wavs_all[i] <= lmax * (1.0 + z))
        and np.all(wavs_all[i] >= lmin * (1 + z))
        and names_all[i] in filter_select
    ]

    wavs = [
        wavs_all[i]
        for i in range(len(names_all))
        if np.all(wavs_all[i] <= lmax * (1.0 + z))
        and np.all(wavs_all[i] >= lmin * (1 + z))
        and names_all[i] in filter_select
    ]

    band_ctrs = np.array([np.median(w) for w in wavs])

    aper_code_units = mocks_cube[f"direction_{idir:d}"]["cube_side"]

    aper_cm = aper_code_units * l * (1e6 * ramses_pc)
    distl = sim.cosmo_model.luminosity_distance(z).value * (1e6 * ramses_pc)
    # aper_as = np.arctan(aper_cm / distl) * 180.0 / np.pi * 3600.0
    aper_as = (
        aper_code_units * l * 1e3
    ) * sim.cosmo_model.arcsec_per_kpc_comoving(z).value

    aper_sr = (aper_as * 3600. * np.pi/180)**2

    cube = mocks_cube[f"direction_{idir:d}"]

    cube_size = cube['cube'].shape[0]
    cube_ctr = np.int32([0.5*cube_size]*2)

    tgt_r_px = int(np.ceil(tgt_aper/aper_as * cube_size * 0.5))

    cube_to_erg_s_A_cm2_as2(cube, pfs, z, aper_as**2)
    print('converted cube to useful units')

    lambda_bins = lambda_bins_rf * (1.0 + z)

    nuv_arg = np.argmin(np.abs(nuv_wav - band_ctrs/1e4/(1.0 + z)))
    J_arg = np.argmin(np.abs(J_wav - band_ctrs/1e4/(1.0 + z)))
    r_arg = np.argmin(np.abs(r_wav - band_ctrs/1e4/(1.0 + z)))

    used_names, used_ctrs, imgs = gen_imgs(
        z, lambda_bins, cube, [[names[J_arg],names[r_arg]], [wavs[J_arg],wavs[r_arg]], [trans[J_arg],trans[r_arg]]]
    )

    print("getting wavelengths")

    Ts = np.asarray(
    [T_IGM_Inoue2014(band_ctr, z) for band_ctr in used_ctrs]
    )

    imgs_MJy_wIGM = [
        flamb_fnu(img, band_ctr)
        * (aper_as**2)
        / aper_sr
        / 1e-23
        / 1e6
        * T
        for img, band_ctr, T in zip(imgs, used_ctrs, Ts)
    ]

    zpt = 8.9

    pixel_mag_imgs = [-2.5*np.log10(i[cube_ctr[0]-tgt_r_px:cube_ctr[0]+tgt_r_px,cube_ctr[1]-tgt_r_px:cube_ctr[1]+tgt_r_px]) + zpt  for i,T in zip(imgs_MJy_wIGM,Ts)]
    print('made pixel maps')

    total_mags = [-2.5*np.log10(i[cube_ctr[0]-tgt_r_px:cube_ctr[0]+tgt_r_px,cube_ctr[1]-tgt_r_px:cube_ctr[1]+tgt_r_px].sum())+ zpt for i,T in zip(imgs_MJy_wIGM,Ts)]

    rMJ_map = pixel_mag_imgs[1] - pixel_mag_imgs[0]
    rMJ_tot = total_mags[1] - total_mags[0]



    cur_res = aper_as / pixel_mag_imgs[0].shape[0]

    degrade = cur_res/tgt_res_as

    maps.append(rMJ_map)

    # rMJ_map[np.isfinite(rMJ_map)==False] = 0
    # maps.append(zoom(rMJ_map,degrade))

    # if im ==0:

    prof_bins = np.arange(0,tgt_r_px*2,2)
    smc_prof = np.zeros(prof_bins.shape[0])
    X,Y = np.meshgrid(np.arange(imgs_MJy_wIGM[0].shape[0]),np.arange(imgs_MJy_wIGM[0].shape[0]))
    R = np.sqrt((X-cube_ctr[0])**2 + (Y-cube_ctr[1])**2)
    for ir,r in enumerate(prof_bins[:-1]):
        r_arg = (R<prof_bins[ir+1])
        
        smc_prof[ir] = np.diff([-2.5*np.log10(i[r_arg].sum())+ zpt for i,T in zip(imgs_MJy_wIGM,Ts)],axis=0)[0]

    if im==0:
        fig_prof,ax_prof = plt.subplots(1,1)
        # ax_prof.axhline(rMJ_tot,color='k',linestyle='--',label='total')
        # ax_prof.axvline(tgt_aper,color='k',linestyle='--',label='total')
        ax_prof.axvline(gprops['r50']*sim.cosmo.cMpc*1e3*sim.arcsec_per_kpc_comoving(cur_z),color='grey',linestyle=':',label='r50')

    ax_prof.plot(prof_bins[:-1]*tgt_aper/tgt_r_px,smc_prof[:-1],label=model_name)

    tots.append(rMJ_tot)

ax_prof.set_xlabel("Radius (arcsec)")
ax_prof.set_ylabel("r-J (mag)")

fig_prof.savefig("rMJ_prof_smc.png")


nmaps = len(maps) + 2

nrows = int(np.sqrt(nmaps))
ncols = int(np.ceil(nmaps/nrows))

fig,ax = plt.subplots(ncols,nrows,figsize=(4*nrows,4*ncols))

ax = np.ravel(ax)

for ia,a in enumerate(ax):

    if ia>=nmaps-2:
        # a.axis('off')
        continue

    m = maps[ia]

    vmin,vmax=np.percentile(m[np.isfinite(m)],[33,66])

    im=a.imshow(m,vmin=vmin,vmax=vmax,extent=tgt_aper*np.array([-0.5,0.5,-0.5,0.5]))

    plt.colorbar(im,ax=a,label='r-J')

    a.set_title(f"{model_names[ia]}:{tots[ia]:.2f}")





#get code units resolution
code_res = sim.cosmo.lcMpc/2**sim.namelist['amr_params']['levelmax']
sim.init_cosmo()
code_to_as = sim.cosmo_model.arcsec_per_kpc_comoving(cur_z) * (sim.cosmo.lcMpc*1e3)

r_code = (aper_as / code_to_as * 0.5).value



pos = gprops['pos']
hid = gprops['host hid']

rascas_dirs = get_directions_cart(hp.npix2nside(12))

plot_fields("stellar age",
            fig,ax[-2],1/(cur_z+1.),
            rascas_dirs[idir],
            pos,r_code,sim,
            hid=hid,debug=False,read_part_ball_fct=read_part_ball_NH,
            mode='mean',
            cmap='Spectral_r')


ax[-2].set_xlim(-r_code*sim.cosmo.lcMpc*1e3,r_code*sim.cosmo.lcMpc*1e3)
ax[-2].set_ylim(-r_code*sim.cosmo.lcMpc*1e3,r_code*sim.cosmo.lcMpc*1e3)

# plot_fields("metallicity",
#             fig,ax[-1],1/(cur_z+1.),
#             rascas_dirs[idir],
#             pos,r_code,sim,
#             hid=hid,debug=False,read_part_ball_fct=read_part_ball_NH,
#             mode='mean',
#             cmap='viridis')


fig.savefig('test_nuvrJ_maps')