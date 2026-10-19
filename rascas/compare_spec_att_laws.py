import numpy as np
import matplotlib.pyplot as plt
import os

from gremlin.read_sim_params import ramses_sim

from zoom_analysis.rascas.igm import T_IGM_Inoue2014

from zoom_analysis.rascas.rascas_steps import filt_file_to_dict, get_directions_cart
from zoom_analysis.rascas.read_rascas import (
    cube_to_erg_s_A_cm2_as2,
    read_compress_mock,
    read_mock_dump,
    spec_to_erg_s_A_cm2,
    mock_spe_dump,
)

from zoom_analysis.rascas.rascas_plots import gen_spec


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

dirs = np.arange(0,12)

sim = ramses_sim(sim_dir, param_save=False, full_check=False)

l = sim.cosmo.lcMpc

aexp = sim.get_snap_exps(416)[0]
z = 1 / aexp - 1

model_names = np.array(["SMC", "MW5p5", "MW3p1", "LMCavg", "noDust", "MW3p1_Zref5EXPm2","MW3p1_Zref1EXPm3"])
# model_names = ["SMC", "MW5p5"]  # , "MW3p1", "LMCavg"]

specs = []

names_all, wavs_all, trans_all = read_transmissions()

for model_name in model_names:

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
            os.path.join(rascas_dir, "mocks.h5")
        )
        pfs = pfs_header

    else:
        pfs_path = os.path.join(rascas_dir, "PFSDump", f"pfsdump")
        pfs = pfs_path
        mocks_spe = read_mock_dump(
            os.path.join(rascas_dir, "mock_test.spectrum"), ndir=ndir
        )
        # mocks_cube = read_mock_dump(
        #     os.path.join(rascas_dir, "mock_test.cube"), ndir=ndir
        # )

    aper_code_units = mocks_cube["direction_0"]["cube_side"]



    aper_cm = aper_code_units * l * (1e6 * ramses_pc)
    distl = sim.cosmo_model.luminosity_distance(z).value * (1e6 * ramses_pc)
    # aper_as = np.arctan(aper_cm / distl) * 180.0 / np.pi * 3600.0
    aper_as = (
        aper_code_units * l * 1e3
    ) * sim.cosmo_model.arcsec_per_kpc_comoving(z).value

    print("getting wavelengths")

    for idir in dirs:

        spec = mocks_spe[f"direction_{idir:d}"]

        lmin = spec["lambda_min"]
        lmax = spec["lambda_max"]
        nlamb = int(spec["lambda_npix"])
        lambda_bins_rf = np.linspace(lmin, lmax, nlamb)
        lambda_bins = lambda_bins_rf * (1.0 + z)

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

        spec_obs = spec_to_erg_s_A_cm2(spec, pfs, z)
        spec_obs_igm = spec_obs * T_IGM_Inoue2014(lambda_bins, z)

        filt_names, mag_spec, mags_no_err, mags, mag_errs = gen_spec(
            aexp, spec_obs_igm, lambda_bins, [names, wavs, trans], rf=False
        )

        specs.append((model_name,lambda_bins, mag_spec))

# specs = np.array(specs)

atts_to_add = ["salim18.dat","SB_calzetti.dat","prevot.dat","extlaw_0.9.dat","SB_calzetti.dat","salim18.dat"]
names_to_add = ['salim18_att','Calzetti_att','Prevot_att','extlaw0p9_att','Calzetti_E(B-V)*5','salim_E(B-V)*5']

for att_to_add, name_to_add in zip(atts_to_add,names_to_add):

    # new model based on calzetti where I match NUV with the existing model
    new = read_law(att_to_add)

    nodust_arg = np.where(model_names=='noDust')[0][0]
    SMC_arg = np.where(model_names=='SMC')[0][0]

    _,SMC_lamb,SMC_spec = specs[SMC_arg]

    b_SMC = SMC_spec[np.argmin(np.abs(b_ctr-SMC_lamb/1e4))]
    v_SMC = SMC_spec[np.argmin(np.abs(v_ctr-SMC_lamb/1e4))]

    _,nodust_lamb,nodust_spec = specs[nodust_arg]

    b_nodust = nodust_spec[np.argmin(np.abs(b_ctr-nodust_lamb/1e4))]
    v_nodust = nodust_spec[np.argmin(np.abs(v_ctr-nodust_lamb/1e4))]

    ebv_SMC = (b_SMC-b_nodust) - (v_SMC-v_nodust)

    if not name_to_add[-2:]=="*5":

        Anew = new['klambda'] * ebv_SMC
        A_SMC = nodust_spec - SMC_spec
        lam_new = new['lambdaangstroms']

        interp_Anew = np.interp(SMC_lamb,lam_new,Anew)
        #match extinction in bluest part of spectrum
        interp_Anew = interp_Anew  * (A_SMC/interp_Anew)[0] * 1.5 #boost a bit
        spec_new = nodust_spec -  interp_Anew
    else:
        Anew = new['klambda'] * ebv_SMC
        A_SMC = nodust_spec - SMC_spec
        lam_new = new['lambdaangstroms']

        interp_Anew = np.interp(SMC_lamb,lam_new,Anew)
        #match extinction in bluest part of spectrum
        interp_Anew = interp_Anew  * (A_SMC/interp_Anew)[0] * 5 #boost a bit
        spec_new = nodust_spec -  interp_Anew

        # Anew = new['klambda']
        # lam_new = new['lambdaangstroms']



        # interp_Anew = np.interp(SMC_lamb,lam_new,Anew)

        # Anew_r = (interp_Anew+nodust_spec)[np.argmin(np.abs(SMC_lamb/1e4-r_wav))]
        # Anew_J = (interp_Anew+nodust_spec)[np.argmin(np.abs(SMC_lamb/1e4-J_wav))]
        # interp_Anew = interp_Anew * 1.5/(Anew_r-Anew_J)        
        # spec_new = nodust_spec -  interp_Anew
        

    specs.append((name_to_add,nodust_lamb,spec_new))

    model_names = np.array(model_names.tolist()+[name_to_add])

# plot spectra

markers = np.array(['o','v','s','D','H','P','x','<','>']*2)
colors = np.array(['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:gray','tab:olive','tab:cyan','gold','brown','magenta','darkgreen','indigo'])

fig, ax = plt.subplots(figsize=(12, 4))

done=[]

for spec in specs:

    name=spec[0]
    name_arg = np.where(model_names==name)[0][0]

    color=colors[name_arg]
    marker=markers[name_arg]

    if name not in done:
        ax.plot(spec[1]/1e4, spec[2],c=color,alpha=0.3)
        done.append(name)
        ax.plot([],[], label=name,c=color)
    else:
        ax.plot(spec[1]/1e4, spec[2],c=color,alpha=0.3)
    # if name not in done:
    #     ax.plot(spec[1]/1e4, spec[2],c=color,marker=marker,alpha=0.3)
    #     done.append(name)
    #     ax.plot([],[], label=name,c=color,marker=marker)
    # else:
    #     ax.plot(spec[1]/1e4, spec[2],c=color,marker=marker,alpha=0.3)


ax.legend(framealpha=0.0)

ax.axvline(nuv_wav,c='k',ls='--')
ax.axvline(r_wav,c='k',ls='--')
ax.axvline(J_wav,c='k',ls='--')

ax.set_xscale("log")

ax.set_xlabel("wavelength, microns")
ax.set_ylabel("AB magnitude")

ax.grid()

ax.tick_params(which="both", direction="in")

ax.invert_yaxis()

fig.savefig("specs_diff_att_laws")


fig_nuvjr, ax_nuvjr = plt.subplots(1, 1)

done=[]
for spec in specs:

    name = spec[0]
    wavs = spec[1] * aexp / 1e4
    mags = spec[2]

    name_arg = np.where(model_names==name)[0][0]

    color=colors[name_arg]
    marker=markers[name_arg]    

    nuv = mags[np.argmin(np.abs(wavs - nuv_wav))]
    r = mags[np.argmin(np.abs(wavs - r_wav))]
    J = mags[np.argmin(np.abs(wavs - J_wav))]

    if name not in done:
        ax_nuvjr.scatter(r - J, nuv - r,c=color,marker=marker,alpha=0.3)
        ax_nuvjr.scatter([],[], label=name,c=color,marker=marker)
        done.append(name)
    else:
        ax_nuvjr.scatter(r - J, nuv - r,c=color,marker=marker)




ax_nuvjr.set_xlabel("r-J")
ax_nuvjr.set_ylabel("NUV-r")

ax_nuvjr.grid()

ax_nuvjr.tick_params(which="both", direction="in")

ax_nuvjr.set_xlim(-0.1, 1.5)
ax_nuvjr.set_ylim(-0.1, 8)

ax_nuvjr.legend(framealpha=0.0)

xlim = ax_nuvjr.get_xlim()
xs = np.linspace(*xlim, 1000)
qu_lim = [max(3.1, 3 * rj + 1) for rj in xs]
ax_nuvjr.plot(xs, qu_lim, ls="--", c="k")


fig_nuvjr.savefig("nuvjr_diff_att_laws")


arg_nodust = np.where(np.array(model_names) == "noDust")[0][0] * ndir

wavs_noDust = specs[arg_nodust][1] * aexp / 1e4
mags_noDust = specs[arg_nodust][2]

bnoDust = interp_spec(b_ctr, wavs_noDust, mags_noDust)
vnoDust = interp_spec(v_ctr, wavs_noDust, mags_noDust)
ebv_noDust = bnoDust - vnoDust

fig_k, ax_k = plt.subplots()

done=[]

for spec in specs:

    name = spec[0]

    if name == "noDust":
        continue
    else:
        print(name)

    wavs_rest = spec[1] / 1e4 * aexp
    mag = spec[2]

    bdusty = interp_spec(b_ctr, wavs_rest, mag)
    vdusty = interp_spec(v_ctr, wavs_rest, mag)

    arg_rascas_norm = np.argmin(np.abs(wavs_rest - norm_tgt))

    eb =  - b_nodust + bdusty
    ev =  - v_nodust + vdusty

    ebv = eb-ev

    ext_rascas = mag - mags_noDust
    ext_rascas = ext_rascas / ebv
    # ext_rascas = ext_rascas / ebv_dusty

    # print(ebv, ebv_dusty, ebv_noDust)
    # print(ext_rascas)

    # ax_k.plot(wavs_rest, ext_rascas, label=model)

    name_arg = np.where(model_names==name)[0][0]

    color=colors[name_arg]
    marker=markers[name_arg]    

    if name not in done:
        # ax_k.plot(wavs_rest, ext_rascas / np.abs(ext_rascas[arg_rascas_norm]),alpha=0.3,color=color)
        ax_k.plot(wavs_rest, ext_rascas,alpha=0.3,color=color)
        ax_k.plot([],[], label=name,color=color)
        done.append(name)
    else:
        # ax_k.plot(wavs_rest, ext_rascas / np.abs(ext_rascas[arg_rascas_norm]),color=color,alpha=0.3)
        ax_k.plot(wavs_rest, ext_rascas,color=color,alpha=0.3)

law_dir = "/home/jlewis/attenuation_laws"

to_load = ["calzetti.dat","salim18.dat","extlaw_0.9.dat"]

lss=['--',':','-.']

for ilaw,ftgt in enumerate(to_load):
    wavs_ang,k=np.genfromtxt(os.path.join(law_dir,ftgt),skip_header=1).T

    arg_wav = np.argmin(np.abs(wavs_ang/1e4 - norm_tgt))


    ax_k.plot(wavs_ang/1e4,k,label=ftgt,c='k',ls=lss[ilaw])
    # ax_k.plot(wavs_ang/1e4,k/k[arg_wav],label=ftgt,c='k',ls=lss[ilaw])


ax_k.legend(framealpha=0.0)
ax_k.set_xscale("log")
# ax_k.set_yscale("log")
ax_k.set_ylim(-0.25,50)
ax_k.grid()
ax_k.set_xlabel("rest-frame wavelength, microns")
ax_k.set_ylabel("K(lambda)")

ax_k.set_xlim(5e-2, 4)


ax_k.axvline(nuv_wav,c='k',ls='--')
ax_k.axvline(r_wav,c='k',ls='--')
ax_k.axvline(J_wav,c='k',ls='--')

fig_k.savefig("compare_attenuation_att_laws")
