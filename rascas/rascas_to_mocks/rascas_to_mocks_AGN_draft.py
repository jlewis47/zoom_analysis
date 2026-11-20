import os
import sys
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.signal import fftconvolve
from pathlib import Path

# ======================================================================
# PATHS SETUP & IMPORTS
# ======================================================================
HERE = Path(__file__).resolve().parent  
extra_paths = [
    "/home/maegey/M2/src/zoom_toolbox", 
    "/home/maegey/M2/src/Rascas_toolbox", 
    "/home/maegey/M2/src/obs"
]
for path in extra_paths:
    if path not in sys.path: 
        sys.path.append(path)

from gremlin.read_sim_params import ramses_sim
from zoom_analysis.rascas.read_rascas import read_mock_dump
import obs_utils as obs
from Rascas_utils import (
    cube_to_mock_obs, 
    convert_cgs_to_mjy_sr, 
    inject_mock_into_background_inplace, 
#    load_psf, 
    load_background, 
    get_mock_px_as, 
    rebin_psf, 
    instrumental_rebinning
)

from Rascas_to_mocks_utils import (
    get_mock_geometry,
)

from zoom_analysis.rascas.rascas_steps import (
    filt_file_to_dict,
    get_directions_cart,
#    micron_to_fwhm_as,
)

from zoom_analysis.rascas.filts.filts import (
    read_transmissions,
)

from zoom_analysis.rascas.field_fits import (
    load_psf,
    # rebin_factor
    # rebin
    dilate,
)

# ======================================================================
# CONFIGURATION
# ======================================================================
# Target configurations
fedd_list = [1, 10, 30, 100]
agn_scenarios = ['no_agn'] + [f'fedd{f}' for f in fedd_list]
#target_filters = ['NIRCAM-F277', 'NIRCAM-F444', 'MIRI-F770']

# Paths
galaxies_csv_path = "/home/maegey/M2/data/stats/Selection_galaxies_all_z_Mbh1e5_with_apertures.csv"
positions_txt_path = "/home/maegey/M2/data/stats/2dir_5fedd_mag27" # <-- UPDATED PATH

output_dir = "/home/maegey/n23data1/omaegey/sim_data/NewHorizon/analysis"
os.makedirs(output_dir, exist_ok=True)

#filters_path = os.path.expanduser("~/M2/data/filters")
filt_dict_file = HERE.parent/"filts"/"filter_psf_res.csv"
dir_filters = HERE.parent/"filts"
#!!!!! dir_psfs to be updated to the correct path where PSF files are stored
# On Infinity : root_psf="/automnt/data101/jlewis/marko_psfs"
dir_psfs = "/home/maegey/M2/data/psfs" 


#Filters selection
filter_select = [
    # "MegaCam-u_new",
    # "HSC-g",
    # "HSC-i",
    # "HSC-r",
    # "HSC-y",
    # "HSC-z",
    # "NIRCAM-F115",
    # "NIRCAM-F150",
    "NIRCAM-F277",
    "NIRCAM-F444",
    "MIRI-F770",
    # "HST-f814_res",
    # "SC-L427",
    # "SC-L484",
    # "SC-L505",
    # "SC-L527",
    # "SC-L574",
    # "SC-L624",
    # "SC-L679",
    # "SC-L709",
    # "SC-L711",
    # "SC-L738",
    # "SC-L767",
    # "SC-L816",
    # "SC-L827",
    # "UVISTA-H",
    # "UVISTA-J",
    # "UVISTA-Ks",
    # "UVISTA-NB118",
    # "UVISTA-Y",
]

#Read the filter table (PSF file, resolution, depth) into a dictionary.
# {filter_name: {"psf_fname": str, "res_as": float,
#         "depth_aper": float, "aper_depth_as": float}}

filt_dict = filt_file_to_dict(floc=filt_dict_file)

# Read filter transmissions
names_all, wavs_all, trans_all = read_transmissions(
    tgt_bands=None,
    NIRCAM=True,
    HST=False,
    HSC=False,
    IRAC=False,
    SuprimeCam=False,
    ULTRAVISTA=False,
    MIRI=True,
    VISTA=False,
    NISP=False,
    VIS=False,
    LSST=False,
    GALEX=False,
    MegaCam=False,
    Euclid_EXT=False,
    CLUTCH=False,
    root=dir_filters
)

# # Filter dictionaries
# filt_dict = {
#     "NIRCAM-F277": {"lambda_eff": 27700.0, "psf_fname": "PSF_NIRCam_in_flight_opd_filter_F277W.fits", "fwhm_jwst_as": 0.092, "inst_px_as": 0.063, "bg_path": "/home/maegey/n23data2/analyze_gal_sims/residual_images/background_image_f277w.fits"},
#     "NIRCAM-F444": {"lambda_eff": 44400.0, "psf_fname": "PSF_NIRCam_in_flight_opd_filter_F444W.fits", "fwhm_jwst_as": 0.145, "inst_px_as": 0.063, "bg_path": "/home/maegey/n23data2/analyze_gal_sims/residual_images/background_image_f444w.fits"},
#     "MIRI-F770":   {"lambda_eff": 77000.0, "psf_fname": "PSF_MIRI_in_flight_opd_filter_F770W.fits",   "fwhm_jwst_as": 0.269, "inst_px_as": 0.110, "bg_path": "/home/maegey/n23data2/analyze_gal_sims/residual_images/background_image_MIRI-F770W_B3.fits"}
# }

# ======================================================================
# DATA LOADING & PRE-INITIALIZATION
# ======================================================================
# 1. Initialize simulation and cosmology
sim = ramses_sim("/n23data1/jlewis/sim_data/NewHorizon", full_snaps=False)
sim.init_cosmo()
#filters_df = obs.load_filters_to_dataframe(filters_path)

# 2. Load the input catalogs
df_galaxies = pd.read_csv(galaxies_csv_path)
df_positions = pd.read_csv(positions_txt_path, sep=r"\s+") 

# Ensure index is treated as integer for proper mapping
# df_positions['index'] = df_positions['index'].astype(int)

# 3. Pre-load composite PSFs
#composites = {}
#bg_headers = {}
#bg_px_as = {}
#wcs_dict = {}
psfs_norm = {}
psf_res_as = {}

print("--- Pre-loading PSFs and normalizing")
for f_name in filter_select:
    psf_fname = filt_dict[f_name]["psf_fname"]
    #fwhm_jwst_as = filt_dict[f_name]["fwhm_jwst_as"]
    
    # Load PSF file and resolution. Caution : the resolution is obtained from the get_res_from_img function (read from the background image)
    img, res_as = load_psf(psf_fname,root_psf=dir_psfs)
        
    psfs_norm[f_name] = img / np.sum(img) 
    
    psf_res_as[f_name] = res_as
    
   
    #bg_path = filt_dict[f_name]["bg_path"]
    #this_bg_data, this_bg_headers, this_bg_px_as = load_background(bg_path, f_name)
    #bg_headers[f_name] = this_bg_headers
    #bg_px_as[f_name] = this_bg_px_as
    
    #composites[f_name] = np.copy(this_bg_data)
    #wcs_dict[f_name] = wcs.WCS(this_bg_headers)

# ======================================================================
# INJECTION PIPELINE
# ======================================================================
print(f"\n--- Starting Injection Pipeline ({len(df_positions)} total injections) ---")

n_injected = 0

# Loop through every row in the pre-calculated positions file
for i_row, row in df_positions.iterrows():
    
    # 1. Identify Galaxy, Scenario, and Direction dynamically
    gal_idx = int(row['index'])
    ra, dec = row['ra'], row['dec']
    
    # Calculate scenario and direction based on row position.
    # Assuming 10 rows per galaxy: 5 scenarios for Dir 1, then 5 scenarios for Dir 2.
    scenario_idx = i_row % len(agn_scenarios)
    dir_idx = (i_row // len(agn_scenarios)) % 2 
    
    scenario = agn_scenarios[scenario_idx]
    ndir_fortran = dir_idx + 1                 # Fortran uses 1-based indexing (1 or 2)
    direction_key = f"direction_{dir_idx}"     # Dict keys are 0-based ("direction_0" or "direction_1")
    
    # Fetch physical properties from the galaxy catalog
    gal_data = df_galaxies.iloc[gal_idx]
    tgt_snap, galid = int(gal_data['snap']), int(gal_data['gid'])
    redshift = (1.0 / sim.get_snap_exps(tgt_snap)[0]) - 1
    
    print(f"[{i_row+1}/{len(df_positions)}] GalID: {galid} | Scen: {scenario} | Dir: {ndir_fortran} | RA: {ra:.4f}, DEC: {dec:.4f}")

    # 2. Path definitions
    galaxy_dir = f"/n23data1/omaegey/sim_data/NewHorizon/snap_{tgt_snap:05d}/gal_{galid:07d}"
    pfsdump_path = os.path.join(galaxy_dir, "PFSDump/pfsdump")
    stars_cube_path = os.path.join(galaxy_dir, "RASCASDump/mock_stars.cube")
    
    if not os.path.exists(stars_cube_path):
        print(f"   ⚠️ WARNING: Stars mock missing for GalID {galid}. Skipping.")
        continue

    # 3. Load Raw RASCAS Data Cubes
    full_stars_dump = read_mock_dump(stars_cube_path, ndir=ndir_fortran)
    stars_cube_raw = full_stars_dump[direction_key]
    aper_as_stars,px_as_stars,aper_sr_stars,px_sr_stars = get_mock_geometry(sim, redshift, full_stars_dump, direction_key)
    
    agn_cube_raw = None
    px_as_agn = None
    if scenario != 'no_agn':
        fedd_val = int(scenario.replace('fedd', ''))
        agn_path = os.path.join(galaxy_dir, "RASCASDump", f"mock_agn_fedd{fedd_val}.cube")
        pfagn_path = os.path.join(galaxy_dir, "PFAGNDump", f"agn_fedd{fedd_val}.dat")
        
        if os.path.exists(agn_path):
            full_agn_dump = read_mock_dump(agn_path, ndir=ndir_fortran)
            agn_cube_raw = full_agn_dump[direction_key]
            aper_as_agn,px_as_agn,aper_sr_agn,px_sr_agn = get_mock_geometry(sim, redshift, full_agn_dump, direction_key)
        else:
            print(f"   ⚠️ WARNING: AGN mock missing for {scenario}. Injecting stars only.")

    # 4. Filter Loop
    for filter_name in filter_select:
        lam_eff = filt_dict[filter_name]["lambda_eff"]
        
        img_stars_cgs = cube_to_mock_obs(
            stars_cube_raw, filter_name, filters_path, pfsdump_path, 
            redshift, sim.cosmo_model, sim.cosmo.lcMpc
        )
        img_total = convert_cgs_to_mjy_sr(img_stars_cgs, lam_eff)

        if agn_cube_raw is not None and np.isclose(px_as_stars, px_as_agn, rtol=1e-3):
            img_agn_cgs = cube_to_mock_obs(
                agn_cube_raw, filter_name, filters_path, pfagn_path, 
                redshift, sim.cosmo_model, sim.cosmo.lcMpc
            )
            img_agn = convert_cgs_to_mjy_sr(img_agn_cgs, lam_eff)
            img_total += img_agn

        psf_rebinned = rebin_psf(psfs_norm[filter_name], psf_res_as[filter_name], px_as_stars)
        img_conv = fftconvolve(img_total, psf_rebinned, mode='same')
        img_final = instrumental_rebinning(img_conv, px_as_stars, bg_px_as[filter_name])
        
        coord = SkyCoord(ra * u.deg, dec * u.deg)
        x_px, y_px = wcs_dict[filter_name].world_to_pixel(coord)
        
        inject_mock_into_background_inplace(img_final, composites[filter_name], int(x_px), int(y_px))

    n_injected += 1

print(f"\n✅ {n_injected} / {len(df_positions)} mock scenarios successfully injected.")

# ======================================================================
# SAVE MASTER IMAGES
# ======================================================================
print("\n--- Saving Master Images ---")
for filter_name in filter_select:
    filepath = os.path.join(output_dir, f"master_cosmosweb_injected_{filter_name}.fits")
    final_data = composites[filter_name]
    clean_data = np.nan_to_num(final_data, nan=0.0, posinf=0.0, neginf=0.0)
    fits.writeto(filepath, clean_data, bg_headers[filter_name], overwrite=True)
    print(f"Saved: {filepath}")