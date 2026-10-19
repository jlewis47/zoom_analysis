import numpy as np
from zoom_analysis.rascas.field_fits import fit_galaxies, gals_to_fit_pos
from gremlin.read_sim_params import ramses_sim
from zoom_analysis.halo_maker.assoc_fcts import get_gal_props_snap

fname="Selection_galaxies_all_z_Mbh1e5_with_apertures.csv"

tgt_gals = np.genfromtxt(fname,dtype=None,names=True,delimiter=',')

rtgts_as = tgt_gals['aperture_as'] * 0.5 #size of galaxy is half aperture

sim = ramses_sim("/data101/jlewis/sims/NH_for_rascas", nml="cosmo.nml",full_check=False)

sim.init_cosmo()

aper_eff_as= np.zeros(len(rtgts_as))
last_snap=-1


for igal in range(len(aper_eff_as)):

    cur_snap = int(tgt_gals['snap'][igal])
    gid = int(tgt_gals['gid'][igal])
    zmock = float(tgt_gals['z_target'][igal])

    if (last_snap==-1) or (cur_snap!=last_snap):

        gal_cat = get_gal_props_snap(sim.path, cur_snap)

    this_gal = gal_cat['gids']==gid

    this_gals_subs = (gal_cat['host gid']==gid) * (gal_cat['gids']!=gid) * (gal_cat['mass']>=1e6)
    if this_gals_subs.sum()>0:
        subs_pos = gal_cat['pos'][:,this_gals_subs]
        subs_rs = gal_cat['rmax'][this_gals_subs]
        # subs_rs = all_gal_props['r90'][this_gals_subs] * 1.5

        dists_to_subs = np.linalg.norm(gal_cat['pos'][:,this_gal] - subs_pos,axis=0)
        max_extent_subs = dists_to_subs + subs_rs

        kpc50_lim = max_extent_subs< (50/(sim.cosmo.lcMpc*1e3))

        if np.any(kpc50_lim):
            max_containinig_subs = max_extent_subs[kpc50_lim].max()

            cur_rmax = max(max_containinig_subs, gal_cat['r90'][igal])
            # print("some close sub gal make aper larger")
        else:

            cur_rmax = gal_cat['r90'][igal] * 1.5


    else:

        cur_rmax = gal_cat['r90'][igal] * 1.5

    aper_eff_as[igal] = cur_rmax * 1e3 * sim.cosmo.lcMpc * sim.cosmo_model.arcsec_per_kpc_comoving(zmock).value

# rtgts_as[:] = 2.0
#suggest adding maximum opening... huge size mocks are mostly empty any way.
# size_lim_as=3.
# rtgts_as = np.where(aper_eff_as>size_lim_as,size_lim_as,aper_eff_as)
rtgts_as = aper_eff_as


idx = np.arange(len(rtgts_as))

nb_directions = 12
nb_fedd = 5

rtgts_as = np.repeat(rtgts_as,nb_directions*nb_fedd)
all_idx = np.repeat(idx,nb_directions*nb_fedd)

miri_mask_fname = "/data101/jlewis/mask_MIRI_F770_non_zero.fits"

(
    tile_code,
    tile_shape_px,
    found_pos_px,
    all_rgals_px,
    found_all,
    done_bools,
    wcs,
) = gals_to_fit_pos(
    rtgts_as,
    miri_mask = miri_mask_fname
)

found_pos_deg = wcs.pixel_to_world(
    found_pos_px[done_bools, 0], found_pos_px[done_bools, 1]
)


np.savetxt('test_pos',np.transpose([all_idx[done_bools],rtgts_as[done_bools],found_pos_deg.ra,found_pos_deg.dec]))