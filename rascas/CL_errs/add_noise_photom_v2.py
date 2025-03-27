from astropy.cosmology import Planck13 as cosmo
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import yaml
import sys
import matplotlib

from scipy.stats import gaussian_kde


def AddErrorOnFlux(flux, f, depth, fact_err, nsig):
    '''
    f is the filter name
    depth, fact_err, nsig are dictionnaries which give the depth at nsig sigma, and fact_err is the square root of the inverse of the gain
    
    it returns the perturbed flux, flux error, perturbed magnitudes, magnitudes errors
    '''
    onesig = depth[f]+2.5*np.log10(nsig[f])
    onesig = 10**((-onesig+23.9)/2.5)

    fluxerr = np.sqrt(np.random.normal(loc=onesig,scale=onesig/4.,size=len(flux))**2)
    newflux = np.random.normal(loc=flux,scale=fluxerr,size=len(flux))
    
    fluxerr = np.sqrt((fluxerr)**2+flux*fact_err[f]**2)

    mag = -2.5*np.log10(newflux)+23.9
    magerr = 1.086*np.abs(fluxerr/newflux)
    return newflux, fluxerr, mag, magerr


def estimate_pseudogain(flux, flux_err):
    w = np.where((flux_err>0)*(flux > 20) *(flux <600))[0]

    gain = np.mean(flux_err[w]**2/flux[w])
    return 1./gain


def Plot_magerr(mag, fluxerr, f, magbins_pdf=[18.,20.,22.,24.,26.,28.,30.], ax=[],leg=1, pl=1):
    if (pl == 1):
        
        fig, ax = plt.subplots(1,1,figsize=(8,6))
    
    
    for k in range(len(magbins_pdf)-1):
        cond = (mag>magbins_pdf[k]) & (mag<magbins_pdf[k+1])
        h1,binedg = np.histogram(fluxerr[cond],range=[0.,0.3],bins=nb_pdf,density=True)

        xx=(binedg[1:]+binedg[:-1])/2.
        ax.plot(xx,h1,label=r''+str(magbins_pdf[k])+r'<m$_{\rm '+f+'}$<'+str(magbins_pdf[k+1]))
    if (leg==1):
        leg = plt.legend(loc='lower right', fontsize=16)
        leg.get_frame().set_linewidth(0.0)
    if (pl==1):
        ax.set_xlabel(r'error on the flux $\mu$ Janskies')

    
    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14) 

    return ax

def CreateFITSPHOTOM_HzAGN(outname, depth,fact_err, nsig):
    
   
    f = fits.open('/n07data/Horizon-AGN/JWST-like/HorizonAGN_COSMOS-Web_v2.0.fits')
    tab = f[1].data
    
    list_filt = ['HSTF814W_TOTnoerr', 'F444W_TOTnoerr','F115W_TOTnoerr','F150W_TOTnoerr','F277W_TOTnoerr','F770W_TOTnoerr', 'u_TOTnoerr', 'gHSC_TOTnoerr', 'rHSC_TOTnoerr', 'iHSC_TOTnoerr', 'zHSC_TOTnoerr', 'yHSC_TOTnoerr', 'Y_TOTnoerr', 'Ks_TOTnoerr', 'H_TOTnoerr', 'J_TOTnoerr', 'IB427_TOTnoerr', 'IB467_TOTnoerr', 'IB484_TOTnoerr', 'IB505_TOTnoerr', 'IB527_TOTnoerr', 'IB574_TOTnoerr', 'IB624_TOTnoerr', 'IB679_TOTnoerr', 'IB709_TOTnoerr', 'IB738_TOTnoerr', 'IB767_TOTnoerr','IB827_TOTnoerr', 'NB711_TOTnoerr', 'NB816_TOTnoerr', 'B_TOTnoerr', 'V_TOTnoerr', 'r_TOTnoerr', 'ip_TOTnoerr', 'zpp_TOTnoerr', 'ch1_TOTnoerr','ch2_TOTnoerr', 'NUV_TOTnoerr','FUV_TOTnoerr']
    inter_filt = ['IB427_TOTnoerr', 'IB467_TOTnoerr', 'IB484_TOTnoerr', 'IB505_TOTnoerr', 'IB527_TOTnoerr', 'IB574_TOTnoerr', 'IB624_TOTnoerr', 'IB679_TOTnoerr', 'IB709_TOTnoerr', 'IB738_TOTnoerr', 'IB767_TOTnoerr','IB827_TOTnoerr']
    
    col_list = []
    col_list.append(fits.Column(name='ID', array=tab.field('ID'), format='K'))
    
    for f in list_filt:
        if (f == 'IB527_TOTnoerr'):
            fb = 'I527_TOTnoerr'
        elif (f in inter_filt):
            fb = 'IA'+f.split('IB')[1]
        else:
            fb = f
           
        flux = 10**((-tab.field(fb)+23.9)/2.5)
        newflux,fluxerr,mag,magerr = AddErrorOnFlux(flux,f.split('_')[0],depth,fact_err, nsig)
        col_list.append(fits.Column(name=f.split('_')[0]+'_FLUX_noerr', array=flux, format='D'))
        col_list.append(fits.Column(name=f.split('_')[0]+'_FLUX', array=newflux, format='D'))
        col_list.append(fits.Column(name=f.split('_')[0]+'_FLUXERR', array=fluxerr, format='D'))
        col_list.append(fits.Column(name=f.split('_')[0]+'_MAG_noerr', array=tab.field(fb), format='D'))
        col_list.append(fits.Column(name=f.split('_')[0]+'_MAG', array=mag, format='D'))
        col_list.append(fits.Column(name=f.split('_')[0]+'_MAGERR', array=magerr, format='D'))

    cols = fits.ColDefs(col_list)
    f = fits.BinTableHDU.from_columns(cols)
            
    f.writeto(outname,overwrite=1)

def CreateFITSPHOTOM_dream(outname, depth,fact_err, nsig):
    
    f =fits.open('/n07data/laigle/JWST/DREAM-JWST/DREaM_photo2_b.fits')
    tab = f[1].data
    f =fits.open('/n07data/laigle/JWST/DREAM-JWST/DREaM_photo.fits')
    tab2 = f[1].data
    col_list = []
    col_list.append(fits.Column(name='ID', array=tab.field('ID'), format='K'))
    col_list.append(fits.Column(name='RA', array=tab.field('RA'), format='D'))
    col_list.append(fits.Column(name='Dec', array=tab.field('Dec'), format='D'))    
    ######u######
    
    for f in ['g','r','i','z']:
        flux = 10**((-tab.field('subaru_hsc_'+f)+23.9)/2.5)
        newflux,fluxerr,mag,magerr=AddErrorOnFlux(flux,f,depth,fact_err, nsig)

        col_list.append(fits.Column(name='HSC_'+f+'_FLUX_noerr', array=flux, format='D'))
        col_list.append(fits.Column(name='HSC_'+f+'_FLUX', array=newflux, format='D'))
        col_list.append(fits.Column(name='HSC_'+f+'_FLUXERR', array=fluxerr, format='D'))
        col_list.append(fits.Column(name='HSC_'+f+'_MAG', array=mag, format='D'))
        col_list.append(fits.Column(name='HSC_'+f+'_MAGERR', array=magerr, format='D'))
    for f in ['u']:
        flux = 10**((-tab.field('megacam_'+f)+23.9)/2.5)
        newflux,fluxerr,mag,magerr=AddErrorOnFlux(flux,f,depth,fact_err, nsig)

        col_list.append(fits.Column(name='CFHT_'+f+'_FLUX_noerr', array=flux, format='D'))
        col_list.append(fits.Column(name='CFHT_'+f+'_FLUX', array=newflux, format='D'))
        col_list.append(fits.Column(name='CFHT_'+f+'_FLUXERR', array=fluxerr, format='D'))
        col_list.append(fits.Column(name='CFHT_'+f+'_MAG', array=mag, format='D'))
        col_list.append(fits.Column(name='CFHT_'+f+'_MAGERR', array=magerr, format='D'))
    for f in ['F814W']:
        
        flux = 10**((-tab2.field('WFC_ACS_'+f)+23.9)/2.5)
        newflux,fluxerr,mag,magerr=AddErrorOnFlux(flux,f,depth,fact_err, nsig)

        col_list.append(fits.Column(name='WFC_ACS_'+f+'_FLUX_noerr', array=flux, format='D'))
        col_list.append(fits.Column(name='WFC_ACS_'+f+'_FLUX', array=newflux, format='D'))
        col_list.append(fits.Column(name='WFC_ACS_'+f+'_FLUXERR', array=fluxerr, format='D'))
        col_list.append(fits.Column(name='WFC_ACS_'+f+'_MAG', array=mag, format='D'))
        col_list.append(fits.Column(name='WFC_ACS_'+f+'_MAGERR', array=magerr, format='D'))

    cols = fits.ColDefs(col_list)
    f = fits.BinTableHDU.from_columns(cols)
            
    f.writeto(outname,overwrite=1)