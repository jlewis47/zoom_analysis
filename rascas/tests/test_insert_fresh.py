import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import binned_statistic_2d
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.ndimage import map_coordinates, shift, zoom, convolve



lowres_as=0.168
highres_as=0.033

#create test source

r_src_as = 1.0 
source_size = int(np.ceil(r_src_as/highres_as))
if source_size%2==0:
    source_size+=1

source_image = np.zeros((source_size,source_size))
source_ctr = int(0.5*source_size)

#cross in the centre
# source_image[source_ctr,:]=1
# source_image[:,source_ctr]=1

#box in the centre
source_image[source_ctr-5:source_ctr+6,source_ctr-5:source_ctr+6]=1


#gaussian
# source_image = (np.exp())

fig,ax=plt.subplots(3,2,sharex="row",sharey='row',layout='constrained',figsize=(8,12))


n_coarse  = int(np.ceil((r_src_as)/lowres_as)) 
if n_coarse*lowres_as < r_src_as:
    n_coarse+=1
if n_coarse%2==0:
    n_coarse+=1

r_coarse_as=n_coarse*lowres_as

xfine = np.linspace(0,source_size*highres_as,source_size)
coarse_bins = np.linspace(0,r_coarse_as,n_coarse)

xg,yg = np.meshgrid(xfine,xfine)

coarse_image = binned_statistic_2d(np.ravel(xg),np.ravel(yg),np.ravel(source_image),'sum',bins=[coarse_bins,coarse_bins]).statistic


# vmax_img= coarse_image.max()

vmin=0
vmax=20

ax[0,0].imshow((source_image.T),extent=[0,r_src_as,0,r_src_as],origin="lower",vmin=vmin,vmax=vmax)
ax[0,0].set_title('fine')
ax[0,1].set_title('coarse')

im=ax[0,1].imshow((coarse_image.T),extent=[0,r_coarse_as,0,r_coarse_as],origin="lower",vmin=vmin,vmax=vmax)

ax[0,1].set_xlim(0,r_src_as)
ax[0,1].set_ylim(0,r_src_as)

# plt.colorbar(im,ax=ax[0,:])


#Create mosaic into which we insert

mosaic_size_as=(3,3)

nmosaic_fine = np.int32(np.ceil(np.array(mosaic_size_as)/highres_as))
if np.any(nmosaic_fine*highres_as<mosaic_size_as):
    nmosaic_fine+=1
mosaic_fine = np.zeros(nmosaic_fine)

nmosaic_coarse = np.int32(np.ceil(np.array(mosaic_size_as)/lowres_as))
if np.any(nmosaic_coarse*lowres_as<mosaic_size_as):
    nmosaic_coarse+=1
mosaic_coarse = np.zeros(nmosaic_coarse)


#pick insertion point

ctr_tgt_as = np.random.uniform(size=2,low=r_src_as,high=mosaic_size_as[0]-r_src_as)

# ctr_tgt_as = np.array([2.5*5.090909090909091*lowres_as]*2)
# ctr_tgt_as=np.array([1.70093247, 1.41129425])
# ctr_tgt_as=np.array([1.27095955, 1.71823873])
print(ctr_tgt_as)
# ax[1,0].scatter(*ctr_tgt_as)
# ax[1,1].scatter(*ctr_tgt_as)

#insert into fine mosaic


closest_fine_cell = ctr_tgt_as//highres_as
closest_fine_as = closest_fine_cell * highres_as

ax[1,0].scatter(closest_fine_as[0]+0.5*highres_as,closest_fine_as[1]+0.5*highres_as,marker="x")
ax[1,1].scatter(*(closest_fine_as+0.5*highres_as),marker='x')


mosaic_fine[int(closest_fine_cell[0]-source_ctr):int(closest_fine_cell[0]+source_ctr)+1,
            int(closest_fine_cell[1]-source_ctr):int(closest_fine_cell[1]+source_ctr)+1] = source_image

#insert into coarse mosaic


# closest_coarse_cell = ctr_tgt_as//lowres_as
closest_coarse_cell_x = np.argmin(np.abs((closest_fine_cell[0]+0.5)*highres_as-(np.linspace(0,nmosaic_coarse[0]*lowres_as,nmosaic_coarse[0]+1)+0.5*lowres_as)))
closest_coarse_cell_y = np.argmin(np.abs((closest_fine_cell[1]+0.5)*highres_as-(np.linspace(0,nmosaic_coarse[1]*lowres_as,nmosaic_coarse[1]+1)+0.5*lowres_as)))
closest_coarse_cell=np.array([closest_coarse_cell_x,closest_coarse_cell_y])
closest_coarse_as = closest_coarse_cell * lowres_as
# closest_coarse_as = ctr_tgt_as//lowres_as * lowres_as

ax[1,1].scatter((closest_coarse_cell[0]+0.5)*lowres_as,(closest_coarse_cell[1]+0.5)*lowres_as,marker="x")

diff_to_tgt_as = (closest_coarse_as+0.5*lowres_as)-(closest_fine_as+0.5*highres_as)

margin=1

nxcoarse = int(((r_coarse_as+margin*lowres_as) - (-margin*lowres_as))//lowres_as)
if nxcoarse%2==0:
    nxcoarse+=1
nycoarse = int(((r_coarse_as+margin*lowres_as) - (-margin*lowres_as))//lowres_as)
if nycoarse%2==0:
    nycoarse+=1

xcoarse_bins = np.linspace(-margin*lowres_as,r_coarse_as+margin*lowres_as,nxcoarse)
ycoarse_bins = np.linspace(-margin*lowres_as,r_coarse_as+margin*lowres_as,nycoarse)
# ycoarse_bins = np.arange(-abs(diff_to_tgt_as[1])-lowres_as,r_coarse_as+abs(diff_to_tgt_as[1])+lowres_as,lowres_as)

nxcoarse_ctr = int(0.5*nxcoarse)
nycoarse_ctr = int(0.5*nycoarse)


offcenter_xg = xg - diff_to_tgt_as[0]
offcenter_yg = yg - diff_to_tgt_as[1]


rescale=np.round(lowres_as/highres_as)

kern_size=int(rescale)
kernel = np.ones((kern_size,kern_size))
kernel/=kernel.sum()

fine_degraded = convolve(source_image,kernel,mode='constant',cval=0.0)

coarse_in_mosaic = binned_statistic_2d(np.ravel(offcenter_xg),np.ravel(offcenter_yg),np.ravel(fine_degraded),'sum',bins=[xcoarse_bins,ycoarse_bins]).statistic

#conserve number of photons
# coarse_in_mosaic/= rescale**2


mosaic_coarse[int(closest_coarse_cell[0]-nxcoarse_ctr)+1:int(closest_coarse_cell[0]+nxcoarse_ctr)+1,
            int(closest_coarse_cell[1]-nycoarse_ctr)+1:int(closest_coarse_cell[1]+nycoarse_ctr)+1] = coarse_in_mosaic


alphas_fine = np.zeros_like(mosaic_fine)
alphas_fine[mosaic_fine>0]=0.5
alphas_coarse = np.zeros_like(mosaic_coarse)
alphas_coarse[mosaic_coarse>0]=0.5
ax[1,0].imshow((mosaic_fine.T),extent=[0,mosaic_fine.shape[0]*highres_as,0,mosaic_fine.shape[1]*highres_as],origin="lower",vmin=vmin,vmax=vmax)#,alpha=alphas_fine.T
ax[2,0].imshow((fine_degraded.T),extent=[0,mosaic_fine.shape[0]*highres_as,0,mosaic_fine.shape[1]*highres_as],origin="lower",vmin=vmin,vmax=vmax)
# ax[1,1].imshow(mosaic_fine.T,extent=[0,mosaic_fine.shape[0]*highres_as,0,mosaic_fine.shape[1]*highres_as],origin="lower",alpha=alphas_fine.T,cmap='Reds')
ax[1,1].imshow((mosaic_coarse.T),extent=[0,mosaic_coarse.shape[0]*lowres_as,0,mosaic_coarse.shape[1]*lowres_as],origin="lower",vmin=vmin,vmax=vmax)#,alpha=alphas_coarse.T)


ax[1,1].set_xlim(0,mosaic_size_as[0])
ax[1,1].set_ylim(0,mosaic_size_as[1])

ax[1,0].axvline(closest_fine_cell[0]*highres_as+0.5*highres_as,c='tab:blue',ls='--')
ax[1,0].axhline(closest_fine_cell[1]*highres_as+0.5*highres_as,c='tab:blue',ls='--')
ax[1,1].axvline(closest_fine_cell[0]*highres_as+0.5*highres_as,c='tab:blue',ls='--')
ax[1,1].axhline(closest_fine_cell[1]*highres_as+0.5*highres_as,c='tab:blue',ls='--')

fig.savefig('test_insert2.png')


plt.close()