import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import binned_statistic_2d
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.ndimage import map_coordinates, shift, zoom



lowres_as=0.168
highres_as=0.033

image_size_as = 3.0

# ctr_tgt_as = (0.6632,1.1256)

ctr_tgt_as = np.random.uniform(size=2,low=lowres_as,high=image_size_as-lowres_as)
# ctr_tgt_as = (5.,5.)

print(ctr_tgt_as)

highres_mock = np.zeros((13,13))
ctr=6
highres_mock[ctr,ctr-1:ctr+2]=1
highres_mock[ctr-1:ctr+2,ctr]=1

highres_mock_size = highres_mock.shape
highres_mock_size_as = np.array(highres_mock_size)*highres_as

xhighs = np.arange(0,highres_mock_size_as[0],highres_as)#+0.5*highres_as
yhighs = np.arange(0,highres_mock_size_as[1],highres_as)#+0.5*highres_as
# xhighs = np.arange(highres_as,highres_mock_size_as[0],highres_as)
# yhighs = np.arange(highres_as,highres_mock_size_as[1],highres_as)

xhigh_grid,yhigh_grid = np.meshgrid(xhighs,yhighs,indexing='ij')

highres_interpolator = LinearNDInterpolator(points=(np.ravel(xhigh_grid),np.ravel(yhigh_grid)),values=np.ravel(highres_mock))
# highres_interpolator = RegularGridInterpolator(points=(xhigh_grid,yhigh_grid),values=highres_mock)

# half_size_mock = highres_mock.shape
highres_mock_size_as = np.array(highres_mock.shape) * highres_as

xlows = np.arange(0,highres_mock_size_as[0],lowres_as)#/highres_as
ylows = np.arange(0,highres_mock_size_as[1],lowres_as)#/highres_as


# xlow_grid,ylow_grid = np.meshgrid(xlows,ylows,indexing='ij')

# lowres_mock = highres_interpolator((xlow_grid,ylow_grid))

lowres_mock = binned_statistic_2d(np.ravel(xhigh_grid),np.ravel(yhigh_grid),np.ravel(highres_mock),'sum',bins=[xlows,ylows]).statistic


imgs = [lowres_mock,highres_mock]



sky_images=[]
printed_images=[]
ress=[lowres_as,highres_as]

for ires,(res,img) in enumerate(zip(ress,imgs)):

    npix = int(image_size_as/res)
    if npix*res<image_size_as:npix+=1

    npix_img = img.shape[0]
    half_pix = int(0.5*npix_img)
    if 2*half_pix<npix_img:half_pix+=1

    sky_images.append(np.zeros((npix,npix)))

    # image_ctr = int(0.5*npix)
    # px_ctrs = np.array(np.mgrid[:npix,:npix])/npix * res  + res*0.5
    img_sh=img.shape

    full_res_img_sh = highres_mock.shape



    true_xs = np.linspace(ctr_tgt_as[0]-0.5*full_res_img_sh[0]*highres_as-highres_as,ctr_tgt_as[0]+0.5*full_res_img_sh[0]*highres_as+highres_as,full_res_img_sh[0])
    true_ys = np.linspace(ctr_tgt_as[1]-0.5*full_res_img_sh[1]*highres_as-highres_as,ctr_tgt_as[1]+0.5*full_res_img_sh[1]*highres_as+highres_as,full_res_img_sh[1])

    # print(ctr_tgt_as[0]-0.5*full_res_img_sh[0]*highres_as-highres_as,ctr_tgt_as[0]+0.5*full_res_img_sh[0]*highres_as+highres_as,full_res_img_sh[0])

    xpxs = np.arange(npix)*res +0.5*res
    # x_closest = np.argmin(np.abs(ctr_tgt_as[0]-xpxs))
    x_closest = int((ctr_tgt_as[0]-xpxs[0]-0.5*res)//res)+1
    ypxs = np.arange(npix)*res +0.5*res
    y_closest = int((ctr_tgt_as[1]-ypxs[0]-0.5*res)//res)+1
    # y_closest = np.argmin(np.abs(ctr_tgt_as[1]-ypxs))

    x_closest_as = xpxs[x_closest]
    y_closest_as = ypxs[y_closest]
    
    # xhighs = np.arange(0,highres_mock_size_as[0],highres_as)
    # yhighs = np.arange(0,highres_mock_size_as[1],highres_as)
    xhigh_grid,yhigh_grid = np.meshgrid(true_xs,true_ys,indexing='ij')
    # insert_xbins = np.arange(x_closest_as-0.5*highres_mock_size_as[0],x_closest_as+0.5*highres_mock_size_as[0]+res,res)
    # insert_ybins = np.arange(y_closest_as-0.5*highres_mock_size_as[1],y_closest_as+0.5*highres_mock_size_as[1]+res,res)
    insert_xbins = np.arange(x_closest_as-(half_pix+1)*res,x_closest_as+(half_pix+1)*res+res,res)
    insert_ybins = np.arange(y_closest_as-(half_pix+1)*res,y_closest_as+(half_pix+1)*res+res,res)
    # insert_xbins = np.linspace(x_closest_as-0.5*img_sh[0],x_closest_as+0.5*img_sh[0],img_sh[0]+1)
    # insert_ybins = np.linspace(y_closest_as-0.5*img_sh[1],y_closest_as+0.5*img_sh[1],img_sh[1]+1)

    true_xg,true_yg=np.meshgrid(true_xs,true_ys)

    # if len(insert_xbins)<len(insert_ybins):
    #     print(len(insert_xbins),len(insert_ybins))
    #     print(1)
    #     insert_xbins = np.arange(x_closest_as-half_pix*res,x_closest_as+half_pix*res+2*res,res)
    # elif len(insert_xbins)>len(insert_ybins):
    #     print(len(insert_xbins),len(insert_ybins))
    #     print(2)
        
    #     insert_ybins = np.arange(y_closest_as-half_pix*res,y_closest_as+half_pix*res+2*res,res)

    

    this_res_mock = binned_statistic_2d(np.ravel(true_xg),np.ravel(true_yg),np.ravel(highres_mock),'sum',bins=[insert_xbins,insert_ybins]).statistic
    this_res_mock_sh = this_res_mock.shape

    print(true_xs)

    print(ires,res,this_res_mock.shape)
    print(x_closest_as-half_pix*res,x_closest_as+half_pix*res+res,res,len(insert_xbins))
    print(insert_xbins)
    print(y_closest_as-half_pix*res,y_closest_as+half_pix*res+res,res,len(insert_ybins))
    print(insert_ybins)

    print(this_res_mock)


    printed_images.append(this_res_mock)

    # res_shift =  1*(np.array([x_closest_as,y_closest_as])-ctr_tgt_as)
    # print(x_closest_as,y_closest_as,ctr_tgt_as,res_shift,res)

    # recentrd_high_res_img = shift(highres_mock,res_shift,mode='nearest')

    # this_res_mock = zoom(recentrd_high_res_img,(highres_as/res))
    
    this_res_mock_sh = this_res_mock.shape

    xmin = x_closest-half_pix
    ymin = y_closest-half_pix
    xmax = x_closest+half_pix
    ymax = y_closest+half_pix

    sky_images[ires][xmin:xmin+this_res_mock_sh[0],ymin:ymin+this_res_mock_sh[1]]=this_res_mock
    
    # break


    # sky_images[ires][xmin:xmin+img_sh[0],ymin:ymin+img_sh[1]]+=img


# fig,ax=plt.subplots(3,2,sharex="row",sharey='row')
fig,ax=plt.subplots(2,2,sharex="row",sharey='row')



ax[0][0].imshow(sky_images[0].T,origin='lower',extent=[0,sky_images[0].shape[0]*ress[0],0,sky_images[0].shape[1]*ress[0]],vmin=0,vmax=5)
ax[0][0].scatter(ctr_tgt_as[0],ctr_tgt_as[1],marker='o',facecolors='none',edgecolors='red')
ax[0][1].imshow(sky_images[1].T,origin='lower',extent=[0,sky_images[1].shape[0]*ress[1],0,sky_images[1].shape[1]*ress[1]],vmin=0,vmax=5)
ax[0][1].scatter(ctr_tgt_as[0],ctr_tgt_as[1],marker='o',facecolors='none',edgecolors='red')
# im=ax[1][0].imshow(imgs[0].T,origin='lower',extent=[0,highres_mock_size_as[0],0,highres_mock_size_as[1]],vmin=0,vmax=5)
# im=ax[1][1].imshow(imgs[1].T,origin='lower',extent=[0,highres_mock_size_as[0],0,highres_mock_size_as[1]],vmin=0,vmax=5)
im=ax[1][0].imshow(printed_images[0].T,origin='lower',extent=[0,printed_images[0].shape[0]*ress[0],0,printed_images[0].shape[1]*ress[0]],vmin=0,vmax=5)
ax[1][1].imshow(printed_images[1].T,origin='lower',extent=[0,highres_mock_size_as[0],0,highres_mock_size_as[1]],vmin=0,vmax=5)
plt.colorbar(im,ax=ax)

fig.savefig('test_insert')