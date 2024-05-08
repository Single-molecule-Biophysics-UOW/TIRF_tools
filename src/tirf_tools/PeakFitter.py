# -*- coding: utf-8 -*-
"""
Created on Mon May 29 12:56:49 2023

@author: smueller

peak finder and fitter for python!

"""

from tkinter import filedialog
from tkinter import Tk
from dask_image import imread
import io

from HMM_barcoding.image_utils import io_all
from dask import array as da
from dask_image import ndfilters
import napari
import skimage
import numpy as np
import scipy.optimize as opt
import tqdm
from dask.diagnostics import ProgressBar
from scipy.spatial import distance
# from scipy import 
# from ome_zarr.io import parse_url
# from ome_zarr.reader import Reader
import napari
from concurrent.futures import ThreadPoolExecutor
import dask.config as cfg
pool = ThreadPoolExecutor()
cfg.set(pool=pool)


#%% find pareticles
#use difference of gaussian algorithm

def peak_finder(data, roi= [0,0,512,512], min_dist = 10,**kwargs):
    #TODO: make sure its only one frame!
    #kwords: max_sigma=20, threshold_rel=0.05, overlap = 1.
        
    blobs_dog = skimage.feature.blob_log(data, **kwargs)
    
    peaks = blobs_dog[:,0:2]
    
    peaks = peaks[(peaks[:,0] > roi[0]) & (peaks[:,1] > roi[1])]
    peaks = peaks[(peaks[:,0] < roi[2]) & (peaks[:,1] < roi[3])]
    
    mask = np.ones(peaks.shape, dtype = bool)
    dist = distance.cdist(peaks,peaks)
    
    for i,d in enumerate(dist):
        mind = np.min(d[d!=0])
        if mind < min_dist:
            mask[i] = False
    new_points = peaks[mask].reshape(-1,2)
    print('found {} spots'.format(new_points.shape[0]))
    return new_points


def make_projection(data, projection = 'mean'):
    peaks = {}
    projs = {}
    for im in data:
        #make projection
        projs[im] = getattr(type(data[im]),projection)(data[im], axis=0)
        
    return projs

def make_single_projection(data, projection = 'mean'):
    
    
    
        #make projection
    proj = getattr(type(data),projection)(data, axis=0)
        
    return proj



#fit peaks:
def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset): 
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                                + c*((y-yo)**2)))
    return g.ravel()


def peak_fitter(data, points, r, intitial_guess = [2,2,0,0]):
    
    #loops through spots:    
    spots = []
    f=0
    for roi in tqdm.tqdm(points):
        x,y = roi[0],roi[1]
        # print(data.shape)
        fit_region = data[x-r:x+r,y-r:y+r]
        #check boundaries:
        if fit_region.shape != (2*r,2*r):
            tqdm.tqdm.write('bad shape')
            # print(fit_region)
            continue
        #create meshgrid for fitting
        xm = np.linspace(x-r, x+r, 2*r)
        ym = np.linspace(y-r, y+r, 2*r)
        xm, ym = np.meshgrid(xm, ym)
        # #flatten the image
        flat_data = fit_region.ravel()
        #set a resonable initial guess, i.e. the middle
        initial_guess = (np.max(fit_region),x,y,*intitial_guess)
        bounds=((0,x-r,y-r,0,0,-np.inf, -np.inf),(1e06,x+r,y+r,1e06,1e06,1e06,1e06))
        try:
            popt, pcov = opt.curve_fit(twoD_Gaussian, (xm, ym), flat_data, p0=initial_guess, xtol=1e-03)#, bounds = bounds)
        except RuntimeError:
            tqdm.tqdm.write('fit failed for pot at {},{}'.format(x,y))
            continue
        #TODO except OptimizeWarning
        # #write result in a list
        spots.append([popt[0],popt[1], popt[2], popt[3],popt[4], popt[5], popt[6]])
    return spots
        
#%%    
if __name__ == "__main__":    
    with ProgressBar(): #load full image file in memory
        data = [x for x in io_all.load_image(sigma = 20, darkframe = 480)]
        
    
    #%%
    if len(data) == 1:
        data = data[0]
    #%%
    #blur in slice direction before projecting:
    blurred = ndfilters.gaussian_filter(data,  (10,0,0))
    
    #%%
    with ProgressBar():
        proj = da.mean(data, axis = 0).compute()
        proj_blur = da.mean(blurred, axis = 0).compute()
    #%%

    #%%
    with ProgressBar():
        proj_std = da.std(blurred, axis = 0).compute()
        # proj_std_n = da.std(data, axis = 0).compute()

    #%%
    points = peak_finder(proj_std, max_sigma =10,threshold_rel=0.02, overlap = 1.,roi = [10,10,502,502], min_dist = 3)
    #%%
    points_blur = peak_finder(proj_std_n, max_sigma =10,threshold_rel=0.02, overlap = 1.,roi = [10,10,502,502], min_dist = 3)
    #%%
    
    viewer = napari.Viewer()
    #%%
    viewer.add_image(proj_std)
    #%%
    viewer.add_image(data)
    
    #%%
    viewer.add_points(points,edge_color = 'green', face_color='transparent', size = 8, blending = 'additive', name = 'z blur before proj')
    #%%
    viewer.add_points(points_blur,edge_color = 'magenta', face_color='transparent', size = 8, blending = 'additive',name = 'std proj')
    
    #%%
    fitted_points = peak_fitter(proj,points,5)





    viewer.add_points(fitted_points,edge_color = 'red', face_color='red',symbol ='cross' , size = 2, edge_width = 0.05)