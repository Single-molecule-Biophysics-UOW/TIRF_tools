# -*- coding: utf-8 -*-
"""
Created on Mon May 29 12:56:49 2023

@author: smueller

peak finder and fitter for python!

"""

from tkinter import filedialog
from tkinter import Tk
from dask_image import imread
from tirf_tools import io, corrections

from HMM_barcoding.image_utils import io_all
from dask import array as da
from dask_image import ndfilters
from napari import Viewer
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


def chooseXY(data, dim_order = 'TCZYX'):
    num_dims = data.ndim
    print(num_dims)
    Xindex,Yindex = dim_order.index('X'), dim_order.index('Y')
    slices = [slice(0)] * num_dims
    slices[Xindex] = slice(data.shape[Xindex])
    slices[Yindex] = slice(data.shape[Yindex])
    data = data[tuple(slices)]
    print(data.shape)

def peak_finder(data, roi= [0,0,512,512], min_dist = 10,**kwargs):
    #TODO: make sure its only one frame!
    #kwords: max_sigma=20, threshold_rel=0.05, overlap = 1.
    #only 2D is supported for now
    #assumption: Y and X are the two last dimensions in the array
    try:
        data=data.reshape(data.shape[-2],data.shape[-1])
    except ValueError:
        print('''failed to reshape the array into (1,Y,X,)\n
              if you tried to find peaks on a multi color image try 
              to separate the colors using indexing: color1 = data[0,0,:,:] etc.''')
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

def projection(data, projection = 'mean', dim_order = 'TCZYX', dimension = 'T'):
    #find axis index
    axis = dim_order.index(dimension)
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
    #only 2D is supported for now
    #assumption: Y and X are the two last dimensions in the array
    data=data.reshape(data.shape[-2],data.shape[-1])
    #loops through spots:    
    spots = []
    f=0
    for roi in tqdm.tqdm(points):
        x,y = int(roi[0]),int(roi[1])
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
    im = io.load_image()
    corrections.correct_data(im, sigma= 20, darkframe=488)
    im['std_proj'] = projection(im['corr_data'],projection='std')
    #%%
    spot_threshold = 0.05
    peaks = peak_finder(im['std_proj'],
                                   max_sigma =2,
                                   threshold_rel=spot_threshold,
                                   roi = [10,10,502,502], 
                                   min_dist = 4)
    #%%
    
    fitted = np.array(peak_fitter(im['std_proj'],peaks,5))
    #%%
    fitted_pos = fitted[:,1:3]
#%%
    v = Viewer()
    #%%
    v.add_image(im['std_proj'], name = 'std')
    #%%
    # v.add_image(im['data'], name = im['filename'])
    v.add_points(peaks,edge_color = 'yellow', face_color='transparent', size = 7, edge_width = 0.05)
    #%%
    v.add_points(fitted_pos,edge_color = 'yellow', face_color='yellow', opacity=0.5 , size = fitted[:,4], edge_width = 0.05)