# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:01:47 2023

@author: smueller
"""


from dask_image import imread

from dask import array as da
import dask as dask

from dask import delayed
import napari

import numpy as np

from HMM_barcoding.image_utils import PeakFitter as pf
import time
import pandas as pd
from dask.diagnostics import ProgressBar
from concurrent.futures import ThreadPoolExecutor
import dask.config as cfg
# from dask_image.ndfilters import gaussian as gaussian
#%%
pool = ThreadPoolExecutor()
cfg.set(pool=pool)
#%%

def stack_trajs(x,y):
    return da.hstack([x,y])

# @delayed
def make_selection(data1,x,y, max_x):
    selection = data1[:,x-max_x:x+max_x+1,y-max_x:y+max_x+1]
    return selection

@delayed
def integrate(bg_selection,i_r, bg_r,inner_area ,outer_area, peak, i):
    #make selection from bg_selection:
    middle = i_r+bg_r
    selection = bg_selection[:,middle-i_r:middle+i_r+1,middle-i_r:middle+i_r+1]
    #sum the inner area
    traj = np.sum(selection,axis=(1,2))
    #sum the bg area, then subtract inner area and divide by outer_area
    bg_total = np.sum(bg_selection,axis=(1,2))
    bg = (np.subtract(bg_total,traj)/outer_area)
    #finally compute the corrected intensity and make an array with trajectory number
    #intensity x,y, etc.
    traj_corr = np.subtract(traj,bg*inner_area)
    traj_bg = np.vstack([np.ones(bg.shape)*i,traj_corr,bg, np.ones(bg.shape)*peak[0],np.ones(bg.shape)*peak[1],np.arange(0,bg.shape[0])])
    return traj_bg

def integrate_trajectories(data, peaks,inner_radius, outer_radius):
    #the integrator expects (1,512,512)
    if len(data.shape) <3:
        data = data.reshape(1,512,512)  #this is bad practise
    s = time.time()    
    #pixel area calculations
    inner_area = (inner_radius+1+inner_radius)**2
    outer_area = (inner_radius+1+inner_radius+2*outer_radius)**2 - inner_area
    #delay data for parallel processing
    ddata = delayed(data)
    all_bg_rois = []
    for i,[x,y] in enumerate(peaks):
        # selection = make_selection(ddata,int(x), int(y), max_x)
        #only make bg roi, since the inner roi is included
        bg_selection = make_selection(ddata,int(x), int(y), inner_radius+outer_radius)
        # all_rois.append(selection)
        all_bg_rois.append(bg_selection)
    #compute the roi selections
    print('collect peak rois \n')
    with ProgressBar():
        all_bg_rois = dask.compute(*all_bg_rois)
    #now loop through peaks again and integrate lazily   
    result = []
    for j,(bg,peak) in enumerate(zip(all_bg_rois,peaks)):
        result.append(integrate(bg,inner_radius,outer_radius, inner_area, outer_area, peak,j))
    #compute integrations
    print('integrate rois \n')
    with ProgressBar():
        all_traj = dask.compute(*result)
        
    #some shuffling to create a nice dataframe
    array = np.concatenate(all_traj,axis=1)
    df = pd.DataFrame(array.T,columns=['trajectory','Intensity','bg','x','y', 'slice'])
    df = df.astype({"trajectory": "int16", "Intensity": "float32", "bg": "float16", 
                                    "x": "int64", "y": "int64", "slice":"int16"},
                                   copy=False)
    print('integrated in {}s'.format(time.time()-s))
    return df

def integrate_trajectories_with_drift(data, peaks,inner_radius, outer_radius,drift):
    s = time.time()    
    #pixel area calculations
    inner_area = (inner_radius+1+inner_radius)**2
    outer_area = (inner_radius+1+inner_radius+2*outer_radius)**2 - inner_area
    #delay data for parallel processing
    ddata = delayed(data)
    all_bg_rois = []
    for i,[x,y] in enumerate(peaks):
        # selection = make_selection(ddata,int(x), int(y), max_x)
        #only make bg roi, since the inner roi is included
        bg_selection = make_selection(ddata,int(x)+drift[1], int(y)+drift[0], inner_radius+outer_radius)
        # all_rois.append(selection)
        all_bg_rois.append(bg_selection)
    #compute the roi selections
    print('collect peak rois \n')
    with ProgressBar():
        all_bg_rois = dask.compute(*all_bg_rois)
    #now loop through peaks again and integrate lazily   
    result = []
    for j,(bg,peak) in enumerate(zip(all_bg_rois,peaks)):
        result.append(integrate(bg,inner_radius,outer_radius, inner_area, outer_area, peak,j))
    #compute integrations
    print('integrate rois \n')
    with ProgressBar():
        all_traj = dask.compute(*result)
        
    #some shuffling to create a nice dataframe
    array = np.concatenate(all_traj,axis=1)
    df = pd.DataFrame(array.T,columns=['trajectory','Intensity','bg','x','y', 'slice'])
    df = df.astype({"trajectory": "int16", "Intensity": "float32", "bg": "float16", 
                                    "x": "int64", "y": "int64", "slice":"int16"},
                                   copy=False)
    print('integrated in {}s'.format(time.time()-s))
    return df



#%%
if __name__ == "__main__":
    
    #load 2 datasets and make z-projections
    # s = time.time()
    # data = io.load_ome_zarr(path='load')
    
    data = imread.imread(r'Z:\Barcoding_subgroup\Current_experiments\230703_1\corr/corrected_ClpS_B6_60nMLAR6g_OD15_003.tif')
    #%%
    
    data = data.compute()
    #%%
    data = np.nan_to_num(data, posinf=0.0)
    
    #%%
    v = napari.Viewer()
    v.add_image(data)
    #%%
    with ProgressBar():        
        proj = da.mean(data,axis=0).compute()
    # proj = np.mean(data,axis=0)
        
    #%%
    v.add_image(proj)
    #%%
    s = time.time()
    with ProgressBar():        
        #kwords: max_sigma=20, threshold_rel=0.05, overlap = 1.
        peaks = pf.peak_finder(proj, max_sigma =60,threshold_rel=0.02, overlap = 1.,roi = [10,10,502,502], min_dist = 5)
    print('found peaks in {} seconds'.format(time.time()-s))
    
    
    v.add_points(peaks, opacity = 0.5, face_color = 'transparent',edge_color = 'red')
    
    #%%
    with ProgressBar(): 
        data = data.compute()
    #%%
    df = integrate_trajectories(data,peaks,2, 3)

    
    #%%
    io.save_data(df,'corrected_ClpS_B6_60nMLAR6g_OD15_003.h5',path='load')
