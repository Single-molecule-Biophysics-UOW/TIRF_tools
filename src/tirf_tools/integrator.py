# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:01:47 2023

@author: smueller
"""


from dask_image import imread

from dask import array as da
import dask as dask

from dask import delayed


import numpy as np
from tirf_tools import io, corrections,PeakFitter
import warnings
import time
import pandas as pd
from dask.diagnostics import ProgressBar
from concurrent.futures import ThreadPoolExecutor
import dask.config as cfg
# from dask_image.ndfilters import gaussian as gaussian
# #%%
#%%
pool = ThreadPoolExecutor()
cfg.set(pool=pool)



def chooseTXY(data, dim_order = 'TCZYX'):
    """
    convenience method to select TXY 3d array from an N-d image
    """
    #select all dimensions with shape >1:
    data = np.squeeze(data)
    if len(data.shape)>3:
        warnings.warn("""Only 3D arrays (TYX) are currently supported. 
If you tried to integrate multi-color or multi-series data, try splitting them first""")
        num_dims = len(data.shape)
        index_tuple = []
        for i in range(num_dims):
            if i in [0,num_dims-2,num_dims-1]:
                index_tuple.append(slice(None,None,None))
            else:
                index_tuple.append(0)
        index_tuple = tuple(index_tuple)
        data = data[index_tuple]
    elif len(data.shape)==2:
        data = np.expand_dims(data,axis=0)
    # Xindex,Yindex, Tindex = dim_order.index('X'), dim_order.index('Y'), dim_order.index('T')
    # Xsize, Ysize, Tsize = data.shape[Xindex],data.shape[Yindex],data.shape[Tindex]
    return data#.reshape(Tsize,Ysize,Xsize)

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
    #the integrator expects (T,X,Y)
    #multi color is not directly supported yet
    data = chooseTXY(data)
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
    #the integrator expects (T,X,Y)
    #multi color is not directly supported yet
    data = chooseTXY(data)
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
    
    
    data = io.load_image()
    #%%
    
    corrections.correct_data(data, sigma= 20, darkframe=488)
    
    
    #%%
    data['std_proj'] = PeakFitter.projection(data['corr_data'],projection='std')
    data['peaks'] = PeakFitter.peak_finder(data['std_proj'], 
                                           max_sigma =2,
                                           threshold_rel=0.02, 
                                           roi = [10,10,502,502],
                                           min_dist = 5)
    #%%
    v = napari.Viewer()
    #%%
    # v.add_points(data['peaks'], face_color='transparent', edge_color='yellow', symbol='square', size = 4)
    v.add_image(f)
    
    #%%
    test = chooseTXY(data['corr_data'])
    #%%
    
    data['integration'] = integrate_trajectories(f,data['peaks'],2, 3)

    

