# -*- coding: utf-8 -*-
"""
Created on Mon May 29 14:18:51 2023

@author: smueller
"""

from tkinter import filedialog
from tkinter import Tk
# from dask_image import imread
from dask.diagnostics import ProgressBar
from dask import array as da
from napari import Viewer
from tirf_tools import io, corrections,PeakFitter
import pandas as pd
# import skimage
from dask import delayed
import numpy as np
# import scipy.optimize as opt
# import tqdm
from HMM_barcoding.image_utils import PeakFitter as pf
from HMM_barcoding.image_utils import io_all, integrator
import time
# from matplotlib import pyplot as plt
# import pandas as pd
from scipy.spatial import distance
import itertools
# import time
#%%

def make_bbox(bbox_extents):
    """Get the coordinates of the corners of a
    bounding box from the extents

    Parameters
    ----------
    bbox_extents : list (4xN)
        List of the extents of the bounding boxes for each of the N regions.
        Should be ordered: [min_row, min_column, max_row, max_column]

    Returns
    -------
    bbox_rect : np.ndarray
        The corners of the bounding box. Can be input directly into a
        napari Shapes layer.
    """
    minr = bbox_extents[0]
    minc = bbox_extents[1]
    maxr = bbox_extents[2]
    maxc = bbox_extents[3]

    bbox_rect = np.array(
        [[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]]
    )
    bbox_rect = np.moveaxis(bbox_rect, 2, 0)

    return bbox_rect

def get_coloc_peaks(p1,p2, threshold):
    #TODO delayed?
    #this creates all pairwise distances for all spots in the two dataFrames
    #so for N spots indf1 and M spots df2 we get NxM distances in a matrix
    dist_matrix = distance.cdist(p1, p2, metric='euclidean')
    #the closest spot in df2 to every spot in df1 is the row-wise minimum over the matrix:
    min_dist = np.min(dist_matrix, axis=1)
    #so for M spots in the MxN matrix we get a M-dim column vector
    #now to find the coordinates to every distance we got get the index of this element in the matrix:
    condition = np.where(np.min(dist_matrix,axis=1)<threshold)
    spot_in_peaks2 = np.argmin(dist_matrix[condition],axis=1)
    # print(spot_in_peaks2)
    # print(spot_in_peaks2)
    # new = p2[spot_in_peaks2]
    # print(new)
    return spot_in_peaks2
    
    return p2[spot_in_peaks2]


def get_coloc_peaks_pd(p1,p2, threshold, name1 = 'peaks1', name2 = 'peaks2'):
    """
    build a dataframe for the colocalised peaks, containing the coordinates of the peak1 and peaks2
    The resulting dataframe will have columns name1_x, name1_y, name2_x, name2_y
    
    """
    #TODO delayed?
    #this creates all pairwise distances for all spots in the two dataFrames
    #so for N spots indf1 and M spots df2 we get NxM distances in a matrix
    dist_matrix1 = distance.cdist(p1, p2, metric='euclidean')
    dist_matrix2 = dist_matrix1.T
    #the closest spot in df2 to every spot in df1 is the row-wise minimum over the matrix:
    #so for M spots in the MxN matrix we get a M-dim column vector
    #now to find the coordinates to every distance we got get the index of this element in the matrix:
    #The resulting coordinates will be the original peak pf p2! p1 can change by +-threshold
    condition1 = np.where(np.min(dist_matrix1,axis=1)<threshold)
    condition2 = np.where(np.min(dist_matrix2,axis=1)<threshold)
    
    spot_in_peaks2 = np.argmin(dist_matrix1[condition1],axis=1)
    spot_in_peaks1 = np.argmin(dist_matrix2[condition2],axis=1)
    
    #the actual peaks:
    coloc_peaks1 = p1[spot_in_peaks1]
    coloc_peaks2 = p2[spot_in_peaks2]
    
    # print(coloc_peaks1[0],coloc_peaks1[1])
    # print(coloc_peaks1[0:1,:])
    dist_matrix_p0 = distance.cdist(coloc_peaks1[0:1,:], coloc_peaks2, metric='euclidean')
    
    coloc_peaks2_coords = []
    for p in range(coloc_peaks1.shape[0]):
        corrd_in_peak2 = np.argmin(distance.cdist(coloc_peaks1[p:p+1,:], coloc_peaks2, metric='euclidean'))
        coloc_peaks2_coords.append(corrd_in_peak2)
        

    
    df = pd.DataFrame({name1+'_x':coloc_peaks1[:,0], name1+'_y':coloc_peaks1[:,1], 
                       name2+'_x':coloc_peaks2[coloc_peaks2_coords][:,0], name2+'_y':coloc_peaks2[coloc_peaks2_coords][:,1]})
    
    # df1 = pd.DataFrame({name1+'_x':coloc_peaks1[:,0], name1+'_y':coloc_peaks1[:,1]})
    # df2 = pd.DataFrame({name2+'_x':coloc_peaks2[:,0], name2+'_y':coloc_peaks2[:,1]})
    
    #just sort by x index and secondary sort by y value, there might be a better way but this should work
    # df1_s = df1.sort_values([name1+'_x',name1+'_y'])
    # df2_s = df2.sort_values([name2+'_x',name2+'_y'])
    #now join them:
    
    # df1_s.reset_index(inplace=True, drop=True)
    # df2_s.reset_index(inplace=True, drop=True)
    # coloc = pd.concat([df1_s,df2_s], axis = 1)
    return df
    

def find_drift(p1,p2,precision = 0.5, max_x = 5, max_y = 5, name1 = 'peaks1', name2 = 'peaks2'):
    """
    This function expects a numpy array of shape (N,2) where N is the number of peaks.
    
    Returns
    -------
    n : int
        number of colocalised peaks
    drift : tuple
        drift between peaks
    coloc : TYPE
        pd.df of colocalised peaks
    """    
    #TODO: delayed?
    n = 0    
    drift = (0,0)
    coloc = np.array([])
    for i,(x,y) in enumerate(itertools.product(np.arange(-max_x,max_x+1,precision),np.arange(-max_y,max_y+1,precision))):
        # print(i,(x,y))
        # print(x,y)
        #translate peaks2:
        drift_p2 = np.ones(p2.shape)
        drift_p2[:,0]=p2[:,0]+x
        drift_p2[:,1]=p2[:,1]+y
        #find colocalised spots:
        # print(drift_p2[0],x,y)
        #change to pandas here. should still work
        coloc_peaks = get_coloc_peaks_pd(p1,drift_p2,1,name1 = name1, name2 = name2)
        # print(drift_p2.shape)
        n_peaks = len(coloc_peaks)
        if n_peaks > n:
            n = n_peaks
            drift = (x,y)
            coloc = coloc_peaks
    #we want to return the original peak for each color => undo the drift!
    
    
    # coloc[name2+'_x'] = coloc[name2+'_x']-drift[0]
    # coloc[name2+'_y'] = coloc[name2+'_y']-drift[1]
    
    print('found {} colocalised peaks for drift {},{}'.format( n,drift[0],drift[1]))
    return n, drift, coloc
# @delayed
def find_drift_v2(p1,p2,threshold = 1,precision = 1, max_x = 7, max_y = 7, name1 = 'peaks1', name2 = 'peaks2'):
    """
    This function expects a numpy array of shape (N,2) where N is the number of peaks.
    
    Returns
    -------
    n : int
        number of colocalised peaks
    drift : tuple
        drift between peaks
    coloc : TYPE
        pd.df of colocalised peaks
    """    
    #TODO: delayed?
    n = 0    
    drift = (0,0)
    coloc = np.array([])
    for i,(x,y) in enumerate(itertools.product(np.arange(-max_x,max_x+1,precision),np.arange(-max_y,max_y+1,precision))):
        
        
        #translate peaks2:
        drift_p2 = np.ones(p2.shape)
        drift_p2[:,0]=p2[:,0]+x
        drift_p2[:,1]=p2[:,1]+y
        
        #find colocalised spots:
        # print(drift_p2[0],x,y)
        #change to pandas here. should still work
        coloc_peaks = get_coloc_peaks_pd(p1,drift_p2,threshold,name1 = name1, name2 = name2)
        
        n_peaks = len(coloc_peaks)
        if n_peaks > n:
            n = n_peaks
            drift = (x,y)
            coloc = coloc_peaks
    #we want to return the original peak for each color => undo the drift!
    
    
    coloc[name2+'_x'] = coloc[name2+'_x']-drift[0]
    coloc[name2+'_y'] = coloc[name2+'_y']-drift[1]
    
    print('found {} colocalised peaks for drift {},{}'.format( n,drift[0],drift[1]))
    return n, drift, coloc

def coloc_n_peaks(peaks, names = None, precision = 1, threshold = 1):
    if names is None:
        names = list(map(str,range(len(peaks))))
    #colocalise the first set of peaks with all others:
    coloc_with_0 = []
    for i,name in zip(peaks[1:],names[1:]):
        # print(peaks[0])
        # print(i)
        coloc_with_0.append(find_drift_v2(peaks[0],i,
                      name1 = names[0],
                      name2 = name, 
                      precision = precision, 
                      threshold = threshold)[2])
    return coloc_with_0


#%%    
if __name__ == "__main__":
    

    im = io.load_image()
    corrections.correct_data(im, sigma= 20, darkframe=488)
    #%%
    im[0]['std_proj'] = PeakFitter.projection(im[0]['corr_data'],projection='std')
    im[1]['std_proj'] = PeakFitter.projection(im[1]['corr_data'],projection='std')
    im[2]['std_proj'] = PeakFitter.projection(im[2]['corr_data'],projection='std')
    #%%
    spot_threshold = 0.05
    im[0]['c1'] = PeakFitter.peak_finder(im[0]['std_proj'],
                                   max_sigma =2,
                                   threshold_rel=spot_threshold,
                                   roi = [10,10,502,502], 
                                   min_dist = 4)
    im[1]['c2'] = PeakFitter.peak_finder(im[1]['std_proj'],
                                   max_sigma =2,
                                   threshold_rel=spot_threshold,
                                   roi = [10,10,502,502], 
                                   min_dist = 4)
    im[2]['c3'] = PeakFitter.peak_finder(im[2]['std_proj'],
                                   max_sigma =2,
                                   threshold_rel=spot_threshold,
                                   roi = [10,10,502,502], 
                                   min_dist = 4)
    #%%
    coloc_list = coloc_n_peaks([im[0]['c1'],im[1]['c2'],im[2]['c3']],threshold = 3)
    import functools as ft
    df_final = ft.reduce(lambda left, right: pd.merge(left, right,on=['0_x','0_y']), coloc_list)    
    
    
    
    #%%

    
    #%%
    v = Viewer()
    
    v.add_image(im[0]['std_proj'], name = im[0]['filename'])
    v.add_image(im[1]['std_proj'], name = im[1]['filename'])
    v.add_image(im[2]['std_proj'], name = im[2]['filename'])
    #%%
    # v.add_image(im['data'], name = im['filename'])
    v.add_points(im[0]['c1'],edge_color = 'yellow', face_color='transparent', size = 7, edge_width = 0.05, name ='c1')
    v.add_points(im[1]['c2'],edge_color = 'green', face_color='transparent', size = 7, edge_width = 0.05, name ='c2')
    v.add_points(im[2]['c3'],edge_color = 'green', face_color='transparent', size = 7, edge_width = 0.05, name ='c3')
    #%%
    v.add_points(df_final[['0_x','0_y']],edge_color = 'magenta', face_color='transparent', size = 7, edge_width = 0.05)
    #%%
    