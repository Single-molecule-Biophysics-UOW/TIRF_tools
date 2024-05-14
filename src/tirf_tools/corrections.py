# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:07:49 2023

@author: smueller
"""


from dask_image import imread
from tirf_tools import io
from dask_image import ndfilters
from scipy.ndimage.filters import gaussian_filter
from dask import array as da
import os
import napari
import skimage
import numpy as np
from dask import delayed
from dask.diagnostics import ProgressBar

#%%
# @delayed
def div_by_gauss_da(data,sigma=20,offset=0):
    """
    Let's assume dim order: TCZYX. We only blur in x and y.
    
    """
    sigmas = da.zeros(np.array(data.shape).shape)
    sigmas[-2] = sigma
    sigmas[-1] = sigma
    input_dtype = data.dtype
    #subtract electronic offset
    data = data - offset
    #gaussian blur/normalize to 0,1    
    
    #somehow when computing delayed one of the arrays becomes numpy. da.asarray fixes this
    blur = ndfilters.gaussian(da.asarray(data), sigmas) 
    blur = blur/da.max(blur).astype(input_dtype)
    #calculate result (will be 32 bit?)
    res = data / blur
    #normalisation should be to global max
    # res = ((res/da.max(res).astype(input_dtype))*65536).astype(input_dtype)
    #set dtype back to input dtype
    res = da.nan_to_num(res).astype(input_dtype)
    # min_value = np.iinfo(data.dtype).min
    max_value = np.finfo(input_dtype).max
    res.clip(0,max_value)
    return res

def correct_data(data, sigma, darkframe):
    
    if not isinstance(data,list):
        data = [data]
    for i in data:
        with ProgressBar():
            i['corr_data'] = div_by_gauss_da(i['data']).compute()

    
#%%     
if __name__ == '__main__':
    # path = r'Z:/In_vitro_replication/Stefan/test/N41_Q55_vars_Tramp_ProbeA647_OD2_400ms_008.nd2'
    
    file = io.load_image()
    #%%
    
    correct_data(file, sigma= 20, darkframe=488)
    
    
    
    #%%
    from napari import Viewer
    v = Viewer()
    
    
    
    v.add_image(file['corr_data'])
    # v.add_image(frame)