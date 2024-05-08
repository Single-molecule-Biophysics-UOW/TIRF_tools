# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:07:49 2023

@author: smueller
"""


from dask_image import imread
from dask_image import ndfilters
from dask import array as da
import os
import napari
import skimage
import numpy as np
from dask.diagnostics import ProgressBar



def div_by_gauss_da(data,sigma_x=20,sigma_y=20,sigma_z=0,offset=0):
    input_dtype = data.dtype
    #subtract electronic offset
    data = data - offset
    #gaussian blur/normalize to 0,1    
    blur = ndfilters.gaussian(data, (sigma_x,sigma_y))
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

def corr_stack_dask(stack, sigma, offset):
    
    lazy_arrays = [div_by_gauss_da(stack[n], 
                                sigma_x=sigma,
                                sigma_y=sigma,
                                sigma_z=0, 
                                offset=offset) for n in range(0, stack.shape[0])]
    
    dask_stack = da.stack(lazy_arrays, axis=0)
    return dask_stack.astype(np.float32)


# def corr_batch(folder, sigma, offset):
#     files = [x for x in os.listdir(folder) if (x.endswith('nd2')) and os.path.isdir(folder+x)== False]
#     for i in files:
#         print('correcting {}'.format(i))
#         data = imread.imread(folder+i).astype(np.float16)
#         with ProgressBar():
#             corr = corr_stack_dask(data, sigma, offset).compute()
#         save_tiff(folder+'corr_'+i+'.tiff',corr)
#             # da.to_zarr(corr,folder+'corr_'+i, compute = True)

# def save_tiff(path, data,**kwargs):
#     skimage.io.imsave(path, data, **kwargs)


    
    
    