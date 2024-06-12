# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:07:49 2023

@author: smueller
"""



from tirf_tools import io
from dask_image import ndfilters
from dask import array as da
import numpy as np
from dask.diagnostics import ProgressBar
from dask import delayed
from scipy import signal as sig
from scipy import ndimage as ndi

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
        corr = div_by_gauss_da(i['data'])
        with ProgressBar():
            i['corr_data'] = corr.compute()
            
@delayed
def calc_shift(frameN,frame1):
    correlation = sig.correlate(frameN,frame1, mode = 'same')
    c_max = np.unravel_index(correlation.argmax(), correlation.shape)
    shift = np.array([[256,256]])-c_max
    return shift

@delayed
def corr_shift(shift,frame):
    matrix = np.array([[1,0,-shift[0]],
                       [0,1,-shift[1]],
                       [0,0,1]])
    corr_frame = np.expand_dims(ndi.affine_transform(frame, matrix), axis=0)
    return corr_frame

def drift_correct(data, key = 'corr_data'):
    if not isinstance(data,list):
        data = [data]
    for i in data:
        frame1 = i[key][0,0,0,:,:]   #assuming 5D image and T being the first        
        shift = []
        for frame in range(i[key].shape[0]):
            shift.append(calc_shift(i[key][frame,0,0,:,:],frame1))
        print("Calculate drift")
        with ProgressBar():
            final = da.compute(*shift)
        final= np.concatenate(final)    #stack into np.array
        drift_corr = i[key][0,0,0,:,:]
        drift_corr = np.expand_dims(drift_corr,axis=0)
        
        corr_movie = []
        for frame, shift_n in zip(range(1,i[key].shape[0],1),final[1:]):
            corr_frame = corr_shift(shift_n,i[key][frame,0,0,:,:])
            corr_movie.append(corr_frame)
        print('Correct data')
        with ProgressBar():
            movie = da.compute(*corr_movie)    
        movie = np.concatenate(movie)   #stack into one array
        i['drift_corr'] = movie

#%%
    
if __name__ == '__main__':
    # path = r'Z:/In_vitro_replication/Stefan/test/N41_Q55_vars_Tramp_ProbeA647_OD2_400ms_008.nd2'
    
    file = io.load_image()

    correct_data(file, sigma= 20, darkframe=488)
    
    
    from napari import Viewer
    v = Viewer()  
    v.add_image(file['corr_data'])