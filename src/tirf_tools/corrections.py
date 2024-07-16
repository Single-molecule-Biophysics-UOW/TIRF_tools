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
from scipy import fft
import skimage
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
def calc_shift(frameN,frame1, filter_kernel):
    #correlate in frequency domain
    freq_corr =filter_kernel*fft.fft2(frameN).conj() * fft.fft2(frame1)
    ccorr =  fft.ifftshift(
                                fft.ifft2(freq_corr)
                                    ).real
    
    
    c_max = np.unravel_index(ccorr.argmax(), ccorr.shape)
    # print()
    shift = np.array([[256,256]])-c_max
    
    # print(c_max)
    return shift

@delayed
def corr_shift(shift,frame):
    matrix = np.array([[1,0,shift[0]],
                       [0,1,shift[1]],
                       [0,0,1]])
    corr_frame = ndi.affine_transform(frame, matrix)
    return corr_frame

def check_color(data):
    #if 5D we assume TCZYX
    #if 4D: TCYX
    #check how many dimensions are greater than 1:
    n_dims = sum(i>1 for i in data.shape)
    if n_dims>3:
        print('color channels detected')
    #we assume color is dim 1.
    return data.shape[1]
        

def drift_correct(data, key = 'corr_data', color_channel = 0):
    if not isinstance(data,list):
        data = [data]
    #prepare filter
    win = skimage.filters.window('hann', (512,512))
    win = fft.fftshift(win)
    for i in data:
        color = check_color(i[key])
        channel = i[key][:,color_channel,0,:,:]
        
        
        frame1_channel = channel[0,:,:]   #assuming 5D image and T being the first        
        shift = []
                
        #loop through frames and calculate drift
        for frame in range(channel.shape[0]):
            shift.append(calc_shift(channel[frame,:,:],frame1_channel,win))
        
        if color>1:
            print("Calculate drift in channel {}".format(color_channel))
        else:
            print("Calculate drift")
        with ProgressBar():
            final = da.compute(*shift)
        final= np.concatenate(final)    #stack into np.array
        #set first frame for corrected movie
        all_colors = []
        for c in range(color):
            corr_movie = [i[key][0,c,0,:,:]]
            for frame, shift_n in zip(range(1,i[key].shape[0],1),final[1:]):
                corr_frame = corr_shift(shift_n,i[key][frame,c,0,:,:])            
                corr_movie.append(corr_frame)
        
            if color>1:
                print("Apply drift in channel {}".format(c))
            else:
                print('Apply drift')
            with ProgressBar():
                movie = da.compute(*corr_movie)  
            movie = np.stack(movie)   #stack into one array
            
            while len(movie.shape)<len(data[0][key].shape):
                movie= np.expand_dims(movie,axis=-3)    #should become 5D again
            all_colors.append(movie)
        final_m = np.concatenate(all_colors, axis = 1)
        i['drift_corr'] = final_m

#%%
    
if __name__ == '__main__':
    # path = r'Z:/In_vitro_replication/Stefan/test/N41_Q55_vars_Tramp_ProbeA647_OD2_400ms_008.nd2'
    
    file = io.load_image()

    correct_data(file, sigma= 20, darkframe=488)
    
    
    from napari import Viewer
    v = Viewer()  
    v.add_image(file['corr_data'])