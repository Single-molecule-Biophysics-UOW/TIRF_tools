# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:05:18 2023

@author: smueller
"""

from tkinter import filedialog
from tkinter import Tk
from dask_image import imread
import skimage
import numpy as np
from dask.diagnostics import ProgressBar
from collections.abc import Iterable
from HMM_barcoding.image_utils import corrections
import pims
import dask as da
import os
import warnings
import nd2reader      #somehow this line convinces pims to use nd2reader instead of bioformats!
        #%%
def save_tiff(data, path = None,**kwargs):
    if not path:
        root = Tk() 
        root.withdraw()
        root.attributes("-topmost", True)
        file = filedialog.askopenfilename(title = 'choose trajetory file')
    else:
        file = path
    if not file.endswith('.tif'):
        file = path+'.tif'
    print(file)
    skimage.io.imsave(file, data, **kwargs)

def load_dask_image(file, dtype = np.float32):
    data_raw = imread.imread(file).astype(dtype)
    return data_raw

def load_raw(path = None, compute = True, sigma = 50, darkframe = 433, correct = True):
    """
    Convenience method to read single nd2 file

    Returns
    -------
    data

    """
    #load data
    data = load_image(path = path, sigma = sigma, darkframe = darkframe)
    # path = data[0].path
    
    if compute and correct:
        # with ProgressBar():
            data = corrections.corr_stack_dask(data[0],sigma,darkframe).compute()
    if compute and not correct:
        with ProgressBar():
            data = data[0].compute()
    if not compute and correct:
        raise RuntimeError
    if not compute and not correct:
        data = data[0]
    # else:
    #     data = corrections.corr_stack_dask(data[0],sigma,darkframe)
    return data#,path

def check_file_permissions(file_path):
   if (os.access(file_path, os.R_OK)) and (os.access(file_path, os.W_OK)) and (os.access(file_path, os.X_OK)):
      # print(f"Read write and execute permissions granted for file: {file_path}")
      pass
   else:
      print(f"limited permissions for file: {file_path} \n This might be a problem")



def load_image(path = None, compute = True, correct = True, dtype = np.float32,**kwargs):
    """
    Loads one or more n-D images.
    
    Parameters
    ----------
    path : string or iterable, optional
        Path to the image/ or iterable collection of paths
        if None TKinter will be used to prompt a dialog. pyQT should be used
        in the future
    compute : bool
        if True a numpy array will be returned. Else a dask array will be returned
        which can be loaded lazily and computations can be parallelised.
    correct : bool
        if true a correct will be applied to even out non-uniform illumination
        and subtract an electronic offset. Note if correct is True compute is
        set to True as well.
    dtype : numpy.dtype object
        bitdepth of the loaded image. Defaults to float32.
    **kwargs are passed to corrections.corr_stack_dask if correct is True

    Returns
    -------
    dask array

    """
    if not path:
        root = Tk() 
        root.withdraw()
        root.attributes("-topmost", True)
        file = filedialog.askopenfilenames(title = 'open one or more files')
        print(file)
    else:
        file = path
        
    if isinstance(file, Iterable):       
        data = {}
        for f in file:
            check_file_permissions(f)
            im = load_dask_image(f, dtype = dtype)
            im.path = f
            try:       
                with pims.open(f) as img:
                    meta = img.metadata
                im.meta = meta
            except:
                print('metadata could not be extracted')
                meta = None
                pass
                
            data[f] = im
    else:
        check_file_permissions(file)
        im = load_dask_image(file, **kwargs)
        im.path = file
        data[file] = im
    if correct:
        if not compute:            
            warnings.warn(""""The background correction can't be done without
                          loading the whole image in the memory""")
        corr = {}
        for key in data:
            try:
                with ProgressBar():            
                    data_corr = corrections.corr_stack_dask(data[key], kwargs['sigma'],kwargs['darkframe']).compute()
                    corr[key] = data_corr
            except KeyError as e:
                missing = []
                for key in ['sigma','darkframe']:
                    if key not in kwargs:                      
                        missing.append('Parameter {} is missing for non-uniform illumination correction !\n'.format(key))
                        # print('Parameter {} is missing for non-uniform illumination correction !'.format(key))
                # print("""Make sure to pass sigma=value1 and darkframe=value2
                      # Sigma corresponds to the blur and darkframe to the electronic pixel offset""")                      
                raise KeyError(''.join(missing)+"""Make sure to pass sigma=value1 and darkframe=value2
                                           Sigma corresponds to the blur and 
                                           darkframe to the electronic pixel offset""")
        return corr
    else:
        return data
    
def load_image_nogui(file, compute = True, correct = True, dtype = np.float32,**kwargs):
    """
    Loads one or more n-D images.
    
    Parameters
    ----------
    path : string or iterable, optional
        Path to the image/ or iterable collection of paths
        if None TKinter will be used to prompt a dialog. pyQT should be used
        in the future
    compute : bool
        if True a numpy array will be returned. Else a dask array will be returned
        which can be loaded lazily and computations can be parallelised.
    correct : bool
        if true a correct will be applied to even out non-uniform illumination
        and subtract an electronic offset. Note if correct is True compute is
        set to True as well.
    dtype : numpy.dtype object
        bitdepth of the loaded image. Defaults to float32.
    **kwargs are passed to corrections.corr_stack_dask if correct is True

    Returns
    -------
    dask array

    """
    # if not path:
    #     root = Tk() 
    #     root.withdraw()
    #     root.attributes("-topmost", True)
    #     file = filedialog.askopenfilenames(title = 'open one or more files')
    # else:
    #     file = path
    if isinstance(file, Iterable):
        
        data = {}
        for f in file:
            check_file_permissions(f)
            im = load_dask_image(f, dtype = dtype)
            im.path = f
            try:       
                with pims.open(f) as img:
                    meta = img.metadata
                im.meta = meta
            except:
                print('metadata could not be extracted')
                meta = None
                pass
                
            data[f] = im
    else:
        check_file_permissions(file)
        im = load_dask_image(file, **kwargs)
        im.path = file
        data[file] = im
    if correct:
        if not compute:            
            warnings.warn(""""The background correction can't be done without
                          loading the whole image in the memory""")
        corr = {}
        for key in data:
            try:
                # with ProgressBar():            
                data_corr = corrections.corr_stack_dask(data[key], kwargs['sigma'],kwargs['darkframe'])#.compute()
                corr[key] = data_corr
            except KeyError as e:
                missing = []
                for key in ['sigma','darkframe']:
                    if key not in kwargs:                      
                        missing.append('Parameter {} is missing for non-uniform illumination correction !\n'.format(key))
                        # print('Parameter {} is missing for non-uniform illumination correction !'.format(key))
                # print("""Make sure to pass sigma=value1 and darkframe=value2
                      # Sigma corresponds to the blur and darkframe to the electronic pixel offset""")                      
                raise KeyError(''.join(missing)+"""Make sure to pass sigma=value1 and darkframe=value2
                                           Sigma corresponds to the blur and 
                                           darkframe to the electronic pixel offset""")
        return corr
    else:
        return data    

    
#%%     
if __name__ == '__main__':
    file = load_image()
#%%
    pims_data = pims.open(file[0].path)
    #%%
    
    meta = da.array.utils.meta_from_array(file[0].blocks[0])
    
    #%%
    data  = load_image(dtype = np.int16)
    #%%
    with ProgressBar():
        data = data[0].compute()
    
    #%%
    save_tiff(data)





