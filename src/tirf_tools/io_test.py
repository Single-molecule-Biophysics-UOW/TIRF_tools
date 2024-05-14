# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:05:18 2023

@author: smueller
"""

from tkinter import filedialog
from tkinter import Tk
from dask_image import imread
import traceback
# import dask as da
import skimage
import numpy as np
from dask.diagnostics import ProgressBar
# from tirf_tools import corrections
import pims
import glob
import numbers
import warnings

import dask.array as da
# import numpy as np
# import pims
from tifffile import natural_sorted
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
    data_raw = imread(file).astype(dtype)
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

def load_pims(path, order = 'ctxy', series = None):
    with pims.open(path) as im:
        #depending on the data pims will return a <FramesSequenceND>
        # or a FramesSequence or a imageSequence.
        if isinstance(im,pims.FramesSequenceND):
            print(im)
            if 'v' in im.axes:
                im.iter_axes = 'v'    
                im.bundle_axes = ''.join(im.axes).replace('v','')
                if im.sizes['v']<2:
                    return da.array.from_array(im)
                if im.sizes['v']>1 and not series:   #multi series data
                    series=[]
                    for i in im:
                        series.append(da.array.from_array(i))
                    return series
                if im.sizes['v']>1 and series:   #multi series data
                    return da.array.from_array(im[series])
            else:
                im.bundle_axes = ''.join(im.axes)
                return da.array.from_array(im)
        else: return da.array.from_array(im)
        
def load_image_pims(path = None, compute = True, correct = False, Series=1, dtype = np.float32, meta=False,**kwargs):
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
    else:
        file = path
    if hasattr(file, '__iter__'):    
        dataset = []
        for f in file:
            data = {}
            check_file_permissions(f)
            try:
                im = load_pims(f, **kwargs)
                data['path'] = f
                data['name'] = os.path.basename(f)
                data['data'] = im
            except Exception:
                print(traceback.format_exc())
                
            dataset.append(data)
        
    else:
        data = {}
        check_file_permissions(file)
        im = load_pims(f, **kwargs)
        data['path'] = f
        data['name'] = os.path.basename(f)
        data['data'] = im
        dataset = [data]
    if correct:
        if not compute:            
            warnings.warn(""""The background correction can't be done without
                          loading the whole image in the memory""")
        for data in dataset:
            try:
                with ProgressBar():            
                    data['data'] = corrections.corr_stack_dask(data['data'], kwargs['sigma'],kwargs['darkframe']).compute()
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
        
    if len(dataset) == 1:
        return dataset[0]
    else: return dataset
    

def load_image(path = None, compute = True, correct = False, dtype = np.float32,**kwargs):
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
    print('some change')
    if not path:
        root = Tk() 
        root.withdraw()
        root.attributes("-topmost", True)
        file = filedialog.askopenfilenames(title = 'open one or more files')
        print(file)
    else:
        file = path
        
    if hasattr(file, '__iter__'):    
        data = {}
        for f in file:
            check_file_permissions(f)
            im = load_dask_image(f, dtype = dtype)
            im.path = f
            try:       
                img= pims.open(f)# as img:
                print(img)
                
                meta = img.metadata
                im.meta = meta
                
                print(meta)
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
    
def imread(fname, nframes=1, *, arraytype="numpy"):
    """
    Read image data into a Dask Array.

    Provides a simple, fast mechanism to ingest image data into a
    Dask Array.

    Parameters
    ----------
    fname : str or pathlib.Path
        A glob like string that may match one or multiple filenames.
        Where multiple filenames match, they are sorted using
        natural (as opposed to alphabetical) sort.
    nframes : int, optional
        Number of the frames to include in each chunk (default: 1).
    arraytype : str, optional
        Array type for dask chunks. Available options: "numpy", "cupy".

    Returns
    -------
    array : dask.array.Array
        A Dask Array representing the contents of all image files.
    """

    sfname = str(fname)
    if not isinstance(nframes, numbers.Integral):
        raise ValueError("`nframes` must be an integer.")
    if (nframes != -1) and not (nframes > 0):
        raise ValueError("`nframes` must be greater than zero.")

    if arraytype == "numpy":
        arrayfunc = np.asanyarray
    elif arraytype == "cupy":   # pragma: no cover
        import cupy
        arrayfunc = cupy.asanyarray

    with pims.open(sfname) as imgs:
        if 'v' in imgs.axes:
            imgs.iter_axes = 'v'
            ax_bundle = ''.join(imgs.axes).replace('v','')
            ax_bundle_sorted = ax_bundle.replace('x','').replace('y','') + 'xy'
            imgs.bundle_axes = ax_bundle_sorted
            print(''.join(imgs.axes).replace('v',''))
        shape = (len(imgs),) + imgs.frame_shape
        dtype = np.dtype(imgs.pixel_type)

    if nframes == -1:
        nframes = shape[0]

    if nframes > shape[0]:
        warnings.warn(
            "`nframes` larger than number of frames in file."
            " Will truncate to number of frames in file.",
            RuntimeWarning
        )
    elif shape[0] % nframes != 0:
        warnings.warn(
            "`nframes` does not nicely divide number of frames in file."
            " Last chunk will contain the remainder.",
            RuntimeWarning
        )

    # place source filenames into dask array after sorting
    filenames = natural_sorted(glob.glob(sfname))
    if len(filenames) > 1:
        ar = da.from_array(filenames, chunks=(nframes,))
        multiple_files = True
    else:
        ar = da.from_array(filenames * shape[0], chunks=(nframes,))
        multiple_files = False

    # read in data using encoded filenames
    a = ar.map_blocks(
        _map_read_frame,
        chunks=da.core.normalize_chunks(
            (nframes,) + shape[1:], shape),
        multiple_files=multiple_files,
        new_axis=list(range(1, len(shape))),
        arrayfunc=arrayfunc,
        meta=arrayfunc([]).astype(dtype),  # meta overwrites `dtype` argument
    )
    return a



def _map_read_frame(x, multiple_files, block_info=None, **kwargs):

    fn = x[0]  # get filename from input chunk

    if multiple_files:
        i, j = 0, 1
    else:
        i, j = block_info[None]['array-location'][0]

    return _read_frame(fn=fn, i=slice(i, j), **kwargs)


def _read_frame(fn, i, *, arrayfunc=np.asanyarray):
    with pims.open(fn) as imgs:
        if 'v' in imgs.axes:
            imgs.iter_axes = 'v'
            ax_bundle = ''.join(imgs.axes).replace('v','')
            ax_bundle_sorted = ax_bundle.replace('x','').replace('y','') + 'xy'
            imgs.bundle_axes = ax_bundle_sorted
            
        return arrayfunc(imgs[i])
    
#%%     
if __name__ == '__main__':
    file = load_image()

