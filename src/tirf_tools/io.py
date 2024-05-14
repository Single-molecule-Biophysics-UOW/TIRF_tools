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
from tirf_tools import corrections
import pims
import glob
import numbers
import warnings
import aicsimageio
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
    data_raw = imread.imread(file).astype(dtype)
    return data_raw


def check_file_permissions(file_path):
   if (os.access(file_path, os.R_OK)) and (os.access(file_path, os.W_OK)) and (os.access(file_path, os.X_OK)):
      # print(f"Read write and execute permissions granted for file: {file_path}")
      pass
   else:
      print(f"limited permissions for file: {file_path} \n This might be a problem")





    
def _unpack_tuple(x):
    """ Unpacks one-element tuples for use as return values """
    if len(x) == 1:
        return x[0]
    else:
        return x
def load_image(path=None, dtype = np.float32):
    if not path:
        root = Tk() 
        root.withdraw()
        root.attributes("-topmost", True)
        file = filedialog.askopenfilenames(title = 'open one or more files')        
    else:
        file = path
    
    if not hasattr(file, '__iter__'):  
        file = [file]
    data = []
    for f in file:
        print(f)
        basename = os.path.basename(f)
        folder = os.path.dirname(f)
        check_file_permissions(f)
        im = aicsimageio.AICSImage(f)
        #loop through series for multi series data
        for scene in im.scenes:
            im.set_scene(scene)
            dask_ar = im.dask_data.astype(dtype)  
            meta = im.metadata
            dims_order = im.dims.order
            data_dict = {'data':dask_ar,
                         'meta':meta,
                         'filename':basename,
                         'folder':folder,
                         'dimension order': dims_order}
            data.append(data_dict)
    return _unpack_tuple(data)

#%%     
if __name__ == '__main__':
    # path = r"Z:\In_vitro_replication\Stefan\test/N41_Q55_vars_Tramp_ProbeA647_OD2_400ms_008.nd2"
    im = load_image()    
    # im = aicsimageio.AICSImage(path)
    # im = pims.open(path)
    
    

