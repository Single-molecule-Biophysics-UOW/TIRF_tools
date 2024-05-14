# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:28:52 2024

@author: smueller
"""

import io_test as io
from matplotlib import pyplot as plt
import pims
import dask as da
import numpy as np
from dask_image import imread
#%%


def _map_read_frame(x, multiple_files, block_info=None, **kwargs):

    fn = x[0]  # get filename from input chunk

    if multiple_files:
        i, j = 0, 1
    else:
        i, j = block_info[None]['array-location'][0]

    return _read_frame(fn=fn, i=slice(i, j), **kwargs)


def _read_frame(fn, i, *, arrayfunc=np.asanyarray):
    with pims.open(fn) as imgs:
        return arrayfunc(imgs[i])



#%%

from dask_image import imread
import pims
path = r'Z:\In_vitro_replication\Stefan\test\test/multi_series_multiC_001.nd2'
im_dask = imread.imread(path)
im_pims = pims.open(path)

print('dask:',im_dask.shape)
print('pims:',im_pims)



#%%
# test_data/multi_series_multiC_002.nd2
path1 = r'Z:\Barcoding_subgroup\Current_experiments\240509/2colourtest.nd2'
im = imread.imread(path1)
print(im.shape)
with pims.open(path1) as im:
    print(im.shape)
    print(im)
    meta = im.metadata


#%%
# path2 = r'Z:\Barcoding_subgroup\Current_experiments\240502_3/Enzymes_no65_ProbeA647_OD15_200ms_007.nd2'

with pims.open(path) as imgs:    
    shape = (len(imgs),) + imgs.frame_shape
    print(shape)
    #change iteration axis
    imgs.iter_axes = 'v'
    ax_bundle = ''.join(imgs.axes).replace('v','')
    ax_bundle_sorted = ax_bundle.replace('x','').replace('y','') + 'xy'
    imgs.bundle_axes = ax_bundle_sorted
    #get the hsape gain
    shape = (len(imgs),) + imgs.frame_shape
    print(shape)
    
    
#%%


    
#%%

im = io.load_image(compute=False)

#%%
from napari import Viewer
#%%
v = Viewer()
#%%
v.add_image(im['Z:/In_vitro_replication/Stefan/test/test/multi_series_multiC_002.nd2'])


#%%

da_im = imread.imread(path1)
da_im2 = da.array.image.imread(path1)
da.array.image
    # plt.imshow(im[0])
#%%

    
    
    
#%%

image = io.load_image_pims(compute=False, correct=False)
#%%
image = io.load_image(compute=False, correct=False)
#%%

from napari import Viewer
#%%
v = Viewer()
#%%
v.add_image(im)



#%%
import dask as da

image.iter_axes = 'v'
image.bundle_axes = 'cxyt'

series = []
for i in image:
    series.append(da.array.from_array(i))

#%%
plt.figure()
plt.imshow(image['Z:/Barcoding_subgroup/Current_experiments/ClpS_A7_B8D8_probeB_OD13_HGscope_001.nd2'][:,50,50])
plt.show()

#%%
import os
n = os.path.basename('C:/Users/smueller/OneDrive - University of Wollongong/Documents/GitHub/TIRF_tools/src/tirf_tools/untitled3.py')

