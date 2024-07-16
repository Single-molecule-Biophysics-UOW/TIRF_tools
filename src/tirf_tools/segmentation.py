import ruptures as rpt
import ruptures as rpt
from tqdm import tqdm
import pandas as pd
from dask import dataframe as dd
import numpy as np
from scipy.optimize import curve_fit

tqdm.pandas()
def find_changePoints_traj(traj,segmentator=None,sigma = 1, costf = 'rbf'):
    inten = np.array(traj)
    segmentator = rpt.Binseg(model=costf)
    algo = segmentator.fit(inten)
    my_bkps = algo.predict(pen=sigma)
    return my_bkps

def find_ChangePoints(data, costf = 'rbf', sigma = 1):
    # 
    #create segmentation instance
    # segmentators = pd.Series([rpt.Binseg(model=costf) for x in range(len(data))])
    # print(data)
    # data = dd.from_pandas(data)
    # data['segmentators'] = segmentators
    cp_df = data.groupby('trajectory')['Intensity'].progress_apply(find_changePoints_traj,
                                                                   sigma = sigma,
                                                                   costf = costf
                                                                   )
    return cp_df




class SegmentBuilder():
    def lin(x,m,t):
        return m*x+t
    def piecew_c(segment):
        #piecewise constant signal
        # return np.mean(segment)
        return np.ones(segment.shape)*np.mean(segment)
    def piecew_l(segment):
        #piecewise linear signal
        x = np.array(range(len(segment)))
        popt,pcov = curve_fit(SegmentBuilder.lin,x,segment)        
        y_fit = SegmentBuilder.lin(x,*popt)
        return y_fit
        
def construct_traj(data,cp=None, method = 'piecew_c'):
    traj_number = data.name
    changePoints = cp[traj_number]
    segments = []
    data = np.array(data)
    c0 = 0
    total = 0
    for i in changePoints:
        # print(c0,i)
        segment_values = data[c0:int(i)]
        func = getattr(SegmentBuilder,method)
        seg = func(segment_values)
        segments.append(seg)
        total+=len(seg)
        # print(total)
        c0=int(i)
        np_seg = np.concatenate(segments)
    return np_seg

def contruct_segmented(data,cp,method = 'piecew_c'):
    data['seg'] = data.groupby('trajectory')['Intensity'].progress_transform(construct_traj,cp=cp, method = method)