import ruptures as rpt
import ruptures as rpt
from tqdm import tqdm
import pandas as pd
from dask import dataframe as dd
import numpy as np

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


def construct_traj(data,cp,steps = True):
    traj_number = data.name
    changePoints = cp[traj_number]
    segments = []
    data = np.array(data)
    c0 = 0
    total = 0
    for i in changePoints:
        print(c0,i)
        intens = np.mean(data[c0:int(i)])
        seg = np.ones(data[c0:int(i)].shape)*intens
        segments.append(seg)
        total+=len(seg)
        print(total)
        c0=int(i)
        np_seg = np.concatenate(segments)
    print('next')
    # print(len(changePoints),len(segments))
    return np_seg

def contruct_segmented(data,cp,steps=True):
    data['seg'] = data.groupby('trajectory')['Intensity'].transform(construct_traj,cp)
    # return M