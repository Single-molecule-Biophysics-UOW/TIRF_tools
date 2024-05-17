# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:39:15 2022

@author: StefanMueller
"""


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from hmmlearn import hmm    
from scipy.signal import savgol_filter

from scipy.stats import norm
from scipy.optimize import curve_fit
from tqdm import tqdm
# from sklearn_genetic.callbacks import ProgressBar
from hmmlearn.base import ConvergenceMonitor

def filter_garbage(df,traj_column = 'trajectory', intensity_column = 'Intensity', threshold=2.,lower_threshold = 0, plot = False):
    """
    Removes trajectories with maximum intensity higher than threshold*average 
    maximum Intensity over the whole dataset and exclude trajectories with a negative mean value.

    Parameters
    ----------
    df : pd.DataFrame
        contains your data. Needs to have a column specifying the trajectory
        number and one for the Intensity
    traj_column : Key for trajectory column
        pandas column name of trajectory column. The default is 'trajectory'.
    intensity_column : str, Key for intensity column
        pandas column name of intensity column. The default is 'Intensity'.
    threshold : float
        the threshold for filtering in units of standard deviations. The default is 2.
    lower_threshold : float
        sort out trajectories if there are intensity values below lower_threshold.
        The default is -10000
    plot: boolean
        if true a plot of the maximum intensity is created and returned to analyse
        the results of filtering.

    Returns
    -------
    filtered DataFrame
    """
    #first group the data:
    df['Intensity'] = df[intensity_column]
    grouped = df.groupby(traj_column)
    #get all maximum intensities:
    max_dist = grouped[intensity_column].max()
    #determine the cutoff:
    cutoff = (threshold*max_dist.std())+max_dist.mean()
    print(cutoff)
    #filter the dataframe:
    upper_filtered = grouped.filter(lambda g: (g.Intensity < cutoff).all())
    lower_filtered = upper_filtered.groupby(traj_column).filter(lambda g: (g.Intensity.mean() > lower_threshold))
    
    if plot:
        #if plot is true we return a plot comparing filtered vs unfiltered data
        fig,ax = plt.subplots(1,2)
        ax[0].set_title('before')
        # print('bins',int(np.sqrt(len(max_dist))))
        n_bins =  max([int(np.sqrt(len(max_dist))),1])
        ax[0].hist(max_dist,edgecolor='black',bins =n_bins,label='n={}'.format(len(max_dist)))
        max_dist_filtered = lower_filtered.groupby(traj_column)[intensity_column].max()
        ax[1].set_title('after')
        n_bins_filtered =  max([int(np.sqrt(len(max_dist_filtered))),1])
        ax[1].hist(max_dist_filtered,edgecolor='black',bins = n_bins_filtered,label='n={}'.format(len(max_dist_filtered)))        
        ax[1].legend()
        ax[0].legend()
        return lower_filtered, fig
    return lower_filtered



def filter_std(df,traj_column = 'trajectory', intensity_column = 'Intensity', threshold=2., plot = False):
    """
    TODO
    """
    #first group the data:
    df['Intensity'] = df[intensity_column]
    
    grouped = df.groupby(traj_column)
    #get all std intensities:
    std_dist = grouped[intensity_column].std()
    
    #determine the cutoff:
    # cutoff = threshold#(threshold*max_dist.std())+max_dist.mean()
    #filter the dataframe:
    # upper_filtered = grouped.filter(lambda g: (g.Intensity < cutoff).all())
    filtered = df.groupby(traj_column).filter(lambda g: (g.Intensity.std() > threshold))
    
    if plot:
        #if plot is true we return a plot comparing filtered vs unfiltered data
        fig,ax = plt.subplots(1,3)
        ax[0].set_title('before')
        # print('bins',int(np.sqrt(len(max_dist))))
        n_bins =  max([int(np.sqrt(len(std_dist))),1])
        ax[0].hist(std_dist,edgecolor='black',bins =n_bins,label='n={}'.format(len(std_dist)))
        std_dist_filtered = filtered.groupby(traj_column)[intensity_column].std()
        ax[1].set_title('after')
        n_bins_filtered =  max([int(np.sqrt(len(std_dist_filtered))),1])
        ax[1].hist(std_dist_filtered,edgecolor='black',bins = n_bins_filtered,label='n={}'.format(len(std_dist_filtered)))
        # ax[2].hist(std_dist,edgecolor='black',bins = n_bins_filtered,label='n={}'.format(len(std_dist)))
        ax[1].legend()
        ax[0].legend()
        # return lower_filtered, fig
        plt.show()
    return filtered




def filter_negative(df,traj_column = 'trajectory', intensity_column = 'Intensity', plot = False):
    """
    If the local background correction goes wrong for some reason, for example drift out of the ROI
    the Intensity will go fully negative. These trajectories should be filtered out.
    This should be build into filter_garbage.
    Sort out trajectories with mean<0

    Parameters
    ----------
    df : pd.DataFrame
        contains your data. Needs to have a column specifying the trajectory
        number and one for the Intensity
    traj_column : Key for trajectory column
        pandas column name of trajectory column. The default is 'trajectory'.
    intensity_column : str, Key for intensity column
        pandas column name of intensity column. The default is 'Intensity'.
    threshold : float
        the threshold for filtering in units of standard deviations. The default is 2.
    lower_threshold : float
        sort out trajectories if there are intensity values below lower_threshold.
        The default is -10000
    plot: boolean
        if true a plot of the maximum intensity is created and returned to analyse
        the results of filtering.

    Returns
    -------
    filtered DataFrame
    """
    #first group the data:
    df['Intensity'] = df[intensity_column]
    grouped = df.groupby(traj_column)
    #get all mean intensities:
    mean_dist = grouped[intensity_column].mean()
    #determine the cutoff:
    cutoff = 0#(threshold*max_dist.std())+max_dist.mean()
    #filter the dataframe:
    upper_filtered = grouped.filter(lambda g: (g.Intensity.mean() > cutoff))
    
    if plot:
        #if plot is true we return a plot comparing filtered vs unfiltered data
        fig,ax = plt.subplots(1,2)
        ax[0].set_title('before')
        # print('bins',int(np.sqrt(len(max_dist))))
        n_bins =  max([int(np.sqrt(len(mean_dist))),1])
        ax[0].hist(mean_dist,edgecolor='black',bins =n_bins,label='n={}'.format(len(mean_dist)))
        mean_dist_filtered = upper_filtered.groupby(traj_column)[intensity_column].mean()
        ax[1].set_title('after')
        n_bins_filtered =  max([int(np.sqrt(len(mean_dist_filtered))),1])
        ax[1].hist(mean_dist_filtered,edgecolor='black',bins = n_bins_filtered,label='n={}'.format(len(mean_dist_filtered)))
        ax[1].legend()
        ax[0].legend()
        return upper_filtered, fig
    return upper_filtered


def cohens_d(a,b):
    # pooled_standard_deviation  = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    
    s = np.sqrt(((a.shape[0] - 1) * a.var() + (b.shape[0] - 1) * b.var())/ (b.shape[0] + a.shape[0] -2))
    cohens_d = (a.mean() - b.mean())/ s
    return cohens_d
def cohenFilter(x, threshold, intensity_column = 'Intensity'):
    on_dist = x[x['fit'] ==1][intensity_column]
    off_dist = x[x['fit'] ==0][intensity_column]
    d = np.abs(cohens_d(on_dist,off_dist))
    if d > threshold:
        return True
    else:
        return False
    
def cohenFilter_returnd(x, intensity_column = 'Intensity'):
    on_dist = x[x['fit'] ==1][intensity_column]
    off_dist = x[x['fit'] ==0][intensity_column]
    d = np.abs(cohens_d(on_dist,off_dist))
    return d

def cohenFilter_negative(x, threshold, intensity_column = 'Intensity'):
    on_dist = x[x['fit'] ==1][intensity_column]
    off_dist = x[x['fit'] ==0][intensity_column]
    d = np.abs(cohens_d(on_dist,off_dist))
    if d <= threshold:
        return True
    else:
        return False
def filter_SNR(df, threshold = 2, return_bad = False, intensity_column = 'Intensity'):
    """
    estimate SNR by calculating cohen's d between the intensity distributions
    in on and off state.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    filtered = df.groupby('trajectory').filter(lambda x: cohenFilter(x,threshold, intensity_column = intensity_column))
    if return_bad:
        bad = df.groupby('trajectory').filter(lambda x: cohenFilter_negative(x,threshold, intensity_column = intensity_column))
        return filtered, bad
    return filtered


def get_number_of_events(series):
    """find number of binding events in trajectory
    anywhere where diff(fit) (that is the element-wise difference) 
    is non-zero an event starts or stops
    the number of binding events is half that!
    Its not necesarrily an integer!
    """
    
    n = (len((np.where(np.diff(series,append=0)!=0))[0]))/2
    # print(len((np.where(np.diff(series,append=0)!=0))[0]))
    return n
def filter_inactive(df,traj_column = 'trajectory', intensity_column = 'Intensity',fit_column = 'fit', threshold = 5, return_inactive = False):
    """
    filter out trajectories that show less than threshold binding events
    
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    traj_column : TYPE, optional
        DESCRIPTION. The default is 'trajectory'.
    intensity_column : TYPE, optional
        DESCRIPTION. The default is 'Intensity'.

    Returns
    -------
    None.

    """
    #first find the number of changepoints in fit column
    #anywhere where gradient(fit) is non-zero a event starts or stops
    #the number of binding events is half that!
    #first group the data:
    df['fit'] = df[fit_column]
    grouped = df.groupby(traj_column)
    #get all number of binding events, i.e the distribution of number of events
    
    #filter the dataframe:
    filtered = grouped.filter(lambda g: (get_number_of_events(g.fit) >= threshold))
    bad = grouped.filter(lambda g: (get_number_of_events(g.fit) <= threshold))
    if return_inactive:
        return filtered, bad
    else:
        return filtered
    

def filter_longEvents(df,traj_column = 'trajectory', fit_column = 'fit', threshold = 100):
    """
    filter out trajectories that stay in the on-state for very long:
    
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    traj_column : TYPE, optional
        DESCRIPTION. The default is 'trajectory'.
    
    Returns
    -------
    None.

    """
    #first find the number of changepoints in fit column
    #anywhere where gradient(fit) is non-zero a event starts or stops
    #the number of binding events is half that!
    #first group the data:
    df['fit'] = df[fit_column]
    grouped = df.groupby(traj_column)
    for trajn, trajg in grouped:
        #find the longest binding event:
            df['counter'] = df['fit'].diff().ne(0).cumsum()
            #now groiupby the counter and find the longest even group:
            grouped = df.groupby('counter')
            maxL=1
            for name, group in grouped:
                if name % 2 == 0:
                    if len(group) > maxL:
                        maxL = len(group)
                print(maxL)
    #filter the dataframe:
    # filtered = grouped.filter(lambda g: (get_number_of_events(g.fit) >= threshold))
    # bad = grouped.filter(lambda g: (get_number_of_events(g.fit) <= threshold))
    # return filtered, bad
    return maxL

def selectEvents(df,eventList):
    """select events manually
    """       
    df.set_index(['trajectory','slice'],inplace=True)
    
    sortOut = []
    for names,groups in df.groupby('trajectory'):        
        if names not in eventList:                  
            sortOut.append(names)
    filtered = df[df.groupby('trajectory').filter(lambda x: x.name not in sortOut).astype('bool')]
    final = filtered.dropna()
    final.reset_index(inplace=True)
    return final

def func(g):
    if len(g) <200:
        return False
    else:
        return True
def filter_byPulses(df,traj_column = 'aperture_index', intensity_column = 'intensity', min_pulses=300):
    """
    filter out short traces

    """
    grouped = df.groupby(traj_column)
    filtered = grouped.filter(func)
    return filtered
    
    
def filter_by_length(pulses, threshold = 5, method='std'):
    # filter based on pulse stats
    cutoff = pulses['dur'].mean() + threshold * pulses['dur'].std()
    so = max_length(pulses,cutoff)
    pulses = pulses[~pulses['trajectory'].isin(so.index)]
    return pulses

def max_length(df, max_l):
    max_pulses_dur = df.groupby('trajectory')['dur'].max()
    sort_out = max_pulses_dur[max_pulses_dur>max_l]
    print(sort_out)    
    return sort_out


    

def add_index(col):
    return range(len(col))
def add_timeColumn(df, traj_column ='aperture_index'):
    """
    platinum data comes without a slice/frame column. Add it based on index.

    """
    grouped = df.groupby('aperture_index')
    df['time'] = grouped['trajectory'].transform(add_index)

class ThresholdMonitor(ConvergenceMonitor):
         @property
         def converged(self):
             return (self.iter == self.n_iter or
                     self.history[-1] >= self.tol)
def init_hmm(n_components=2,init_params="st", initial_guess = [],n_iter=1000, means = [0,1]):
    """
    

    Parameters
    ----------
    n_components : TYPE, optional
        DESCRIPTION. The default is 2.
    init_params : TYPE, optional
        DESCRIPTION. The default is "".
    initial_guess : iterable
        the initial guess for fitting. We assume it comes from a cruve fir to a
        double gaussian, see find_initial_guess
    n_iter : TYPE, optional
        DESCRIPTION. The default is 1000.
    means : TYPE, optional
        DESCRIPTION. The default is [0,1].

    Returns
    -------
    layer1_hmm : TYPE
        DESCRIPTION.

    """
    if isinstance(initial_guess, (np.ndarray, np.generic)):
        model = hmm.GaussianHMM(n_components=n_components,
                                     init_params=init_params,
                                     n_iter=n_iter, verbose = True)
        # model.monitor_ = ThresholdMonitor(model.monitor_.tol,
                                           # model.monitor_.n_iter,
                                           # model.monitor_.verbose)
        #startprobaility should be set from ratio of weights in initial_guess!
        probability_0 = initial_guess[-2]/(initial_guess[-1]+initial_guess[-2])
        probability_1 = initial_guess[-1]/(initial_guess[-1]+initial_guess[-2])
        
        
        model.startprob_ = np.array([probability_0,probability_1])   
        print(model.startprob_)
        model.means_ = np.array([[initial_guess[0]],
                                      [initial_guess[1]]])
        model.means_weight = np.array([[initial_guess[-1]],
                                            [initial_guess[-2]]])
        
        # layer1_hmm.means_ = np.array(initial_guess[0])
        # layer1_hmm.covars_ = np.array(initial_guess[1])
        return model
    else:
        model = hmm.GaussianHMM(n_components=n_components,
                                     init_params=init_params,
                                     n_iter=n_iter, verbose = True)
        print('no intial guess supplied')
        return model
    
    

def train_hmm(df,model, traj_column='trajectory', intensity_column = "Intensity", fraction = 0.1, showFig = True,qt = False):
    """
    trains a hidden-markov-model (HMM) with gaussian emision probabilities.
    see https://hmmlearn.readthedocs.io/en/latest/ for details of the HMM.
    The HMM needs to be initialized beforehand.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe with the data. needs a column with intensity values, and one
        with the trajectory
    hmm : hmmlearn.hmm
        hidden markov model.
    traj_column : str, optional
        name of pandas column containing the trajectory. The default is 'trajectory'.
    intensity_column : str, optional
        name of pandas column containing the intensity values. The default is "Intensity".
    fraction : float, optional
        value between 0 and 1. The fraction of data to use for training. Beware:
            For large datasets fraction=1 can take loong times to train.
            Future versions of this funciton might allow to choose an absolute number of trajectories
        The default is 0.1.

    Returns
    -------
    None.

    """
    
    #train on fraction of data only:
    df['trajectory'] = df[traj_column]
    #total number of trajs:
    trajectories = df[traj_column].unique()
    Ntraj = len(trajectories)
    #select fraction*Ntraj random columns:    
    selection = np.random.choice(np.array(trajectories), size = int(fraction*Ntraj))
    training_data = df.groupby(traj_column).filter(lambda g: (g.trajectory.iloc[0] in selection))

    #list of trajectories needed. each of them
    # in shape [-1,1]
    grouped_training_data = training_data.groupby(traj_column)
    formatted_trainingData = np.concatenate(
        [(np.array(x[1][intensity_column])).reshape([-1,1]) for x in grouped_training_data])
    # while True:
        # try:
    print('training HMM:\n')
    model.fit(formatted_trainingData)
        # except ValueError:
            # print('hard to fit, keep trying...')
    score = model.score(formatted_trainingData)
    if showFig:
        fig = plot_HMMfit(df[intensity_column], model)
        fig.show()
        return fig, score
    if qt:
        plot_HMMfit_qt(qt,df[intensity_column], model)
    
    return score

def plot_HMMfit_qt(fig,data, model):
    # fig_finalHMM, ax =plt.subplots()

    stationary = model.get_stationary_distribution()
    loc1, scale1, size1 =  model.means_[0][0], np.sqrt(model.covars_[0][0]), int(np.round(1000*stationary[0]))

    loc2, scale2, size2 = model.means_[1][0], np.sqrt(model.covars_[1][0]), int(np.round(1000*stationary[1]))

    x2 = np.concatenate([np.random.normal(loc=loc1, scale=scale1, size=size1),

                         np.random.normal(loc=loc2, scale=scale2, size=size2)])
    x_eval = np.linspace(x2.min() - 1, x2.max() + 1, 10000)
    pdf = norm.pdf

    bimodal_pdf = pdf(x_eval, loc=loc1, scale=scale1) * float(size1)/x2.size + pdf(x_eval, loc=loc2, scale=scale2) * float(size2)/x2.size

    SNR = (np.abs(loc2 - loc1) / (0.5 * (scale1+scale2)))[0]
    # print(SNR)
    ax = fig.add_subplot()
    ax.hist(data, bins = 100, edgecolor='black', density = True)
    ax.set_title("SNR = {:.2f}".format(SNR))
    ax.plot(x_eval, bimodal_pdf, 'r--', label="Actual PDF")
    ax.plot(x_eval,norm.pdf(x_eval, loc=loc1, scale=scale1)*float(size1)/x2.size, color = 'red', ls = '--')
    ax.plot(x_eval,norm.pdf(x_eval, loc=loc2, scale=scale2)*float(size2)/x2.size, color = 'red', ls = '--')
    # return fig_finalHMM


def plot_HMMfit(data, model):
    fig_finalHMM, ax =plt.subplots()

    stationary = model.get_stationary_distribution()
    loc1, scale1, size1 =  model.means_[0][0], np.sqrt(model.covars_[0][0]), int(np.round(1000*stationary[0]))

    loc2, scale2, size2 = model.means_[1][0], np.sqrt(model.covars_[1][0]), int(np.round(1000*stationary[1]))

    x2 = np.concatenate([np.random.normal(loc=loc1, scale=scale1, size=size1),

                         np.random.normal(loc=loc2, scale=scale2, size=size2)])
    x_eval = np.linspace(x2.min() - 1, x2.max() + 1, 10000)
    pdf = norm.pdf

    bimodal_pdf = pdf(x_eval, loc=loc1, scale=scale1) * float(size1)/x2.size + pdf(x_eval, loc=loc2, scale=scale2) * float(size2)/x2.size

    SNR = (np.abs(loc2 - loc1) / (0.5 * (scale1+scale2)))[0]
    # print(SNR)

    ax.hist(data, bins = 100, edgecolor='black', density = True)
    ax.set_title("SNR = {:.2f}".format(SNR))
    ax.plot(x_eval, bimodal_pdf, 'r--', label="Actual PDF")
    ax.plot(x_eval,norm.pdf(x_eval, loc=loc1, scale=scale1)*float(size1)/x2.size, color = 'red', ls = '--')
    ax.plot(x_eval,norm.pdf(x_eval, loc=loc2, scale=scale2)*float(size2)/x2.size, color = 'red', ls = '--')
    return fig_finalHMM



def train_trajectory(traj,hmm):
    training_traj = (np.array(traj)).reshape([-1,1])
    hmm.fit(training_traj)
    print(hmm.means_)
def retrain_hmm(df,model, traj_column='trajectory', intensity_column = "norm_Intensity", params = 'mcs',fraction = 0.1):
    """
    TODO

    """
     
    #train on fraction of data only:
    df['trajectory'] = df[traj_column]
    #total number of trajs:
    trajectories = df[traj_column].unique()
    Ntraj = len(trajectories)
    #select fraction*Ntraj random columns:    
    selection = np.random.choice(np.array(trajectories), size = int(fraction*Ntraj))
    training_data = df.groupby('trajectory').filter(lambda g: (g.trajectory.iloc[0] in selection))

    #list of trajectories needed. each of them
    # in shape [-1,1]
    grouped_training_data = training_data.groupby('trajectory')
    formatted_trainingData = np.concatenate(
        [(np.array(x[1][intensity_column])).reshape([-1,1]) for x in grouped_training_data])
    
    retrainedHmm = hmm.GaussianHMM(n_components=2, n_iter=1000, init_params="", params = "mc")
    retrainedHmm.transmat_ = model.transmat_
    retrainedHmm.means_ = model.means_
    retrainedHmm.covariance_type = model.covariance_type
    retrainedHmm.covars_ = model.covars_.reshape(2,1)

    retrainedHmm.fit(formatted_trainingData)
    score = retrainedHmm.score(formatted_trainingData)
    return retrainedHmm

    
    grouped = df.groupby(traj_column)
    
    grouped[intensity_column].transform(train_trajectory,retrainedHmm)
    
    # formatted_trainingData = np.concatenate(
    #     [(np.array(x[1][intensity_column])).reshape([-1,1]) for x in grouped_training_data])
    # hmm.fit(formatted_trainingData)
    # score = hmm.score(formatted_trainingData)
    return retrainedHmm


def posterior_trajectory(traj, hmm, state):
    
    posterior = hmm.predict_proba(np.array(traj).reshape([-1,1]))
    if state == 0:
        posterior = posterior[:,0]
    if state == 1:
        posterior = posterior[:,1]
    
    return posterior

def posterior_dataset(df,hmm, traj_column = 'trajectory', intensity_column = 'Intensity'):
    grouped = df.groupby(traj_column)
    
    df['posterior_0'] = grouped[intensity_column].transform(posterior_trajectory,hmm,0)
    df['posterior_1'] = grouped[intensity_column].transform(posterior_trajectory,hmm,1)
    # return posterior
def fit_trajectory(traj,hmm):
    result = hmm.predict(np.array(traj).reshape([-1,1]))
    return result

def fit_dataset(df,hmm, traj_column = 'trajectory', intensity_column = 'Intensity', fit_column = 'fit'):
    grouped = df.groupby(traj_column)
    df[fit_column] = grouped[intensity_column].transform(fit_trajectory,hmm)    

def normalize_trajectory(traj):
    normed = traj/traj.max()
    return normed

def normalize_dataset(df,traj_column = 'trajectory', intensity_column = 'Intensity'):
    grouped = df.groupby(traj_column)
    df['norm_'+intensity_column] = grouped[intensity_column].transform(normalize_trajectory)    

def calibrate_trajectory(traj,conversion):
    cal = traj * conversion
    return cal
def calibrate_dataset(df, conversion, traj_column = 'trajectory', time_column = 'slice'):
    grouped = df.groupby(traj_column)
    df['seconds'] = grouped[time_column].transform(calibrate_trajectory, conversion)    
    
def decode_trajectory(traj,hmm,out='fit'):
    score, states = hmm.decode(np.array(traj).reshape([-1,1]))
    # print('score:',score)
    # print('states',states)
    if out == 'fit':
        return states
    elif out == 'score':
        return score
def posterior_trajectory(traj,hmm):
    log_prob, posteriors = hmm.score_samples(np.array(traj).reshape([-1,1]))
    # print(posteriors[0:])
    # print(posteriors[:,1])
    return posteriors[:,1]

def decode_dataset(df,hmm, traj_column = 'trajectory', intensity_column = 'Intensity', fit_column = 'fit'):
    grouped = df.groupby(traj_column)
    df[fit_column] = grouped[intensity_column].transform(decode_trajectory,hmm,out='fit')
    df[fit_column+'_score'] = grouped[intensity_column].transform(decode_trajectory,hmm,out='score')
    df[fit_column+'_posterior'] = grouped[intensity_column].transform(posterior_trajectory,hmm)
    
def map_trajectory(traj,hmm):
    #TODO: only works for 3 state data now!
    means = hmm.means_
    # print(hmm.means_)
    map_dict = {0: means[0][0], 1: means[1][0]}#dict( [(0,means[0]),(1,means[1])])
    # print(traj)
    mapped = traj.map(map_dict)
    # print(mapped)  
    return mapped

def map_states_dataset(df,hmm, traj_column = 'trajectory', fit_column = 'fit', mapped_fit_column = 'mapped_fit'):
    grouped = df.groupby(traj_column)
    df[mapped_fit_column] = grouped[fit_column].transform(map_trajectory,hmm)    
    
    
def dwelltimes(X):
    X = np.array(X)
    dwells = np.diff(np.flatnonzero(np.concatenate(([True], X[1:]!= X[:-1], [True] ))))
    dwells_on = dwells[1::2]
    dwells_off = dwells[0::2]
    return list(dwells_on),list(dwells_off)


def smooth(df,wd=5):
    if isinstance(df,pd.DataFrame):   
        smoothed = df.rolling(wd).mean().fillna(value=0)
        return smoothed
    else:
        df = pd.Series(df)
        smoothed = df.rolling(wd).mean()
        return smoothed
def savgol_filter_smooth(df,wd=2,polyorder=2):
    ydata = np.array(df)
    smoothed = savgol_filter(ydata,wd,polyorder)
    return smoothed
    
def smooth_dataset(df,column,wd,groupName = 'trajectory',result_prefix = 'smooth',polyorder = 3,method = 'savgol'):
    print(df.keys())
    groupedDF = df.groupby(groupName)
    print(groupedDF)
    
    if method == 'savgol':
        df[result_prefix + '_'+column] = groupedDF[column].transform(savgol_filter_smooth,wd=wd,polyorder = polyorder)
    if method == 'window':
        print('window mean')
        df[result_prefix + '_'+column] = groupedDF[column].transform(smooth,wd=wd)
        # df[result_prefix + '_'+column].fillna('backfill')

def get_dwelltimes_trajectory(traj, fit_column = 'fit'):
    changePoints = np.where(traj[fit_column].diff()!=0)
    dwells = traj[['seconds',fit_column]].iloc[changePoints]
    dwells['diff'] = dwells['seconds'].diff()

    ontimes = dwells['diff'][dwells[fit_column] == 0]
    offtimes = dwells['diff'][dwells[fit_column] == 1]
    ontimes.dropna(inplace = True)
    offtimes.dropna(inplace = True)
    return ontimes, offtimes

def get_pulses_trajectory(traj, fit_column = 'fit', intensity_column = 'Intensity', traj_column = 'trajectory', time_conversion =1.):
    # print(traj)
    #first figure out which is on which is off state:
    state1 = traj[traj[fit_column] == 1][intensity_column].mean()
    state0 = traj[traj[fit_column] == 0][intensity_column].mean()
    on_state = np.argmax([state0,state1])
    off_state = np.argmin([state0,state1])

    #find changepoints, i.e. where we change from on to off or vice versa:
    changePoints = np.where(traj[fit_column].diff()!=0)
    # print(changePoints)
    pulses = {'trajectory':[], 'dur':[],'Intensity':[], 'std':[], 'start_frame':[], 'end_frame':[]}
    off = {'trajectory':[],'dur':[],'Intensity':[], 'std':[], 'start_frame':[], 'end_frame':[]}
    start = 0
    for cp in changePoints[0]:
        # print(cp)
        segment = traj.iloc[start:cp]
        # print(segment)
        if len(segment>0):
            if segment['fit'].iloc[0] == on_state:
                #the start value is part of the pulse, not the baseline
                #the last value is not part of the pulse    
                pulses['trajectory'].append(segment[traj_column].iloc[0])
                pulses['dur'].append(len(segment)*time_conversion)
                pulses['Intensity'].append(segment['Intensity'].mean())
                pulses['std'].append(segment['Intensity'].std())
                pulses['start_frame'].append(start)
                pulses['end_frame'].append(cp)
            if segment['fit'].iloc[0] == off_state:
                off['trajectory'].append(segment[traj_column].iloc[0])
                off['dur'].append(len(segment)*time_conversion)
                off['Intensity'].append(segment['Intensity'].mean())
                off['std'].append(segment['Intensity'].std())
                off['start_frame'].append(start)
                off['end_frame'].append(cp)
        start = cp
    #there is a potential last segment miseed when the trajectory ends in the bound state
    
    pulse_df = pd.DataFrame(pulses)
    off_df = pd.DataFrame(off)
    return (pulse_df, off_df)
        
def get_pulses_dataset(data,**kwargs):   
    tqdm.pandas()
    print('generating pulse statistics')
    pulses = data.groupby('trajectory').progress_apply(get_pulses_trajectory, **kwargs)
    all_on = pd.DataFrame()
    all_off = pd.DataFrame()
    for i in pulses:
        all_on = pd.concat([all_on,i[0]])
        all_off = pd.concat([all_off,i[1]])
    
    
    return all_on,all_off


def get_dwelltimes_dataset(data, trajectory_column = 'trajectory', fit_column = 'fit'):    
    all_on_dwells = []
    all_off_dwells = []
    mean_on = []
    mean_off = []
    i = 0
    
    for name, group in data.groupby('trajectory'):
        on_dwells,off_dwells = get_dwelltimes_trajectory(group)
        on_dwells = on_dwells#(on_dwells[on_dwells<np.mean(on_dwells)+1*np.std(on_dwells)])
        off_dwells = off_dwells#(off_dwells[off_dwells<np.mean(off_dwells)+2*np.std(off_dwells)])
        all_on_dwells = np.concatenate([all_on_dwells,on_dwells])
        all_off_dwells = np.concatenate([all_off_dwells,off_dwells])
        mean_on.append(np.mean(on_dwells))
        mean_off.append(np.mean(off_dwells))
        i+=1
    return all_on_dwells, all_off_dwells, mean_on, mean_off


def get_dwelltimes_dataset_pd(data, trajectory_column = 'trajectory', fit_column = 'fit', method = 'mean'):    
    all_on_dwells = []
    all_off_dwells = []
    all_trajectories_column = []
    mean_on = []
    mean_off = []
    mean_N = []
    trajectories = []
    i = 0
    all_dwells = pd.DataFrame({'ondwells':[],'offdwells':[]})
    mean_dwells = pd.DataFrame({'ondwells':[],'offdwells':[], 'N':[], 'trajectory':[]})
    for name, group in data.groupby('trajectory'):
        on_dwells,off_dwells = get_dwelltimes_trajectory(group, fit_column = fit_column)
        # on_dwells = on_dwells#(on_dwells[on_dwells<np.mean(on_dwells)+1*np.std(on_dwells)])
        # off_dwells = off_dwells#(off_dwells[off_dwells<np.mean(off_dwells)+2*np.std(off_dwells)])
        
        all_on_dwells = np.concatenate([all_on_dwells,on_dwells])
        all_off_dwells = np.concatenate([all_off_dwells,off_dwells])
        #add a column for the trajectory:
        trajectory_column = np.ones(len(on_dwells))*name
        all_trajectories_column = np.concatenate([all_trajectories_column,trajectory_column])
        
        if method == 'mean':
            mean_on.append(np.mean(on_dwells))
            mean_off.append(np.mean(off_dwells))
            mean_N.append(len(on_dwells))
            trajectories.append(name)
        else:
            mean_on.append(np.median(on_dwells))
            mean_off.append(np.median(off_dwells))
            mean_N.append(len(on_dwells))
            trajectories.append(name)
        i+=1
    allDwells = pd.concat([pd.Series(all_on_dwells),pd.Series(all_off_dwells),pd.Series(all_trajectories_column)], axis = 1)
    allDwells.rename(columns={allDwells.keys()[0]:'on dwells',allDwells.keys()[1]:'off dwells', allDwells.keys()[2]: 'trajectory'},inplace=True)
    allDwells.dropna(inplace = True)    #If there is more off than on rates the additional ones are deleted!
    
    meanDwells = pd.concat([pd.Series(mean_on), 
                            pd.Series(mean_off), 
                            pd.Series(mean_N),
                            pd.Series(trajectories)], axis = 1)
    meanDwells.rename(columns={meanDwells.keys()[0]:'on dwells',
                               meanDwells.keys()[1]:'off dwells',
                               meanDwells.keys()[2]:'N',
                               meanDwells.keys()[3]:'trajectory'},inplace=True)
    return allDwells, meanDwells

def get_dwelltimes_trajectory_platinum(traj, fit_column = 'fit'):
    """
    TODO write this into a method that creates pulse information a la platinum, i.e.
    mean intensity, pulse duration, bin-ratio etc. for every pulse.

    Parameters
    ----------
    traj : TYPE
        DESCRIPTION.
    fit_column : TYPE, optional
        DESCRIPTION. The default is 'fit'.

    Returns
    -------
    ontimes : TYPE
        DESCRIPTION.
    offtimes : TYPE
        DESCRIPTION.

    """
    changePoints = np.where(traj[fit_column].diff()!=0)
    print(changePoints)
    dwells = traj[['seconds',fit_column]].iloc[changePoints]
    dwells['diff'] = dwells['seconds'].diff()

    ontimes = dwells['diff'][dwells[fit_column] == 0]
    offtimes = dwells['diff'][dwells[fit_column] == 1]
    ontimes.dropna(inplace = True)
    offtimes.dropna(inplace = True)
    return ontimes, offtimes

def get_dwelltimes_dataset_platinum(data, trajectory_column = 'trajectory', fit_column = 'fit', method = 'mean'):    
    all_on_dwells = []
    all_off_dwells = []
    all_trajectories_column = []
    mean_on = []
    mean_off = []
    mean_N = []
    trajectories = []
    i = 0
    all_dwells = pd.DataFrame({'ondwells':[],'offdwells':[]})
    mean_dwells = pd.DataFrame({'ondwells':[],'offdwells':[], 'N':[], 'trajectory':[]})
    for name, group in data.groupby('trajectory'):
        on_dwells,off_dwells = get_dwelltimes_trajectory_platinum(group, fit_column = fit_column)
        # on_dwells = on_dwells#(on_dwells[on_dwells<np.mean(on_dwells)+1*np.std(on_dwells)])
        # off_dwells = off_dwells#(off_dwells[off_dwells<np.mean(off_dwells)+2*np.std(off_dwells)])
        
        all_on_dwells = np.concatenate([all_on_dwells,on_dwells])
        all_off_dwells = np.concatenate([all_off_dwells,off_dwells])
        #add a column for the trajectory:
        trajectory_column = np.ones(len(on_dwells))*name
        all_trajectories_column = np.concatenate([all_trajectories_column,trajectory_column])
        
        if method == 'mean':
            mean_on.append(np.mean(on_dwells))
            mean_off.append(np.mean(off_dwells))
            mean_N.append(len(on_dwells))
            trajectories.append(name)
        else:
            mean_on.append(np.median(on_dwells))
            mean_off.append(np.median(off_dwells))
            mean_N.append(len(on_dwells))
            trajectories.append(name)
        i+=1
    allDwells = pd.concat([pd.Series(all_on_dwells),pd.Series(all_off_dwells),pd.Series(all_trajectories_column)], axis = 1)
    allDwells.rename(columns={allDwells.keys()[0]:'on dwells',allDwells.keys()[1]:'off dwells', allDwells.keys()[2]: 'trajectory'},inplace=True)
    allDwells.dropna(inplace = True)    #If there is more off than on rates the additional ones are deleted!
    
    meanDwells = pd.concat([pd.Series(mean_on), 
                            pd.Series(mean_off), 
                            pd.Series(mean_N),
                            pd.Series(trajectories)], axis = 1)
    meanDwells.rename(columns={meanDwells.keys()[0]:'on dwells',
                               meanDwells.keys()[1]:'off dwells',
                               meanDwells.keys()[2]:'N',
                               meanDwells.keys()[3]:'trajectory'},inplace=True)
    return allDwells, meanDwells


def double_gauss(x,mean_a,mean_b,sig_a, sig_b,weight_a, weight_b):
    gauss_a = weight_a*norm.pdf(x,loc=mean_a, scale= sig_a)
    gauss_b = weight_b *norm.pdf(x,loc=mean_b, scale= sig_b)
    double = gauss_a + gauss_b
    return double
def single_gauss(x,mean,sig,weight):
    gauss = weight*norm.pdf(x,loc=mean, scale= sig)
    return gauss

def find_initial_guess(df, intensity_column = 'Intensity',showFig = True, figName = '', p0 = [0,20000,2000,2000,1,0.05], return_fig = False):    
    """    
    the EM algorithm is inherently prone to getting stuck in local maxima. Therefore
    a good initial guess is important. We use a bi-normal fit for that.
    if this fit fails we have a problem...
    """

    min_int = df[intensity_column].min()
    max_int = df[intensity_column].max()

    x = np.arange(min_int,max_int,10)
    fig_initial, ax = plt.subplots()

    # double = double_gauss(x,10000,30000,5000,5000,2,1)

    bins = np.histogram(df[intensity_column], bins = 100, density = True)

    #even a normal curve fit for a bimodal distribution can get stuck in local
    #maxima so its best to add an initial guess p0
    #the syntax is p0 = [mean1,mean2,sigma1,sigma2, weight1,weight2] where the means and sigmas
    # are the means and deviations of the individual distributions and the weight determines
    # the hieght of them, the standard guesses should usually work though!
    print(p0)
    popt, pcov = curve_fit(double_gauss, bins[1][1:], bins[0], p0 = p0)

    ax.plot(x,double_gauss(x,*popt), color = 'black', lw = 2)

    ax.plot(x,single_gauss(x, popt[0],popt[2],popt[4]), color  ='black', ls = '--')
    ax.plot(x,single_gauss(x, popt[1],popt[3],popt[5]), color  ='black', ls = '--')
    SNR = (np.abs(popt[1] - popt[2]) / (0.5 * (popt[3]+popt[4])))
    ax.hist(df[intensity_column], edgecolor = 'black', bins = 100, density = True, 
            color = 'gold', label = 'estimated SNR: {:.1f}'.format(SNR))
    ax.set_xlabel('Intensity (arb. units)')
    ax.set_ylabel('Frequency')
    ax.set_xlim([min_int,max_int])
    ax.set_title('initial guess for state asignment '+figName)
    ax.legend()
    # fig_initial.show()
    # fig.savefig('intensity_hist_25C_D9_647_DE_1000ms_OD25.svg')
    if return_fig:
        return popt,fig_initial
    # print(SNR)
    return popt
   