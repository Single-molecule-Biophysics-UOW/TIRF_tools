# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 11:13:57 2022

@author: smueller
"""
import os
import numpy as np
from matplotlib import pyplot as plt
import itertools

def plot_random(df, n,random_state = 42):
    """
    Plot random trajectories

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(n)
    
    np.random.seed(random_state)
    grouped = df.groupby('trajectory')
# traj = grouped.get_group(107)
    groups = list(grouped.groups.keys())
    
    choices =np.random.choice(groups,size=n) #52
    for i in range(n):
        
        traj = grouped.get_group(choices[i]) #220 #74
        #you might have to adjust this for a nice plot
        x = np.arange(-10000,30000,1)

        ax[i].plot(traj['seconds'], traj['Intensity'], label = 'Traj {}'.format(choices[i]))        
        ax[i].plot(traj['seconds'], traj['mapped_fit'])
        ax[i].legend()
    return fig

def plot_traj(df, choice, xcolumn = 'seconds', traj_column = 'trajectory' ,
              ycolumns = ['Intensity', 'mapped_fit'], **kwargs):
    """
    

    Parameters
    ----------
    df : DataFrame
        contains input data
    choice : int
        trajectory number
    xcolumn : str, optional
        column to plot on x axis. The default is 'seconds'.
    traj_column : TYPE, optional
        DESCRIPTION. The default is 'trajectory'.
    ycolumns : str, optional
        column to plot on y axis. The default is ['Intensity', 'mapped_fit'].
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    fig : matplotlib figure object
        

    """
    grouped = df.groupby(traj_column)
    traj = grouped.get_group(choice)

    fig, ax = plt.subplots()
    for y in ycolumns:
        ax.plot(traj[xcolumn], traj[y],**kwargs)
        ax.legend()
    return fig
        
def plot_all_trajs(df,path, xcolumn = 'slice', ycolumn = ['Intensity','fit'], trajectory_column = 'trajectory'):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    path : TYPE
        DESCRIPTION.
    xcolumn : TYPE, optional
        DESCRIPTION. The default is 'slice'.
    ycolumn : TYPE, optional
        DESCRIPTION. The default is ['Intensity','fit'].
    trajectory_column : TYPE, optional
        DESCRIPTION. The default is 'trajectory'.

    Returns
    -------
    None.

    """
    #check if the path exists:
    if not os.path.exists(path):
        os.makedirs(path)
    
    plt.ioff()
    fig,ax = plt.subplots(3,3)

    indeces = list(itertools.product([0,1,2],repeat =2))
    n=0
    
    if isinstance(ycolumn,str):        
        for name,group in df.groupby(trajectory_column):                             
            ax[indeces[n]].plot(group[xcolumn],group[ycolumn])
            ax[indeces[n]].set_title('trajectory {}'.format(name))            
            n+=1
            if n == 9:
                #save figure and close it:
                fig.savefig(path+'trajectory_'+str(name)+'.jpg')
                plt.close(fig)
                #make new one:
                fig,ax = plt.subplots(3,3)
                n= 0
            
    if isinstance(ycolumn,list):
        for name,group in df.groupby(trajectory_column):
            for column in ycolumn:
                #print ('n:{}, column:{}'.format(n,column))
                try:
                    t = (indeces[n])
                except:
                    print("indexing error")
                try:
                    t=(group[column])
                except:
                    print('ycolumn problem')
                line, =ax[indeces[n]].plot(group[xcolumn],group[column])
                ax[indeces[n]].set_title('trajectory {}'.format(name))                
            n+=1
            if n == 9:
                #save figure and close it:
                fig.savefig(path+'trajectory_'+str(name)+'.png')
                plt.close(fig)
                #make new one:
                fig,ax = plt.subplots(3,3)
                n= 0
def plot_all_trajs_big(df,path, xcolumn = 'slice', ycolumn = ['Intensity','fit'], trajectory_column = 'trajectory', title = ''):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    path : TYPE
        DESCRIPTION.
    xcolumn : TYPE, optional
        DESCRIPTION. The default is 'slice'.
    ycolumn : TYPE, optional
        DESCRIPTION. The default is ['Intensity','fit'].
    trajectory_column : TYPE, optional
        DESCRIPTION. The default is 'trajectory'.

    Returns
    -------
    None.

    """
    #check if the path exists:
    if not os.path.exists(path):
        os.makedirs(path)
    
    plt.ioff()
    fig,ax = plt.subplots(3)

    # indeces = list(itertools.product([0,1,2],repeat =2))
    n=0
    
    if isinstance(ycolumn,str):        
        for name,group in df.groupby(trajectory_column):                             
            ax[n].plot(group[xcolumn],group[ycolumn])
            ax[n].set_title('trajectory {}'.format(name))            
            n+=1
            if n == 3:
                #save figure and close it:
                fig.savefig(path+'trajectory_'+str(name)+'.jpg')
                plt.close(fig)
                #make new one:
                fig,ax = plt.subplots(3)
                n= 0
            
    if isinstance(ycolumn,list):
        for name,group in df.groupby(trajectory_column):
            for column in ycolumn:
                #print ('n:{}, column:{}'.format(n,column))
                try:
                    t = ([n])
                except:
                    print("indexing error")
                try:
                    t=(group[column])
                except:
                    print('ycolumn problem')
                # print(group[xcolumn])
                # print(group[column])
                # print(np.array(group[column]))
                # print(n)
                # ax[n].set_title('test')
                
                # ax[n].plot()
                
                line, =ax[n].plot(np.array(group[xcolumn]),np.array(group[column]))
                if column == 'Intensity':
                    print(group['Intensity'].std())
                    # line.set_label('state: {:d}-mer'.format(int(group['determined state'].iloc[0])))
                    ax[n].legend()
                ax[n].set_title('trajectory {}'.format(name))                
            n+=1
            if n == 3:
                print('saving at:\n {}'.format(path))
                #save figure and close it:
                fig.savefig(path+'trajectory_'+str(name)+'.png')
                plt.close(fig)
                #make new one:
                fig,ax = plt.subplots(3)
                n= 0
        if n<3:
            try:
                fig.savefig(path+'trajectory_'+str(name)+'.png')
            except UnboundLocalError:
                fig.savefig(path+'trajectory_'+str(n)+'.png')
                
            plt.close(fig)

def dwelltime_distributions(pulses, off, bins = 'fd'):
    fig_dwelltimes, ax = plt.subplots(1,2,figsize=(10,7))
    ax[0].hist(pulses.groupby('trajectory')['dur'].mean(),
             bins = bins, 
             edgecolor='black', 
             label = 'mean = {:.2f} sem = {:.2f}'.format(
                 pulses.groupby('trajectory')['dur'].mean().mean(),
                 pulses.groupby('trajectory')['dur'].mean().sem())
             )
    ax[0].set_xlabel('Duration (s)')
    ax[0].set_ylabel('Probability density')
    ax[0].legend()
    ax[0].set_title('on times')
    ax[1].hist(off.groupby('trajectory')['dur'].mean(),
             bins = 'fd', 
             edgecolor='black', 
             label = 'mean = {:.2f} sem = {:.2f}'.format(
                 off.groupby('trajectory')['dur'].mean().mean(),
                 off.groupby('trajectory')['dur'].mean().sem())
             )
    ax[1].set_xlabel('Duration (s)')
    ax[1].set_ylabel('Probability density')
    ax[1].set_title('off times')
    ax[1].legend()
    return fig_dwelltimes
