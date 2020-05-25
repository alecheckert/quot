#!/usr/bin/env python
"""
trajUtils.py -- functions to compute some common values on 
trajectories

"""
# Numeric
import numpy as np 

# DataFrames
import pandas as pd 

def traj_length(trajs):
    """
    Compute the number of localizations corresponding to 
    each trajectory.

    Unassigned localizations (with a trajectory index of 
    -1) are given traj_len = 0.

    args
    ----
        trajs       :   pandas.DataFrame, localizations with the 
                        'trajectory' column

    returns
    -------
        pandas.DataFrame, the same dataframe with "traj_len"
            column

    """
    # Remove traj_len column if it already exists
    if "traj_len" in trajs.columns:
        trajs = trajs.drop("traj_len", axis=1)

    # Get the lengths 
    trajs = trajs.join(
        trajs.groupby("trajectory").size().rename("traj_len"),
        on="trajectory"
    )

    # Unassigned localizations are given a traj_len of 0
    trajs.loc[trajs['trajectory']==-1, 'traj_len'] = 0
    return trajs  

def radial_disps(trajs, pixel_size_um=0.16, first_only=True):
    """
    Calculate the 2D radial displacement of each jump in every 
    trajectory.

    args
    ----
        trajs           :   pandas.DataFrame
        pixel_size_um   :   float, size of pixels
        first_only      :   float, only compute the first 
                            displacement of each trajectory

    returns
    -------
        pandas.DataFrame, with the new column 'radial_disp_um'

    """
    # If handed an empty dataframe, assign a dead column
    if len(trajs) == 0:
        trajs['radial_disp_um'] = []

    # Order by ascending trajectory, frame
    trajs = trajs.sort_values(by=['trajectory', 'frame'])

    # Filter out singlets
    if not "traj_len" in trajs.columns:
        trajs = traj_length(trajs)
    F0 = np.logical_and(trajs['trajectory']>=0, trajs['traj_len']>1)

    # If no locs remain, return all NaN
    if F0.sum() == 0:
        trajs['radial_disp_um'] = np.nan 

    # Format as ndarray for speed
    S = np.asarray(trajs.loc[F0, ['trajectory', 'frame', 'y', 'x']])

    # Compute Cartesian displacements
    D = S[1:,:] - S[:-1,:]

    # Add extra row at the end for indexing purposes (there is one
    # fewer row in D than in S)
    D = np.concatenate((D, np.array([[1, 1, 1, 1]])), axis=0)

    # Take all displacements that originate from the same trajectory
    # and from a single frame interval
    F1 = np.logical_and(D[:,0]==0, D[:,1]==1)

    # If no disps remain, return all NaN
    if F1.sum() == 0:
        trajs['radial_disp_um'] = np.nan 

    # Compute radial displacements
    R = np.full(F0.sum(), np.nan)
    R[F1] = np.sqrt((D[F1, 2:]**2).sum(axis=1))

    # Assign to the corresponding rows in the original dataframe
    trajs.loc[F0, "radial_disp_um"] = R 

    # If desired, only used the first jump from each trajectory
    if first_only:
        trajs["_ones"] = 1 
        trajs["_first_loc"] = trajs.groupby("trajectory")["_ones"].cumsum()==1 
        trajs.loc[~trajs['_first_loc'], 'radial_disp_um'] = np.nan 
        trajs = trajs.drop(['_ones', '_first_loc'], axis=1)

    # Scale into um
    trajs['radial_disp_um'] = trajs['radial_disp_um'] * pixel_size_um 

    return trajs 

