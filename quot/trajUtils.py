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

def radial_disps(trajs, pixel_size_um=0.16, first_only=False):
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

def radial_disp_histograms(trajs, n_intervals=1, pixel_size_um=0.16,
    first_only=True, n_gaps=0, bin_size=0.001, max_disp=5.0):
    """
    Calculate radial displacement histograms for a set of 
    trajectories.

    args
    ----
        trajs           :   pandas.DataFrame
        n_intervals     :   int, the maximum number of frames
                            over which to compute displacements
        pixel_size_um   :   float, size of pixels (um)
        first_only      :   bool, only take the displacements
                            relative to the first point of 
                            each trajectory
        n_gaps          :   int, the number of gaps allowed
                            in tracking
        bin_size        :   float, the spatial bin size in um
        max_disp        :   float, maximum displacement to 
                            consider

    returns
    -------
        (
            2D ndarray of shape (n_intervals, n_bins), the radial
                displacements in each bin for each interval;
            1D ndarray of shape (n_bins+1), the edges of the
                displacement bins in um
        )

    """
    # Filter out localizations not assigned to a trajectory, or 
    # trajectories that are length 1
    trajs = traj_length(trajs)
    T = trajs.loc[
        (trajs['trajectory']>=0) & (trajs['traj_len']>1), 
        :
    ].copy()

    # Add a column that indicates whether each localization is
    # the first in the corresponding trajectory (computed later)
    T['first_in_traj'] = 0

    # Order the trajectories by ascending frame / trajectory indices.
    # Trajectories are grouped together, and frames are ascending within
    # each trajectory group
    T = T.sort_values(by=['trajectory', 'frame'])

    # Take only the columns we care about in an ndarray. The operations
    # we will perform with ndarrays are much faster than their pandas
    # dataframe equivalents
    X = np.asarray(T[['frame', 'trajectory', 'y', 'x', 'first_in_traj']])

    # Determine which localizations are the first in their respective
    # trajectories
    X[0,4] = 1.0 
    X[1:,4] = ((X[1:,1] - X[:-1,1])>0).astype('float64')

    # Set up the binning scheme
    bin_edges = np.arange(0.0, max_disp+bin_size, bin_size)

    # Set up the output histogram
    H = np.zeros((n_intervals, bin_edges.shape[0]-1), dtype=np.int64)

    # In order to capture every possible displacement in the presence
    # of gaps, we need to check enough delays for the maximum interval
    # (n_intervals) with the maximum number of gaps between every
    # frame.
    n_delays = n_intervals * (n_gaps+1)

    for delay in range(1, n_delays+1):

        # Take the displacements
        D = X[delay:,:] - X[:-delay,:]

        # We're only interested in displacements between localizations
        # in the same trajectory
        in_same_traj = D[:,1] == 0

        # Check for displacements matching any of the desired frame
        # intervals
        for interval in range(1, n_intervals+1):

            if first_only:
                match = (D[:,0]==interval) & (D[:,4]==-1) & in_same_traj 
            else:
                match = (D[:,0]==interval) & in_same_traj 

            # Check if there are any displacements that match these
            # criteria
            if match.sum() > 0:

                # Get the radial displacements
                disps = np.sqrt((D[match,2:4]**2).sum(axis=1)) * pixel_size_um 

                # Bin these into a histogram
                H_int, _edges = np.histogram(disps, bins=bin_edges)

                # Accumulate into the histogram
                H[interval-1,:] = H[interval-1,:] + H_int

    # Return the histogram and the accompanying set of bin 
    # definitions
    return H, bin_edges














