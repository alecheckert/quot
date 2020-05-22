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