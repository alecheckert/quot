#!/usr/bin/env python
"""
test_trajUtils.py -- testing module for quot.trajUtils

"""
# Absolute root dir for the testing module
import os
package_dir = os.path.dirname(os.path.abspath(__file__))

# Testing utilities
import unittest
from numpy import testing 

# Numeric
import numpy as np 

# Dataframes
import pandas as pd

# Test targets
from quot.track import track 
from quot.helper import connected_components
from quot.trajUtils import radial_disps, traj_length

def get_test_data(filename):
    """
    Get a particular file from the testing data.

    """
    ext = os.path.splitext(filename)[-1]
    if ext == '.csv':
        return pd.read_csv("%s/fixtures/%s" % (package_dir, filename))
    elif ext == ".tif":
        return tifffile.imread("%s/fixtures/%s" % (package_dir, filename))

class TestTrajLength(unittest.TestCase):
    """
    Test the function quot.trajUtils.traj_length, which assigns
    a length to each trajectory in a dataset.

    """
    def setUp(self):
        """
        Load the testing data.

        """
        self.locs = get_test_data("localizations_pixel-0.183um_interval-0.01048ms.csv")

        # Tracking settings
        self.track_params = {'method': 'euclidean',
                'pixel_size_um': 0.183, 'frame_interval': 0.01048,
                'search_radius': 2.0, 'max_blinks': 0, 'min_I0': 0.0,
                'debug': False}

    def test_traj_length(self):

        print("\nTesting trajectory length calculations...")

        # Track the localizations
        locs = track(self.locs, **self.track_params)

        # Compute trajectory lengths
        locs = traj_length(locs)

        # Output exists
        assert 'traj_len' in locs.columns

        # Lengths make sense
        print("Do trajectory lengths make physical sense?")
        assert (locs['traj_len']>=1).all()
        print("\tyes")

        # Lengths match the size of each trajectory gropu
        print("Do trajectory lengths match the actual number of instances of that trajectory " \
            "in the set of localizations?")
        x = np.asarray(locs[['trajectory', 'traj_len']])
        unique_traj_indices = np.unique(x[:,0])
        for j in unique_traj_indices:
            assert (x[:,0]==j).sum() == x[x[:,0]==j,1][0]
        print("\tthey do")

class TestRadialDisps(unittest.TestCase):
    """
    Test the function quot.trajUtils.radial_disps, which calculates
    the 2D radial displacement of each loc-loc connection present
    in a set of trajectories.

    """
    def setUp(self):
        """
        Load the testing data.

        """
        self.locs = get_test_data("localizations_pixel-0.183um_interval-0.01048ms.csv")

        # Tracking settings
        self.track_params = {'method': 'euclidean',
                'pixel_size_um': 0.183, 'frame_interval': 0.01048,
                'search_radius': 2.0, 'max_blinks': 0, 'min_I0': 0.0,
                'debug': False}

    def test_radial_disps(self):

        print("\nTesting radial displacement calculations...")

        # Track the localizations
        locs = track(self.locs, **self.track_params)

        # Compute radial displacements
        locs = radial_disps(locs, pixel_size_um=self.track_params['pixel_size_um'],
            first_only=False)

        # Output exists
        assert 'radial_disp_um' in locs.columns

        # All jumps have been computed
        print("Testing whether all jumps have been assigned displacements...")
        for i, traj in locs.groupby('trajectory'):

            # The last localization should always have NaN in its 'radial_disp_um'
            # column, since there is no localization afterward to connect to 
            assert pd.isnull(traj.iloc[-1].radial_disp_um)

            # Everything else should have been assigned a float value
            if traj.shape[0] > 1:
                assert (~pd.isnull(traj.iloc[:-1]['radial_disp_um'])).all()

        print('\tthey have')

        # Jumps are correct
        print("Testing whether jumps are numerically correct...")
        locs = locs[:10000]
        for i, traj in locs.groupby('trajectory'):
            for j in range(len(traj)-1):
                y0, x0 = traj.iloc[j][['y', 'x']]
                y1, x1 = traj.iloc[j+1][['y', 'x']]
                R = np.sqrt(((y0-y1)**2 + (x0-x1)**2)) * self.track_params['pixel_size_um']
                assert np.abs(R-traj.iloc[j]['radial_disp_um']) <= 1.0e-10
        print('\tthey are')

    def test_radial_disps_first_only(self):

        print("\nTesting radial displacement calculations with the first_only flag...")
        # Track the localizations
        locs = track(self.locs, **self.track_params)

        # Compute radial displacements
        locs = radial_disps(locs, pixel_size_um=self.track_params['pixel_size_um'],
            first_only=False)

        # Compute radial displacements
        locs = radial_disps(locs, pixel_size_um=self.track_params['pixel_size_um'],
            first_only=True)

        # All jumps have been computed
        print("Testing whether only the first jump of each trajectory has been assigned a displacement...")
        for i, traj in locs.groupby('trajectory'):

            if len(traj) == 1:
                assert (pd.isnull(traj.iloc[0].radial_disp_um))

            else:

                # First disp should not be NaN
                assert (not pd.isnull(traj.iloc[0].radial_disp_um))

                # The rest should be NaN
                assert pd.isnull(traj.iloc[1:]['radial_disp_um']).all()
        print("\tyes, it's the only one")









