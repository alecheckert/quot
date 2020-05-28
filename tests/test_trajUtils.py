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
from quot.trajUtils import radial_disps, traj_length, radial_disp_histograms
from quot.plot import plotRadialDisps

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

class TestRadialDispHistograms(unittest.TestCase):
    """
    tests:
        quot.trajUtils.radial_disp_histograms

    This function takes a set of trajectories as a DataFrame and 
    returns a set of histograms with the radial displacements 
    binned across the entire dataset.

    The approach in this test is to simulate a set of trajectories
    with defined displacements, format as a DataFrame, then bin
    with radial_disp_histograms(). We can then check against the 
    original distribution.

    """
    def test_radial_disp_histograms_gapless(self):

        print("Testing whether we can accurately compute radial displacement histograms...")

        # The number of trajectories to simulate
        n_trajs = 3000

        # The average trajectory length to simulate
        mean_traj_len = 2.0 

        # Choose a bunch of trajectories with defined lengths
        traj_lens = np.random.poisson(mean_traj_len, n_trajs) + 1
        n_disps = (traj_lens-1).sum()

        # Keep track of trajectories as a list of 2D ndarray
        all_trajs = []

        # Keep track of every displacement
        xy_disps = []

        # Simulate each trajectory
        for traj_idx in range(n_trajs):

            # The length of this trajectory
            traj_len = traj_lens[traj_idx]

            # Create this trajectory as a 2D ndarray. Column 0 is 
            # the frame index, column 1 is the trajectory index, 
            # and columns 2 and 3 are the X and Y positions
            traj = np.zeros((traj_len, 4), dtype='float64')

            # Assign trajectory index
            traj[:,1] = traj_idx

            # Assign frame indices
            traj[:,0] = np.arange(traj_len)

            # If the trajectory is length 1, just choose a random position
            if traj_len == 1:
                traj[0,2:4] = np.random.uniform(0, 10.0, size=2)

            # Otherwise simulate a Brownian motion
            else:
                # Broadcast initial random position over all 
                # subsequent positions
                traj[:,2:4] = np.random.uniform(0, 10.0, size=2)

                # Use the Euler-Maruyama scheme to simulate Brownian motion, keeping
                # track of the individual displacements
                traj_disps = np.random.normal(scale=1.0, loc=0.0, size=(traj_len-1, 2))
                xy_disps.append(traj_disps.copy())
                traj_disps = traj_disps.cumsum(axis=0)

                # Add the initial position (offset) and the subsequent displacements
                traj[1:,2:4] = traj[1:,2:4] + traj_disps 

                # Append to the growing heap of trajectories
                all_trajs.append(traj)

        # Concatenate into a single ndarray
        traj_array = np.concatenate(all_trajs, axis=0)

        # Concatenate displacements into a single array
        all_single_disps = np.concatenate(xy_disps, axis=0)

        # Format as dataframe
        df = pd.DataFrame(traj_array, columns=['frame', 'trajectory', 'y', 'x'])
        df['frame'] = df['frame'].astype('int64')
        df['trajectory'] = df['trajectory'].astype('int64')

        # Manually find the radial displacement histogram
        xy_disps = np.concatenate(xy_disps, axis=0)
        r_disps = np.sqrt((xy_disps**2).sum(axis=1))
        bin_edges = np.linspace(0.0, 5.0, 5001)
        H_man, _edges = np.histogram(r_disps, bins=bin_edges)

        # Do the same thing with quot.trajUtils.radial_disp_histograms
        H_rdh, bin_edges_rdh = radial_disp_histograms(
            df, n_intervals=1, pixel_size_um=1.0, first_only=False,
            n_gaps=0, bin_size=0.001, max_disp=5.0)

        # The histogram will have one extra index
        assert len(H_rdh.shape) == 2
        H_rdh = H_rdh[0,:]

        # Assert equality
        testing.assert_allclose(bin_edges, bin_edges_rdh, atol=1.0e-10)
        testing.assert_allclose(H_man, H_rdh, atol=1.0e-10)

        print("\tsuccess (ungapped displacements)")

        print("Seeing if we can accurately calculate displacements over multiple frame intervals...")
        del xy_disps 
        del r_disps 

        # Manually compute all displacements at 2 and 3 frame intervals 
        # from the set of trajectories
        xy_disps_2 = []  # List of 2D ndarray
        xy_disps_3 = []  # List of 2D ndarray

        for t in all_trajs:

            # Only length-3 and greater have displacements over 2 frame intervals
            if t.shape[0] > 2:
                disps = t[2:,2:4] - t[:-2,2:4]
                xy_disps_2.append(disps.copy())

            # Only length-4 and greater have displacements over 3 frame intervals
            if t.shape[0] > 3:
                disps = t[3:,2:4] - t[:-3,2:4]
                xy_disps_3.append(disps.copy())               

        # Concatenate the displacements into a single array
        xy_disps_2 = np.concatenate(xy_disps_2, axis=0)
        xy_disps_3 = np.concatenate(xy_disps_3, axis=0)

        # Get radial displacements
        r_disps_2 = np.sqrt((xy_disps_2**2).sum(axis=1))
        r_disps_3 = np.sqrt((xy_disps_3**2).sum(axis=1))

        # Accumulate into histograms
        H_man_2, _edges = np.histogram(r_disps_2, bins=bin_edges)
        H_man_3, _edges = np.histogram(r_disps_3, bins=bin_edges)

        # Compute the same with trajUtils.radial_disp_histograms
        H_rdh, bin_edges_rdh = radial_disp_histograms(df,
            n_intervals=3, pixel_size_um=1.0, first_only=False,
            n_gaps=0, bin_size=0.001, max_disp=5.0)

        # Compare numerically 
        testing.assert_allclose(H_rdh[1,:], H_man_2, atol=1.0e-10)
        testing.assert_allclose(H_rdh[2,:], H_man_3, atol=1.0e-10)

        print("\tsuccess (ungapped displacements)")

    def test_radial_disp_histograms_gapped_small(self):
        """
        A small test case.

        """
        # Frame, trajectory, y, x
        X = np.array([
            0,  0,  5.0,    7.7,
            1,  0,  4.5,    7.7,
            3,  0,  4.5,    7.8,
        ])

    def test_radial_disp_histograms_gapped(self):
        """
        Similar test, but with gapped trajectories instead of gapless. 
        Uses fairly small data where we can calculate all of the gaps 
        explicitly.

        """
        print("Testing that we can accurately retrieve displacements in the presence of gaps...")
        # Number of trajectories in the simulation
        n_trajs = 5000

        # Average trajectory length
        mean_traj_len = 3.0

        # Probability of a gap
        gap_prob = 0.25

        # Keep track of all trajectories
        all_trajs = []

        # Keep track of all displacements over 1, 2, and 3 frame
        # intervals
        xy_disps_1 = []
        xy_disps_2 = []
        xy_disps_3 = []

        # Simulate the trajectory lengths
        traj_lens = np.random.poisson(mean_traj_len, size=n_trajs)

        for traj_idx in range(n_trajs):

            # Get the length of this trajectory
            traj_len = traj_lens[traj_idx]

            # Keep this trajectory as a 2D ndarray: column 0 is the
            # frame index, column 1 is the trajectory index, and 
            # columns 2 and 3 are the x and y positions
            traj = np.zeros((traj_len, 4), dtype='float64')

            # Record the trajectory index
            traj[:,1] = traj_idx 

            # Start with a random position
            traj[:,2:4] = np.random.uniform(0, 10.0, size=2)

            # Current frame index
            frame_idx = 1

            # Current position index in this trajectory
            c = 1

            while c < traj_len:
                if np.random.random() < gap_prob:
                    pass 
                else:
                    disp = np.random.normal(loc=0.0, scale=0.5, size=2)
                    traj[c:,2:4] = traj[c:,2:4] + disp 
                    traj[c,0] = frame_idx 
                    c += 1
                frame_idx += 1

            all_trajs.append(traj.copy())

        # Concatenate all trajectories
        traj_array = np.concatenate(all_trajs, axis=0)

        # Format as dataframe
        df = pd.DataFrame(traj_array, columns=['frame', 'trajectory', 'x', 'y'])
        df['frame'] = df['frame'].astype('int64')
        df['trajectory'] = df['trajectory'].astype('int64')

        # Keep track of all displacements over 1, 2, and 3 frame
        # intervals
        xy_disps_1 = []
        xy_disps_2 = []
        xy_disps_3 = []

        # Accumulate displacements the hard, explicit way
        for t in all_trajs:

            # Look for displacements of 1 frame interval
            if t.shape[0] > 1:
                for i in range(t.shape[0]-1):
                    for j in range(i+1, t.shape[0]):
                        if t[j,0] - t[i,0] == 1:
                            xy_disps_1.append(t[j,2:4] - t[i,2:4])

            # Look for displacements of 2 frame intervals
            if t.shape[0] > 1:
                for i in range(t.shape[0]-1):
                    for j in range(i+1, t.shape[0]):
                        if t[j,0] - t[i,0] == 2:
                            xy_disps_2.append(t[j,2:4] - t[i,2:4])

            # Look for displacements of 3 frame intervals
            if t.shape[0] > 1:
                for i in range(t.shape[0]-1):
                    for j in range(i+1, t.shape[0]):
                        if t[j,0] - t[i,0] == 3:
                            xy_disps_3.append(t[j,2:4] - t[i,2:4])                   

        # Concatenate into single arrays
        xy_disps_1 = np.asarray(xy_disps_1)
        xy_disps_2 = np.asarray(xy_disps_2)
        xy_disps_3 = np.asarray(xy_disps_3)

        # Compute radial displacements
        r_disps_1 = np.sqrt((xy_disps_1**2).sum(axis=1))
        r_disps_2 = np.sqrt((xy_disps_2**2).sum(axis=1))
        r_disps_3 = np.sqrt((xy_disps_3**2).sum(axis=1))

        # Bin 
        bin_edges = np.linspace(0.0, 5.0, 5001)
        H1, _edges = np.histogram(r_disps_1, bins=bin_edges)
        H2, _edges = np.histogram(r_disps_2, bins=bin_edges)
        H3, _edges = np.histogram(r_disps_3, bins=bin_edges)

        # Do the same thing with radial_disp_histograms()
        H_rdh, bin_edges_rdh = radial_disp_histograms(
            df, n_intervals=3, pixel_size_um=1.0, first_only=False,
            n_gaps=2, bin_size=0.001, max_disp=5.0)

        # Assert equality between matching histograms
        testing.assert_allclose(H_rdh[0,:], H1, atol=1.0e-10)
        testing.assert_allclose(H_rdh[1,:], H2, atol=1.0e-10)
        testing.assert_allclose(H_rdh[2,:], H3, atol=1.0e-10)

        print("\tsuccess")

















