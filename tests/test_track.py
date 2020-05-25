#!/usr/bin/env python
"""
test_track.py -- test quot.track

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

# Get radial displacements
from quot.trajUtils import radial_disps 

def get_test_data(filename):
    """
    Get a particular file from the testing data.

    """
    ext = os.path.splitext(filename)[-1]
    if ext == '.csv':
        return pd.read_csv("%s/fixtures/%s" % (package_dir, filename))
    elif ext == ".tif":
        return tifffile.imread("%s/fixtures/%s" % (package_dir, filename))

class TestConnectedComponents(unittest.TestCase):
    """
    Test the function quot.helper.connected_components, which is
    responsible for breaking the tracking subproblem into smaller,
    easier-to-solve subproblems.

    """
    def test_0(self):
        """
        Test on completely unconnected data.

        """
        print("\nTESTING THE PROCEDURE FOR BREAKING TRAJ-LOC PROBLEMS INTO SUBPROBLEMS")
        print("Testing semigraph reduction: scenario 1...")

        # Test data
        AM = np.array([
            [1, 0, 0], 
            [0, 1, 0],
            [0, 0, 1]
        ])

        # Run
        subgraphs, y_groups, x_groups, y_singlets, x_singlets = connected_components(AM)

        # Should be three groups
        assert len(subgraphs) == 3
        assert len(subgraphs) == len(y_groups)
        assert len(subgraphs) == len(x_groups)

        # Should be no singlets
        assert len(y_singlets) == 0
        assert len(x_singlets) == 0

        # Test group assignments
        for i, y_group in enumerate(y_groups):
            assert y_group[0] == x_groups[i][0]

    def test_1(self):
        """
        Test when there are two independent subproblems.

        Traj 0 is near locs 0 and 1
        Traj 1 is near loc 1
        Traj 2 is near loc 2

        """
        print("Testing semigraph reduction: scenario 2...")

        AM = np.array([
            [1, 1, 0], 
            [0, 1, 0], 
            [0, 0, 1]
        ])
        subgraphs, y_groups, x_groups, y_singlets, x_singlets = connected_components(AM)

        # Should be two groups
        assert len(subgraphs) == 2

        # Should be no singlets
        assert len(y_singlets) == 0
        assert len(x_singlets) == 0

        # Test group assignments
        for i, y_group in enumerate(y_groups):
            if y_group[0] == 2:
                assert x_groups[i][0] == 2
            else:
                assert (0 in x_groups[i])
                assert (1 in x_groups[i])

    def test_2(self):
        """
        Test when there is a trajectory singlet.

        Traj 0 is near locs 0 and 1
        Traj 1 is near nothing
        Traj 2 is near loc 2

        """
        print("Testing semigraph reduction: scenario 3...")

        AM = np.array([
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 1]
        ])
        subgraphs, y_groups, x_groups, y_singlets, x_singlets = connected_components(AM)

        # Should be two groups
        assert len(subgraphs) == 2

        # Test that traj 0 is connected to locs 0 and 1
        for i, y_group in enumerate(y_groups):
            if y_group[0] == 0:
                assert len(y_group) == 1
                assert len(x_groups[i]) == 2
                assert 0 in x_groups[i]
                assert 1 in x_groups[i]

        # Should be 1 traj singlet
        assert len(y_singlets) == 1
        assert y_singlets[0] == 1
        assert len(x_singlets) == 0

    def test_3(self):
        """
        A more complex scenario with singlet locs.

        """
        print("Testing semigraph reduction: scenario 4...")

        AM = np.array([
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0]
        ])
        subgraphs, y_groups, x_groups, y_singlets, x_singlets = connected_components(AM)

        # Should be two subgraphs
        assert len(subgraphs) == 2

        # Should be two loc singlets
        assert len(x_singlets) == 2
        assert 1 in x_singlets
        assert 3 in x_singlets

        # Should be no traj singlets
        assert len(y_singlets) == 0

        # Specific tests for reconnection
        for i, S in enumerate(subgraphs):
            assert S.shape == (2, 1)

        for i, y_group in enumerate(y_groups):
            assert len(y_group) == 2
            if 0 in y_group:
                assert len(x_groups[i]) == 1
                assert 0 in x_groups[i]
            if 1 in y_group:
                assert len(x_groups[i]) == 1
                assert 0 in x_groups[i]
            if 2 in y_group:
                assert len(x_groups[i]) == 1
                assert 2 in x_groups[i]
            if 3 in y_group:
                assert len(x_groups[i]) == 1
                assert 2 in x_groups[i]

    def test_4(self):
        """
        A more complex problem.

        Traj 0 is close to locs 3 and 6
        Traj 1 is close to locs 3 and 6
        Traj 2 is close to loc 0
        Traj 3 is close to locs 0 and 2
        Traj 4 is not close to any locs
        Traj 5 is close to loc 2
        Traj 6 is close to loc 4
        No trajs are close to locs 1, 5, or 7.

        """
        print("Testing semigraph reduction: scenario 5...")

        # Test data
        AM = np.array(
            [
                [0, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )

        # Run the algorithm
        subgraphs, y_groups, x_groups, y_singlets, x_singlets = connected_components(AM)

        # Check traj singlets
        assert len(y_singlets) == 1
        assert y_singlets[0] == 4

        # Check loc singlets
        assert len(x_singlets) == 3
        assert 1 in x_singlets
        assert 5 in x_singlets
        assert 7 in x_singlets

        # Check the the right number of subgraphs
        assert len(subgraphs) == 3
        assert len(y_groups) == 3
        assert len(x_groups) == 3

        # Check for specific reconnections
        for i, y_group in enumerate(y_groups):

            # First subgraph
            if 0 in y_group:

                assert subgraphs[i].shape == (2, 2)
                assert (subgraphs[i] != 0).all()
                assert 1 in y_group
                assert 3 in x_groups[i]
                assert 6 in x_groups[i]

            # Second subgraph
            if 2 in y_group:

                assert subgraphs[i].shape == (3, 2)
                assert subgraphs[i][0, 0] != 0
                assert subgraphs[i][0, 1] == 0
                assert subgraphs[i][1, 0] != 0
                assert subgraphs[i][1, 1] != 0
                assert subgraphs[i][2, 0] == 0
                assert subgraphs[i][2, 1] != 0
                assert len(y_group) == 3
                assert len(x_groups[i]) == 2
                assert 3 in y_group
                assert 5 in y_group
                assert 0 in x_groups[i]
                assert 2 in x_groups[i]

            # Third subgraph
            if 6 in y_group:

                assert subgraphs[i].shape == (1, 1)
                assert 4 in x_groups[i]

    def test_5(self):
        """
        Adjacency matrix is all zero - should 
        be all singlets.

        """
        print("Testing semigraph reduction: scenario 6...")

        AM = np.zeros((30, 40), dtype="int64")

        # Run the algorithm
        subgraphs, y_groups, x_groups, y_singlets, x_singlets = connected_components(AM)

        # Tests
        assert len(subgraphs) == 0
        assert len(y_groups) == 0
        assert len(x_groups) == 0
        assert len(y_singlets) == 30
        assert len(x_singlets) == 40

    def test_6(self):
        """
        Adjacency matrix is all ones - should 
        be no singlets and a single subgraph.

        """
        print("Testing semigraph reduction: scenario 7...")

        AM = np.ones((36, 12), dtype="int64")

        # Run the algorithm
        subgraphs, y_groups, x_groups, y_singlets, x_singlets = connected_components(AM)

        # Tests
        assert len(subgraphs) == 1
        assert len(y_groups) == 1
        assert len(x_groups) == 1
        assert len(y_singlets) == 0
        assert len(x_singlets) == 0

class TestTrackingOutput(unittest.TestCase):
    """
    Runs a battery of tests on the function quot.track.track, 
    which is the primary tracking function. These are all 
    tests inspired by past tracking problems. When one of these
    goes haywire, usually something is wrong beneath the hood.

    Note that this class doesn't test for the "correctness" of 
    tracking in the physical sense (localizations produced by 
    the same fluorophore were indeed assigned to the same 
    trajectory), but looks for grosser errors that result from 
    mistakes in the handling of the reconnection problem itself.

        - In the list of Trajectory objects, are there cases
            where the same localization has been assigned to
            multiple Trajectories?

        - Has every localization been assigned to a trajectory?

        - If we track with max_blinks = 0, are there gaps in 
            the resulting trajectories?

        - Is the maximum displacement greater than the search
            radius? 

        - Is the same trajectory assigned to more than one localization
            in a given frame?

        - Do we have duplicate trajectories?

        - Were any trajectory indices skipped?

        - Does all of this hold true when tracking with gaps?

    """
    def setUp(self):
        """
        Load the test data. There are no localizations with 
        intensity less than 0 in this dataset.

        """
        self.locs = get_test_data("localizations_pixel-0.183um_interval-0.01048ms.csv")

        # All tracking methods to test
        self.methods = ['diffusion', 'conservative', 'euclidean']

        # Tracking settings for each method
        self.param_sets_gapless = {
            'diffusion': {'method': 'diffusion', 
                'pixel_size_um': 0.183, 'frame_interval': 0.01048, 
                'search_radius': 2.0, 'max_blinks': 0, 'min_I0': 0.0,
                'debug': True},
            'conservative': {'method': 'conservative',
                'pixel_size_um': 0.183, 'frame_interval': 0.01048,
                'search_radius': 2.0, 'max_blinks': 0, 'min_I0': 0.0,
                'debug': True},
            'euclidean': {'method': 'euclidean',
                'pixel_size_um': 0.183, 'frame_interval': 0.01048,
                'search_radius': 2.0, 'max_blinks': 0, 'min_I0': 0.0,
                'debug': True},
        }

        # Tracking settings for each method, in the presence of gaps
        self.param_sets_gapped = {
            'diffusion': {'method': 'diffusion', 
                'pixel_size_um': 0.183, 'frame_interval': 0.01048, 
                'search_radius': 2.0, 'max_blinks': 1, 'min_I0': 0.0,
                'debug': True},
            'conservative': {'method': 'conservative',
                'pixel_size_um': 0.183, 'frame_interval': 0.01048,
                'search_radius': 2.0, 'max_blinks': 1, 'min_I0': 0.0,
                'debug': True},
            'euclidean': {'method': 'euclidean',
                'pixel_size_um': 0.183, 'frame_interval': 0.01048,
                'search_radius': 2.0, 'max_blinks': 1, 'min_I0': 0.0,
                'debug': True},
        }       

    def test_gapless(self):
        """
        Run a battery of common tests on each of the three methods
        when run without gaps.

        """
        print("\nTESTING ALL TRACKING METHODS WITH GAPLESS SETTINGS...")
        for method in self.methods:

            print("\nMETHOD: %s..." % method)

            # Run tracking. locs is the output dataframe, while 
            # trajs is a list of Trajectory object, used internally
            # in tracking
            locs, trajs = track(self.locs, **self.param_sets_gapless[method])

            # There are no NaNs or infs
            print("testing whether all locs have been assigned to tracks...")
            assert (locs['trajectory'] == -1).sum() == 0
            assert pd.isnull(locs['trajectory']).sum() == 0
            assert (locs['trajectory'] >= 0).all()

            print("\tyes, they have")

            # Test that there are no gaps for these settings
            print("testing whether gaps exist for gapless tracking...")

            for t in trajs:
                t_slice = t.get_slice()

                # >1 localization
                if t_slice.shape[0] > 1:
                    if not ((t_slice[1:,1] - t_slice[:-1,1]) == 1).all():
                        print('\n')
                        print(t_slice)
                        raise AssertionError

            print("\tno gaps found")

            # Test that there are no duplicate localizations in the 
            # set of Trajectories
            print("testing whether locs have been assigned to more than 1 track...")

            loc_assign_counts = np.zeros(len(locs), dtype=np.int64)
            for t in trajs:
                loc_assign_counts[t.indices] += 1

            y = np.unique(loc_assign_counts)

            # # DEBUG
            # for i in y:
            #     print(i, (loc_assign_counts==i).sum())
            # print(np.argmax(loc_assign_counts))

            assert (loc_assign_counts == 1).all()
            print("\tthey haven't been")

            # Test that there are no displacements greater than the search radius
            print("testing whether there are displacements greater than the search radius...")

            # Calculate radial displacements
            locs = radial_disps(locs, first_only=False, pixel_size_um=0.183)
            print("\tmax displacement: %f um" % locs['radial_disp_um'].max())
            assert locs['radial_disp_um'].max() <= self.param_sets_gapless[method]['search_radius']
            print("\tthere are not")

            # Check that no trajectories occur more than twice in the same frame
            print("testing whether the same trajectory occurs more than once per frame...")
            for frame, frame_locs in locs.groupby('frame'):
                traj_indices = np.unique(frame_locs['trajectory'])
                assert len(traj_indices) == frame_locs.shape[0]
            print("\tcouldn't find any")

            # Check that every trajectory index between 0 and the maximum trajectory
            # index actually exists in the output dataframe
            print("testing whether any trajectory indices were skipped...")
            possible_indices = pd.Series(np.arange(0, locs['trajectory'].max()+1))
            assert (possible_indices.isin(locs['trajectory'])).all()
            assert (locs.loc[locs['trajectory']>=0, 'trajectory'].isin(possible_indices)).all()
            print('\tno indices were skipped')

    def test_with_gaps(self):
        """
        Run all methods in the presence of gaps.

        """
        print("\nTESTING ALL TRACKING METHODS WITH 1 GAP TOLERATED...")

        for method in self.methods:

            print("\nMETHOD: %s..." % method)

            # Run tracking. locs is the output dataframe, while 
            # trajs is a list of Trajectory object, used internally
            # in tracking
            locs, trajs = track(self.locs, **self.param_sets_gapped[method])

            # There are no NaNs or infs
            print("testing whether all locs have been assigned to tracks...")
            assert (locs['trajectory'] == -1).sum() == 0
            assert pd.isnull(locs['trajectory']).sum() == 0
            assert (locs['trajectory'] >= 0).all()

            print("\tyes, they have")

            # Test that there are no gaps for these settings
            print("testing that the maximum gap is 1...")

            max_gap = 0
            for t in trajs:
                t_slice = t.get_slice()

                # >1 localization
                if t_slice.shape[0] > 1:

                    max_gap_traj = (t_slice[1:,1] - t_slice[:-1,1]).max()
                    if not ((t_slice[1:,1] - t_slice[:-1,1]) >= 1).all():
                        print('\n')
                        print(t_slice)
                        raise AssertionError
                    if max_gap_traj > max_gap:
                        max_gap = max_gap_traj 

            assert max_gap <= 2
            print("\tmaximum gap found:", max_gap-1)

            # Test that there are no duplicate localizations in the 
            # set of Trajectories
            print("testing whether locs have been assigned to more than 1 track...")

            loc_assign_counts = np.zeros(len(locs), dtype=np.int64)
            for t in trajs:
                loc_assign_counts[t.indices] += 1

            y = np.unique(loc_assign_counts)

            # # DEBUG
            # for i in y:
            #     print(i, (loc_assign_counts==i).sum())
            # print(np.argmax(loc_assign_counts))

            assert (loc_assign_counts == 1).all()
            print("\tthey haven't been")

            # Test that there are no displacements greater than the search radius
            print("testing whether there are displacements greater than the search radius...")

            # Calculate radial displacements
            locs = radial_disps(locs, first_only=False, pixel_size_um=0.183)
            print("\tmax displacement: %f um" % locs['radial_disp_um'].max())
            assert locs['radial_disp_um'].max() <= self.param_sets_gapless[method]['search_radius']
            print("\tthere are not")

            # Check that no trajectories occur more than twice in the same frame
            print("testing whether the same trajectory occurs more than once per frame...")
            for frame, frame_locs in locs.groupby('frame'):
                traj_indices = np.unique(frame_locs['trajectory'])
                assert len(traj_indices) == frame_locs.shape[0]
            print("\tcouldn't find any")

            # Check that every trajectory index between 0 and the maximum trajectory
            # index actually exists in the output dataframe
            print("testing whether any trajectory indices were skipped...")
            possible_indices = pd.Series(np.arange(0, locs['trajectory'].max()+1))
            assert (possible_indices.isin(locs['trajectory'])).all()
            assert (locs.loc[locs['trajectory']>=0, 'trajectory'].isin(possible_indices)).all()
            print('\tno indices were skipped')

        print("TESTING ALL TRACKING METHODS WITH 2 GAPS TOLERATED...")
        for k in self.param_sets_gapped.keys():
            self.param_sets_gapped[k]['max_blinks'] = 2

        for method in self.methods:

            print("\nMETHOD: %s..." % method)

            # Run tracking. locs is the output dataframe, while 
            # trajs is a list of Trajectory object, used internally
            # in tracking
            locs, trajs = track(self.locs, **self.param_sets_gapped[method])

            # There are no NaNs or infs
            print("testing whether all locs have been assigned to tracks...")
            assert (locs['trajectory'] == -1).sum() == 0
            assert pd.isnull(locs['trajectory']).sum() == 0
            assert (locs['trajectory'] >= 0).all()

            print("\tyes, they have")

            # Test that there are no gaps for these settings
            print("testing that the maximum gap is 2...")

            max_gap = 0
            for t in trajs:
                t_slice = t.get_slice()

                # >1 localization
                if t_slice.shape[0] > 1:

                    max_gap_traj = (t_slice[1:,1] - t_slice[:-1,1]).max()
                    if not ((t_slice[1:,1] - t_slice[:-1,1]) >= 1).all():
                        print('\n')
                        print(t_slice)
                        raise AssertionError
                    if max_gap_traj > max_gap:
                        max_gap = max_gap_traj 

            assert max_gap <= 3
            print("\tmaximum gap found:", max_gap-1)

            # Test that there are no duplicate localizations in the 
            # set of Trajectories
            print("testing whether locs have been assigned to more than 1 track...")

            loc_assign_counts = np.zeros(len(locs), dtype=np.int64)
            for t in trajs:
                loc_assign_counts[t.indices] += 1

            y = np.unique(loc_assign_counts)

            # # DEBUG
            # for i in y:
            #     print(i, (loc_assign_counts==i).sum())
            # print(np.argmax(loc_assign_counts))

            assert (loc_assign_counts == 1).all()
            print("\tthey haven't been")

            # Test that there are no displacements greater than the search radius
            print("testing whether there are displacements greater than the search radius...")

            # Calculate radial displacements
            locs = radial_disps(locs, first_only=False, pixel_size_um=0.183)
            print("\tmax displacement: %f um" % locs['radial_disp_um'].max())
            assert locs['radial_disp_um'].max() <= self.param_sets_gapless[method]['search_radius']
            print("\tthere are not")

            # Check that no trajectories occur more than twice in the same frame
            print("testing whether the same trajectory occurs more than once per frame...")
            for frame, frame_locs in locs.groupby('frame'):
                traj_indices = np.unique(frame_locs['trajectory'])
                assert len(traj_indices) == frame_locs.shape[0]
            print("\tcouldn't find any")

            # Check that every trajectory index between 0 and the maximum trajectory
            # index actually exists in the output dataframe
            print("testing whether any trajectory indices were skipped...")
            possible_indices = pd.Series(np.arange(0, locs['trajectory'].max()+1))
            assert (possible_indices.isin(locs['trajectory'])).all()
            assert (locs.loc[locs['trajectory']>=0, 'trajectory'].isin(possible_indices)).all()
            print('\tno indices were skipped')











