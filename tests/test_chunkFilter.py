#!/usr/bin/env python
"""
test_chunkFilter.py -- testing suite for 
quot.chunkFilter

"""
# Absolute root dir for the testing module
import os
package_dir = os.path.dirname(os.path.abspath(__file__))

# Testing utilities
import unittest
from numpy import testing 

# Numeric
import numpy as np 

# Test targets
from quot.read import ImageReader, read_config, save_config 
from quot.chunkFilter import ChunkFilter 

# Other file readers
import tifffile
import nd2reader 

# Hard copy
from copy import copy 

def get_test_data(filename):
    """
    Get a particular file from the testing data.

    """
    ext = os.path.splitext(filename)[-1]
    if ext == '.csv':
        return pd.read_csv("%s/fixtures/%s" % (package_dir, filename))
    elif ext == ".tif":
        return tifffile.imread("%s/fixtures/%s" % (package_dir, filename))

class TestImageReader(unittest.TestCase):
    """
    A variety of tests for the quot.imageReader.ImageReader class.

    """
    def setUp(self):
        """
        Get the testing data.

        """
        self.nd2_path = '%s/fixtures/%s' % (package_dir, "continuous_40ms_633nm-1100mW-aotf-2_region_1_pre.nd2")
        self.tif_path = "%s/fixtures/%s" % (package_dir, "sample_movie.tif")

    def test_tif(self):
        """
        Test the ImageReader's ability to deal with TIF data.

        """
        print("\nTesting ImageReader with TIF data...")
        # Make a reader
        reader = ImageReader(self.tif_path, start=10, stop=190)

        # Should have 200 frames
        print("Can we pull frames of the correct shape out?")
        assert reader.n_frames == 200 

        # Each frames should be 128x128 pixels
        for frame_idx in range(reader.n_frames):
            assert reader.get_frame(frame_idx).shape == (128, 128)
        print("\tyes")

        # Iteration works correctly 
        print("Does enumeration on the reader object work correctly?")
        frames = np.arange(10, 190)
        for frame, frame_image in zip(frames, reader):
            testing.assert_allclose(reader.get_frame(frame), frame_image, atol=1.0e-10)
        print("\tyes")

        # get_frame_range works correctly
        print("Can we pull out a range of frames with get_frame_range()?")
        x = reader.get_frame_range(50, 65)
        y = np.asarray([reader.get_frame(i) for i in range(50, 65)])
        testing.assert_allclose(x, y, atol=1.0e-10)
        print("\tyes")

        # subregions work correctly
        print("Are we accurately recovery subregions?")
        x = reader.get_subregion_range(start=60, stop=100, y0=50, y1=80, x0=20, x1=54)
        y = np.asarray([reader.get_frame(i)[50:80, 20:54] for i in range(60, 100)])
        testing.assert_allclose(x, y, atol=1.0e-10)
        print('\tyes')

        # Projections work correctly
        print("Can we compute projections through the image stack accurately?")
        X = reader.imread()
        mi_proj_a = X.max(axis=0)
        sum_proj_a = (X.astype('float64')).sum(axis=0)
        mi_proj_b = reader.max_int_proj()
        sum_proj_b = reader.sum_proj()
        testing.assert_allclose(mi_proj_a, mi_proj_b, atol=1.0e-10)
        testing.assert_allclose(sum_proj_a, sum_proj_b, atol=1.0e-10)
        print("\tboth max int and sum projections work")

        # min_max works correctly
        print("Pulling out min/max is good?")
        stack_min = X[50:150].min()
        stack_max = X[50:150].max()
        assert (stack_min, stack_max) == reader.min_max(start=50, stop=150)
        print('\tyes')

        # with statements are working
        print("__enter__/__exit__ working correctly?")
        frame = reader.get_frame(15)
        reader.close()
        with ImageReader(self.tif_path, start=15) as f:
            testing.assert_allclose(frame, f.get_frame(f.start), atol=1.0e-10)
        print('\tyes')

    def test_nd2(self):
        """
        Test the ImageReader's ability to deal with ND2 data.

        """
        print("\nTesting ImageReader with ND2 data...")
        # Make a reader
        reader = ImageReader(self.nd2_path, start=10, stop=70)

        # Should have 84 frames
        print("Can we pull frames of the correct shape out?")
        assert reader.n_frames == 84

        # Each frames should be 152x154 pixels
        for frame_idx in range(reader.n_frames):
            assert reader.get_frame(frame_idx).shape == (154, 152)
        print("\tyes")

        # Iteration works correctly 
        print("Does enumeration on the reader object work correctly?")
        frames = np.arange(10, 70)
        for frame, frame_image in zip(frames, reader):
            testing.assert_allclose(reader.get_frame(frame), frame_image, atol=1.0e-10)
        print("\tyes")

        # get_frame_range works correctly
        print("Can we pull out a range of frames with get_frame_range()?")
        x = reader.get_frame_range(50, 65)
        y = np.asarray([reader.get_frame(i) for i in range(50, 65)])
        testing.assert_allclose(x, y, atol=1.0e-10)
        print("\tyes")

        # subregions work correctly
        print("Are we accurately recovery subregions?")
        x = reader.get_subregion_range(start=60, stop=80, y0=50, y1=80, x0=20, x1=54)
        y = np.asarray([reader.get_frame(i)[50:80, 20:54] for i in range(60, 80)])
        testing.assert_allclose(x, y, atol=1.0e-10)
        print('\tyes')

        # Projections work correctly
        print("Can we compute projections through the image stack accurately?")
        X = reader.imread()
        mi_proj_a = X.max(axis=0)
        sum_proj_a = (X.astype('float64')).sum(axis=0)
        mi_proj_b = reader.max_int_proj()
        sum_proj_b = reader.sum_proj()
        testing.assert_allclose(mi_proj_a, mi_proj_b, atol=1.0e-10)
        testing.assert_allclose(sum_proj_a, sum_proj_b, atol=1.0e-10)
        print("\tboth max int and sum projections work")

        # min_max works correctly
        print("Pulling out min/max is good?")
        stack_min = X[50:80,:,:].min()
        stack_max = X[50:80,:,:].max()
        assert (stack_min, stack_max) == reader.min_max(start=50, stop=80)
        print('\tyes')

        # with statements are working
        print("__enter__/__exit__ working correctly?")
        frame = reader.get_frame(15)
        reader.close()
        with ImageReader(self.nd2_path, start=15) as f:
            testing.assert_allclose(frame, f.get_frame(f.start), atol=1.0e-10)
        print('\tyes')

class TestWriteSaveConfig(unittest.TestCase):
    """
    Unit test for quot.read.read_config and quot.read.save_config,
    which are used to read and write TOML configuration files.

    """
    def setUp(self):

        # Specify the path to a config file
        self.config_file = '%s/fixtures/sample_config.toml' % package_dir

    def test_write_save_config(self):

        print("Testing reading/writing from TOML-format config files...")

        # Load configuration settings
        config = read_config(self.config_file)

        # Modify the configuration settings a bit
        config_2 = copy(config)
        config_2['localize']['ridge'] = 0.04
        config_2['localize']['err_model'] = 'model_err_poisson' # not an actual setting

        # Write this to a new file
        save_config("_test_out.toml", config_2)

        # Read from this file
        config_3 = read_config("_test_out.toml")

        # Clean up
        os.remove("_test_out.toml")

        # Assert that they're the same as the ones we wrote from
        for k in config_2.keys():
            assert k in config_3.keys()
            for k2 in config_2[k].keys():
                assert k2 in config_3[k].keys()
                assert config_2[k][k2] == config_3[k][k2]

        print("\tsuccess")

class TestChunkFilter(unittest.TestCase):
    """
    A variety of tests for the quot.chunkFilter.ChunkFilter
    class.

    """
    def setUp(self):
        """
        Get the testing data.

        """
        self.nd2_path = "%s/fixtures/continuous_40ms_633nm-1100mW-aotf-2_region_1_pre.nd2" % package_dir
        self.tif_path = "%s/fixtures/sample_movie.tif" % package_dir 

    def test_chunk_filter(self):
        
        print("Testing filtering capabilities with the ChunkFilter class...")

        print("Are we filtering the correct frames? (chunk_size 20, start 10)")
        F = ChunkFilter(self.nd2_path, start=10, chunk_size=20, method='identity')
        frames = np.arange(10, F.n_frames)
        # for frame_idx, frame_0 in zip(frames, F):
        #     testing.assert_allclose(frame_0, F.get_frame(frame_idx), atol=1.0e-10)
        # print("\tyes")

        # print("Are we filtering the correct frames? (chunk_size 15, start 0, stop 40)")
        # F = ChunkFilter(self.nd2_path, start=0, stop=40, chunk_size=15, method='identity')
        # frames = np.arange(0, 40)
        # for frame_idx, frame_0 in zip(frames, F):
        #     testing.assert_allclose(frame_0, F.get_frame(frame_idx), atol=1.0e-10)
        # print("\tyes")

        for j, i in enumerate(F):
            print('Frame %d' % (j+10))
            print((i == F.get_frame(j+10)).all())
            print('\n')

        F.close()



