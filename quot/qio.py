"""
qio.py -- image file reading utilities

"""
# Numerics
import numpy as np 

# File path manipulations
import os

# Config files
import yaml 

# Underlying readers for various image
# file formats
import tifffile
from nd2reader import ND2Reader 
from czifile import CziFile 

def save_config(out_yaml, in_dict):
    """
    Save data in a dictionary to a YAML 
    file.

    args
    ----
        out_yaml : str, .yaml file path
        in_dict : dict, settings to be saved

    """
    with open(out_yaml, 'w') as o:
        o.write(yaml.dump(in_dict))

def read_config(path):
    """
    Read config settings from a .yaml file.

    args
    ----
        path : str, path to .yaml file

    returns
    -------
        dict, the config settings for detection

    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class ImageFileReader(object):
    def __init__(self, file_name, start_iter=0,
        stop_iter=None):
        self.file_name = file_name 

        if '.nd2' in file_name:
            self.ext = 'nd2'
            self.reader = ND2Reader(file_name)
        elif ('.tif' in file_name) or ('.tiff' in file_name):
            self.ext = 'tif'
            self.reader = tifffile.TiffFile(file_name)
        elif ('.czi' in file_name):
            self.ext = 'czi'
            self.reader = CziFile(file_name)
        else:
            raise RuntimeError("Image format %s " \
                "not recognized" % \
                os.path.splitext(file_name)[1])

        # Record the shape of the movie
        self.n_frames, self.height, self.width = \
            self.get_shape()

        # Set the defaults for iterating over this
        # object
        self.start_iter = start_iter 
        self.stop_iter = stop_iter
        if self.stop_iter is None:
            self.stop_iter = self.n_frames        

    def __iter__(self):
        self.c_idx = self.start_iter 
        return self 

    def __next__(self):
        if self.c_idx < self.stop_iter:
            self.c_idx += 1
            return self.get_frame(self.c_idx-1)
        else:
            raise StopIteration

    @property 
    def shape(self):
        return self.get_shape()

    @property 
    def dtype(self):
        return self.get_frame(0).dtype

    def _frame_valid(self, frame_idx):
        return (frame_idx < self.n_frames)

    def _frame_range_valid(self, frame_0, frame_1):
        return (frame_0 <= self.n_frames) and \
            (frame_1 <= self.n_frames) and \
            (frame_0 < frame_1)

    def _process_frame_range(self, start_frame, stop_frame):
        if start_frame is None:
            start_frame = 0
        if stop_frame is None:
            stop_frame = self.n_frames 
        assert self._frame_range_valid(start_frame, stop_frame)
        return start_frame, stop_frame 

    def _subregion_valid(self, y0, y1, x0, x1):
        return (y0<=self.height) and (y1<=self.height) \
            and (x0<=self.width) and (x1<=self.width) \
            and (y0 < y1) and (x0 < x1)

    def _process_subregion(self, y0, y1, x0, x1):
        if y0 is None:
            y0 = 0 
        if x0 is None:
            x0 = 0
        if y1 is None:
            y1 = self.height
        if x1 is None:
            x1 = self.width 
        assert self._subregion_valid(y0, y1, x0, x1)
        return y0, y1, x0, x1

    def close(self):
        self.reader.close()

    def get_shape(self):
        """
        returns
        -------
            (int n_frames,
             int height,
             int width)
        
        """
        if self.ext == 'nd2':
            H = self.reader.metadata['height']
            W = self.reader.metadata['width']
            T = self.reader.metadata['total_images_per_channel']
        elif self.ext == 'tif' or self.ext == 'czi':
            H, W = self.reader.pages[0].shape 
            T = len(self.reader.pages)
        return (T, H, W)

    def get_frame(self, frame_idx):
        """
        args
        ----
            frame_idx :  int

        returns
        -------
            2D ndarray (YX), dtype uint16

        """
        assert self._frame_valid(frame_idx)
        if self.ext == 'nd2':
            return self.reader.get_frame_2D(t=frame_idx)
        elif self.ext == 'tif' or self.ext == 'czi':
            return self.reader.pages[frame_idx].asarray()

    def get_frames(self, frame_indices):
        """
        args
        ----
            frame_indices : list of int

        returns
        -------
            3D ndarray (TYX), dtype uint16

        """
        n = len(frame_indices)
        result = np.empty((n, self.height, self.width),
            dtype='uint16')
        for i, idx in enumerate(frame_indices):
            result[i,:,:] = self.get_frame(idx)
        return result

    def get_frame_range(self, start_frame, stop_frame):
        """
        args
        ----
            start_frame, stop_frame : int, frame indices

        returns
        -------
            3D ndarray (TYX), dtype uint16

        """
        assert self._frame_range_valid(start_frame, stop_frame)
        n = stop_frame-start_frame
        result = np.empty((n, self.height, self.width),
            dtype='uint16')
        for j, idx in enumerate(range(start_frame, stop_frame)):
            result[j,:,:] = self.get_frame(idx)
        return result 

    def sum_proj(self, start_frame=None, stop_frame=None):
        """
        Return the sum intensity projection 
        for the full movie.

        args
        ----
            start_frame, stop_frame : int, limits
                for the frames to use, if 
                desired

        returns
        -------
            2D ndarray (YX), dtype uint16

        """
        start_frame, stop_frame = self._process_frame_range(
            start_frame, stop_frame)
        result = np.zeros((self.height, self.width),
            dtype='float64')
        for frame_idx in range(start_frame, stop_frame):
            result = result + self.get_frame(frame_idx)
        return result 

    def max_int_proj(self, start_frame=None, stop_frame=None):
        """
        Return the maximum intensity projection 
        for the full movie.

        args
        ----
            start_frame, stop_frame : int, limits
                for the frames to use, if 
                desired

        returns
        -------
            2D ndarray (YX), dtype uint16

        """
        start_frame, stop_frame = self._process_frame_range(
            start_frame, stop_frame)
        result = np.zeros((self.height, self.width),
            dtype='float64')
        for frame_idx in range(start_frame, stop_frame):
            result = np.maximum(result,
                self.get_frame(frame_idx))
        return result 

    def min_max(self, start_frame=None, stop_frame=None):
        """
        args
        ----
            start_frame, stop_frame : int

        returns
        -------
            (int, int), the minimum and maximum pixel
                intensities in the stack

        """
        start_frame, stop_frame = self._process_frame_range(
            start_frame, stop_frame)
        cmax, cmin = 0, 0
        for frame_idx in range(start_frame, stop_frame):
            img = self.get_frame(frame_idx)
            cmax = max([img.max(), cmax])
            cmin = min([img.min(), cmin])
        return cmin, cmax

    def frame_subregion(self, frame_idx, y0=None, y1=None,
        x0=None, x1=None):
        """
        Return a subregion of a single frame.

        args
        ----
            frame_idx : int, the frame index
            y0, y1 : int, y limits of rectangular 
                subregion
            x0, x1 : int, x limits of rectangular
                subregion

        returns
        -------
            2D ndarray (YX), dtype uint16, the
                desired subregion

        """
        y0, y1, x0, x1 = self._process_subregion(y0, y1, x0, x1)
        return self.get_frame(frame_idx)[y0:y1,x0:x1]

    def subregion(self, y0=None, y1=None, x0=None, x1=None,
        start_frame=None, stop_frame=None):
        """
        Return a subregion of the movie.

        args
        ----
            y0, y1 : int, the y limits of the rectangular
                subregion
            x0, x1 : int, the x limits of the rectangular
                subregion
            start_frame, stop_frame : int, the temporal
                limits on the subregion

        returns
        -------
            3D ndarray (TYX), dtype uint16

        """
        y0, y1, x0, x1 = self._process_subregion(y0, y1, x0, x1)
        start_frame, stop_frame = self._process_frame_range(
            start_frame, stop_frame)
        T = stop_frame - start_frame
        N = y1 - y0
        M = x1 - x0 
        result = np.empty((T, N, M), dtype='uint16')
        for j, frame_idx in enumerate(range(start_frame, stop_frame)):
            result[j,:,:] = self.get_frame(frame_idx)[y0:y1,x0:x1]
        return result 

    def imsave(self, out_tif, **subregion_kwargs):
        """
        Save a portion of the image to a TIF 
        file.

        args
        ----
            out_tif : str
            subregion_kwargs : to self.subregion(),
                which rectangular subregion to save

        returns
        -------
            None

        """
        tifffile.imsave(
            out_tif,
            self.subregion(**subregion_kwargs)
        )






