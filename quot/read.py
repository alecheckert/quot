#!/usr/bin/env python
"""
quot/read.py -- image file readers

"""
# ndarrays
import numpy as np 

# File paths
import os

# Read config files in TOML format
import toml 

# Underlying file readers for each file format
import tifffile
from nd2reader import ND2Reader 

def read_config(path):
    """
    Read a config file.

    args
        path    :   str

    returns
        dict

    """
    assert os.path.isfile(path), \
        "read_config: path %s does not exist" % path
    return toml.load(path)

def save_config(path, config):
    """
    Save data in a dict to a TOML file.

    args
        path    :   str
        config  :   dict

    """
    with open(path, 'w') as o:
        toml.dump(config, o)

class ImageReader(object):
    """
    An image file reader that aims to provide a single API for
    several image types, relying on other packages to read 
    individual file types. Focuses especially on methods to 
    return temporal frames.

    *start* and *stop* are the frame limits when iterating over
    the ImageReader. They do not prohibit the user from accessing
    frames outside this range with ImageReader.get_frame() or 
    related methods.

    init
        path        :   str
        start       :   int, limits of iteration. If not set
                        defaults to 0.
        stop        :   int, limits of iteration. If not set
                        defaults to the last frame.
        subregion   :   keys y0, y1, x0, x1 (all int), the 
                        subregion to use for iteration if 
                        desired

    """
    def __init__(self, path, start=None, stop=None, **subregion):

        assert os.path.isfile(path), "ImageReader.__init__: " \
            "path %s does not exist" % path 
        self.path = path

        if '.nd2' in path:
            self.ext = '.nd2'
            self._reader = ND2Reader(path)
        elif ('.tif' in path) or ('.tiff' in path):
            self.ext = '.tif'
            self._reader = tifffile.TiffFile(path)
        #elif ('.czi' in path):
        #    self.ext = '.czi'
        #    self._reader = CziFile(path)
        else:
            raise RuntimeError("Image format %s not recognized" % \
                os.path.splitext(path)[1])

        # Record movie shape
        self.n_frames, self.height, self.width = self.shape 

        # Set defaults for iteration 
        if start is None:
            start = 0
        if stop is None:
            stop = self.n_frames 
        self.start = start 
        self.stop = stop 

    def __iter__(self):
        self._c = self.start 
        return self 

    def __next__(self):
        if self._c < self.stop:
            self._c += 1
            return self.get_frame(self._c-1)
        else:
            raise StopIteration

    def __enter__(self):
        return self 

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        """ Close the filestream. """
        self._reader.close()

    def _frame_valid(self, frame_index):
        """
        True of the corresponding frame exists.

        """
        return (frame_index>=0) and (frame_index<self.n_frames)

    def _frame_range_valid(self, start, stop):
        """
        True if all frames in the range exist.

        """
        return start<=self.n_frames and \
            stop<=self.n_frames and \
            (start<stop)

    def _subregion_valid(self, **subregion):
        """
        True if the subregion lies within the limits of 
        the movie.

        """
        return subregion.get('y0',0)>=0 and \
            subregion.get('x0',0)>=0 and \
            subregion.get('y1',self.height)<=self.height and \
            subregion.get('x1',self.width)<=self.width

    def _process_frame_range(self, **kwargs):
        """
        Truncate a frame range into the permissible limits 
        for this movie.

        kwargs
            start, stop     :   int, the start and stop frame
                                indices for the range. Added
                                if they don't exist.

        returns
            dict, modified kwargs

        """
        kwargs['start'] = max(0, kwargs.get('start', 0))
        kwargs['stop'] = min(self.n_frames, kwargs.get('stop', self.n_frames))
        return kwargs 

    def _process_subregion(self, **kwargs):
        """
        Truncate a subregion into the permissible limits for this
        movie.

        kwargs
            y0, y1, x0, x1  :   int, subregion limits

        returns
            dict, modified subregion kwargs

        """
        kwargs['y0'] = max(kwargs.get('y0', 0), 0)
        kwargs['x0'] = max(kwargs.get('x0', 0), 0)
        kwargs['y1'] = min(kwargs.get('y1', self.height), self.height)
        kwargs['x1'] = min(kwargs.get('x1', self.width), self.width)
        return kwargs

    def _process_subregion_range(self, **kwargs):
        """
        Similar to _process_subregion(), but also specifies the 
        *start* and *stop* kwargs for multiple frames.

        args
            y0, y1, x0, x1, start, stop :   int, subregion limits

        returns
            dict of keys y0, y1, x0, x1, start, stop

        """
        return self._process_frame_range(**self._process_subregion(**kwargs))

    @property 
    def dtype(self):
        """
        Return the dtype for the arrays produced by 
        self.get_frame() and related methods.

        returns
            str

        """
        if hasattr(self, "_dtype"):
            return self._dtype

        self._dtype = self.get_frame(0).dtype
        return self._dtype 

    @property 
    def shape(self):
        """
        Return the number of frames, height, and width of the 
        image movie.

        returns
            (int, int, int)

        """
        if hasattr(self, "_shape"):
            return self._shape 

        if self.ext == ".nd2":
            self._shape = (
                self._reader.metadata["total_images_per_channel"],
                self._reader.metadata["height"],
                self._reader.metadata["width"],
            )
        elif self.ext == ".tif":
            self._shape = (
                len(self._reader.pages),
                *self._reader.pages[0].shape, 
            )
        return self._shape 

    def get_frame(self, frame_index, c=0):
        """
        Return a single frame from the file.

        args
            frame_index     :   int
            c               :   int, channel index. Currently only 
                                implemented for ND2 files.

        returns
            2D ndarray (YX)

        """
        assert self._frame_valid(frame_index)
        if (self.ext == ".nd2") and (c == 0):
            return self._reader.get_frame_2D(t=frame_index)
        elif (self.ext == ".nd2") and (c > 0):
            return self._reader.get_frame_2D(t=frame_index, c=c)
        elif (self.ext == ".tif"):
            return self._reader.pages[frame_index].asarray()

    def get_frame_range(self, start, stop):
        """
        Get a contiguous range of frames as a 3D stack.

        args
            start, stop     :   int

        returns
            3D ndarray (TYX)

        """
        kwargs = self._process_frame_range(start=start, stop=stop)
        n = kwargs['stop'] - kwargs['start']
        result = np.empty((n, self.height, self.width), dtype=self.dtype)
        for i, frame_index in enumerate(range(kwargs['start'], kwargs['stop'])):
            result[i,:,:] = self.get_frame(frame_index)
        return result 

    def get_subregion(self, frame_index, **subregion):
        """
        Get a subregion of a single frame.

        args
            frame_index     :   int
            subregion       :   keys y0, y1, x0, x1,
                                the subregion limits

        returns
            3D ndarray (TYX)

        """
        subregion = self._process_subregion(**subregion)
        return self.get_frame(frame_index)[
            subregion['y0']:subregion['y1'],
            subregion['x0']:subregion['x1'],
        ]

    def get_subregion_range(self, **kwargs):
        """
        Get a subregion of multiple frames.

        kwargs
            y0, y1, x0, x1, start, stop :   int, the 
                subregion limits

        returns
            3D ndarray (TYX)

        """
        kwargs = self._process_subregion_range(**kwargs)
        T = kwargs['stop'] - kwargs['start']
        N = kwargs['y1'] - kwargs['y0']
        M = kwargs['x1'] - kwargs['x0']
        result = np.empty((T, N, M), dtype=self.dtype)
        for j, frame_index in enumerate(range(kwargs['start'], kwargs['stop'])):
            try:
                result[j,:,:] = self.get_frame(frame_index)[
                    kwargs['y0']:kwargs['y1'],
                    kwargs['x0']:kwargs['x1']
                ]
            except:
                result[j,:,:] = 0 
        return result 

    def max_int_proj(self, **kwargs):
        """
        Take a maximum intensity projection of the movie.

        kwargs
            start, stop :   int, the first and last frames to use

        returns
            2D ndarray (YX)

        """
        kwargs = self._process_frame_range(**kwargs)
        I = self.get_frame(kwargs['start'])
        for i in range(kwargs['start']+1, kwargs['stop']):
            I = np.maximum(I, self.get_frame(i))
        return I 

    def sum_proj(self, **kwargs):
        """
        Take a sum projection of the movie.

        kwargs
            start, stop :   int, the first and last frames to use

        returns
            2D ndarray (YX), dtype float64

        """
        kwargs = self._process_frame_range(**kwargs)
        I = self.get_frame(kwargs['start']).astype(np.float64)
        for i in range(kwargs['start']+1, kwargs['stop']):
            I += self.get_frame(i).astype(np.float64)
        return I 

    def min_max(self, **kwargs):
        """
        Get the minimum and maximum pixel intensities for a 
        frame range.

        kwargs
            start, stop :   int, the first and last frames to use

        returns
            (int min, int max)

        """
        kwargs = self._process_frame_range(**kwargs)
        cmax, cmin = 0, 1e10
        for i in range(kwargs['start'], kwargs['stop']):
            I = self.get_frame(i)
            cmax = max(cmax, I.max())
            cmin = min(cmin, I.min())
        return cmin, cmax 

    def imread(self):
        """
        Read the entire stack into memory.

        returns
            3D ndarray (TYX)

        """
        if self.ext == '.nd2':
            return self.get_frame_range(0, self.n_frames)
        elif self.ext == '.tif':
            return self._reader.asarray()

    def imsave(self, path, **kwargs):
        """
        Save part of the movie to a TIF.

        args
            path    :   str, out TIF
            kwargs  :   subregion kwargs

        """
        tifffile.imsave(path, 
            self.get_subregion_range(**kwargs).astype('uint16'))

