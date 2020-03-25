"""
image_filter.py -- apply various filtering utilities
to an image

"""
# Numerics
import numpy as np 

# Computer vision utilities
from scipy import ndimage as ndi 

# File path manipulations
import os 

# Custom utilities
from quot.utils import set_neg_to_zero 

class ImageFilterer(object):
    """
    A filterer for a raw image filestream. 

    init
    ----
        ImageFileReader : quot.qio.ImageFileReader
            object, the filestream reader

        method : str, key to one of the methods in
            FILTER_METHODS

        block_size : int, the temporal size of the 
            image block to use for filtering

        **method_kwargs : to the method


    """
    def __init__(self, ImageFileReader, method=None, 
        block_size=200, start_iter=0, stop_iter=None,
        **method_kwargs):
        self.ImageFileReader = ImageFileReader 
        self.method = method 
        self.block_size = block_size 
        self.method_kwargs = method_kwargs

        # If the user passes a str for ImageFileReader,
        # see if we can interpret it as a file path and
        # read with qio.ImageFileReader
        if isinstance(ImageFileReader, str) and \
            os.path.isfile(ImageFileReader):
            self.ImageFileReader = io.ImageFileReader(
                self.ImageFileReader)

        # If no method passed, default to raw movie
        if self.method is None:
            self.method_f = lambda x: x

        # Otherwise, find desired filtering method
        else:
            try:
                self.method_f = FILTER_METHODS[self.method]
            except KeyError:
                raise RuntimeError("ImageFilterer.__init__: " \
                    "method %s not found; available methods " \
                    "are %s" % (method, ", ".join(FILTER_METHODS.keys())))

        # Get the movie length and shape
        self.n_frames, self.height, self.width = \
            self.ImageFileReader.get_shape()

        # Set the defaults for iteration
        self.start_iter = start_iter
        self.stop_iter = stop_iter 
        if self.start_iter is None:
            self.start_iter = 0
        if self.stop_iter is None:
            self.stop_iter = self.n_frames

    def __iter__(self):
        # Global frame index
        self.frame_idx = self.start_iter

        # Load the first block
        self._update_block(self.frame_idx)

        return self 

    def __next__(self):
        if self.frame_idx < self.stop_iter:

            # Load and filter a new block, if necessary
            if self.frame_idx_in_block == self.block_size:
                self._update_block(self.frame_idx)

            # Get next frame from this block
            result = self.block[self.frame_idx_in_block]

            # Update global and block frame indices
            self.frame_idx += 1
            self.frame_idx_in_block += 1

            return result 

        else:
            raise StopIteration

    def _update_block(self, start_frame):
        """
        Load and filter a new image block, holding it
        at self.block.

        If the start_frame is within one block width
        of the end of the movie, adjust the block limits
        so that we filter over a block of the full width.
        This involves some redundant filtering.

        Parameters
        ----------
            start_frame, stop_frame :  int, frame indices

        """
        # If we're too close to the end, adjust 
        # block limits
        if start_frame > self.n_frames-self.block_size:
            self.frame_idx_in_block = start_frame - \
                self.n_frames + self.block_size 
            start_frame = self.n_frames - self.block_size
        else:
            self.frame_idx_in_block = 0

        # Load the raw images 
        imgs = self.ImageFileReader.get_frame_range(
            start_frame, start_frame+self.block_size)

        # Perform the filtering routine
        self.block = self.method_f(
            imgs, **self.method_kwargs)

        # Record the current block start
        self.block_start = start_frame 

    def filter_all(self):
        """
        Read the full filtered movie into an 
        ndarray.

        Returns
        -------
            3D ndarray (TYX), the filtered movie

        """
        result = np.empty(self.ImageFileReader.get_shape(),
            dtype='float64')
        for frame_idx, frame in enumerate(self):
            result[frame_idx,:,:] = frame 
        return result 

    def filter_frame(self, frame_idx):
        """
        Return a single filtered frame from a movie.

        Parameters
        ----------
            frame_idx : int

        Returns
        -------
            2D ndarray (YX)

        """
        assert self.ImageFileReader._frame_valid(frame_idx)

        # Check to see if we already have the 
        # filtered block in memory
        if not self._in_current_block(frame_idx):

            # Load the appropriate block 
            block_start = self.block_size * (
                frame_idx // self.block_size)
            self._update_block(block_start)

        # Return the desired frame
        return self.block[frame_idx-self.block_start]

    def filter_range(self, start_frame, stop_frame):
        """
        Return a set of filtered frames. Does not
        include *stop_frame*.

        Parameters
        ----------
            start_frame, stop_frame : int

        Returns
        -------
            3D ndarray (TYX)

        """
        assert self.ImageFileReader._frame_range_valid(
            start_frame, stop_frame)
        result = np.empty((stop_frame-start_frame, self.height,
            self.width), dtype='float64')
        for j, frame_idx in enumerate(range(start_frame, stop_frame)):
            result[j,:,:] = self.filter_frame(frame_idx)
        return result

    def _in_current_block(self, frame_idx):
        """
        Return True if the desired frame corresponds
        to a frame currently at self.block.

        Parameters
        ----------
            frame_idx : int

        Returns
        ------- 
            bool

        """
        return hasattr(self, 'block_start') and \
            (frame_idx >= self.block_start) and \
            (frame_idx < self.block_start+self.block_size)

## 
## FILTERING METHODS
## 
def sub_median(imgs, scale=1.0):
    """
    Subtract the movie's median w.r.t time 
    from each frame.

    Parameters
    ----------
        imgs : 3D ndarray (TYX)
        scale : float, scale the avg image by 
            this amount before subtraction

    Returns
    -------
        3D ndarray (TYX), filtered movie

    """
    return set_neg_to_zero(imgs - \
        scale * np.median(imgs, axis=0))

def sub_scaled_median(imgs, scale=1.0):
    """
    Subtract the movie's median w.r.t time
    from each frame, scaled by that frame's
    median.

    Parameters
    ----------
        imgs : 3D ndarray (TYX)
        scale : float, scale the avg image by 
            this amount before subtraction

    Returns
    -------
        3D ndarray (TYX), filtered movie

    """
    # Get the average frame
    img_med = np.median(imgs, axis=0)

    # Get each frame's median
    meds = np.median(imgs, axis=(1,2))

    # Scale the medians
    _max, _min = meds.max(), meds.min()
    if _max == _min:
        meds = scale * np.ones(meds.shape[0])
    else:
        meds = scale * (meds-_min)/(_max-_min)

    # Subtract from each frame
    return set_neg_to_zero(np.asarray([
        imgs[i,:,:]-img_med for \
            i, m in enumerate(meds)
    ]))

def sub_moving_min(imgs, min_window=10, scale=1.0):
    """
    Subtract a minimum projection of the stack
    from each frame.

    Parameters
    ----------
        imgs : 3D ndarray (TYX)
        min_window : int, number of frames to
            take the minimum projection over
        scale : float, intensity modifier

    Returns
    -------
        3D ndarray (TYX), filtered movie

    """
    T, N, M = imgs.shape 
    min_window = int(min([min_window, T]))

    # Assign each frame to a min block
    nu = T // min_window 

    # Perform the subtraction
    result = np.empty((T, N, M), dtype='float64')
    for i in range(nu-1):
        f0 = i*min_window
        f1 = (i+1)*min_window
        result[f0:f1] = imgs[f0:f1] - \
            scale*imgs[f0:f1].min(axis=0)

    # For the last block, make sure that we're 
    # taking a full window for the min calculation
    f0 = T-min_window
    f1 = T 
    result[f0:f1] = imgs[f0:f1] - \
        scale*imgs[f0:f1].min(axis=0)

    return result 

def sub_gauss_filt_avg(imgs, k=2.0, scale=1.0):
    """
    Subtract a Gaussian-filtered mean image
    from each frame.

    Parameters
    ----------
        imgs : 3D ndarray (TYX)
        k : float, width of Gaussian kernel
        scale : float, intensity modifier 

    Returns
    -------
        3D ndarray (TYX), filtered movie

    """
    return set_neg_to_zero(imgs - scale * \
        ndi.gaussian_filter(imgs.mean(axis=0), k))

def sub_gauss_filt_med(imgs, k=2.0, scale=1.0):
    """
    Subtract a Gaussian-filtered median image
    from each frame.

    **WARNING: Untested. I don't think ndarrays 
    have the .med() method.

    Parameters
    ----------
        imgs : 3D ndarray (TYX)
        k : float, width of Gaussian kernel
        scale : float, intensity modifier

    Returns
    -------
        3D ndarray (TYX), filtered movie

    """
    return set_neg_to_zero(imgs - scale * \
        ndi.gaussian_filter(imgs.med(axis=0), k))

def sub_gauss_filt_moving_min(imgs, k=2.0, scale=1.0,
    min_window=10):
    """
    Subtract a Gaussian-filtered rolling min image
    from each frame. 

    Parameters
    ----------
        imgs : 3D ndarray (TYX)
        k : float, width of Gaussian kernel
        scale : float, intensity modifier
        min_window : int, the temporal window
            size for minimum calculation

    Returns
    -------
        3D ndarray (TYX), filtered movie

    """
    T, N, M = imgs.shape 
    min_window = min([min_window, T])

    # Assign each frame to a temporal subwindow
    nu = T // min_window 

    # Perform the subtraction
    result = np.empty((T, N, M), dtype='float64')
    for i in range(nu-1):
        f0 = i*min_window
        f1 = (i+1)*min_window
        result[f0:f1] = imgs[f0:f1] - \
            scale*ndi.gaussian_filter(
                imgs[f0:f1].min(axis=0), k)

    # For the last block, make sure that we're 
    # taking a full window for the min calculation
    f0 = T-min_window
    f1 = T 
    result[f0:f1] = imgs[f0:f1] - \
        scale*ndi.gaussian_filter(
            imgs[f0:f1].min(axis=0), k)

    return result 

##
## Filtering methods currently implemented
##
FILTER_METHODS = {
    'sub_med' : sub_median,
    'sub_med_scale' : sub_scaled_median,
    'sub_moving_min' : sub_moving_min,
    'sub_gauss_filt_avg' : sub_gauss_filt_avg,
    'sub_gauss_filt_med' : sub_gauss_filt_med,
    'sub_gauss_filt_moving_min' : sub_gauss_filt_moving_min,
}



