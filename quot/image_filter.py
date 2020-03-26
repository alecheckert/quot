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

class SubregionFilterer(object):
    """
    A filterer for an image subregion, designed
    to be used with gui.GUI.

    Most of the filtering techniques operate on a
    `block` of several subsequent frames. 

    Unlike the ImageFilterer class, the SubregionFilterer
    class holds the entire raw `block` in memory.
    It will only perform filtering on frames as necessary,
    when the user passes changes in the filtering method.

    init
    ----
        ImageFileReader : quot.qio.ImageFileReader

        subregion : [(int, int), (int, int)], subregion
            limits in format [(y0, y1), (x0, x1)]

        method : str, method to use for filtering

        block_size : int, the temporal size of blocks
            to use for filtering

        **method_kwargs : to the filtering method

    """
    def __init__(self, ImageFileReader, subregion=None,
        method=None, block_size=100, start_iter=0,
        stop_iter=None, **filter_kwargs):
        self.ImageFileReader = ImageFileReader 
        self.method = method 
        self.block_size = block_size
        self.filter_kwargs = filter_kwargs

        # If no subregion passed, default to the whole
        # frame
        if subregion is None:
            self.subregion = [[0, self.ImageFileReader.height],
                [0, self.ImageFileReader.width]]
        else:
            self.subregion = subregion 

        # Determine whether the subregion limits are
        # valid for this reader
        assert self._subregion_valid(self.subregion)

        # If no method passed, default to raw movie
        if (self.method is None) or (self.method == 'None'):
            self.method = 'unfiltered'

        # Find desired filtering method
        try:
            self.method_f = FILTER_METHODS[self.method]
        except KeyError:
            raise RuntimeError("SubregionFilterer.__init__: " \
                "method %s not found; available methods " \
                "are %s" % (method, ", ".join(FILTER_METHODS.keys())))

        # Get the movie length and shape 
        self.n_frames = self.ImageFileReader.n_frames 
        self.height = self.subregion[0][1] - self.subregion[0][0]
        self.width = self.subregion[1][1] - self.subregion[1][0]

        # Save the frames at which to start / stop iteration
        if start_iter is None:
            start_iter = 0
        if stop_iter is None:
            stop_iter = self.n_frames 
        self.start_iter = start_iter 
        self.stop_iter = stop_iter 

        # Set the initial frame to zero
        self.frame_idx = 0
        self._update_block(self.frame_idx)

    def __iter__(self):
        self.c_idx = self.start_iter 
        return self 

    def __next__(self):
        if self.c_idx < self.stop_iter:
            self.c_idx += 1
            return self.filter_frame(self.c_idx-1)
        else:
            raise StopIteration

    def _set_block_size(self, block_size):
        """
        Change the block size for this filtering
        method.

        """
        self.block_size = block_size 
        block_start = self._get_block_start(self.frame_idx)
        self._update_block(block_start)

    def _set_filter_kwargs(self, **kwargs):
        """
        Make modifications to the current 
        set of filter keyword arguments.

        """
        for k in kwargs.keys():
            self.filter_kwargs[k] = kwargs[k]

    def _update_block(self, start_frame):
        """
        Load a new image block into memory, 
        filtering it as necessary.

        args
        ----
            start_frame : int, the frame that
                starts the block. If *None*,
                defaults to the current frame.

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
        self.block = self.ImageFileReader.subregion(
            start_frame=start_frame,
            stop_frame=start_frame+self.block_size,
            y0=self.subregion[0][0],
            y1=self.subregion[0][1],
            x0=self.subregion[1][0],
            x1=self.subregion[1][1]
        )

        # Perform the preprocessing routine, calculating
        # the filtered versions of the block
        self._filter_block()

        # Record the current block start
        self.block_start = start_frame 

    def _get_block_start(self, frame_idx):
        """
        Given the current value of self.block_size, 
        return the starting frame for the block
        corresponding to *frame_idx*.

        """
        return min([
            self.n_frames - self.block_size,
            self.block_size * (frame_idx // self.block_size)
        ])

    def _filter_block(self):
        """
        Preprocess the current block for filter 
        functions.

        This means calculating a set of standard 
        functions on the block, including mean,
        median, and min images.

        Calculating these in advance speeds up
        the filtering of individual frames when
        applying changes in the filtering routine.

        """
        self.filt_imgs = {
            'mean': self.block.mean(axis=0).astype('float64'),
            'min': self.block.min(axis=0).astype('float64'),
            'median': np.median(self.block, axis=0).astype('float64'),
        }

    def _in_current_block(self, frame_idx):
        """
        Return True if the frame is in the 
        current block.

        """
        return hasattr(self, 'block_start') and \
            (frame_idx >= self.block_start) and \
            (frame_idx < self.block_start+self.block_size)

    def _subregion_valid(self, subregion):
        """
        Return True if the subregion is within the
        size limits of frame size.

        """
        H = self.ImageFileReader.height
        W = self.ImageFileReader.width 
        return all([
            subregion[0][0] <= H,
            subregion[0][1] <= H,
            subregion[1][0] <= W,
            subregion[1][1] <= W, 
            subregion[0][0] <= subregion[0][1],
            subregion[1][0] <= subregion[1][1]
        ])

    def _change_filter_method(self, method, **filter_kwargs):

        # If None, default to no filtering
        if (method is None) or (method == 'None'):
            method = 'unfiltered'
        self.method = method

        # Find the desired filter method
        try:
            self.method_f = FILTER_METHODS[self.method]
        except KeyError:
            raise RuntimeError("SubregionFilterer.__init__: " \
                "method %s not found; available methods " \
                "are %s" % (method, ", ".join(FILTER_METHODS.keys())))

        # Reset the filtering keyword arguments to their
        # defaults
        self.filter_kwargs = {}

        # Set newly passed filtering keyword arguments
        if len(filter_kwargs) > 0:
            self._set_filter_kwargs(**filter_kwargs)

    def filter_frame_and_change_block_size(self, frame_idx,
        block_size, **filter_kwargs):
        """
        First change the block size of the filterer, 
        then return a specific frame. 

        args
        ----
            frame_idx : int
            block_size : int
            **filter_kwargs : to the filtering method

        returns
        -------
            2D ndarray, filtered frame

        """
        self.frame_idx = frame_idx 
        self._set_block_size(block_size)
        return self.filter_frame(frame_idx, **filter_kwargs)

    def filter_frame(self, frame_idx, **filter_kwargs):
        """
        Return a single filtered frame from a 
        movie.

        args
        ----
            frame_idx : int
            **filter_kwargs : any CHANGES to the kwargs
                to the current filtering function. These
                are recorded and used for subsequent
                filtering.

        returns
        -------
            2D ndarray, the filtered frame

        """
        assert self.ImageFileReader._frame_valid(frame_idx)

        # Check to see if we already have the
        # filtered block in memory
        if not self._in_current_block(frame_idx):

            # Load the appropriate block
            block_start = self.block_size * (
                frame_idx // self.block_size)
            self._update_block(block_start)

        # Record the current frame index in the context
        # of this block
        self.frame_idx = frame_idx 
        self.frame_idx_in_block = self.frame_idx - self.block_start

        # Save the filtering keyword arguments
        if len(filter_kwargs) > 0:
            self._set_filter_kwargs(**filter_kwargs)

        # Get the subtraction image according to the
        # present method
        sub_img = self.filt_imgs[FILTER_METHOD_INPUTS[self.method]]

        # Apply the filtering routine
        return self.method_f(
            self.block[self.frame_idx_in_block],
            sub_img, 
            **self.filter_kwargs
        )


#
# FILTERING METHODS
#
def unfiltered(img, no_img):
    """
    Do not perform filtering.

    """
    return img 

def sub_median(img, median_img, scale=1.0):
    """
    Subtract the movie's median with respect to 
    time from the frame.

    """
    return set_neg_to_zero(img-scale*median_img)

def sub_min(img, min_img, scale=1.0):
    """
    Subtract a pixelwise minimum from each 
    frame.

    """
    return set_neg_to_zero(img-scale*min_img)

def sub_mean(img, mean_img, scale=1.0):
    """
    Subtract the pixelwise mean from each
    frame.

    """
    return set_neg_to_zero(img-scale*mean_img)

def sub_gauss_filt_median(img, median_img, k=2.0,
    scale=1.0):
    """
    Subtract a Gaussian-filtered median image
    from the present image.

    """
    return set_neg_to_zero(img - scale * \
        ndi.gaussian_filter(median_img, k))

def sub_gauss_filt_min(img, min_img, k=2.0,
    scale=1.0):
    """
    Subtract a Gaussian-filtered minimum image
    from the present image.

    """
    return set_neg_to_zero(img - scale * \
        ndi.gaussian_filter(min_img, k))

def sub_gauss_filt_mean(img, mean_img, k=2.0,
    scale=1.0):
    """
    Subtract a Gaussian-filtered mean image
    from the present image.

    """
    return set_neg_to_zero(img - scale * \
        ndi.gaussian_filter(mean_img, k))

#
# Filtering methods currently implemented
#
FILTER_METHODS = {
    'unfiltered': unfiltered,
    'sub_median': sub_median,
    'sub_min': sub_min,
    'sub_mean': sub_mean,
    'sub_gauss_filt_median': sub_gauss_filt_median,
    'sub_gauss_filt_min': sub_gauss_filt_min,
    'sub_gauss_filt_mean': sub_gauss_filt_mean,
}

# 
# The input images expected by each method
#
FILTER_METHOD_INPUTS = {
    'unfiltered': 'median',
    'sub_median': 'median',
    'sub_min': 'min',
    'sub_mean': 'mean',
    'sub_gauss_filt_median': 'median',
    'sub_gauss_filt_min': 'min',
    'sub_gauss_filt_mean': 'mean',
}



