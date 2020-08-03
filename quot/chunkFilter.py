#!/usr/bin/env python
"""
quot.filters -- background subtraction and related filters
to clean up dirty data for detection and localization

"""
# Numeric 
import numpy as np 

# Gaussian filtering
from scipy.ndimage import gaussian_filter 

# File reader
from .read import ImageReader

class ChunkFilter(ImageReader):
    """
    A chunkwise file reader that also performs common filtering
    operations, like subtracting background from each frame.

    It is convenient to integrate file reading and filtering, since
    filtering often involves computing some kind of function on
    a chunk of frames. For instance, we can perform a rudimentary
    kind of background subtraction by subtracting the minimum value
    of each pixel in a 100-frame block from the corresponding pixel
    in each individual frame.

    When iterating over this object, the ChunkFilter does the
    following:

        1. Load a new chunk, cached at self.chunk.
        2. Perform some precomputations on the chunk - for instance,
            find the through-time minimum of each pixel in frame.
        3. Return individual filtered frames using the calculated
            precomputations.
        4. Load new chunks as necessary.

    init
        path            :   str, path to an image file
        start           :   int, the start frame when iterating
        stop            :   int, the stop frame when iterating
        method          :   str, the filtering method to use
        chunk_size      :   int, the block size to use for filtering
        method_static   :   bool, the method is unlikely to change
                            during the existence of this ChunkFilter.
                            This is not true, for instance, when using
                            the ChunkFilter in a GUI setting where the
                            user can rapidly change between different 
                            filtering settings. If False, the ChunkFilter
                            caches information about the chunk much more
                            aggressively. This makes switching between 
                            filtering methods much faster at the cost of 
                            memory.
        init_load       :   bool, load the first chunk during init.
                            Generally keep this True unless you have a 
                            very good reason not to.
        **method_kwargs :   to the filtering method

    attributes
        self.cache
        self.block_starts (1D ndarray, int)
        self.block_start (int), the first frame for the current
            block

    """
    def __init__(self, path, start=None, stop=None, method='identity',
        chunk_size=40, method_static=True, init_load=True, **method_kwargs):

        # Open file reader
        super(ChunkFilter, self).__init__(path, start=start, stop=stop)

        # If the filtering method is unset, default to the 
        # raw image
        if method is None:
            method = 'identity'

        self.method = method 
        self.chunk_size = chunk_size 
        self.method_static = method_static 
        self.method_kwargs = method_kwargs 

        # If the chunk size is larger than the number of frames,
        # set it to the number of frames
        if self.chunk_size > self.n_frames:
            self.chunk_size = self.n_frames

        # Initially, set no subregion on the image.
        self.sub_kwargs = {}

        # Assign each frame index to a block
        self._init_chunks()

        # Set the initial chunk
        self.chunk_start = self.get_chunk_start(self.start)

        # Load the initial chunk 
        if init_load:
            self.load_chunk(self.chunk_start)

        # Set the method keyword arguments
        self.set_method_kwargs(method=self.method, **method_kwargs)

    def __next__(self):
        if self._c < self.stop:
            self._c += 1
            return self.filter_frame(self._c-1)
        else:
            raise StopIteration 

    def _init_chunks(self):
        """
        Choose which frames correspond to which chunks by instantiating
        an array with the starting frame index of each chunk.

        """
        n_chunks = self.n_frames//self.chunk_size + 1
        self.chunk_starts = np.zeros(n_chunks+1, dtype='int64')
        self._offset = self.start % self.chunk_size
        self.chunk_starts[1:] = np.arange(n_chunks)*self.chunk_size+self._offset
        self.chunk_starts = self.chunk_starts[self.chunk_starts<self.n_frames]

        # Special case: last chunk 
        self.chunk_starts[-1] = self.n_frames - self.chunk_size

    def _in_chunk(self, frame_index):
        """
        Return True if the frame is in the current chunk.

        """
        return (frame_index>=self.chunk_start) and \
            (frame_index<self.chunk_start+self.chunk_size)

    def _generate_cache(self):
        """
        Cache some factors calculated on the current chunk.

        If *method_static* is True, then only compute factors
        necessary for the current filtering method.

        Otherwise compute factors necessary for ANY filtering 
        method from the chunk. This is useful when the user is
        expected to change quickly between methods - for instance,
        in a GUI setting.

        """
        if not self.method_static:
            self.cache = {
                'mean_img': self.chunk.mean(axis=0),
                'median_img': np.median(self.chunk, axis=0),
                'min_img': self.chunk.min(axis=0),
            }
            for i in ['mean_img', 'median_img', 'min_img']:
                self.cache['gauss_filt_%s' % i] = gaussian_filter(self.cache[i], self.method_kwargs.get('k', 5.0))
        else:
            self.cache = {}
            if FILTER_CACHES[self.method] == 'mean_img':
                self.cache['mean_img'] = self.chunk.mean(axis=0)
            elif FILTER_CACHES[self.method] == 'median_img':
                self.cache['median_img'] = np.median(self.chunk, axis=0)
            elif FILTER_CACHES[self.method] == 'min_img':
                self.cache['min_img'] = self.chunk.min(axis=0)
            elif FILTER_CACHES[self.method] == 'gauss_filt_mean_img':
                self.cache['gauss_filt_mean_img'] = gaussian_filter(
                    self.chunk.mean(axis=0),
                    self.method_kwargs.get('k', 5.0)
                )
            elif FILTER_CACHES[self.method] == 'gauss_filt_min_img':
                self.cache['gauss_filt_min_img'] = gaussian_filter(
                    self.chunk.min(axis=0),
                    self.method_kwargs.get('k', 5.0)
                )               
            elif FILTER_CACHES[self.method] == 'gauss_filt_median_img':
                self.cache['gauss_filt_median_img'] = gaussian_filter(
                    np.median(self.chunk, axis=0),
                    self.method_kwargs.get('k', 5.0)
                )
            else:
                pass 

    def _recache_filtered_image(self):
        """
        A somewhat specialized method. Recache only one of the Gaussian-
        filtered images, like "gauss_filt_min_img", "gauss_filt_median_img",
        etc. This is useful when setting method kwargs that will change
        the kernel size for filtering.

        This is only permissible when self.method_static is False, since
        it requires using one of the basic images ("min_img" or "median_img",
        in the two examples above).

        """
        if self.method_static or (not (self.method in ['sub_gauss_filt_min',
            'sub_gauss_filt_median', 'sub_gauss_filt_mean'])):
            print("ChunkFilter._recache_filtered_image: has no effect " \
                "when method is static or does not involve Gaussian filtering")
        else:
            image_name = FILTER_CACHES[self.method]
            self.cache[image_name] = gaussian_filter(self.cache[ORIGIN_IMAGES[image_name]],
                self.method_kwargs.get('k', 5.0))
            self.method_kwargs[image_name] = self.cache[image_name]

    def set_method_kwargs(self, method=None, recache_filtered_image=False,
        **kwargs):
        """
        Set the kwargs passed to the underlying filtering 
        method. This includes factors that are precomputed
        for each chunk.

        If *method* is None, then use the current method.

        Sets the self.method_kwargs variable.

        args
        ----
            method  :   str, the name of the method to use
            recache :   bool, force the ChunkFilter to recache
                        the Gaussian filtered image, if it exists
            kwargs  :   to the method

        """
        self.method_kwargs = kwargs 

        # Regenerate the cache for a new method if necessary
        if not (method is None):
            self.method = method 
            if self.method_static:
                self._generate_cache()

        # Force re-cache, if desired
        if recache_filtered_image:
            self._recache_filtered_image()
        
        # Set cached factors required as keyword arguments
        if not (FILTER_CACHES[self.method] is None):
            self.method_kwargs[FILTER_CACHES[self.method]] = \
                self.cache[FILTER_CACHES[self.method]]

    def set_chunk_size(self, chunk_size):
        """
        Change the chunk size for this ChunkFilter.

        args
        ----
            chunk_size  :   int

        """
        self.chunk_size = chunk_size
        self._init_chunks()
        self.load_chunk(self.chunk_start)
        self._generate_cache()

    def set_subregion(self, **kwargs):
        """
        Set a subregion for this ChunkFilter. All filtering methods will
        return only this subregion.

        args
            y0, y1, x0, x1 (int), the limits of the subregion

        """
        self.sub_kwargs = self._process_subregion(**kwargs)
        self.chunk_height = self.sub_kwargs['y1'] - \
            self.sub_kwargs['y0']
        self.chunk_width = self.sub_kwargs['x1'] - \
            self.sub_kwargs['x0']

    def get_chunk_start(self, frame_index):
        """
        Return the starting frame for the corresponding chunk.

        """
        return self.chunk_starts[(frame_index-self._offset)//self.chunk_size+1]

    def load_chunk(self, start, recache=True):
        """
        Load a new image chunk starting at the frame *start*.
        Saves the chunk at self.chunk.

        args
            start       :   int
            recache     :   bool, recompute cached factors for the
                            new chunk

        """
        # If too close to the end of the movie, pick up 
        # frames before *start* to have a full chunk
        if start > (self.n_frames-self.chunk_size):
            self.local_index = start-self.n_frames+self.chunk_size
            start = self.n_frames-self.chunk_size 
        else:
            self.local_index = 0

        # Read the chunk from the file
        self.chunk = self.get_subregion_range(start=start,
            stop=start+self.chunk_size,
            **self.sub_kwargs).astype('float64')

        # Precompute some values for the current filtering method
        if recache:
            self._generate_cache()
            cache_key = FILTER_CACHES[self.method]
            if not cache_key is None:
                self.method_kwargs[cache_key] = self.cache[cache_key]

    def filter_frame(self, frame_index):
        """
        Return a single filtered frame from the movie.

        args
            frame_index     :   int

        returns
            2D ndarray (YX)

        """
        assert self._frame_valid(frame_index)

        if not self._in_chunk(frame_index):
            self.chunk_start = self.get_chunk_start(frame_index)
            self.load_chunk(self.chunk_start)

        return FILTER_METHODS.get(self.method)(
            self.chunk[frame_index-self.chunk_start], 
            **self.method_kwargs)

#######################
## FILTERING METHODS ##
#######################

def identity(img, **kwargs):
    """
    Do not filter the image.

    args
    ----
        img     :   2D ndarray (YX)

    returns
    -------
        2D ndarray (YX)

    """
    return img

def simple_sub(img, sub_img, scale=1.0):
    """
    Ffrom each pixel in *img*, subtract the corresponding
    pixel in *sub_img* multiplied by *scale*. Set all 
    negative values to zero.

    args
    ----
        img     :   2D ndarray (YX)
        sub_img :   2D ndarray (YX)
        scale   :   float

    returns
    -------
        2D ndarray (YX)

    """
    return img - scale * sub_img 

def sub_min(img, min_img, scale=1.0):
    """
    Wrapper for simple_sub() that uses the pixelwise minimum.

    args
    ----
        img     :   2D ndarray (YX)
        min_img :   2D ndarray (YX)
        scale   :   float

    returns
    -------
        2D ndarray (YX)

    """
    return simple_sub(img, min_img, scale=scale)

def sub_median(img, median_img, scale=1.0):
    """
    Wrapper for simple_sub() that uses the pixelwise median.

    args
    ----
        img        :   2D ndarray (YX)
        median_img :   2D ndarray (YX)
        scale      :   float

    returns
    -------
        2D ndarray (YX)

    """
    return simple_sub(img, median_img, scale=scale)   

def sub_mean(img, mean_img, scale=1.0):
    """
    Wrapper for simple_sub() that uses the pixelwise mean.

    args
    ----
        img        :   2D ndarray (YX)
        mean_img   :   2D ndarray (YX)
        scale      :   float

    returns
    -------
        2D ndarray (YX)

    """
    return simple_sub(img, mean_img, scale=scale)   

def sub_gauss_filt_min(img, gauss_filt_min_img, k=5.0, scale=1.0):
    """
    Subtract a Gaussian-filtered minimum image from this
    frame.

    args
    ----
        img         :   2D ndarray (YX)
        min_img     :   2D ndarray (YX)
        k           :   float, kernel sigma
        scale       :   float

    returns
    -------
        2D ndarray

    """
    return simple_sub(img, gauss_filt_min_img, scale=scale)

def sub_gauss_filt_median(img, gauss_filt_median_img, k=5.0, scale=1.0):
    """
    Subtract a Gaussian-filtered median image from this
    frame.

    args
    ----
        img         :   2D ndarray (YX)
        median_img  :   2D ndarray (YX)
        k           :   float, kernel sigma
        scale       :   float

    returns
    -------
        2D ndarray

    """
    return simple_sub(img, gauss_filt_median_img, scale=scale) 

def sub_gauss_filt_mean(img, gauss_filt_mean_img, k=5.0, scale=1.0):
    """
    Subtract a Gaussian-filtered mean image from this
    frame.

    args
    ----
        img         :   2D ndarray (YX)
        mean_img    :   2D ndarray (YX)
        k           :   float, kernel sigma
        scale       :   float

    returns
    -------
        2D ndarray

    """
    return simple_sub(img, gauss_filt_mean_img, scale=scale) 

# Factors to cache for each filtering method,
# removing redundant computations
FILTER_CACHES = {
    'identity': None,
    'sub_min': 'min_img',
    'sub_median': 'median_img',
    'sub_mean': 'mean_img',
    'sub_gauss_filt_min': 'gauss_filt_min_img',
    'sub_gauss_filt_median': 'gauss_filt_median_img',
    'sub_gauss_filt_mean': 'gauss_filt_mean_img',
}

# The base images for derived images
ORIGIN_IMAGES = {
    'gauss_filt_min_img': 'min_img',
    'gauss_filt_median_img': 'median_img',
    'gauss_filt_mean_img': 'mean_img',
}

# All available filtering methods
FILTER_METHODS = {
    'identity': identity,
    'sub_min': sub_min,
    'sub_median': sub_median,
    'sub_mean': sub_mean,
    'sub_gauss_filt_min': sub_gauss_filt_min,
    'sub_gauss_filt_median': sub_gauss_filt_median,
    'sub_gauss_filt_mean': sub_gauss_filt_mean,
}
