"""
utils.py

"""
# Numerics
import numpy as np 

# Nd image analysis
from scipy import ndimage as ndi 

def set_neg_to_zero(ndarray):
    """
    Set all negative values in a 
    numpy.ndarray to zero.

    """
    ndarray[ndarray<0.0] = 0.0
    return ndarray 

def threshold_image(image, t):
    """
    Return a binary image based on an 
    intensity threshold.

    """
    return (image>=t).astype('uint8')

def overlay_spots(image, positions,
    crosshair_len=4):
    """
    Make a copy of an image and write crosshairs
    over it at a set of defined positions.

    args
    ----
        image : 2D ndarray, the image 
        positions : 2D ndarray of shape
            n_points, 2), the positions
            of the points
        crosshair_len : int, size of the 
            crosshairs

    returns
    -------
        2D ndarray, copy of the image with
            the overlayed crosshairs

    """
    I = image.copy()

    # If no positions are passed, return the same image
    if (positions.shape[0] == 0) or (len(positions.shape) < 2):
        return I 

    I_max = I.max()
    N, M = I.shape 

    for j in range(-crosshair_len, crosshair_len+1):

        # Extend crosshair in y direction
        PY = positions[:,0] + j 
        PX = positions[:,1]
        inside = (PY>=0) & (PY<N) & (PX>=0) & (PX<M)
        I[PY[inside], PX[inside]] = I_max 

        # Extend crosshair in x direction
        PY = positions[:,0]
        PX = positions[:,1] + j 
        inside = (PY>=0) & (PY<N) & (PX>=0) & (PX<M)
        I[PY[inside], PX[inside]] = I_max 

    return I 

def label_binary_spots(img_bin, img_int=None):
    """
    Find the centers of contiguous nonzero objects
    in a binary image, returning the coordinates
    of the spots as a 2D ndarray.

    If *img_int* is passed, then the coordinates
    are the nearest pixels to the centroid of 
    *img_int*. Otherwise the coordinates are the
    nearest pixels to the mean position of the 
    binary spot.

    args
    ----
        img_bin : 2D ndarray, binary spot image
        img_int : 2D ndarray, the intensities
            for (optional) centroid calculations

    returns
    -------
        2D ndarray of shape (n_spots, 2) and
            dtype int64, the YX coordinates of 
            each spot

    """
    img_lab, N = ndi.label(img_bin)
    index = np.arange(1,N+1)
    if img_int is None:
        positions = np.asarray([ndi.center_of_mass(
            img_bin, labels=img_lab, index=index)])
    else:
        positions = np.asarray(ndi.center_of_mass(
            img_int, labels=img_lab, index=index))
    return positions.astype('int64')

def zero_pad(img, H, W, mode='ceil'):
    """
    Pad a kernel with zeroes for subsequent
    convolution.

    args
    ----
        img : 2D ndarray, kernel
        H : int, desired height
        W : int, desired width
        mode : adjustment for nearest-pixel
            effects. 'ceil' is consisent
            with scipy.ndimage.uniform_filter

    returns
    -------
        2D ndarray

    """
    H_in, W_in = img.shape
    out = np.zeros((H, W))
    if mode == 'ceil':
        hc = np.ceil(H / 2 - H_in / 2).astype(int)
        wc = np.ceil(W / 2 - W_in / 2).astype(int)
    elif mode == 'floor':
        hc = np.floor(H / 2 - H_in / 2).astype(int)
        wc = np.floor(W / 2 - W_in / 2).astype(int)
    out[hc : hc + H_in, wc : wc + W_in] = img
    return out

def rfftconvolve(image, kernel):
    """
    Convolve an image with a kernel.

    args
    ----
        image : 2D ndarray
        kernel : 2D ndarray, equal or smaller
                in size to *image*

    returns
    -------
        2D ndarray

    """
    image_rft = np.fft.rfft2(image)
    kernel_rft = np.fft.rfft2(zero_pad(kernel, *image.shape))
    return np.fft.fftshift(np.fft.irfft2(image_rft * kernel_rft))

def get_gaussian_kernel(w, k):
    """
    Build a Gaussian kernel of sigma *k*
    in a window of size (w, w). The Gaussian is
    normalized to sum to 1.

    args
    ----
        w : int, window size
        k : float, kernel width

    returns
    -------
        2D ndarray, dtype float64, shape (w, w)

    """
    result = np.exp(-((np.indices((w, w)) - \
        (w-1)/2)**2).sum(axis=0) / (2*(k**2)))
    return result / result.sum()

def stable_divide(N, D, inf=1.0):
    """
    Divide two arrays, replacing any divide-by-zero
    errors with the value *inf*. 

    args
    ----
        N : numerator matrix
        D : denominator matrix, assumed positive
        inf : the value to replace any divide-by-
            zero errors with

    returns
    -------
        2D ndarray

    """
    # Make sure the shapes coincide
    assert N.shape == D.shape 

    # Format output
    result = np.full(N.shape, inf, dtype='float64')

    # Determine where the denominator is zero
    nonzero = D > 0.0

    # Perform division
    result[nonzero] = N[nonzero] / D[nonzero]
    return result







