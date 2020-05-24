#!/usr/bin/env python
"""
findSpots.py -- detect spots in 2D images

"""
# Numeric
import numpy as np 

# Real-valued FFT
from numpy.fft import rfft2, irfft2, fftshift 

# Image processing / filtering
from scipy import ndimage as ndi 

# Caching
from functools import lru_cache 

# Custom utilities
from .helper import (
    pad,
    threshold_image,
    stable_divide_array,
    assign_methods,
)

def gauss(I, k=1.0, w=9, t=200.0, return_filt=False):
    """
    Convolve the image with a simple Gaussian kernel, then apply
    a static threshold

    args
    ----
        I           :   2D ndarray (YX)
        k           :   float, kernel sigma
        w           :   int, kernel window size
        t           :   float, threshold
        return_filt :   bool, also return the filtered image
                        and boolean image

    returns
    -------
        if return_filt:
        (
            2D ndarray, the post-convolution image;
            2D ndarray, the thresholded binary image;
            2D ndarray, shape (n_spots, 2), the y and x 
                coordinates of each spot
        )
        else
            2D ndarray, shape (n_spots, 2), the y and x
                coordinates of each spot

    """
    # Compute the transfer function
    G_rft = _gauss_setup(*I.shape, k, w)
    return threshold_image(fftshift(irfft2(rfft2(I)*G_rft, s=I.shape)), 
        t=t, return_filt=return_filt)

@lru_cache(maxsize=1)
def _gauss_setup(H, W, k, w):
    """
    Make the transfer function for Gaussian filtering
    of real-valued images.

    args
    ----
        H, W    :   int, height and width of image
        k       :   float, kernel sigma
        w       :   int, window size for the kernel

    returns
    -------
        2D ndarray, dtype complex128, the RFFT2 of
            the Gaussian kernel

    """
    S = 2*(k**2)
    g = np.exp(-((np.indices((w,w))-(w-1)/2)**2).sum(0)/S)
    return rfft2(pad(g/g.sum(), H, W))

def centered_gauss(I, k=1.0, w=9, t=200.0, return_filt=False):
    """
    Convolve the image with a mean-subtracted Gaussian kernel, 
    then apply a static threshold.

    args
    ----
        I           :   2D ndarray (YX)
        k           :   float, kernel sigma
        w           :   int, kernel window size
        t           :   float, threshold
        return_filt :   bool, also return the filtered image
                        and boolean image

    returns
    -------
        if return_filt:
        (
            2D ndarray, the post-convolution image;
            2D ndarray, the thresholded binary image;
            2D ndarray, shape (n_spots, 2), the y and x 
                coordinates of each spot
        )
        else
            2D ndarray, shape (n_spots, 2), the y and x
                coordinates of each spot

    """
    # Compute the transfer function
    G_rft = _centered_gauss_setup(*I.shape, k, w)
    return threshold_image(fftshift(irfft2(rfft2(I)*G_rft, s=I.shape)), 
        t=t, return_filt=return_filt)

@lru_cache(maxsize=1)
def _centered_gauss_setup(H, W, k, w):
    """
    Make the transfer function for convolution of real-valued
    images with a centered (mean-subtracted) Gaussian kernel.

    args
    ----
        H, W    :   int, height and width of image
        k       :   float, kernel sigma
        w       :   int, window size for the kernel

    returns
    -------
        2D ndarray, dtype complex128, the RFFT2 of
            the kernel

    """
    S = 2*(k**2)
    g = np.exp(-((np.indices((w,w))-(w-1)/2)**2).sum(0)/S)
    g = g / g.sum()
    return rfft2(pad(g-g.mean(), H, W))

def mle_amp(I, k=1.0, w=9, t=200.0, return_filt=False):
    """
    Convolve the image with a mean-subtracted Gaussian kernel, 
    square the result and normalize by the then apply a static threshold. This is equivalent to 

    args
    ----
        I           :   2D ndarray (YX)
        k           :   float, kernel sigma
        w           :   int, kernel window size
        t           :   float, threshold
        return_filt :   bool, also return the filtered image
                        and boolean image

    returns
    -------
        if return_filt:
        (
            2D ndarray, the post-convolution image;
            2D ndarray, the thresholded binary image;
            2D ndarray, shape (n_spots, 2), the y and x 
                coordinates of each spot
        )
        else
            2D ndarray, shape (n_spots, 2), the y and x
                coordinates of each spot

    """
    # Compute the transfer function and normalization factor
    G_rft, Sgc2 = _mle_amp_setup(*I.shape, k, w)

    # Perform filtering
    return threshold_image((fftshift(irfft2(rfft2(I)*G_rft, s=I.shape))**2)/Sgc2, 
        t=t, return_filt=return_filt)

@lru_cache(maxsize=1)
def _mle_amp_setup(H, W, k, w):
    """
    Make the transfer function for convolution of real-valued
    images with a centered (mean-subtracted) Gaussian kernel,
    along with a normalization term to convert to the max-likelihood
    estimate for the intensities of Gaussian spots.

    args
    ----
        H, W    :   int, height and width of image
        k       :   float, kernel sigma
        w       :   int, window size for the kernel

    returns
    -------
        (
            2D ndarray, dtype complex128, the RFFT2 of
                the kernel
            float, normalization factor
        )

    """
    S = 2*(k**2)
    g = np.exp(-((np.indices((w,w))-(w-1)/2)**2).sum(0)/S)
    g = g / g.sum()
    gc = g - g.mean()
    Sgc2 = (gc**2).sum()
    return rfft2(pad(gc, H, W)), Sgc2 

def dog(I, k0=1.0, k1=3.0, w=9, t=200.0, return_filt=False):
    """
    Convolve the image with a difference-of-Gaussians kernel,
    then apply a static threshold.

    args
    ----
        I           :   2D ndarray
        k0          :   float, positive kernel sigma
        k1          :   float, negative kernel sigma
        w           :   int, kernel size
        t           :   float, threshold
        return_filt :   bool, also return the filtered image

    returns
    -------
        if return_filt:
        (
            2D ndarray, the post-convolution image;
            2D ndarray, the thresholded binary image;
            2D ndarray, shape (n_spots, 2), the y and x 
                coordinates of each spot
        )
        else
            2D ndarray, shape (n_spots, 2), the y and x
                coordinates of each spot

    """
    # Generate the transfer function
    dog_tf = _dog_setup(*I.shape, k0, k1, w)

    # Perform the convolution
    return threshold_image(fftshift(irfft2(rfft2(I)*dog_tf, s=I.shape)), 
        t=t, return_filt=return_filt)

@lru_cache(maxsize=1)
def _dog_setup(H, W, k0, k1, w):
    """
    Generate the transfer function for DoG filtering.

    args
    ----
        H, W        :   int, height and width of the image
                        to be convolved
        k0          :   float, positive kernel sigma
        k1          :   float, negative kernel sigma
        w           :   int, kernel size

    returns
    -------
        2D ndarray, RFFT of the DoG kernel 

    """
    S0 = 2*(k0**2)
    S1 = 2*(k1**2)
    g1 = np.exp(-((np.indices((int(w),int(w)))-(int(w)-1)/2)**2).sum(0)/S0)
    g1 = g1 / g1.sum()
    g2 = np.exp(-((np.indices((int(w),int(w)))-(int(w)-1)/2)**2).sum(0)/S1)
    g2 = g2 / g2.sum()   
    return rfft2(pad(g1-g2, H, W))

def log(I, k=1.0, w=11, t=200.0, return_filt=False):
    """
    Detect spots by Laplacian-of-Gaussian filtering.

    args
    ----
        I           :   2D ndarray
        k           :   float, kernel sigma
        w           :   int, kernel size 
        t           :   float, threshold
        return_filt :   bool, also return the filtered image

    returns
    -------
        if return_filt:
        (
            2D ndarray, the post-convolution image;
            2D ndarray, the thresholded binary image;
            2D ndarray, shape (n_spots, 2), the y and x 
                coordinates of each spot
        )
        else
            2D ndarray, shape (n_spots, 2), the y and x
                coordinates of each spot

    """
    # Generate the transfer function
    G_rft = _log_setup(*I.shape, k, w)

    # Perform the convolution
    return threshold_image(fftshift(irfft2(rfft2(I)*G_rft, s=I.shape)), 
        t=t, return_filt=return_filt)

def _log_setup(H, W, k, w):
    """
    Generate a Laplacian-of-Gaussian (LoG) transfer function
    for subsequent convolution with an image.

    args
    ----
        H, W    :   int, height and width of target image
        k       :   float, kernel sigma
        w       :   int, kernel size 

    returns
    -------
        2D ndarray, dtype complex128, the transfer function

    """
    S = 2*(k**2)
    g = np.exp(-((np.indices((w,w))-(w-1)/2)**2).sum(0)/S)
    g = g/g.sum()
    log_k = -ndi.laplace(g)
    return rfft2(pad(log_k, H, W))

def dou(I, w0=3, w1=11, t=200.0, return_filt=False):
    """
    Uniform-filter an image with two different kernel sizes and
    subtract them. This is like a poor-man's version of DoG
    filtering.

    args
    ----
        I           :   2D ndarray
        w0          :   int, positive kernel size
        w1          :   int, negative kernel size
        t           :   float, threshold
        return_filt :   bool, also return the filtered image

    returns
    -------
        if return_filt:
        (
            2D ndarray, the post-convolution image;
            2D ndarray, the thresholded binary image;
            2D ndarray, shape (n_spots, 2), the y and x 
                coordinates of each spot
        )
        else
            2D ndarray, shape (n_spots, 2), the y and x
                coordinates of each spot

    """
    return threshold_image(ndi.uniform_filter(I, w0) - \
        ndi.uniform_filter(I, w1), t=t, return_filt=return_filt)

def min_max(I, w=9, t=200.0, mode='constant', return_filt=False,
    **kwargs):
    """
    Use the difference between the local maximum and local minimum
    in square subwindows to identify spots in an image.

    args
    ----
        I           :   2D ndarray
        w           :   int, window size for test
        t           :   float, threshold for spot detection
        mode        :   str, behavior at boundaries (see scipy.ndimage
                        documentation)
        kwargs      :   to ndimage.maximum_filter/minimum_filter

    returns
    -------
        if return_filt:
        (
            2D ndarray, the post-convolution image;
            2D ndarray, the thresholded binary image;
            2D ndarray, shape (n_spots, 2), the y and x 
                coordinates of each spot
        )
        else
            2D ndarray, shape (n_spots, 2), the y and x
                coordinates of each spot

    """
    size = (w, w)
    I_filt = ndi.maximum_filter(I, size=size, mode=mode, **kwargs) - \
        ndi.minimum_filter(I, size=size, mode=mode, **kwargs)

    # Set the probability of detection near the border to zero
    hw = w//2
    I_filt[:hw,:] = t-1
    I_filt[:,:hw] = t-1
    I_filt[-hw:,:] = t-1
    I_filt[:,-hw:] = t-1

    # Threshold the result
    return threshold_image(I_filt, t=t, return_filt=return_filt)

def gauss_filt_min_max(I, k=1.0, w=9, t=200.0, mode='constant',
    return_filt=False, **kwargs):
    """
    Similar to min_max_filter(), but perform a Gaussian convolution
    prior to looking for spots by min/max filtering.

    args
    ----
        I           :   2D ndarray
        k           :   float, Gaussian kernel sigma
        w           :   int, window size for test
        t           :   float, threshold for spot detection
        mode        :   str, behavior at boundaries (see scipy.ndimage
                        documentation)
        kwargs      :   to ndimage.maximum_filter/minimum_filter

    returns
    -------
        if return_filt:
        (
            2D ndarray, the post-convolution image;
            2D ndarray, the thresholded binary image;
            2D ndarray, shape (n_spots, 2), the y and x 
                coordinates of each spot
        )
        else
            2D ndarray, shape (n_spots, 2), the y and x
                coordinates of each spot

    """
    # Generate the kernel for convolution
    G_rft = _gauss_setup(*I.shape, k, w)

    # Perform the convolution and do min/max filtering on
    # the result
    return min_max(fftshift(irfft2(rfft2(I)*G_rft, s=I.shape)),
        w=w, t=t, mode=mode, return_filt=return_filt, **kwargs)

def llr(I, k=1.0, w=9, t=20.0, return_filt=False):
    """
    Perform a log-likelihood ratio test for the presence of 
    spots. This is the ratio of likelihood of a Gaussian spot
    in the center of the subwindow, relative to the likelihood
    of flat background with Gaussian noise.

    args
    ----
        I           :   2D ndarray
        k           :   float, Gaussian kernel sigma
        w           :   int, window size for test
        t           :   float, threshold for spot detection
        return_filt :   bool, also return filtered image

    returns
    -------
        if return_filt:
        (
            2D ndarray, the post-convolution image;
            2D ndarray, the thresholded binary image;
            2D ndarray, shape (n_spots, 2), the y and x 
                coordinates of each spot
        )
        else
            2D ndarray, shape (n_spots, 2), the y and x
                coordinates of each spot

    """
    # Generate the convolution kernel and normalization factor
    G_rft, Sgc2 = _mle_amp_setup(*I.shape, k, w)
    n_pixels = w**2

    # Perform the convolutions for detection
    A = ndi.uniform_filter(I, w)
    B = ndi.uniform_filter(I**2, w)
    C = fftshift(irfft2(rfft2(I)*G_rft, s=I.shape))**2

    # Evaluate the log likelihood ratio for presence of a Gaussian spot
    L = 1.0 - stable_divide_array(C, n_pixels*Sgc2*(B-A**2), zero=0.001)

    # Set probability of detection close to edges to zero
    hw = w//2
    L[:hw,:] = 1.0
    L[:,:hw] = 1.0
    L[-hw:,:] = 1.0
    L[:,-hw:] = 1.0

    return threshold_image(-(n_pixels/2.0)*np.log(L), t=t,
        return_filt=return_filt)

def llr_rect(I, w0=3, w1=11, t=20.0, return_filt=False):
    """
    Perform a log-likelihood ratio test for the presence of 
    square spots of size *w0* in an image using a test equivalent
    to llr(). While squares only roughly approximate real spots,
    the test can be performed extremely fast due to the fact that
    only uniform filtering is required.

    args
    ----
        I           :   2D ndarray
        w0          :   int, spot kernel size
        w1          :   int, background kernel size
        t           :   float, threshold for spot detection
        return_filt :   bool, also return filtered image

    returns
    -------
        if return_filt:
        (
            2D ndarray, the post-convolution image;
            2D ndarray, the thresholded binary image;
            2D ndarray, shape (n_spots, 2), the y and x 
                coordinates of each spot
        )
        else
            2D ndarray, shape (n_spots, 2), the y and x
                coordinates of each spot

    """
    n_pixels = w1**2 

    # Compute normalization factor
    Suc2 = n_pixels * ((1.0/(w0**2)) - (1.0/(w1**2)))

    # Perform convolutions for detection
    A = ndi.uniform_filter(I, w1)
    C = (ndi.uniform_filter(I, w0) - A)**2
    B = ndi.uniform_filter(I**2, w1)

    # Calculate likelihood of squares centered on each pixel
    L = 1.0 - stable_divide_array(C, Suc2*(B-(A**2)), zero=0.0001)

    # Set probability of detection close to edges to zero
    hw = w1//2
    L[:hw,:] = 1.0
    L[:,:hw] = 1.0
    L[-hw:,:] = 1.0
    L[:,-hw:] = 1.0

    # Take log-likelihood and threshold to find spots
    return threshold_image(-(n_pixels/2.0)*np.log(L), t=t,
        return_filt=return_filt)

#############################
## MAIN DETECTION FUNCTION ##
#############################

# All available detection methods
METHODS = {
    'gauss': gauss,
    'centered_gauss': centered_gauss,
    'mle_amp': mle_amp,
    'dog': dog,
    'log': log,
    'dou': dou,
    'min_max': min_max,
    'gauss_filt_min_max': gauss_filt_min_max,
    'llr': llr,
    'llr_rect': llr_rect,
}

def detect(I, method=None, **kwargs):
    """
    Run spot detection on a single image according to a
    detection method.

    args
    ----
        I       :   2D ndarray (YX), image frame
        method  :   str, a method name in METHODS
        kwargs  :   special argument to the method

    returns
    -------
        2D ndarray of shape (n_spots, 2), the Y and 
            X coordinates of identified spots

    """
    # Get the desired method
    method_f = METHODS.get(method)
    if method_f is None:
        raise KeyError("Method %s not available; available " \
            "methods are %s" % (method, ", ".join(METHODS.keys())))

    # Enforce float64, required for some methods
    I = I.astype(np.float64)

    # Run detection
    return method_f(I, **kwargs)




