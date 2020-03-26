"""
detect.py -- methods for spot detection
on 2D images

"""
# Numerics
import numpy as np 

# Filtering utilities
from scipy import ndimage as ndi 

# File path stuff
import os

# Find the Otsu threshold of an image
from skimage.filters import threshold_otsu 

# Custom utilities
from quot import utils

def dog_filter(img, k0=1.5, k1=8.0, t=None,
    return_filt=True):
    """
    Find spots by DoG-filtering an input
    image.

    args
    ----
        img : 2D ndarray 
        k0 : float, spot kernel width
        k1 : float, BG kernel width 
        t : float, threshold (if None, 
            defaults to Otsu)
        return_filt : bool. In addition to
            the spot positions, also return
            the filtered and binary images 

    returns
    -------
        If *return_filt*:
            (
                2D ndarray, filtered image;
                2D ndarray, binary image;
                2D ndarray of shape (n_spots, 2),
                    coordinates of spots
            )
        else:
            2D ndarray of shape (n_spots, 2),
                coordinates of spots

    """
    img = img.astype('float64')

    # Filter 
    img_filt = utils.set_neg_to_zero(
        ndi.gaussian_filter(img, k0) - \
        ndi.gaussian_filter(img, k1),
    )

    # Threshold
    if t is None:
        t = threshold_otsu(img_filt)
    img_bin = img_filt > t 

    # Find spots 
    positions = utils.label_binary_spots(img_bin,
        img_int=img_filt)

    if return_filt:
        return img_filt, img_bin, positions 
    else:
        return positions 

def dou_filter(img, k0=3, k1=9, t=None,
    return_filt=True):
    """
    Find spots by difference of uniforming-
    filtering.

    args
    ----
        img : 2D ndarray
        k0 : float, spot kernel width 
        k1 : float, BG kernel width 
        t : float, threshold (if None,
            defaults to Otsu)
        return_filt : bool. In addition to
            the spot positions, also return
            the filtered and binary images 

    returns
    -------
        If *return_filt*:
            (
                2D ndarray, filtered image;
                2D ndarray, binary image;
                2D ndarray of shape (n_spots, 2),
                    coordinates of spots
            )
        else:
            2D ndarray of shape (n_spots, 2),
                coordinates of spots

    """
    img = img.astype('float64')

    # Filter
    img_filt = utils.set_neg_to_zero(
        ndi.uniform_filter(img, k0) - \
        ndi.uniform_filter(img, k1)
    )

    # Threshold
    if t is None:
        t = threshold_otsu(img_filt)
    img_bin = img_filt > t

    # Find spots 
    positions = utils.label_binary_spots(img_bin,
        img_int=img_filt)

    if return_filt:
        return img_filt, img_bin, positions 
    else:
        return position 

def gauss_filter(img, k=1.0, t=None,
    return_filt=True):
    """
    A simple Gaussian filter followed by 
    thresholding.

    args
    ----
        img : 2D ndarray
        k : float, kernel size
        t : float, threshold (if None,
            defaults to Otsu)
        return_filt : bool. In addition to
            the spot positions, also return
            the filtered and binary images 

    returns
    -------
        If *return_filt*:
            (
                2D ndarray, filtered image;
                2D ndarray, binary image;
                2D ndarray of shape (n_spots, 2),
                    coordinates of spots
            )
        else:
            2D ndarray of shape (n_spots, 2),
                coordinates of spots


    """
    img = img.astype('float64')

    # Filter
    img_filt = ndi.gaussian_filter(img, k)

    # Threshold
    if t is None:
        t = threshold_otsu(img_filt)
    img_bin = img_filt > t

    # Find spots 
    positions = utils.label_binary_spots(img_bin,
        img_int=img_filt)

    if return_filt:
        return img_filt, img_bin, positions 
    else:
        return position 

def gauss_filter_sq(img, k=1.0, t=None,
    return_filt=True):
    """
    A simple Gaussian filter, followed by 
    squaring the image and taking all pixels
    above a threshold.

    args
    ----
        img : 2D ndarray
        k : float, kernel size
        t : float, threshold

    returns
    -------
        If *return_filt*:
            (
                2D ndarray, filtered image;
                2D ndarray, binary image;
                2D ndarray of shape (n_spots, 2),
                    coordinates of spots
            )
        else:
            2D ndarray of shape (n_spots, 2),
                coordinates of spots


    """
    img = img.astype('float64')

    # Filter
    img_filt = ndi.gaussian_filter(img, k) ** 2

    # Threshold
    if t is None:
        t = threshold_otsu(img_filt)
    img_bin = img_filt > t

    # Find spots 
    positions = utils.label_binary_spots(img_bin,
        img_int=img_filt)

    if return_filt:
        return img_filt, img_bin, positions 
    else:
        return position 


def min_max_filter(img, w=9, t=None, mode='constant',
    return_filt=True, **kwargs):
    """
    Use the difference between the local maximum
    and local minimum to find spots in an image.

    args
    ----
        img : 2D ndarray
        w : int, window size (rectangular)
        t : float, threshold. If None, defaults
            to Otsu.
        mode : str, the behavior at boundaries.
            See scipy.ndimage documentation for
            details.
        **kwargs : to scipy.ndimage.maximum_filter
            and scipy.ndimage.minimum_filter

    returns
    -------
        If *return_filt*:
            (
                2D ndarray, filtered image;
                2D ndarray, binary image;
                2D ndarray of shape (n_spots, 2),
                    coordinates of spots
            )
        else:
            2D ndarray of shape (n_spots, 2),
                coordinates of spots

    """
    # Find the difference between local maximum
    # and local minimum
    img_filt = ndi.maximum_filter(img, size=(w,w),
            mode=mode, **kwargs) - \
        ndi.minimum_filter(img, size=(w,w),
            mode=mode, **kwargs)

    # Threshold the image to find spots
    if t is None:
        t = threshold_otsu(img_filt)
    img_bin = img_filt > t 

    # Find spots 
    positions = utils.label_binary_spots(img_bin,
        img_int=img_filt)

    if return_filt:
        return img_filt, img_bin, positions 
    else:
        return position 

def llr(img, w=9, k=1.0, t=None, return_filt=True):
    """
    Use a log-likelihood ratio test for the presence
    of a Gaussian spot in the presence of Gaussian-
    distributed noise, relative to the likelihood
    of flat background with Gaussian-distributed 
    noise.

    args
    ----
        img : 2D ndarray
        w : int, window size (rectangular)
        k : float, width of Gaussian kernel
        t : float, threshold. If None, defaults
            to Otsu.

    returns
    -------
        If *return_filt*:
            (
                2D ndarray, filtered image;
                2D ndarray, binary image;
                2D ndarray of shape (n_spots, 2),
                    coordinates of spots
            )
        else:
            2D ndarray of shape (n_spots, 2),
                coordinates of spots

    """
    # Enforce integer window size 
    w = int(w)

    # Enforce float64 input
    img = img.astype('float64')

    # Get the number of pixels in the kernel
    n_pixels = w**2

    # Generate Gaussian kernel 
    g = utils.get_gaussian_kernel(w, k)

    # Center the kernel
    gc = g - g.mean()

    # Get normalization factor
    Sgc2 = (gc**2).sum()

    # Convolve image with Gaussian
    C = utils.rfftconvolve(img, gc)

    # Convolve image with uniform kernel
    A = ndi.uniform_filter(img, w) * n_pixels
    B = ndi.uniform_filter(img**2, w) * n_pixels

    # Calculate the likelihood of a spot in
    # each pixel, avoiding divide-by-zero errors
    L = 1.0 - utils.stable_divide(
        C**2,
        Sgc2*(B-(A**2)/n_pixels),
        inf=0.0,
    )

    # Set the probability of detection close to 
    # the edges to zero
    hw = w//2
    L[:hw,:] = 1.0
    L[:,:hw] = 1.0
    L[-hw:,:] = 1.0
    L[:,-hw:] = 1.0

    # Set pathological negatives to zero
    # L[L<0.0] = 1.0

    # Take log likelihood ratio test
    img_filt = -(n_pixels/2)*np.log(L)

    # Apply threshold
    if t is None:
        t = threshold_otsu(img_filt)
    img_bin = img_filt > t

    # Find spots 
    positions = utils.label_binary_spots(
        img_bin, img_int=img_filt)

    if return_filt:
        return img_filt, img_bin, positions 
    else:
        return position 

def simple_threshold(img, t=None, return_filt=True):
    """
    Find spots by applying a simple threshold
    to the image.

    args
    ----
        img : 2D ndarray
        t : float, threshold. If None, defaults
            to Otsu.

    returns
    -------
        If *return_filt*:
            (
                2D ndarray, filtered image;
                2D ndarray, binary image;
                2D ndarray of shape (n_spots, 2),
                    coordinates of spots
            )
        else:
            2D ndarray of shape (n_spots, 2),
                coordinates of spots

    """
    # Threshold image 
    if t is None:
        t = threshold_otsu(img)
    img_bin = img > t 

    # Find spots 
    positions = utils.label_binary_spots(
        img, img_int=img)

    if return_filt:
        return img, img_bin, positions 
    else:
        return position









