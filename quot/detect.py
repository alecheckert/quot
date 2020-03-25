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
from .utils import set_neg_to_zero, label_binary_spots

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
    img_filt = set_neg_to_zero(
        ndi.gaussian_filter(img, k0) - \
        ndi.gaussian_filter(img, k1),
    )

    # Threshold
    if t is None:
        t = threshold_otsu(img_filt)
    img_bin = img_filt > t 

    # Find spots 
    positions = label_binary_spots(img_bin,
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
    img_filt = set_neg_to_zero(
        ndi.uniform_filter(img, k0) - \
        ndi.uniform_filter(img, k1)
    )

    # Threshold
    if t is None:
        t = threshold_otsu(img_filt)
    img_bin = img_filt > t

    # Find spots 
    positions = label_binary_spots(img_bin,
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
    positions = label_binary_spots(img_bin,
        img_int=img_filt)

    if return_filt:
        return img_filt, img_bin, positions 
    else:
        return position 


