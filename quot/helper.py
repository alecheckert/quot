#!/usr/bin/env python
"""
quot.helper.py -- low-level utilities

"""
# Numeric
import numpy as np 

# Image processing utilities
from scipy import ndimage as ndi 

#############
## GENERAL ##
#############

def assign_methods(methods):
    """
    Decorator that designates a wrapper function for 
    other methods. Adds a new kwarg to the decorated 
    function ("method") that looks up the corresponding
    method in the *methods* dict.

    Example:
        def add(x, y):
            return x + y
        def mult(x, y):
            return x * y
        METHODS = {'add': add, 'mult': mult}

        do_operation = assign_methods(METHODS)(lambda a, b: None)

    This would be called:
        do_operation(2, 3, method='add')  # is 5
        do_operation(2, 3, method='mult') # is 6

    args
    ----
        methods     :   dict that keys strings to methods

    returns
    -------
        function, wrapper for the methods

    """ 
    def dec(f):
        def apply_method(*args, method=None, **kwargs):
            method_f = methods.get(method)
            if method_f is None:
                raise KeyError("Available methods are {}".format(
                    ", ".join(methods.keys())))
            return method_f(*args, **kwargs)
        return apply_method
    return dec

def stable_divide_array(N, D, zero=0.001):
    """
    Divide array N by array D, setting zeros in D
    to *zero*. Assumes nonzero D.

    args
    ----
        N, D    :   ndarrays of the same shape
        zero    :   float

    returns
    -------
        ndarray of the same shape as N and D

    """
    return N/np.clip(D, zero, np.inf)

###############
## DETECTION ##
###############

def pad(I, H, W, mode='ceil'):
    """
    Pad an array with zeroes around the edges, placing 
    the original array in the center. 

    args
    ----
        I       :   2D ndarray, image to be padded
        H       :   int, height of final image
        W       :   int, width of final image
        mode    :   str, either 'ceil' or 'floor'. 'ceil'
                    yields convolution kernels that function
                    similarly to scipy.ndimage.uniform_filter.

    returns
    -------
        2D ndarray, shape (H, W)

    """
    H_in, W_in = I.shape
    out = np.zeros((H, W))
    if mode == "ceil":
        hc = np.ceil(H / 2 - H_in / 2).astype(int)
        wc = np.ceil(W / 2 - W_in / 2).astype(int)
    elif mode == "floor":
        hc = np.floor(H / 2 - H_in / 2).astype(int)
        wc = np.floor(W / 2 - W_in / 2).astype(int)
    out[hc : hc + H_in, wc : wc + W_in] = I
    return out

def label_spots(binary_img, intensity_img=None, mode="max"):
    """
    Find continuous nonzero objects in a binary image,
    returning the coordinates of the spots.

    If the objects are larger than a single pixel,
    then to find the central pixel do
        1. use the center of mass (if mode == 'centroid')
        2. use the brightest pixel (if mode == 'max')
        3. use the mean position of the binary spot
            (if img_int is not specified)

    args
    ----
        binary_img      :   2D ndarray (YX), dtype bool
        intensity_img   :   2D ndarray (YX)
        mode            :   str, 'max' or 'centroid'

    returns
    -------
        2D ndarray (n_spots, 2), dtype int64,
            the Y and X coordinate of each spot

    """
    # Find and label every nonzero object
    img_lab, n = ndi.label(binary_img)
    index = np.arange(1, n + 1)

    # Find the centers of the spots
    if intensity_img is None:
        positions = np.asarray(ndi.center_of_mass(binary_img, 
            labels=img_lab, index=index))
    elif mode == "max":
        positions = np.asarray(ndi.maximum_position(intensity_img,
            labels=img_lab, index=index))
    elif mode == "centroid":
        positions = np.asarray(ndi.center_of_mass(intensity_img,
            labels=img_lab, index=index))

    return positions.astype("int64")

def threshold_image(I, t=200.0, return_filt=False, mode='max'):
    """
    Take all spots in an image above a threshold *t*
    and return their approximate centers.

    If *return_filt* is set, the function also returns the 
    raw image and binary image. This is useful as a 
    back door when writing GUIs.

    args
    ----
        I           :   2D ndarray, image
        t           :   float, threshold
        return_filt :   bool
        mode        :   str, either 'max' or 'centroid'

    returns
    -------
        If *return_filt* is set:
        (
            2D ndarray, same as I;
            2D ndarray, binary image;
            2D ndarray, shape (n_spots, 2), the spot
                coordinates
        )
        else
            2D ndarray of shape (n_spots, 2), 
                the spot coordinates

    """
    I_bin = I > t
    pos = label_spots(I_bin, intensity_img=I, mode=mode)
    if return_filt:
        return I, I_bin, pos 
    else:
        return pos 





