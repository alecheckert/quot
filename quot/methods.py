"""
methods.py -- raw methods for spot detection

Each method takes input (image, **kwargs) and
outputs 

    (
        2D ndarray [detection_idx, parameter_idx], the detections,
        list of str, the parameter attributes
    )

`y` and `x` must be among the parameter attributes.

"""
import numpy as np 
import pandas as pd 
from scipy import ndimage as ndi 
from skimage.morphology import threshold_otsu 

def otsu_threshold(
    image,
    weight='intensity',
):
    """
    Apply a simple Otsu threshold to an image and 
    take the thresholded objects as spots.

    Return the centers of mass of each spot and 
    the integrated intensity of each spot.

    args
    ----
        image :  2D ndarray
        weight :  str, either `intensity` or 
            `binary`. If `binary`, all pixels
            are given equal weight in the center
            of mass calculation

    returns
    -------
        (
            2D ndarray, parameter values of each spot,
            list of parameter identities
        )

    """
    # Threshold the image 
    binary = image >= threshold_otsu(image)
    spots, n_spots = ndi.label(binary)
    spot_index = np.arange(1, n_spots+1)

    # Get the center of each thresholded object
    # by using the center of mass, either weighting
    # each pixel uniformly (`binary`) or by the 
    # intensity in that pixel
    if weight == 'binary':
        com = ndi.center_of_mass(binary, labels=spots,
            index=spot_index)
    elif weight == 'intensity':
        com = ndi.center_of_mass(image, labels=spots,
            index=spot_index)

    com = np.asarray(com)

    # Calculate the intensity of each spot
    spot_sums = ndi.sum(image, labels=spots, index=spot_index)

    result = np.concatenate((com, np.array([spot_sum]).T), axis=1)
    return result, ['y', 'x', 'intensity']


    

