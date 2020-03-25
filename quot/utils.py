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









