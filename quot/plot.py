"""
plot.py

"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os

def overlay_dots(image, dot_df, plot=False):
    """
    Visualize dots by overlaying onto 
    the raw image, returning the overlay.

    args
    ----
        image :  2D ndarray
        dot_df :  pandas.DataFrame

    returns
    -------
        2D ndarray, image with overlay

    """
    result = image.copy()
    yx = np.asarray(dot_df[['y', 'x']])
    for y, x in yx:
        for c in range(-3, 4):
            result[y+c, x] = result.max()
            result[y, x+c] = result.max()
    return result 

