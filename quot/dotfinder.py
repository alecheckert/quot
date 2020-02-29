"""
dotfinder.py -- main function for 
dot-finding in an image

"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

METHODS = {}

def find_dots(image, method, **kwargs):
    """
    args
    ----
        image :  2D ndarray
        method :  str
        **kwargs :  to methods.method

    returns
    -------
        pandas.DataFrame, the detections

    """
    # Check to see if this method is implemented 
    if method not in METHODS.keys():
        raise RuntimeError("quot.dotfinder.find_dots: " \
            "method '%s' must be one of %s" % \
                ', '.join(METHODS.keys()))

    # Get the corresponding function 
    f = METHODS[method]

    # Run on the image
    par_values, par_names = f(image, **kwargs)

    # Format output as pandas.DataFrame
    df = pd.DataFrame(par_values, columns=par_names)

    return df








