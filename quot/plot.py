#!/usr/bin/env python
"""
plot.py -- simple plotting utilities for quot

"""
# Numeric
import numpy as np 

# Core matplotlib library
import matplotlib.pyplot as plt 

def imshow(*imgs, vmax_mod=1.0):
    """
    Show a variable number of images side-by-side in a 
    temporary window.

    args
    ----
        imgs        :   2D ndarrays
        vmax_mod    :   float, modifier for the
                        white point as a fraction of 
                        max intensity

    """
    n = len(imgs)
    if n == 1:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(imgs[0], cmap='gray', vmax=vmax_mod*imgs[0].max())
    else:
        fig, ax = plt.subplots(1, n, figsize=(3*n, 3))
        for j in range(n):
            ax[j].imshow(imgs[j], cmap='gray', vmax=vmax_mod*imgs[j].max())
    plt.show(); plt.close()

    